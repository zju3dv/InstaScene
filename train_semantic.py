import os.path
from argparse import ArgumentParser
from random import randint

from arguments import ModelParams, PipelineParams, OptimizationParams
from spatial_track.spatialtrack import GausCluster
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.contrastive_utils import *
from utils.general_mesh_utils import *

from tqdm import tqdm

from vis_utils.color_utils import generate_semantic_colors


class SegSplatting:
    def __init__(self, modelparams: ModelParams, optimparams: OptimizationParams, pipelineparams: PipelineParams):
        self.modelparams = modelparams
        self.data_dir = modelparams.source_path
        self.optimparams = optimparams
        self.pipelineparams = pipelineparams
        # gaussian
        self.gaussians = GaussianModel(sh_degree=3)
        self.gaussians.pipelineparams = pipelineparams
        self.gaussians.set_segfeat_params(modelparams)
        self.gaussians.load_ply(os.path.join(self.data_dir, 'point_cloud.ply'))

        self.model_path = os.path.join("output", self.modelparams.source_path.split("/")[-2],
                                       self.modelparams.source_path.split("/")[-1],
                                       self.modelparams.model_path)

    @torch.no_grad()
    def RobustSemanticPriors(self):
        print("\033[91mRunning Mask Clustering with Spatial Gaussian Tracker... \033[0m")

        if os.path.exists(self.modelparams.preload_robust_semantic):
            segment_save_dir = self.modelparams.preload_robust_semantic
        else:
            segment_save_dir = os.path.join(self.model_path, f"semantic_association")
            os.makedirs(segment_save_dir, exist_ok=True)

        scene = Scene(self.modelparams, self.gaussians, loaded_gaussian=True)
        viewpoint_stack = scene.getTrainCameras().copy()
        self.gausclustering = GausCluster(self.gaussians, viewpoint_stack)

        if not os.path.exists(os.path.join(segment_save_dir, "output_dict.npy")):
            self.gausclustering.maskclustering(segment_save_dir)  # TODO: cluster init in SFM ?

        self.robust_semantic_priors = np.load(os.path.join(segment_save_dir, "output_dict.npy"),
                                              allow_pickle=True).item()
        self.Seg3D_masks = self.robust_semantic_priors["mask_3d_labels"]
        self.Seg3D_labels = torch.argmax(torch.tensor(self.Seg3D_masks, dtype=torch.int16), dim=1).cuda()

        self.Seg2D_masks = self.robust_semantic_priors['mask_2d_clusters']
        if not os.path.exists(os.path.join(self.data_dir, "sam/mask_sorted")):
            self.gausclustering.rearrange_mask(os.path.join(self.data_dir, "sam/mask"), self.Seg2D_masks)

        # Filter undersegment
        self.undersegment_masks = self.robust_semantic_priors["underseg_mask_ids"]
        ## TODO: fix error undersegment -> 3D mask tracking
        if not os.path.exists(os.path.join(self.data_dir, "sam/mask_filtered")):
            self.gausclustering.filter_undersegment_mask(os.path.join(self.data_dir, "sam/mask"),
                                                         self.undersegment_masks)

        self.scene = Scene(self.modelparams, self.gaussians, loaded_gaussian=True)

        self.gaussians.set_3d_feat(self.Seg3D_masks, gram_feat=self.optimparams.gram_feat_3d)

    ################################## Train Segment Feature ##################################
    def train_segfeat(self):
        print("\n\033[91mRunning Spatial Contrastive Learning... \033[0m")

        if os.path.exists(
                os.path.join(self.model_path, "point_cloud/iteration_{}".format(self.optimparams.iterations))):
            return

        self.gaussians.training_setup(self.optimparams)

        bg_color = [1, 1, 1] if self.modelparams.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        first_iter = 0
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        viewpoint_stack = None
        progress_bar = tqdm(range(first_iter, self.optimparams.iterations), desc="Training progress")
        first_iter += 1

        for iteration in range(first_iter, self.optimparams.iterations + 1):
            iter_start.record()

            if not viewpoint_stack:
                viewpoint_stack = self.scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

            render_pkg = render(viewpoint_cam, self.gaussians, self.pipelineparams, background)

            image, seg_feature, viewspace_point_tensor, visibility_filter, radii = \
                render_pkg["render"], render_pkg["seg_feature"], render_pkg["viewspace_points"], \
                    render_pkg["visibility_filter"], render_pkg["radii"]

            singleview_contra_loss = 0
            mask_type_cnts = 0
            if self.gaussians.class_feat is not None:
                segmap_lists = [viewpoint_cam.segmap.squeeze().cuda(), viewpoint_cam.sorted_segmap.squeeze().cuda()]
            else:
                segmap_lists = [viewpoint_cam.segmap.squeeze().cuda()]  # mask with undersegment filter
            for gt_segmap in segmap_lists:
                batchsize = self.optimparams.sample_batchsize
                valid_labels_mask = gt_segmap > 0
                # ! Note: May all invalid
                if valid_labels_mask.sum() > 0:
                    valid_seg_feature = seg_feature[:, valid_labels_mask]
                    valid_seg_labels = gt_segmap[valid_labels_mask]
                    sampled_idx = torch.randint(0, len(valid_seg_labels), size=(batchsize,), device="cuda")

                    sampled_segfeat = valid_seg_feature[:, sampled_idx].T
                    sampled_labels = valid_seg_labels[sampled_idx]

                    single_view_weight = 1 if mask_type_cnts == 1 else 0.5  # mv with larger weight

                    singleview_contra_loss = singleview_contra_loss + contrastive_loss(
                        sampled_segfeat, sampled_labels,
                        # mv can use pre-defined feat
                        predef_u_list=self.gaussians.class_feat if mask_type_cnts == 1 else None,
                    ) * self.optimparams.lambda_singview_contras * single_view_weight
                else:
                    print("Invalid View: ", viewpoint_cam.image_name)
                mask_type_cnts += 1

            # cross-view contrastive learning
            multiview_contra_loss = 0
            if self.optimparams.lambda_multiview_contras > 0 and iteration % 10 == 0:
                num_sample_views = self.optimparams.sample_mv_frames
                sampled_views = self.scene.getTrainCameras()
                sampled_view_id = np.random.randint(0, len(sampled_views) - num_sample_views)
                sampled_views = [sampled_views[view_idx] for view_idx in
                                 range(sampled_view_id, sampled_view_id + num_sample_views)]  # or can using match pair
                seg_feature_list = []
                seg_labels_list = []
                for sample_view in sampled_views:
                    render_pkg = render(sample_view, self.gaussians, self.pipelineparams, background)
                    seg_feature = render_pkg["seg_feature"]
                    seg_feature_list.append(seg_feature)
                    seg_labels_list.append(sample_view.sorted_segmap.cuda())
                seg_feature_list = torch.stack(seg_feature_list, dim=0)
                seg_labels_list = torch.stack(seg_labels_list, dim=0).squeeze()

                batchsize = self.optimparams.sample_batchsize
                # 筛选出大于0的
                valid_labels_mask = seg_labels_list > 0
                valid_seg_feature = seg_feature_list.permute(1, 0, 2, 3)[:, valid_labels_mask]
                valid_seg_labels = seg_labels_list[valid_labels_mask]
                sampled_idx = torch.randint(0, len(valid_seg_labels), size=(batchsize,), device="cuda")
                sampled_segfeat = valid_seg_feature[:, sampled_idx].T
                sampled_labels = valid_seg_labels[sampled_idx]
                multiview_contra_loss = contrastive_loss(sampled_segfeat,
                                                         sampled_labels,
                                                         predef_u_list=self.gaussians.class_feat
                                                         ) * self.optimparams.lambda_multiview_contras

            # 3D contrastive learning
            visibility_segfeat = self.gaussians.get_seg_feature[visibility_filter]
            visibility_labels_3d = self.Seg3D_labels[visibility_filter]

            contra_3d_loss = 0
            if self.optimparams.lambda_3D_contras > 0:
                # Global supervision
                batchsize_3d = self.optimparams.sample_batchsize

                valid_labels_mask = visibility_labels_3d > 0
                if valid_labels_mask.sum() > 0:
                    valid_seg_feature = visibility_segfeat[valid_labels_mask]
                    valid_seg_labels = visibility_labels_3d[valid_labels_mask]

                    sampled_idx = torch.randint(0, len(valid_seg_labels), size=(batchsize_3d,), device="cuda")
                    sampled_segfeat_3d = valid_seg_feature[sampled_idx]
                    sampled_labels_3d = valid_seg_labels[sampled_idx]

                    contra_3d_loss = contrastive_loss(sampled_segfeat_3d,
                                                      sampled_labels_3d,
                                                      predef_u_list=self.gaussians.class_feat
                                                      ) * self.optimparams.lambda_3D_contras
                else:
                    print("Invalid View: ", viewpoint_cam.image_name)

            total_loss = singleview_contra_loss + \
                         multiview_contra_loss + \
                         contra_3d_loss

            total_loss.backward()
            iter_end.record()

            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            with torch.no_grad():
                # Progress bar
                if iteration % 10 == 0:
                    loss_dict = {
                        "SV_ContraLoss": f"{singleview_contra_loss:.{3}f}",
                        "MV_ContraLoss": f"{multiview_contra_loss:.{3}f}",
                        "3D_ContraLoss": f"{contra_3d_loss:.{3}f}"
                    }
                    progress_bar.set_postfix(loss_dict)
                    progress_bar.update(10)

                if iteration % 200 == 0:
                    viewpoint = self.scene.getTrainCameras()[0]
                    render_pkg = render(viewpoint, self.gaussians, self.pipelineparams, background)

                    _, seg_feature, _, _, _ = render_pkg["render"], render_pkg["seg_feature"], \
                        render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    os.makedirs(self.scene.model_path, exist_ok=True)
                    Image.fromarray(feature_to_rgb(seg_feature)).save(f"{self.scene.model_path}/{iteration}_feat.png")

                if iteration % 2500 == 0:
                    self.scene.save(iteration)
                    self.export_segment_results(iteration)

                if iteration == self.optimparams.iterations:
                    progress_bar.close()

        self.export_segment_results(iteration, use_hdbscan=True, note=f"iteration_{iteration}_hdbscan")

    @torch.no_grad()
    def export_segment_results(self, iteration, score_threshold=0.9, use_hdbscan=False, note=None):
        if note is None:
            save_dir = os.path.join(self.model_path, f"point_cloud/iteration_{iteration}")
        else:
            save_dir = os.path.join(self.model_path, f"point_cloud/{note}")
        os.makedirs(save_dir, exist_ok=True)
        save_partial_dir = os.path.join(save_dir, "label_pointclouds")
        os.makedirs(save_partial_dir, exist_ok=True)
        if use_hdbscan:
            # Following Garfield https://github.com/chungmin99/garfield
            from cuml.cluster.hdbscan import HDBSCAN
            from sklearn.neighbors import NearestNeighbors

            positions = self.gaussians.get_xyz.detach()
            group_feats = self.gaussians.get_seg_feature.cpu().numpy()  # (N, 32)
            positions = positions.cpu().numpy()

            vec_o3d = o3d.utility.Vector3dVector(positions)
            pc_o3d = o3d.geometry.PointCloud(vec_o3d)
            min_bound = np.clip(pc_o3d.get_min_bound(), -1, 1)
            max_bound = np.clip(pc_o3d.get_max_bound(), -1, 1)

            downsample_size = 0.01
            pc, _, ids = pc_o3d.voxel_down_sample_and_trace(max(downsample_size, 0.0001), min_bound, max_bound)
            while len(pc.points) > 1000000:
                downsample_size = downsample_size * 2
                pc, _, ids = pc_o3d.voxel_down_sample_and_trace(max(downsample_size, 0.0001), min_bound, max_bound)

            id_vec = np.array([points[0] for points in ids])  # indices of gaussians kept after downsampling
            group_feats_downsampled = group_feats[id_vec]
            positions_downsampled = np.array(pc.points)

            print(f"HDBScan for {group_feats_downsampled.shape[0]} gaussians... ", end="", flush=True)

            # Run cuml-based HDBSCAN
            clusterer = HDBSCAN(cluster_selection_epsilon=0.1,
                                min_samples=30,
                                min_cluster_size=30,
                                allow_single_cluster=True).fit(group_feats_downsampled)

            non_clustered = np.ones(positions.shape[0], dtype=bool)
            non_clustered[id_vec] = False
            labels = clusterer.labels_.copy()
            clusterer.labels_ = -np.ones(positions.shape[0], dtype=np.int32)
            clusterer.labels_[id_vec] = labels

            # Assign the full gaussians to the spatially closest downsampled gaussian, with scipy NearestNeighbors.
            positions_np = positions[non_clustered]
            if positions_np.shape[0] > 0:  # i.e., if there were points removed during downsampling
                k = 1
                nn_model = NearestNeighbors(
                    n_neighbors=k, algorithm="auto", metric="euclidean"
                ).fit(positions_downsampled)
                _, indices = nn_model.kneighbors(positions_np)
                clusterer.labels_[non_clustered] = labels[indices[:, 0]]

            labels = clusterer.labels_

            noise_mask = labels == -1
            if noise_mask.sum() != 0 and (labels >= 0).sum() > 0:
                # if there is noise, but not all of it is noise, relabel the noise
                valid_mask = labels >= 0
                valid_positions = positions[valid_mask]
                k = 1
                nn_model = NearestNeighbors(
                    n_neighbors=k, algorithm="auto", metric="euclidean"
                ).fit(valid_positions)
                noise_positions = positions[noise_mask]
                _, indices = nn_model.kneighbors(noise_positions)
                # for now just pick the closest cluster
                noise_relabels = labels[valid_mask][indices[:, 0]]
                labels[noise_mask] = noise_relabels
                clusterer.labels_ = labels

            labels = clusterer.labels_

            pclds = None
            instance_colors = generate_semantic_colors(len(np.unique(labels)))

            for label in np.unique(labels):
                pcld = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(positions[labels == label]))
                pcld.paint_uniform_color(instance_colors[label])
                pclds = pcld if pclds is None else pcld + pclds
                o3d.io.write_point_cloud(os.path.join(save_partial_dir, f"{label}.ply"), pcld)

            o3d.io.write_point_cloud(os.path.join(save_dir, "point_cloud_labels.ply"), pclds)
            np.save(os.path.join(save_dir, "point_cloud_labels.npy"), labels)

        else:
            # Use Coarse 3D Mask to get instance pointcloud
            instance_colors = generate_semantic_colors(self.Seg3D_masks.shape[1])

            scene_pclds = self.gaussians.get_xyz.detach().cpu().numpy()
            pclds = None
            global_feat = self.gaussians.get_seg_feature.cpu()
            for sampled_3d_labels in range(self.Seg3D_masks.shape[1]):
                selected_pseudo_3d_feat = self.gaussians.get_seg_feature[self.Seg3D_masks[:, sampled_3d_labels]]
                selected_pseudo_3d_feat_mean = selected_pseudo_3d_feat.mean(0).cpu()
                feat_score = selected_pseudo_3d_feat_mean @ global_feat.T
                selected_points_mask = (feat_score >= score_threshold)

                if selected_points_mask.sum() == 0:
                    selected_points_mask = (self.Seg3D_labels == sampled_3d_labels).cpu()

                pcld = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene_pclds[selected_points_mask.numpy()]))
                color = instance_colors[sampled_3d_labels]
                pcld.paint_uniform_color(color)
                pclds = pcld if pclds is None else pclds + pcld
                o3d.io.write_point_cloud(os.path.join(save_partial_dir, f"{sampled_3d_labels}.ply"), pcld)

            o3d.io.write_point_cloud(os.path.join(save_dir, "point_cloud_labels.ply"), pclds)

    @torch.no_grad()
    def render_views(self, save_mask=False, view_idx=[]):
        save_dir = os.path.join(self.scene.model_path, "render")
        os.makedirs(save_dir, exist_ok=True)
        for folder in ["segfeat", "segmask"]:
            os.makedirs(os.path.join(save_dir, folder), exist_ok=True)

        bg_color = [1, 1, 1] if self.modelparams.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if save_mask:
            instance_feats = []
            instance_colors = []
            for sampled_3d_labels in range(self.Seg3D_masks.shape[1]):
                selected_pseudo_3d_feat = self.gaussians.get_seg_feature[self.Seg3D_masks[:, sampled_3d_labels]]
                instance_feats.append(selected_pseudo_3d_feat.mean(0))
                instance_colors.append(np.random.rand(3))

            instance_feats = torch.stack(instance_feats, dim=0)
            instance_colors = np.stack(instance_colors, axis=0) * 0.7 + 0.3

        viewpoints = self.scene.getTrainCameras() if view_idx == [] else \
            [self.scene.getTrainCameras()[idx] for idx in view_idx]

        for viewpoint_view in tqdm(viewpoints):
            if os.path.exists(f"{save_dir}/segfeat/{viewpoint_view.image_name}.npy"):
                seg_feature = torch.from_numpy(
                    np.load(f"{save_dir}/segfeat/{viewpoint_view.image_name}.npy")).float().cuda().permute(2, 0, 1)
            else:
                render_result = render(viewpoint_view, self.gaussians, self.pipelineparams, background)
                seg_feature = render_result["seg_feature"]
                np.save(f"{save_dir}/segfeat/{viewpoint_view.image_name}.npy",
                        seg_feature.permute(1, 2, 0).cpu().numpy())

            # pca
            if not hasattr(self, "pca_proj_mat"):
                def pca(X, n_components=3):
                    n = X.shape[0]
                    mean = torch.mean(X, dim=0)
                    X = X - mean
                    covariance_matrix = (1 / n) * torch.matmul(X.T, X).float()
                    # An old torch bug: matmul float32->float16,
                    eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix)
                    idx = torch.argsort(-eigenvalues.type(torch.float32))
                    eigenvectors = eigenvectors[:, idx]
                    proj_mat = eigenvectors[:, 0:n_components]

                    return proj_mat.type(torch.float32)

                sems = self.gaussians._seg_feature.clone().squeeze()
                N, C = sems.shape
                torch.manual_seed(0)
                randint = torch.randint(0, N, [200_000])
                sems /= (torch.norm(sems, dim=1, keepdim=True) + 1e-6)
                sem_chosen = sems[randint, :]
                self.pca_proj_mat = pca(sem_chosen, n_components=3)

            Image.fromarray(feature_to_rgb(seg_feature, self.pca_proj_mat)).save(
                f"{save_dir}/segfeat/{viewpoint_view.image_name}.png")

            # segmask
            if save_mask:
                seg_feature = seg_feature / seg_feature.norm(dim=0, keepdim=True)
                instance_feat_score = seg_feature.permute(1, 2, 0) @ instance_feats.permute(1, 0)
                instance_feat_score = instance_feat_score.reshape(-1, instance_feat_score.shape[-1])
                seg_feature_instance = instance_feat_score.argmax(-1)
                seg_feature_instance[instance_feat_score[
                                         torch.arange(len(instance_feat_score)), instance_feat_score.argmax(
                                             -1)] < 0.75] = 0

                seg_mask_colormap = instance_colors[seg_feature_instance.cpu().numpy()]
                seg_mask_colormap[seg_feature_instance.cpu().numpy() == 0] = 0

                Image.fromarray(
                    np.uint8(255.0 * seg_mask_colormap.reshape(seg_feature.shape[1], seg_feature.shape[2], 3))
                ).save(f"{save_dir}/segmask/{viewpoint_view.image_name}.png")

            torch.cuda.empty_cache()


if __name__ == "__main__":
    # python train_semantic.py -s data/3dovs/bed -m train_semanticgs --use_seg_feature --iterations 10000 --load_filter_segmap
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])

    segsplat = SegSplatting(lp.extract(args), op.extract(args), pp.extract(args))
    segsplat.args = args
    segsplat.RobustSemanticPriors()
    segsplat.train_segfeat()
    print("\nTraining complete.")

    '''
    python train_semantic.py -s data/360_v2/counter/ -m train_semanticgs_grad \
        --use_seg_feature --iterations 10000 --load_filter_segmap \
        --preload_robust_semantic output/360_v2/counter/train_semanticgs/semantic_association/ \
        --gram_feat_3d \
    '''
