import os.path
from argparse import ArgumentParser
from random import randint

from arguments import ModelParams, PipelineParams, OptimizationParams
from spatialtrack.spatialtrack import GraphClustering
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.contrastive_utils import *
from utils.general_mesh_utils import *
from utils.graphics_utils import focal2fov
from utils.image_utils import crop_image
from utils.visual_instance_utils import *


class BackgroundRemoval:
    def __init__(self, device='cuda'):
        from carvekit.api.high import HiInterface
        self.interface = HiInterface(
            object_type="object",  # Can be "object" or "hairs-like".
            batch_size_seg=5,
            batch_size_matting=1,
            device=device,
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=231,
            trimap_dilation=30,
            trimap_erosion_iters=5,
            fp16=True,
        )

    @torch.no_grad()
    def __call__(self, image):
        # image: [H, W, 3] array in [0, 255].
        image = Image.fromarray(image)
        image = self.interface([image])[0]
        image = np.array(image)
        return image


class SegSplatting:
    def __init__(self, modelparams: ModelParams, optimparams: OptimizationParams, pipelineparams: PipelineParams):
        self.modelparams = modelparams
        self.data_dir = modelparams.source_path
        self.optimparams = optimparams
        self.pipelineparams = pipelineparams
        # gaussian
        self.gaussians = GaussianModel(sh_degree=3)
        self.gaussians.pipelineparams = pipelineparams
        self.gaussians.set_segfeat_params(modelparams)  # 设置超参use_seg_feature seg_feat_dim load_seg_feat
        self.gaussians.load_ply(os.path.join(self.data_dir, 'point_cloud.ply'))

        self.model_path = os.path.join("output", self.modelparams.source_path.split("/")[-2],
                                       self.modelparams.source_path.split("/")[-1],
                                       self.modelparams.model_path)

    @torch.no_grad()
    def GraphBasedClustering(self):
        segment_save_dir = os.path.join(self.model_path, f"semantic_association")
        os.makedirs(segment_save_dir, exist_ok=True)

        rearange_not = False
        if not os.path.exists(os.path.join(segment_save_dir, "mask3d.npz")):
            rearange_not = True
            scene = Scene(self.modelparams, self.gaussians)
            viewpoint_stack = scene.getTrainCameras().copy()

            self.graphclustering = GraphClustering(self.gaussians, viewpoint_stack)
            self.graphclustering.graphclustering()  # TODO: 假如我在SFM的时候就开始聚类？->可能有些点是空的
            self.graphclustering.export(segment_save_dir)

        self.clusteringSeg3D_masks = \
        np.load(os.path.join(segment_save_dir, "output/object/mask3d.npz"), allow_pickle=True)["pred_masks"]
        '''
        pclds = []
        for i in range(self.clusteringSeg3D_masks.shape[1])[1:]:
            pcld = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
                self.gaussians.get_xyz[self.clusteringSeg3D_masks[:, i]].detach().cpu().numpy()))
            color = np.random.rand(3) * 0.7 + 0.3
            pcld.paint_uniform_color(color)
            pclds.append(pcld)
        o3d.visualization.draw_geometries(pclds)
        '''
        # 将点少的筛掉
        # TODO: 0有可能是第一个instance!
        self.clusteringSeg3D_masks, self.valid_3d_labels = filter_3d_mask(self.clusteringSeg3D_masks)
        self.clusteringSeg3D_labels = torch.argmax(torch.tensor(self.clusteringSeg3D_masks, dtype=torch.int16),
                                                   dim=1).cuda()  # 转换为label

        # TODO：每个点云不止一个label
        self.clusteringSeg2D_masks = np.load(os.path.join(render_cache_dir, "output/object/object_dict.npy"),
                                             allow_pickle=True).item()  # 没有underseg_mask
        # 将无效的3D分割结果在2D分割结果中进行筛除
        # TODO: 会有一些mask重复!
        self.clusteringSeg2D_masks = update_2d_mask(self.clusteringSeg2D_masks, self.valid_3d_labels)
        # filter_2d_mask()

        # 更新segmap_sortedmap
        if rearange_not or not os.path.exists(os.path.join(self.data_dir, "maskclustering/mask_sorted")):
            rearange_mask(os.path.join(self.data_dir, "maskclustering/mask"), self.clusteringSeg2D_masks)
        self.scene = Scene(self.modelparams, self.gaussians)

        self.undersegment_masks = []
        if os.path.exists(os.path.join(render_cache_dir, "output/object/underseg_masks.npy")):
            self.undersegment_masks = np.load(os.path.join(render_cache_dir, "output/object/underseg_masks.npy"))
        # save_undersegmask(self.undersegment_masks, self.scene.getTrainCameras(),save_dir=os.path.join(self.data_dir, "maskclustering/mask_undersegment"))
        # TODO: 如果segmap和分割结果没有任何交集，则将其视为invalid
        if True:
            if os.path.exists(os.path.join(render_cache_dir, "output/object/invalid_masks.npy")):
                self.undersegment_masks = np.load(os.path.join(render_cache_dir, "output/object/invalid_masks.npy"))
                if os.path.exists(os.path.join(render_cache_dir, f'output/object/mask_instance_match.npz')):
                    self.mask_instance_match_list = np.load(
                        os.path.join(render_cache_dir, f'output/object/mask_instance_match.npz'), allow_pickle=True)
            else:
                # 增强undersegment_mask
                self.undersegment_masks, self.mask_instance_match_list = filter_invalid_mask_segmap(
                    self.clusteringSeg3D_masks, self.gaussians,
                    self.scene.getTrainCameras(), render_func=render,
                    save_dir=os.path.join(self.data_dir, "maskclustering/mask_filter"))

                np.savez(os.path.join(render_cache_dir, f'output/object/mask_instance_match'),
                         **self.mask_instance_match_list)
                np.save(os.path.join(render_cache_dir, "output/object/invalid_masks.npy"), self.undersegment_masks)

        # save_enhance_instance_mask(self.mask_instance_match_list, self.scene.getTrainCameras(),os.path.join(self.data_dir, "maskclustering/mask_enhance_sorted"))

        self.gaussians.set_clustering3d_feat(self.clusteringSeg3D_masks, gram_feat=self.optimparams.add_3d_prior)
        if not self.gaussians.load_seg_feat:
            if not os.path.exists(os.path.join(self.data_dir, "point_cloud_only3d.ply")):
                self.gaussians.save_ply(os.path.join(self.data_dir, "point_cloud_only3d.ply"))
        '''
        color_path = os.path.join(self.render_cache_dir, "output/object/instance_colors.npy")
        if os.path.exists(color_path):
            instance_colors = np.load(color_path)
        pclds = None
        self.clusteringSeg3D_masks_filter = np.zeros_like(self.clusteringSeg3D_masks)
        for instance_id, mask3d in tqdm(enumerate(self.clusteringSeg3D_masks.T)):
            pcld = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.gaussians.get_xyz.cpu().numpy()[mask3d]))
            # cl, ind = pcld.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
            # pcld = pcld.select_by_index(ind)
            labels = np.array(pcld.cluster_dbscan(eps=0.1, min_points=30)) + 1  # -1 for noise
            label_lists, labels_cnts = np.unique(labels, return_counts=True)
            suitable_label = label_lists[labels_cnts.argsort()[-1]]
            ind = np.arange(len(labels))[labels == suitable_label]
            pcld = pcld.select_by_index(ind)
            pcld.paint_uniform_color(instance_colors[instance_id])
            pclds = pcld if pclds is None else pclds + pcld

            self.clusteringSeg3D_masks_filter[
                np.arange(len(self.clusteringSeg3D_masks))[mask3d][np.array(ind)], instance_id] = True
        o3d.io.write_point_cloud(
            "/home/bytedance/Projects/3DGS/FunctionSplatting++/2d-gaussian-splatting/data/ZipNeRF/alameda/meetingroom/segment.ply",
            pclds)
        self.gaussians.set_clustering3d_feat(self.clusteringSeg3D_masks_filter, gram_feat=self.optimparams.add_3d_prior)
        if not self.gaussians.load_seg_feat:
            self.gaussians.save_ply(os.path.join(self.data_dir, "point_cloud_only3d.ply"))
        '''

    ################################## Train Segment Feature ##################################
    def train_segfeat(self):
        # self.scene.save(0)
        # self.export_segment_results(0)
        # 过滤掉undersegment的数据

        '''
        Ablation:
            * only 2D noisy prior
            * only 3D global prior
            * add filter 2D noisy prior
            * add mv 2D prior
        Returns:

        '''
        # 是否过滤无效mask
        if self.optimparams.add_filter_2d:
            for frame_id, mask_id in self.undersegment_masks:  # 欠分割的过滤掉
                segmap = self.scene.train_cameras[1.0][frame_id].segmap
                segmap[segmap == mask_id] = 0

        # self.scene.save_segmap(os.path.join(self.data_dir, "maskclustering/mask_filter"))

        # clusteringSeg2D_mask也要过滤

        # 还要更新train_cameras

        '''
        for image_id in range(len(self.scene.getTrainCameras()))[::5]:
            origin_image = np.uint8(
                (self.scene.train_cameras[1.0][image_id].original_image.permute(1, 2, 0).numpy() * 255.0))
            for frame_id, mask_id in self.undersegment_masks:  # 欠分割的过滤掉
                if frame_id == image_id:
                    segmap = self.scene.train_cameras[1.0][frame_id].segmap
                    segmap_mask = segmap == mask_id
                    segmap_mask = segmap_mask.squeeze().numpy()
                    origin_image[segmap_mask] = np.uint8(
                        origin_image[segmap_mask] * 0.3 + np.random.random(3) * 255.0 * 0.7)
            Image.fromarray(origin_image).show()
        '''
        # 初始化feature
        self.gaussians.training_setup(self.optimparams)

        ### 可视化3d mask和2d mask是否一致
        '''
        save_dir = os.path.join(self.scene.model_path, "label_pointclouds")
        os.makedirs(save_dir, exist_ok=True)
        segmaps = torch.stack([viewpoint_cam.sorted_segmap.squeeze() for viewpoint_cam in self.scene.getTrainCameras()])
        images = torch.stack([viewpoint_cam.original_image.squeeze() for viewpoint_cam in self.scene.getTrainCameras()])
        for class_id in tqdm(range(self.clusteringSeg3D_masks.shape[1])[1:]):
            instance_savedir = os.path.join(save_dir, f"{class_id}")
            os.makedirs(instance_savedir, exist_ok=True)
            pcld = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
                self.gaussians.get_xyz[self.clusteringSeg3D_masks[:, class_id]].cpu().numpy()))
            color = np.random.rand(3) * 0.7 + 0.3
            pcld.paint_uniform_color(color)
            o3d.io.write_point_cloud(os.path.join(instance_savedir, f"{class_id}.ply"), pcld)

            for frame_idx, segmap in tqdm(enumerate(segmaps)):
                if (segmap == class_id).any():
                    segmap_mask = segmap == class_id
                    image = images[frame_idx].permute(1, 2, 0).numpy()
                    image_tmp = image.copy() * 0.5
                    image_tmp[segmap_mask] = image[segmap_mask] * 0.2 + color * 0.8
                    Image.fromarray(np.uint8(255 * image_tmp)).save(os.path.join(instance_savedir, f"{frame_idx}.jpg"))
        '''
        # 去掉重复的mask->已经训了一段时间了
        if False:  # self.gaussians.load_seg_feat:
            self.clusteringSeg3D_masks = filter_repeat_gs_labels(self.clusteringSeg3D_masks,
                                                                 self.gaussians.get_seg_feature,
                                                                 self.gaussians.get_xyz)
            self.clusteringSeg3D_labels = torch.argmax(torch.tensor(self.clusteringSeg3D_masks, dtype=torch.int16),
                                                       dim=1).cuda()  # 转换为label

        # groupingloss
        if self.modelparams.use_grouping_loss:
            num_classes = len(self.clusteringSeg2D_masks)
            classifier = torch.nn.Conv2d(self.gaussians.seg_feat_dim, num_classes, kernel_size=1)
            cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
            cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4)
            classifier.cuda()

        bg_color = [1, 1, 1] if self.modelparams.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        first_iter = 0
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        viewpoint_stack = None
        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(first_iter, self.optimparams.iterations), desc="Training progress")
        first_iter += 1

        contrast_batchsize = self.optimparams.sample_batchsize  # 8 * 1024  # 32*1024

        for iteration in range(first_iter, self.optimparams.iterations + 1):
            iter_start.record()

            if not viewpoint_stack:
                viewpoint_stack = self.scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

            render_pkg = render(viewpoint_cam, self.gaussians, self.pipelineparams, background)
            # 3d feat已经被归一化过
            image, seg_feature, viewspace_point_tensor, visibility_filter, radii = \
                render_pkg["render"], render_pkg["seg_feature"], render_pkg["viewspace_points"], \
                    render_pkg["visibility_filter"], render_pkg["radii"]
            '''
            from PIL import Image
            import numpy as np
            Image.fromarray(np.uint8(image.clamp(0, 1).permute(1, 2, 0).cpu().detach().numpy() * 255.0)).show()
            '''
            singleview_contra_loss = 0
            feat_norm_loss = 0
            multiview_contra_loss = 0
            contra_3d_loss = 0
            consist_3d_loss = 0
            clustering3D_filter_loss = 0

            # TODO: 问题在于，如何解决没有mask的区域？
            # BUG: 有可能全部都取到了0对应的mask
            if not self.modelparams.only_3d_supervised:
                # TODO: 如果sorted_map
                count = 0  # 0: sorted_segmap; 1: segmap
                segmap_lists = [viewpoint_cam.segmap.squeeze().cuda()]
                if self.optimparams.add_mv_2d:  # 多视图一致的segmap
                    segmap_lists.append(viewpoint_cam.sorted_segmap.squeeze().cuda())
                for gt_segmap in segmap_lists:
                    # gt_segmap = viewpoint_cam.sorted_segmap.squeeze().cuda()  # viewpoint_cam.segmap.squeeze().cuda()
                    # TODO: 直接将gt_segmap换成sorted
                    # 1. Contrastive Learning 2D
                    # x = torch.randint(0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device)
                    batchsize = contrast_batchsize  # 32 * 1024

                    # 直接选取有效的feat和label
                    ## 如果是mvseg，不考虑0，single_view_seg考虑0
                    if count == 1 or gt_segmap.sum() == 0:  # 只考虑有效mask
                        valid_labels_mask = gt_segmap > 0
                    else:  # 把负样本加入
                        ## 排除全为0的情况
                        valid_labels_mask = torch.ones_like(gt_segmap, dtype=torch.bool).cuda()

                    # ! Note: 有可能全为0
                    if valid_labels_mask.sum() > 0:
                        valid_seg_feature = seg_feature[:, valid_labels_mask]
                        valid_seg_labels = gt_segmap[valid_labels_mask]
                        sampled_idx = torch.randint(0, len(valid_seg_labels), size=(batchsize,), device="cuda")

                        sampled_segfeat = valid_seg_feature[:, sampled_idx].T
                        sampled_labels = valid_seg_labels[sampled_idx]
                        '''
                        sampled_y = torch.randint(0, viewpoint_cam.image_height, size=(batchsize,), device="cuda")
                        sampled_x = torch.randint(0, viewpoint_cam.image_width, size=(batchsize,), device="cuda")
                        # importance_sampling(gt_segmap, 10000)
                        sampled_segfeat = seg_feature[:, sampled_y, sampled_x].T
                        sampled_labels = gt_segmap[sampled_y, sampled_x]
                        '''
                        single_view_weight = 1 if count == 1 else 0.5  # mv权重更大
                        singleview_contra_loss = singleview_contra_loss + contrastive_loss(
                            sampled_segfeat, sampled_labels,
                            predef_u_list=self.gaussians.class_feat if count == 1 else None,  # mv的时候才考虑，用预先定义好的feat
                            consider_negative=count == 0  # segmap要考虑0
                        ) * self.optimparams.lambda_singview_contras * single_view_weight
                    else:
                        print("Invalid View: ", viewpoint_cam.image_name)
                    count += 1

                # 2. grouping loss
                if self.optimparams.lambda_multiview_contras > 0 and self.optimparams.add_mv_2d:
                    if self.modelparams.use_grouping_loss:  # gsgrouping
                        gt_sorted_segmap = viewpoint_cam.sorted_segmap.squeeze().cuda().long()
                        valid_mask = gt_sorted_segmap > 0  # 不考虑0
                        logits = classifier(seg_feature)
                        multiview_contra_loss = cls_criterion(logits[:, valid_mask].unsqueeze(0),
                                                              gt_sorted_segmap[valid_mask].unsqueeze(0)
                                                              ).squeeze().mean() / torch.log(torch.tensor(num_classes))
                        multiview_contra_loss = multiview_contra_loss * self.optimparams.lambda_multiview_contras
                    else:
                        if iteration % 10 == 0:
                            # 随机渲染5个view
                            num_sample_views = self.optimparams.sample_mv_frames
                            sampled_views = self.scene.getTrainCameras()
                            sampled_view_id = np.random.randint(0, len(sampled_views) - num_sample_views)
                            sampled_views = [sampled_views[view_idx] for view_idx in
                                             range(sampled_view_id, sampled_view_id + num_sample_views)]
                            seg_feature_list = []
                            seg_labels_list = []
                            for sample_view in sampled_views:
                                render_pkg = render(sample_view, self.gaussians, self.pipelineparams, background)
                                seg_feature = render_pkg["seg_feature"]
                                seg_feature_list.append(seg_feature)
                                seg_labels_list.append(sample_view.sorted_segmap.cuda())
                            seg_feature_list = torch.stack(seg_feature_list, dim=0)
                            seg_labels_list = torch.stack(seg_labels_list, dim=0).squeeze()

                            batchsize = contrast_batchsize
                            # 筛选出大于0的
                            valid_labels_mask = seg_labels_list > 0
                            valid_seg_feature = seg_feature_list.permute(1, 0, 2, 3)[:, valid_labels_mask]
                            valid_seg_labels = seg_labels_list[valid_labels_mask]
                            sampled_idx = torch.randint(0, len(valid_seg_labels), size=(batchsize,), device="cuda")
                            sampled_segfeat = valid_seg_feature[:, sampled_idx].T
                            sampled_labels = valid_seg_labels[sampled_idx]
                            '''
                            sampled_view_ids = torch.randint(0, num_sample_views, size=(batchsize,), device="cuda")
                            sampled_y = torch.randint(0, viewpoint_cam.image_height, size=(batchsize,), device="cuda")
                            sampled_x = torch.randint(0, viewpoint_cam.image_width, size=(batchsize,), device="cuda")
                            # importance_sampling(gt_segmap, 10000)
                            sampled_segfeat = seg_feature_list[sampled_view_ids, :, sampled_y, sampled_x]
                            sampled_labels = seg_labels_list[sampled_view_ids, sampled_y, sampled_x]
                            '''
                            multiview_contra_loss = contrastive_loss(sampled_segfeat,
                                                                     sampled_labels,
                                                                     predef_u_list=self.gaussians.class_feat
                                                                     ) * self.optimparams.lambda_multiview_contras

            # 3. 3D Consistency
            """
            if iteration > 5000:  # not hasattr(self, "clusteringSeg3D_masks_update"):  # TODO: 3DMask有重复
                if not hasattr(self, "clusteringSeg3D_masks_update"):
                    if not os.path.exists(os.path.join(self.modelparams.model_path, "object/update_mask3d.npy")):
                        os.makedirs(os.path.join(self.modelparams.model_path, "object"), exist_ok=True)
                        self.clusteringSeg3D_masks_update, self.origin2map2update = reclustering_3d_mask(
                            self.clusteringSeg3D_labels,
                            self.gaussians.get_seg_feature,
                            self.gaussians.get_xyz)
                        np.save(os.path.join(self.modelparams.model_path, "object/update_mask3d"),
                                self.clusteringSeg3D_masks_update)
                        np.save(os.path.join(self.modelparams.model_path, "object/origin2update.npy"),
                                self.origin2map2update, allow_pickle=True)
                    else:
                        self.clusteringSeg3D_masks_update = np.load(
                            os.path.join(self.modelparams.model_path, "object/update_mask3d.npy"))
                        self.origin2map2update = np.load(
                            os.path.join(self.modelparams.model_path, "object/origin2update.npy"),
                            allow_pickle=True).item()

                    '''
                    self.clusteringSeg3D_masks_update = filter_repeat_gs_labels(self.clusteringSeg3D_masks_update,
                                                                                                self.gaussians.get_seg_feature,
                                                                                                self.gaussians.get_xyz)
                    '''
                    self.clusteringSeg3D_labels_update = torch.argmax(
                        torch.tensor(self.clusteringSeg3D_masks_update, dtype=torch.uint8),
                        dim=1).cuda()

                    self.origin2map2update_multi = {}
                    label_cnt = 0
                    for labels_id in self.origin2map2update.values():
                        if len(labels_id) > 1:
                            self.origin2map2update_multi[label_cnt] = labels_id
                            label_cnt += 1
            """

            visibility_segfeat = self.gaussians.get_seg_feature[visibility_filter]
            visibility_labels_3d = self.clusteringSeg3D_labels[visibility_filter]
            # self.clusteringSeg3D_labels[visibility_filter]  # 直接记录了每个点的label
            visibility_gs_xyz = self.gaussians.get_xyz[visibility_filter]
            '''
            pclds = []
            for i in visibility_labels_3d.unique()[1:]:
                pcld = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
                    visibility_gs_xyz[visibility_labels_3d == i].detach().cpu().numpy()))
                color = np.random.rand(3) * 0.7 + 0.3
                pcld.paint_uniform_color(color)
                pclds.append(pcld)
            o3d.visualization.draw_geometries(pclds)
            '''
            if self.optimparams.lambda_3D_clustering_filter > 0:  # and iteration > 5000:
                # TODO: current view
                ### 筛选相似的3d
                if not hasattr(self, "similar_masks") or iteration % 500 == 0:
                    global_feat = self.gaussians.get_seg_feature.clone()
                    global_feat = global_feat / (global_feat.norm(dim=-1, keepdim=True) + 1e-9)
                    class_feats = []
                    for i in range(self.clusteringSeg3D_masks.shape[1]):
                        class_feats.append(global_feat[self.clusteringSeg3D_masks[:, i]].mean(dim=0))  # 注意class 0没有
                    class_feats = torch.stack(class_feats, dim=0)
                    similar_scores = class_feats @ class_feats.T
                    similar_masks = similar_scores > 0.7 if iteration <= 500 else similar_scores > 0.8
                    similar_masks.fill_diagonal_(False)
                    self.similar_masks = similar_masks
                    optimize_label_mask = torch.where(similar_masks.sum(-1))[0]
                    print(f"Need Update {len(optimize_label_mask)} Instance!")
                    self.optimize_label_mask = [torch.cat(
                        [torch.where(self.similar_masks[label])[0],
                         torch.Tensor([label]).cuda().type(torch.uint8)],
                        -1).cpu().numpy() for label in optimize_label_mask]

                if len(self.optimize_label_mask) > 0:
                    sampled_3d_labels = self.optimize_label_mask[randint(0, len(self.optimize_label_mask) - 1)]
                    selected_gs_masks = self.clusteringSeg3D_masks[:, sampled_3d_labels].any(-1)
                    selected_3d_feats = self.gaussians.get_seg_feature[selected_gs_masks]
                    selected_3d_labels = self.clusteringSeg3D_labels[selected_gs_masks]  # 防止为0

                    clustering3D_filter_loss = contrastive_loss(selected_3d_feats,
                                                                selected_3d_labels) * self.optimparams.lambda_3D_clustering_filter

                '''
                sampled_3d_labels = self.origin2map2update_multi[
                    randint(0, len(self.origin2map2update_multi.keys()) - 1)]
                selected_gs_masks = self.clusteringSeg3D_masks_update[:, sampled_3d_labels].any(-1)
                selected_3d_feats = self.gaussians.get_seg_feature[selected_gs_masks]
                selected_3d_labels = self.clusteringSeg3D_labels_update[selected_gs_masks] + 1  # 防止为0
                clustering3D_filter_loss = contrastive_loss(selected_3d_feats,
                                                            selected_3d_labels,
                                                            min_pixnum=5) * self.optimparams.lambda_3D_clustering_filter
                '''

                """
                if len(sampled_3d_labels) == 1:
                    clustering3D_filter_loss = 0
                    '''
                    selected_3d_feat = self.gaussians.get_seg_feature[
                                           self.clusteringSeg3D_masks_update[:, sampled_3d_labels[0]]][::5]
                    selected_3d_feat = selected_3d_feat / (selected_3d_feat.norm(dim=-1, keepdim=True) + 1e-9)
                    feat_simi_score = torch.triu(selected_3d_feat @ selected_3d_feat.T, diagonal=1)
                    feat_simi_score = feat_simi_score[feat_simi_score != 0]
                    clustering3D_filter_loss = (1 - feat_simi_score).mean()
                    '''
                else:
                    selected_gs_masks = self.clusteringSeg3D_masks_update[:, sampled_3d_labels].any(-1)
                    selected_3d_feats = self.gaussians.get_seg_feature[selected_gs_masks]
                    selected_3d_labels = self.clusteringSeg3D_labels_update[selected_gs_masks] + 1
                    clustering3D_filter_loss = contrastive_loss(selected_3d_feats,
                                                                selected_3d_labels,
                                                                min_pixnum=5) * self.optimparams.lambda_3D_clustering_filter
                """
                '''
                sampled_3d_labels = self.valid_3d_labels[
                    torch.randint(0, len(self.valid_3d_labels), size=(1,), device="cuda")]
                selected_pseudo_3d_feat = self.gaussians.get_seg_feature[
                    self.clusteringSeg3D_labels == sampled_3d_labels]
                selected_pseudo_3d_feat = selected_pseudo_3d_feat / (
                        selected_pseudo_3d_feat.norm(dim=1, keepdim=True) + 1e-9)
                selected_pseudo_3d_feat_mean = selected_pseudo_3d_feat.mean(0)
                clustering3D_filter_loss = filter_clustering_3d_loss(selected_pseudo_3d_feat_mean,
                                                                     self.gaussians.get_seg_feature,
                                                                     self.gaussians.get_xyz) * self.optimparams.lambda_3D_clustering_filter
                '''
                '''
                import open3d as o3d
                pcld = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
                    self.gaussians.get_xyz[self.clusteringSeg3D_labels == sampled_3d_labels].detach().cpu().numpy()))
                pcld.paint_uniform_color([1, 0, 0])
                bbox = o3d.geometry.OrientedBoundingBox.create_from_points(pcld.points)
                bbox.color = np.array([1, 0, 0])
                o3d.visualization.draw_geometries([pcld, bbox, o3d.io.read_point_cloud(
                    "/home/bytedance/Projects/Datasets/ZipNeRF/nyc/meetingroom/sparse/0/points3D.ply")])
                '''
            # 将具体太远的分开
            if self.optimparams.lambda_3D_contras > 0 and self.optimparams.add_3d_prior:
                # 全局监督
                batchsize_3d = contrast_batchsize

                valid_labels_mask = visibility_labels_3d > 0
                if valid_labels_mask.sum() > 0:
                    valid_seg_feature = visibility_segfeat[valid_labels_mask]
                    valid_seg_labels = visibility_labels_3d[valid_labels_mask]

                    sampled_idx = torch.randint(0, len(valid_seg_labels), size=(batchsize_3d,), device="cuda")
                    sampled_segfeat_3d = valid_seg_feature[sampled_idx]
                    sampled_labels_3d = valid_seg_labels[sampled_idx]
                    '''
                    sampled_points_idx = torch.randint(0, len(visibility_segfeat), size=(batchsize_3d,), device="cuda")
                    sampled_segfeat_3d = visibility_segfeat[sampled_points_idx]
                    sampled_labels_3d = visibility_labels_3d[sampled_points_idx]
                    '''
                    contra_3d_loss = contrastive_loss(sampled_segfeat_3d,
                                                      sampled_labels_3d,
                                                      predef_u_list=self.gaussians.class_feat
                                                      ) * self.optimparams.lambda_3D_contras
                else:
                    print(viewpoint_cam.image_name)

            if self.optimparams.lambda_3D_consist > 0:
                # 找到离得最近的点云，最小化其feature
                batchsize_3d = 128
                '''
                以下完全属于bug
                if len(visibility_segfeat) == 0:
                    print(viewpoint_cam.image_name)
                '''
                sampled_points_idx = torch.randint(0, len(visibility_segfeat), size=(batchsize_3d,), device="cuda")
                sampled_segfeat_3d = visibility_segfeat[sampled_points_idx]
                sampled_gaussian_xyz = visibility_gs_xyz[sampled_points_idx]
                consist_3d_loss = consist_3d_feat_loss(sampled_segfeat_3d, sampled_gaussian_xyz,
                                                       visibility_segfeat, visibility_gs_xyz
                                                       ) * self.optimparams.lambda_3D_consist

            # 4. norm loss
            if self.optimparams.lambda_feat_norm > 0:
                feat_norm_loss = (((torch.norm(visibility_segfeat, dim=-1, keepdim=True) - 1.0) ** 2).mean() + \
                                  ((torch.norm(seg_feature, dim=0) - 1.0) ** 2).mean()
                                  ) * self.optimparams.lambda_feat_norm

            total_loss = singleview_contra_loss + \
                         multiview_contra_loss + \
                         contra_3d_loss + \
                         consist_3d_loss + \
                         feat_norm_loss + \
                         clustering3D_filter_loss

            total_loss.backward()
            iter_end.record()

            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            if self.modelparams.use_grouping_loss:
                cls_optimizer.step()
                cls_optimizer.zero_grad()
            torch.cuda.empty_cache()

            with torch.no_grad():
                # Progress bar
                if iteration % 10 == 0:
                    loss_dict = {
                        "SV_ContraLoss": f"{singleview_contra_loss:.{3}f}",
                        "MV_ContraLoss": f"{multiview_contra_loss:.{3}f}",
                        "3D_ContraLoss": f"{contra_3d_loss:.{3}f}",
                        "3D_ConsistLoss": f"{consist_3d_loss:.{3}f}",
                        "SemNorm_Loss": f"{feat_norm_loss:.{3}f}",
                        "3D_ClusterFilter_Loss": f"{clustering3D_filter_loss:.{3}f}",
                    }
                    progress_bar.set_postfix(loss_dict)
                    progress_bar.update(10)

                if iteration % 200 == 0:
                    viewpoint = self.scene.getTrainCameras()[2]
                    render_pkg = render(viewpoint, self.gaussians, self.pipelineparams, background)

                    _, seg_feature, _, _, _ = \
                        render_pkg["render"], render_pkg["seg_feature"], render_pkg["viewspace_points"], \
                            render_pkg["visibility_filter"], render_pkg["radii"]
                    os.makedirs(self.scene.model_path, exist_ok=True)
                    Image.fromarray(feature_to_rgb(seg_feature)).save(
                        f"{self.scene.model_path}/{iteration}_feat.png")

                    if self.modelparams.use_grouping_loss:
                        logits = classifier(seg_feature)
                        pred_label = torch.argmax(logits, dim=0)
                        Image.fromarray(mask_to_rgb(pred_label)).save(
                            f"{self.scene.model_path}/{iteration}_seg.png")

                if iteration % 2500 == 0:
                    self.scene.save(iteration)
                    self.export_segment_results(iteration)

                if iteration == self.optimparams.iterations:
                    progress_bar.close()

    @torch.no_grad()
    def render_views(self, save_mask=False, view_idx=[]):
        save_dir = os.path.join(self.scene.model_path, "render")
        os.makedirs(save_dir, exist_ok=True)
        for folder in ["segfeat", "segmask"]:
            os.makedirs(os.path.join(save_dir, folder), exist_ok=True)

        bg_color = [1, 1, 1] if self.modelparams.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 获取instance feature
        if save_mask:
            instance_feats = []
            instance_colors = []
            for sampled_3d_labels in range(self.clusteringSeg3D_masks.shape[1]):
                selected_pseudo_3d_feat = self.gaussians.get_seg_feature[
                    self.clusteringSeg3D_masks[:, sampled_3d_labels]]
                selected_pseudo_3d_feat = selected_pseudo_3d_feat / (
                        selected_pseudo_3d_feat.norm(dim=1, keepdim=True) + 1e-9)
                instance_feats.append(selected_pseudo_3d_feat.mean(0))
                instance_colors.append(np.random.rand(3))

            instance_feats = torch.stack(instance_feats, dim=0)
            instance_colors = np.stack(instance_colors, axis=0) * 0.7 + 0.3

        viewpoints = self.scene.getTrainCameras() if view_idx == [] else [self.scene.getTrainCameras()[idx] for idx in
                                                                          view_idx]

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
                    covariance_matrix = (1 / n) * torch.matmul(X.T,
                                                               X).float()  # An old torch bug: matmul float32->float16,
                    eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix)
                    # eigenvalues = torch.norm(eigenvalues, dim=1)
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
                # 找到每个feat最大的score对应的
                instance_feat_score = instance_feat_score.reshape(-1, instance_feat_score.shape[-1])
                seg_feature_instance = instance_feat_score.argmax(-1)
                seg_feature_instance[
                    instance_feat_score[
                        torch.arange(len(instance_feat_score)), instance_feat_score.argmax(-1)] < 0.75] = 0

                seg_mask_colormap = instance_colors[seg_feature_instance.cpu().numpy()]
                seg_mask_colormap[seg_feature_instance.cpu().numpy() == 0] = 0

                Image.fromarray(
                    np.uint8(255.0 * seg_mask_colormap.reshape(seg_feature.shape[1], seg_feature.shape[2], 3))
                ).save(f"{save_dir}/segmask/{viewpoint_view.image_name}.png")

            torch.cuda.empty_cache()

    @torch.no_grad()
    def export_segment_results(self, iteration, score_threshold=0.9, ignore_labels=[]):
        # 颜色
        color_path = os.path.join(self.render_cache_dir, "output/object/instance_colors.npy")
        if os.path.exists(color_path):
            instance_colors = np.load(color_path)
        else:
            '''
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            num_classes = self.clusteringSeg3D_masks.shape[1]
            colors = plt.get_cmap('hsv', num_classes)
            norm = mcolors.Normalize(vmin=0, vmax=num_classes - 1)
            instance_colors = colors(norm(np.arange(num_classes)))[:, :3] * 0.7 + 0.3
            '''
            instance_colors = np.random.rand(self.clusteringSeg3D_masks.shape[1], 3) * 0.7 + 0.3
            np.save(color_path, instance_colors)

        save_dir = os.path.join(self.scene.model_path, f"point_cloud/iteration_{iteration}")
        os.makedirs(save_dir, exist_ok=True)
        save_partial_dir = os.path.join(save_dir, "label_pointclouds")
        os.makedirs(save_partial_dir, exist_ok=True)

        pclds = None
        global_feat = self.gaussians.get_seg_feature / (
                self.gaussians.get_seg_feature.norm(dim=-1, keepdim=True) + 1e-9)
        global_feat = global_feat.cpu()
        for sampled_3d_labels in range(self.clusteringSeg3D_masks.shape[1])[1:]:
            if sampled_3d_labels in ignore_labels:
                continue
            selected_pseudo_3d_feat = self.gaussians.get_seg_feature[self.clusteringSeg3D_masks[:, sampled_3d_labels]]
            selected_pseudo_3d_feat = selected_pseudo_3d_feat / (
                    selected_pseudo_3d_feat.norm(dim=1, keepdim=True) + 1e-9)
            selected_pseudo_3d_feat_mean = selected_pseudo_3d_feat.mean(0).cpu()
            feat_score = selected_pseudo_3d_feat_mean @ global_feat.T
            # feat_sq_score = ((global_feat - selected_pseudo_3d_feat_mean) ** 2).sum(-1)
            selected_points_mask = (feat_score >= score_threshold)  # & (feat_sq_score <= score_sq_threshold)

            pcld = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
                self.gaussians.get_xyz[selected_points_mask].detach().cpu().numpy()))
            color = instance_colors[sampled_3d_labels]  # np.random.rand(3) * 0.7 + 0.3
            pcld.paint_uniform_color(color)
            '''
            labels = np.array(pcld.cluster_dbscan(eps=0.3, min_points=4)) + 1
            pclds = []
            for label in np.unique(labels):
                chosen_label_mask = labels == label

                pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
                    self.gaussians.get_xyz[selected_points_mask][chosen_label_mask].detach().cpu().numpy()))
                pc.paint_uniform_color(np.random.rand(3) * 0.7 + 0.3)
                pclds.append(pc)
            o3d.visualization.draw_geometries(pclds)
            '''

            # bbox = o3d.geometry.OrientedBoundingBox.create_from_points(pcld.points)
            # bbox.color = color
            # pclds.append(bbox)
            pclds = pcld if pclds is None else pclds + pcld

            if len(pcld.points) > 0:
                o3d.io.write_point_cloud(os.path.join(save_partial_dir, f"{sampled_3d_labels}.ply"),
                                         pcld)
            else:
                pcld = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
                    self.gaussians.get_xyz[self.clusteringSeg3D_labels == sampled_3d_labels].detach().cpu().numpy()))
                pcld.paint_uniform_color(color)

                o3d.io.write_point_cloud(os.path.join(save_partial_dir, f"{sampled_3d_labels}_empty.ply"),
                                         pcld)

        o3d.io.write_point_cloud(os.path.join(save_dir, "point_cloud_labels.ply"),
                                 pclds)


if __name__ == "__main__":
    # Set up command line argument parser
    # -s data/ZipNeRF/nyc/kitchen_dsfm -r 4 -m test++
    # -s /home/bytedance/Projects/Datasets/LERF_Dataset/lerf_ovs/waldo_kitchen -m test
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])

    segsplat = SegSplatting(lp.extract(args), op.extract(args), pp.extract(args))
    segsplat.args = args
    segsplat.GraphBasedClustering()
    segsplat.train_segfeat()
    print("\nTraining complete.")
