import os
from tqdm import tqdm
from PIL import Image

from types import SimpleNamespace
from scene import GaussianModel
from scene.cameras import Camera

from typing import List

from spatial_track.modules.init_tracker import *
from spatial_track.modules.iterative_cluster import iterative_clustering
from spatial_track.modules.post_process import post_process


### Refer from https://github.com/PKU-EPIC/MaskClustering with Gaussian-based Tracker ###
class GausCluster:
    def __init__(self, gaussian: GaussianModel, viewcams: List[Camera], debug=True):
        # 构建init node
        self.gaussian = gaussian
        self.viewcams = viewcams

        clustering_args = {
            "mask_visible_threshold": 0.5,  #
            "undersegment_filter_threshold": 0.4,
            "contained_threshold": 0.6,
            "view_consensus_threshold": 0.9,
            "point_filter_threshold": 0.5
        }

        self.clustering_args = SimpleNamespace(**clustering_args)

        self.debug = debug

    def maskclustering(self, save_dir=None):
        ## Init the Mask's Gaussian Tracker
        init_mask_assocation = construct_mask2gs_tracker(self.gaussian, self.viewcams, self.clustering_args,
                                                         save_dir, self.debug)
        ## Cluster the Mask's Gaussian Tracker
        update_mask_assocation = iterative_clustering(init_mask_assocation, self.clustering_args)

        ## Use DBScan to Filter Noisy Points from maskclustering
        final_mask_assocation = post_process(self.gaussian, update_mask_assocation, self.clustering_args)

        self.export(final_mask_assocation, save_dir=save_dir)

    def export(self, mask_assocation, save_dir, export_vis=True):
        # undersegment, mask3d, mv-consist masks
        os.makedirs(save_dir, exist_ok=True)

        total_point_num = len(self.gaussian.get_xyz)

        mask_3d_labels = []
        for i, (point_ids, mask_list) in enumerate(zip(mask_assocation["total_point_ids_list"],
                                                       mask_assocation["total_mask_list"])):
            binary_mask = np.zeros(total_point_num, dtype=bool)
            binary_mask[list(point_ids)] = True
            mask_3d_labels.append(binary_mask)

        mask_3d_labels = np.stack(mask_3d_labels, axis=1)

        underseg_mask_ids = np.stack([list(mask_assocation['global_frame_mask_list'][id]) for id in
                                      mask_assocation["undersegment_mask_ids"]], axis=0)

        output_dict = {
            "mask_3d_labels": mask_3d_labels,
            "underseg_mask_ids": underseg_mask_ids,
            "mask_2d_clusters": mask_assocation["total_mask_list"]
        }

        np.save(os.path.join(save_dir, 'output_dict.npy'), output_dict, allow_pickle=True)

        if export_vis:
            import open3d as o3d
            from vis_utils.color_utils import generate_semantic_colors
            scene_points = self.gaussian.get_xyz.cpu().numpy()
            pcld_colors = generate_semantic_colors(mask_3d_labels.shape[1], normalize=True)

            color_pts = None
            for i in range(mask_3d_labels.shape[1]):
                curr_mask = mask_3d_labels[:, i]
                if curr_mask.sum() == 0 or curr_mask.sum() < 10:
                    continue
                point_ids = np.where(curr_mask)[0]

                points = scene_points[point_ids]
                instance_pts = o3d.geometry.PointCloud()
                instance_pts.points = o3d.utility.Vector3dVector(points)
                instance_pts.paint_uniform_color(pcld_colors[i])

                color_pts = color_pts + instance_pts if color_pts is not None else instance_pts

            # o3d.visualization.draw_geometries([color_pts])
            o3d.io.write_point_cloud(os.path.join(save_dir, 'segment.ply'), color_pts)

    def rearrange_mask(self, mask_folder, mask_assocation_info):
        save_dir = os.path.join(os.path.dirname(mask_folder), "mask_sorted")
        os.makedirs(save_dir, exist_ok=True)

        masks_origin = []
        for viewcam in self.viewcams:
            mask_file = os.path.join(mask_folder, viewcam.image_name + ".png")
            masks_origin.append(np.array(Image.open(mask_file)))

        masks_origin = np.stack(masks_origin)
        masks_new = np.zeros_like(masks_origin, dtype=np.int16)

        for cluster_id, cluster_info in enumerate(mask_assocation_info):
            cluster_id = cluster_id + 1  # 从1开始
            for frame_mask_id in cluster_info:
                frame_id, mask_id = frame_mask_id[:2]
                masks_new[frame_id][masks_origin[frame_id] == mask_id] = cluster_id  # 更新为instance

        for mask_id in range(len(masks_origin)):
            save_path = os.path.join(save_dir, self.viewcams[mask_id].image_name + ".png")
            Image.fromarray(masks_new[mask_id]).save(save_path)

    def filter_undersegment_mask(self, mask_folder, undersegment_masks):
        save_dir = os.path.join(os.path.dirname(mask_folder), "mask_filtered")
        os.makedirs(save_dir, exist_ok=True)

        save_undersegment_dir = os.path.join(os.path.dirname(mask_folder), "mask_undersegment")
        os.makedirs(save_undersegment_dir, exist_ok=True)

        masks_origin = []
        for viewcam in self.viewcams:
            mask_file = os.path.join(mask_folder, viewcam.image_name + ".png")
            masks_origin.append(np.array(Image.open(mask_file)))

        masks_origin = np.stack(masks_origin)
        masks_new = masks_origin.copy()
        masks_undersegment = np.zeros_like(masks_origin, dtype=np.int16)

        for underseg_frame_mask_ids in undersegment_masks:
            frame_id, mask_id = underseg_frame_mask_ids[:2]
            masks_new[frame_id][masks_origin[frame_id] == mask_id] = 0
            masks_undersegment[frame_id][masks_origin[frame_id] == mask_id] = mask_id

        for mask_id in range(len(masks_origin)):
            save_path = os.path.join(save_dir, self.viewcams[mask_id].image_name + ".png")
            Image.fromarray(masks_new[mask_id]).save(save_path)

            save_underseg_path = os.path.join(save_undersegment_dir, self.viewcams[mask_id].image_name + ".png")
            Image.fromarray(masks_undersegment[mask_id]).save(save_underseg_path)
