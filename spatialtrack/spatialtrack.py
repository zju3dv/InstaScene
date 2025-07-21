from types import SimpleNamespace

import torch.cuda

from spatialtrack.graph.construction import *
from spatialtrack.graph.iterative_clustering import *
from spatialtrack.graph.node import *
from spatialtrack.post_process import *
from scene import GaussianModel


### Refer from https://github.com/PKU-EPIC/MaskClustering ###
class GraphClustering:
    def __init__(self, gaussian: GaussianModel, viewcams):
        # 构建init node
        self.gaussian = gaussian
        self.viewcams = viewcams

        clustering_args = {
            "mask_visible_threshold": 0.3,  # 0.3
            "undersegment_filter_threshold": 0.3,  # 0.2 越大越不会被识别为欠分割
            "contained_threshold": 0.8,  # 0.8 如果另一个视角的mask在当前mask中占据了0.8，则认为包含
            "view_consensus_threshold": 0.9,  # 0.9
            "point_filter_threshold": 0.5
        }
        self.clustering_args = SimpleNamespace(**clustering_args)

    def graphclustering(self):
        self.mask_graph_construction()
        self.iterative_clustering()
        '''
        import open3d as o3d
        import numpy as np
        scene_points = self.gaussian.get_xyz.cpu().numpy()
        pclds = []
        for object in self.nodes:
            pointcloud = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(scene_points[np.array(list(object.point_ids))]))
            pointcloud.paint_uniform_color(np.random.random(3))
            pclds.append(pointcloud)
        o3d.visualization.draw_geometries(pclds)
        '''
        self.post_process()

    def mask_graph_construction(self):
        print("\n## Start Init the Graph ##")
        '''
            boundary_points: 出现在某一帧的多个Mask中，作为边界Points考虑
            point_in_mask_matrix: point_i exist in mask_j [N_pts, N_frames] 点云对应到frame的第几个Mask
            mask_point_clouds:{"frameID_maskID":Points_id}
            point_frame_matrix: [N_pts, N_masks] 点云i是否在framej可见
            global_frame_mask_list: mask list [frame_ID,mask_ID]
        '''
        print("### Organize the mask & gaussians ###")
        boundary_points, point_in_mask_matrix, mask_point_clouds, point_frame_matrix, global_frame_mask_list = build_point_in_mask_matrix(
            self.gaussian, self.viewcams)

        print("### Compute the visibility & undersegment ###")
        visible_frames, contained_masks, undersegment_mask_ids = process_masks(self.viewcams, global_frame_mask_list,
                                                                               point_in_mask_matrix, boundary_points,
                                                                               mask_point_clouds, self.clustering_args)
        observer_num_thresholds = get_observer_num_thresholds(visible_frames)  # nk的阈值
        nodes = init_nodes(global_frame_mask_list, visible_frames, contained_masks, undersegment_mask_ids,
                           mask_point_clouds)

        self.nodes = nodes
        self.observer_num_thresholds = observer_num_thresholds
        self.mask_point_clouds = mask_point_clouds
        self.point_frame_matrix = point_frame_matrix

        self.undersegment_mask_ids = undersegment_mask_ids
        self.global_frame_mask_list = global_frame_mask_list

    def iterative_clustering(self):
        print("## Start Clustering ##")
        nodes = self.nodes
        for iterate_id, observer_num_threshold in enumerate(self.observer_num_thresholds):
            print(f'Iterate {iterate_id}: observer_num', observer_num_threshold, ', number of nodes', len(nodes))
            graph = update_graph(nodes, observer_num_threshold,
                                 self.clustering_args.view_consensus_threshold)  # connect_threshold: 0.9
            nodes = cluster_into_new_nodes(iterate_id + 1, nodes, graph)
            torch.cuda.empty_cache()

        self.nodes = nodes

    def post_process(self):
        print("## Start Post-Process ##")
        # For each cluster, we follow OVIR-3D to i) use DBScan to split the disconnected point cloud into different objects
        # ii) filter the points that hardly appear within this cluster, i.e. the detection ratio is lower than a threshold
        total_point_ids_list, total_bbox_list, total_mask_list = [], [], []
        scene_points = self.gaussian.get_xyz.cpu().numpy()

        for node in tqdm(self.nodes):
            if len(node.mask_list) < 2:  # objects merged from less than 2 masks are ignored
                continue
            pcld, point_ids = node.get_point_cloud(scene_points)
            if True:
                pcld_list, point_ids_list = dbscan_process(pcld, point_ids, DBSCAN_THRESHOLD=0.1,
                                                           min_points=4)  # split the disconnected point cloud into different objects
            else:
                pcld_list, point_ids_list = [pcld], [np.array(point_ids)]
            point_ids_list, bbox_list, mask_list = filter_point(self.point_frame_matrix, node, pcld_list,
                                                                point_ids_list,
                                                                self.mask_point_clouds,
                                                                self.clustering_args)

            total_point_ids_list.extend(point_ids_list)
            total_bbox_list.extend(bbox_list)
            total_mask_list.extend(mask_list)

        # merge objects that have larger than 0.8 overlapping ratio 根据两个point bbox相交的比例
        # TODO: 优化合并策略 A和B，A在B的BBox中的点，占B的40%，和B在A的BBox中的点，占A的40%
        total_point_ids_list_merge, total_mask_list_merge, invalid_object_mask = merge_overlapping_objects(
            total_point_ids_list, total_bbox_list, total_mask_list, overlapping_ratio=0.8)
        # export(dataset, total_point_ids_list, total_mask_list, args)
        self.total_point_ids_list = total_point_ids_list_merge
        self.total_mask_list = total_mask_list_merge

    def export(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print("## Start Export ##")
        total_point_num = len(self.gaussian.get_xyz)
        class_agnostic_mask_list = []
        object_dict = {}
        for i, (point_ids, mask_list) in enumerate(zip(self.total_point_ids_list, self.total_mask_list)):
            def find_represent_mask(mask_list):
                mask_list.sort(key=lambda x: x[2], reverse=True)
                return mask_list[:5]

            object_dict[i] = {
                'point_ids': point_ids,
                'mask_list': mask_list,
                'repre_mask_list': find_represent_mask(mask_list),
            }
            binary_mask = np.zeros(total_point_num, dtype=bool)
            binary_mask[list(point_ids)] = True
            class_agnostic_mask_list.append(binary_mask)

        pred_masks = np.stack(class_agnostic_mask_list, axis=1)
        pred_dict = {
            "pred_masks": pred_masks
        }

        np.savez(os.path.join(save_dir, f'mask3d.npz'), **pred_dict)
        np.save(os.path.join(save_dir, 'object_dict.npy'), object_dict, allow_pickle=True)

        underseg_mask_ids = [list(self.global_frame_mask_list[id]) for id in self.undersegment_mask_ids]
        if len(underseg_mask_ids) > 0:
            np.save(os.path.join(save_dir, 'underseg_masks.npy'), np.stack(underseg_mask_ids, axis=0))

        # 保存点云
        scene_points = self.gaussian.get_xyz.cpu().numpy()
        color_pts = None
        for i in tqdm(range(pred_masks.shape[1])):
            curr_mask = pred_masks[:, i]
            if curr_mask.sum() == 0 or curr_mask.sum() < 10:
                print(f"Idx {i} is skipped...")
                continue
            point_ids = np.where(curr_mask)[0]

            def vis_one_object(point_ids, scene_points):
                points = scene_points[point_ids]
                color = np.random.rand(3) * 0.7 + 0.3
                colors = np.tile(color, (points.shape[0], 1))
                return points, colors

            points, colors = vis_one_object(point_ids, scene_points)
            instance_pts = o3d.geometry.PointCloud()
            instance_pts.points = o3d.utility.Vector3dVector(points)
            instance_pts.colors = o3d.utility.Vector3dVector(colors)

            color_pts = color_pts + instance_pts if color_pts is not None else instance_pts

        # o3d.visualization.draw_geometries([color_pts])
        o3d.io.write_point_cloud(os.path.join(save_dir, 'segment.ply'), color_pts)
