import os

import torch
import numpy as np
from typing import List
from tqdm import tqdm
from scipy.sparse import csr_matrix

from gaussian_renderer import render
from scene import GaussianModel
from scene.cameras import Camera

from spatial_track.modules.node import Node


def get_segmap_gaussians(gaussian: GaussianModel, view: Camera):
    '''
    Get each mask's corresponding gaussian points with rasterization-based gaussian tracing
    !!! Since GS's rendered-depth exists severe mv-inconsistency, use rasterization tracing to get corresponding gaussians
    :param gaussian:
    :param view:
    :return:
    '''
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    gau_related_pixels = render(view, gaussian, gaussian.pipelineparams, background)['gau_related_pixels']

    gaus_ids = gau_related_pixels[:, 0]  # gs_idx
    pixel_ids = gau_related_pixels[:, 1]  # pixel_idx

    mask_image = view.segmap.cuda().reshape(-1)
    ids = torch.unique(mask_image).cpu().numpy()
    ids.sort()

    mask_info = {}  # id: pts_id
    frame_gaussian_ids = set(gaus_ids.tolist())
    for mask_id in ids:
        if mask_id == 0:
            continue
        segmentation = mask_image == mask_id
        valid_mask = segmentation[pixel_ids]

        if len(set(gaus_ids[valid_mask].tolist())) < 50:
            continue

        mask_info[mask_id] = set(gaus_ids[valid_mask].tolist())
    '''
    import open3d as o3d
    scene_points = gaussian.get_xyz
    pclds = None
    for mask_id in mask_info:
        pcld_ids = mask_info[mask_id]
        pcld = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(scene_points.cpu().numpy()[np.array(list(pcld_ids))]))
        pcld.paint_uniform_color(np.random.rand(3))
        if pclds is None:
            pclds = pcld
        else:
            pclds += pcld
    # o3d.visualization.draw_geometries(pclds)
    view.view_o3d([pclds])
    o3d.io.write_point_cloud("t1.ply", pclds)
    '''
    return mask_info, list(frame_gaussian_ids)


def compute_mask_visible_frame(global_gaussian_in_mask_matrix, gaussian_in_frame_matrix, threshold=0.3):
    '''
    50% points occurs in frame -> visible
    Args:
        global_point_in_mask_matrix: [N_pts,N_masks]
        point_frame_matrix: [N_pts,N_Frames]

    Returns:

    '''
    A = csr_matrix(global_gaussian_in_mask_matrix, dtype=np.float32)  # shape: [N_pts, N_masks]
    B = csr_matrix(gaussian_in_frame_matrix, dtype=np.float32)  # shape: [N_pts, N_frames]

    intersection_counts = A.T @ B  # still sparse->交集

    mask_point_counts = np.array(A.sum(axis=0)).ravel() + 1e-6  # 防止除以0

    intersection_counts = intersection_counts.tocoo()
    visible_mask = (intersection_counts.data / mask_point_counts[intersection_counts.row]) > threshold  # 30%点在该Frame可见

    result = csr_matrix(
        (np.ones(visible_mask.sum(), dtype=bool),
         (intersection_counts.row[visible_mask], intersection_counts.col[visible_mask])),
        shape=(A.shape[1], B.shape[1])
    )
    return result.toarray()


def construct_mask2gs_tracker(gaussian: GaussianModel, viewcams: List[Camera], clustering_args, save_dir, debug):
    # Extract each mask's corresponding gaussian
    if debug:
        save_tracker_dir = os.path.join(save_dir, "tracker")
        os.makedirs(save_tracker_dir, exist_ok=True)

    iterator = tqdm(enumerate(viewcams), total=len(viewcams), desc="Extracting Gaussian Tracker")

    # gaussian point in each frame's correspond mask
    gaussian_in_frame_maskid_matrix = np.zeros((len(gaussian.get_xyz), len(viewcams)), dtype=np.uint16)

    gaussian_in_frame_matrix = np.zeros((len(gaussian.get_xyz), len(viewcams)), dtype=bool)
    global_frame_mask_list = []
    mask_gaussian_pclds = {}

    for frame_cnt, view in iterator:
        # find the gaussian contained in the frame
        if debug:
            tracker_path = os.path.join(save_dir, "tracker", view.image_name.split(".")[0] + ".npy")
            if not os.path.exists(tracker_path):
                mask_dict, frame_gaussian_ids = get_segmap_gaussians(gaussian, view)
                view_info = {
                    "mask_dict": mask_dict,
                    "frame_gaussian_ids": frame_gaussian_ids,
                }
                np.save(tracker_path, view_info, allow_pickle=True)
            else:
                view_info = np.load(tracker_path, allow_pickle=True).item()
                mask_dict = view_info['mask_dict']
                frame_gaussian_ids = view_info['frame_gaussian_ids']
        else:
            mask_dict, frame_gaussian_ids = get_segmap_gaussians(gaussian, view)

        gaussian_in_frame_matrix[frame_gaussian_ids, frame_cnt] = True  # gs在frame可见

        for mask_id, mask_point_cloud_ids in mask_dict.items():
            mask_gaussian_pclds[f'{frame_cnt}_{mask_id}'] = mask_point_cloud_ids
            # note: gs point may correspond to multi mask (boundary)
            gaussian_in_frame_maskid_matrix[list(mask_point_cloud_ids), frame_cnt] = mask_id
            global_frame_mask_list.append((frame_cnt, mask_id))
        torch.cuda.empty_cache()

    global_gaussian_in_mask_matrix = np.zeros((len(gaussian.get_xyz), len(global_frame_mask_list)), dtype=bool)
    for mask_idx, frame_mask_id in enumerate(mask_gaussian_pclds):
        global_gaussian_in_mask_matrix[np.array(list(mask_gaussian_pclds[frame_mask_id])), mask_idx] = True

    # Filter Undersegment
    visible_frames = []
    contained_masks = []  # each mask contained mask id
    undersegment_mask_ids = []

    # 计算各个Mask的visible frames
    mask_visible_frames = compute_mask_visible_frame(global_gaussian_in_mask_matrix, gaussian_in_frame_matrix)
    mask_cnts = 0
    iterator = tqdm(global_frame_mask_list, total=len(global_frame_mask_list), desc="Filtering Undersegment Masks")
    for frame_id, mask_id in iterator:
        valid, contained_mask, visible_frame = judge_single_mask(gaussian_in_frame_maskid_matrix,
                                                                 mask_gaussian_pclds,
                                                                 f'{frame_id}_{mask_id}',
                                                                 mask_visible_frames[mask_cnts],
                                                                 viewcams,
                                                                 global_frame_mask_list,
                                                                 clustering_args)

        contained_masks.append(contained_mask)
        visible_frames.append(visible_frame)
        if not valid:
            global_mask_id = global_frame_mask_list.index((frame_id, mask_id))
            undersegment_mask_ids.append(global_mask_id)
        torch.cuda.empty_cache()
        mask_cnts += 1

    contained_masks = np.stack(contained_masks, axis=0)
    visible_frames = np.stack(visible_frames, axis=0)  # mask_visible_frames
    '''
    import open3d as o3d
    valid_indices = np.where(contained_masks[0])[0]
    pcld_1 = mask_gaussian_pclds[list(mask_gaussian_pclds.keys())[valid_indices[0]]]
    pcld_2 = mask_gaussian_pclds[list(mask_gaussian_pclds.keys())[valid_indices[-1]]]

    scene_pts = gaussian.get_xyz.detach().cpu().numpy()
    pcld_1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene_pts[np.array(list(pcld_1))]))
    pcld_2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene_pts[np.array(list(pcld_2))]))
    pcld_1.paint_uniform_color(np.random.random(3))
    pcld_2.paint_uniform_color(np.random.random(3))
    o3d.visualization.draw_geometries([pcld_1, pcld_2])
    '''
    for global_mask_id in undersegment_mask_ids:  # remove undersegment
        frame_id, _ = global_frame_mask_list[global_mask_id]
        global_frame_id = frame_id
        mask_projected_idx = np.where(contained_masks[:, global_mask_id])[0]  # 被包含在
        contained_masks[:, global_mask_id] = False  # 排除掉该mask的影响
        visible_frames[mask_projected_idx, global_frame_id] = False

    # 构造Graph
    ## Use Maskclustering to cluster the masks
    contained_masks = torch.from_numpy(contained_masks).float().cuda()
    visible_frames = torch.from_numpy(visible_frames).float().cuda()

    observer_num_thresholds = get_observer_num_thresholds(visible_frames)  # nk的阈值
    nodes = init_nodes(global_frame_mask_list, visible_frames, contained_masks, undersegment_mask_ids,
                       mask_gaussian_pclds)

    return {
        "nodes": nodes,
        "observer_num_thresholds": observer_num_thresholds,
        "mask_gaussian_pclds": mask_gaussian_pclds,
        "global_frame_mask_list": global_frame_mask_list,
        "gaussian_in_frame_matrix": gaussian_in_frame_matrix,
        "undersegment_mask_ids": undersegment_mask_ids
    }


def judge_single_mask(gaussian_in_mask_matrix,
                      mask_gaussian_pclds,
                      frame_mask_id,
                      mask_visible_frame,
                      viewcams: List[Camera],
                      global_frame_mask_list,
                      clustering_args):
    '''
    :param gaussian_in_mask_matrix: gaussian in frame's mask id
    :param mask_gaussian_pcld:
    :param viewcams:
    :param global_frame_mask_list:
    :param clustering_args:
    :return:
    '''
    mask_gaussian_pcld = mask_gaussian_pclds[frame_mask_id]

    visible_frame = np.zeros(len(viewcams), dtype=bool)
    contained_mask = np.zeros(len(global_frame_mask_list), dtype=bool)

    mask_gaussians_info = gaussian_in_mask_matrix[list(mask_gaussian_pcld), :]  # 每个点在各个frame mask中的idx

    split_num = 0  # frame number of undersegment
    visible_num = 0  # frame number of visible

    for frame_id in np.where(mask_visible_frame)[0]:
        # overlap masks in current frame
        overlap_mask_ids, overlap_mask_cnts = np.unique(mask_gaussians_info[:, frame_id], return_counts=True)
        sorted_idx = np.argsort(overlap_mask_cnts)[::-1]
        overlap_mask_ids, overlap_mask_cnts = overlap_mask_ids[sorted_idx], overlap_mask_cnts[sorted_idx]

        # if most are invisible, continue
        if 0 in overlap_mask_ids:
            invalid_gaussian_cnts = overlap_mask_cnts[np.where(overlap_mask_ids == 0)[0]]
            if invalid_gaussian_cnts / overlap_mask_cnts.sum() > clustering_args.mask_visible_threshold:  # 0.5
                continue

        visible_num += 1

        if overlap_mask_ids[0] == 0:
            overlap_mask_ids = overlap_mask_ids[1:]
            overlap_mask_cnts = overlap_mask_cnts[1:]

        if len(overlap_mask_ids) == 0:
            continue

        contained_ratio = overlap_mask_cnts[0] / overlap_mask_cnts.sum()
        if contained_ratio > clustering_args.contained_threshold:  # check max overlap mask
            frame_mask_idx = global_frame_mask_list.index((frame_id, overlap_mask_ids[0]))
            contained_mask[frame_mask_idx] = True  # contained in this mask
            visible_frame[frame_id] = True
        else:
            split_num += 1

    if visible_num == 0 or split_num / visible_num > clustering_args.undersegment_filter_threshold:  # 30% Frame exist undersegment
        return False, contained_mask, visible_frame  # undersegment
    else:
        return True, contained_mask, visible_frame


##
def get_observer_num_thresholds(visible_frames):
    '''
        Compute the observer number thresholds for each iteration. Range from 95% to 0%.
    '''
    observer_num_matrix = torch.matmul(visible_frames, visible_frames.transpose(0, 1))  # 各个mask之间都可见的frame数量
    observer_num_list = observer_num_matrix.flatten()
    observer_num_list = observer_num_list[observer_num_list > 0].cpu().numpy()
    observer_num_thresholds = []
    for percentile in range(95, -5, -5):
        observer_num = np.percentile(observer_num_list, percentile)
        if observer_num <= 1:
            if percentile < 50:
                break
            else:
                observer_num = 1
        observer_num_thresholds.append(observer_num)
    return observer_num_thresholds


def init_nodes(global_frame_mask_list, mask_project_on_all_frames, contained_masks, undersegment_mask_ids,
               mask_point_clouds):
    nodes = []
    for global_mask_id, (frame_id, mask_id) in enumerate(global_frame_mask_list):
        if global_mask_id in undersegment_mask_ids:
            continue
        mask_list = [(frame_id, mask_id)]
        frame = mask_project_on_all_frames[global_mask_id]
        frame_mask = contained_masks[global_mask_id]
        point_ids = mask_point_clouds[f'{frame_id}_{mask_id}']
        node_info = (0, len(nodes))
        node = Node(mask_list, frame, frame_mask, point_ids, node_info, None)
        nodes.append(node)
    return nodes
