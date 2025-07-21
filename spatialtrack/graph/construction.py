import torch
import numpy as np
from tqdm import tqdm
import open3d as o3d
from gaussian_renderer import render
from spatialtrack.graph.node import Node


@torch.no_grad()
def get_segmap_point(gaussian, view, frame_id):
    # TODO: 只需要 [mask_id:point_id], frame_pts_id
    # 超参
    DISTANCE_THRESHOLD = 0.03
    COVERAGE_THRESHOLD = 0.3

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    gau_related_pixels = render(view, gaussian, gaussian.pipelineparams, background)['gau_related_pixels']
    # 很有可能和pixel数量不对应
    gaus_ids = gau_related_pixels[:, 0]
    pixel_ids = gau_related_pixels[:, 1]
    # pixel_ids_y = gau_related_pixels[:, 1] // view.image_width  # pix_id = W * pix.y + pix.x
    # pixel_ids_x = gau_related_pixels[:, 1] % view.image_height

    scene_points = gaussian.get_xyz
    view_points = scene_points[gaus_ids]
    # pcld = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(view_points))
    # o3d.visualization.draw_geometries([pcld])

    mask_image = view.segmap.cuda().reshape(-1)
    ids = torch.unique(mask_image).cpu().numpy()
    ids.sort()

    mask_info = {}  # id: pts_id
    frame_point_ids = set(gaus_ids.tolist())
    for mask_id in ids:
        if mask_id == 0:
            continue
        segmentation = mask_image == mask_id
        valid_mask = segmentation[pixel_ids]

        mask_info[mask_id] = set(gaus_ids[valid_mask].tolist())
        # mask_points = view_points[valid_mask]

    return mask_info, list(frame_point_ids)

    '''
    mask_points_list = []
    mask_points_num_list = []
    scene_points_list = []
    scene_points_num_list = []
    selected_point_ids_list = []
    initial_valid_mask_ids = []

    for mask_id in ids:
        if mask_id == 0:
            continue
        segmentation = mask_image == mask_id
        valid_mask = segmentation[pixel_ids]

        mask_points = view_points[valid_mask]

        def crop_scene_points():
            x_min, x_max = torch.min(mask_points[:, 0]), torch.max(mask_points[:, 0])
            y_min, y_max = torch.min(mask_points[:, 1]), torch.max(mask_points[:, 1])
            z_min, z_max = torch.min(mask_points[:, 2]), torch.max(mask_points[:, 2])

            selected_point_mask = (scene_points[:, 0] > x_min) & (scene_points[:, 0] < x_max) & (
                    scene_points[:, 1] > y_min) & (
                                          scene_points[:, 1] < y_max) & (scene_points[:, 2] > z_min) & (
                                          scene_points[:, 2] < z_max)
            selected_point_ids = torch.where(selected_point_mask)[0]
            cropped_scene_points = scene_points[selected_point_ids]
            return cropped_scene_points, selected_point_ids

        cropped_scene_points, selected_point_ids = crop_scene_points() # 每个instance区域内的点云
        initial_valid_mask_ids.append(mask_id)
        mask_points_list.append(mask_points)
        scene_points_list.append(cropped_scene_points)
        mask_points_num_list.append(len(mask_points))
        scene_points_num_list.append(len(cropped_scene_points))
        selected_point_ids_list.append(selected_point_ids)

    mask_points_tensor = torch.nn.utils.rnn.pad_sequence(mask_points_list, batch_first=True, padding_value=0)
    scene_points_tensor = torch.nn.utils.rnn.pad_sequence(scene_points_list, batch_first=True, padding_value=0)
    lengths_1 = torch.tensor(mask_points_num_list).cuda()  # mask对应的点
    lengths_2 = torch.tensor(scene_points_num_list).cuda()  # 场景crop的点

    _, neighbor_in_scene_pcld, _ = ball_query(mask_points_tensor, scene_points_tensor, lengths_1, lengths_2, K=20,
                                              radius=DISTANCE_THRESHOLD, return_nn=False)

    valid_mask_ids = []
    mask_info = {}
    frame_point_ids = set()

    for i, mask_id in enumerate(initial_valid_mask_ids):  # 有效的点
        mask_neighbor = neighbor_in_scene_pcld[i]  # P, 20
        mask_point_num = mask_points_num_list[i]  # Pi
        mask_neighbor = mask_neighbor[:mask_point_num]  # Pi, 20

        valid_neighbor = mask_neighbor != -1  # Pi, 20
        neighbor = torch.unique(mask_neighbor[valid_neighbor])
        neighbor_in_complete_scene_points = selected_point_ids_list[i][neighbor].cpu().numpy()
        coverage = torch.any(valid_neighbor, dim=1).sum().item() / mask_point_num

        if coverage < COVERAGE_THRESHOLD:  # 0.3
            continue
        valid_mask_ids.append(mask_id)
        mask_info[mask_id] = set(neighbor_in_complete_scene_points)
        frame_point_ids.update(mask_info[mask_id])

    return mask_info, list(frame_point_ids)
    '''


def build_point_in_mask_matrix(gaussian, viewstack, filter_repeat_points=False):
    '''
    Args:
        gaussian:
        viewstack:

    Returns:
        boundary_points: a set of points that are contained by multiple masks in a frame and thus are on the boundary of the masks. We will not consider these points in the following computation of view consensus rate.
        point_in_mask_matrix: the 'point in mask' matrix.
        mask_point_clouds: a dict where each key is the mask id in a frame, and the value is the point ids that are in this mask.
        point_frame_matrix: a matrix of size (scene_points_num, frame_num). For point i and frame j, if point i is visible in frame j, then M[i,j] = True. Otherwise, M[i,j] = False.
        global_frame_mask_list: a list of masks in the whole sequence. Each tuple contains the frame id and the mask id in this frame.

    '''
    iterator = tqdm(enumerate(viewstack), total=len(viewstack))

    boundary_points = set()
    point_in_mask_matrix = np.zeros((len(gaussian.get_xyz), len(viewstack)), dtype=np.uint16)
    point_frame_matrix = np.zeros((len(gaussian.get_xyz), len(viewstack)), dtype=bool)
    global_frame_mask_list = []
    mask_point_clouds = {}

    for frame_cnt, view in iterator:
        # 找到当前view包含的点云
        mask_dict, frame_point_cloud_ids = get_segmap_point(gaussian, view, frame_cnt)
        point_frame_matrix[frame_point_cloud_ids, frame_cnt] = True  # gs在frame可见
        appeared_point_ids = set()
        frame_boundary_point_index = set()
        for mask_id, mask_point_cloud_ids in mask_dict.items():
            frame_boundary_point_index.update(mask_point_cloud_ids.intersection(appeared_point_ids))  # 查看是否有重叠
            mask_point_clouds[f'{frame_cnt}_{mask_id}'] = mask_point_cloud_ids
            point_in_mask_matrix[list(mask_point_cloud_ids), frame_cnt] = mask_id
            appeared_point_ids.update(mask_point_cloud_ids)
            global_frame_mask_list.append((frame_cnt, mask_id))
        if filter_repeat_points:
            point_in_mask_matrix[list(frame_boundary_point_index), frame_cnt] = 0
            boundary_points.update(frame_boundary_point_index)
        torch.cuda.empty_cache()

    return boundary_points, point_in_mask_matrix, mask_point_clouds, point_frame_matrix, global_frame_mask_list


def process_masks(viewstack, global_frame_mask_list, point_in_mask_matrix, boundary_points, mask_point_clouds,
                  clustering_args):
    '''
        For each mask, compute the frames that it is visible and the masks that contains it.
        Meanwhile, we judge whether this mask is undersegmented.
    '''
    visible_frames = []
    contained_masks = []
    undersegment_mask_ids = []

    iterator = tqdm(global_frame_mask_list)  # 遍历mask
    for frame_id, mask_id in iterator:
        # 每一个mask可见的frame & 包含的mask
        valid, visible_frame, contained_mask = process_one_mask(point_in_mask_matrix, boundary_points,
                                                                mask_point_clouds[f'{frame_id}_{mask_id}'], viewstack,
                                                                global_frame_mask_list,
                                                                clustering_args)
        visible_frames.append(visible_frame)
        contained_masks.append(contained_mask)
        if not valid:
            global_mask_id = global_frame_mask_list.index((frame_id, mask_id))
            undersegment_mask_ids.append(global_mask_id)  # 该mask在某些视角下被分割为多个mask，则欠分割
        torch.cuda.empty_cache()

    visible_frames = torch.stack(visible_frames, dim=0).cuda()  # (mask_num, frame_num)
    contained_masks = torch.stack(contained_masks, dim=0).cuda()  # (mask_num, mask_num)

    # Undo the effect of undersegment observer masks to avoid merging two objects that are actually separated
    for global_mask_id in undersegment_mask_ids:  # 移除undersegment
        frame_id, _ = global_frame_mask_list[global_mask_id]
        global_frame_id = frame_id
        mask_projected_idx = torch.where(contained_masks[:, global_mask_id])[0]
        contained_masks[:, global_mask_id] = False  # 排除掉该mask的影响
        visible_frames[mask_projected_idx, global_frame_id] = False

    return visible_frames, contained_masks, undersegment_mask_ids


def process_one_mask(point_in_mask_matrix, boundary_points, mask_point_cloud, frame_list, global_frame_mask_list, args):
    '''
        point_in_mask_matrix:[N_pts,N_frames]: 点云在Frame的哪个Mask中
        For a mask, compute the frames that it is visible and the masks that contains it.
    '''
    visible_frame = torch.zeros(len(frame_list))
    contained_mask = torch.zeros(len(global_frame_mask_list))

    valid_mask_point_cloud = mask_point_cloud - boundary_points  # 很细的对象可能被移除
    mask_point_cloud_info = point_in_mask_matrix[list(valid_mask_point_cloud), :]  # 每个点在各个frame mask中的idx
    # 看当前mask包含的点在各个frame中包含的情况
    possibly_visible_frames = np.where(np.sum(mask_point_cloud_info, axis=0) > 0)[0]  # 当前点在各个frame有效

    split_num = 0
    visible_num = 0

    for frame_id in possibly_visible_frames:
        # 当前mask点云在各个frame下属于哪些mask
        mask_id_count = np.bincount(mask_point_cloud_info[:, frame_id])  # 数组索引表示整数值，具体值表示出现的次数 统计这些点云在各个视角出现的情况
        invisible_ratio = mask_id_count[0] / np.sum(mask_id_count)  # 0 means that this point is invisible in this frame
        # If in a frame, most points in this mask are missing, then we think this mask is invisible in this frame.
        if 1 - invisible_ratio < args.mask_visible_threshold and (np.sum(mask_id_count) - mask_id_count[0]) < 500:
            continue  # 大部分点不可见；万一总共都没500点？
        visible_num += 1
        mask_id_count[0] = 0
        max_mask_id = np.argmax(mask_id_count)  # 峰值，该mask在另一个frame下投影最多的mask
        contained_ratio = mask_id_count[max_mask_id] / np.sum(mask_id_count)  # 峰值
        if contained_ratio > args.contained_threshold:  # 0.8
            # 如果另一个视角的mask占了该mask的0.8，则说明包含
            visible_frame[frame_id] = 1
            frame_mask_idx = global_frame_mask_list.index((frame_id, max_mask_id))
            contained_mask[frame_mask_idx] = 1
        else:
            split_num += 1  # This mask is splitted into two masks in this frame

    if visible_num == 0 or split_num / visible_num > args.undersegment_filter_threshold:
        return False, visible_frame, contained_mask
    else:
        return True, visible_frame, contained_mask


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
