# import umap
import copy
import glob
import os
import sys
from collections import defaultdict

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import tqdm
from PIL import Image
from sklearn.decomposition import PCA


def importance_sampling(semantic_map, num_samples=1000):
    """
    根据语义图的标签比例进行重要性采样，采样 num_samples 个像素，且不使用 for 循环
    semantic_map: Tensor, 大小为 (H, W)，表示每个像素的标签
    num_samples: int, 要采样的像素数量
    """
    # 获取语义图的高度和宽度
    H, W = semantic_map.shape

    # 统计每个标签的数量
    unique_labels, counts = torch.unique(semantic_map, return_counts=True)

    # 计算每个标签的比例
    probabilities = counts.float() / counts.sum()

    # 将语义图展平为一维向量
    semantic_map_flat = semantic_map.view(-1).type(torch.int64)  # (H * W)

    # 创建与语义图大小一致的权重张量，初始值为0
    weights = torch.zeros_like(semantic_map_flat, dtype=torch.float32)

    # 使用 scatter_ 来直接为每个标签分配权重
    weights.scatter_(0, semantic_map_flat, probabilities[semantic_map_flat])

    # 归一化权重，使其和为 1
    weights = weights / weights.sum()

    # 使用权重进行随机采样，采样 num_samples 个像素
    sampled_indices = torch.multinomial(weights, num_samples, replacement=False)

    # 将采样到的索引还原为二维坐标 (H, W)
    sampled_y = sampled_indices // W
    sampled_x = sampled_indices % W

    return sampled_y, sampled_x


def contrastive_loss(features, masks, predef_u_list=None, min_pixnum=0, temp_lambda=1000):
    '''
    :param features:
    :param masks:
    :param predef_u_list:
    :param min_pixnum:
    :param temp_lambda:
    :return:
    '''
    valid_semantic_idx = masks > 0  # note: no consider negative masks
    # 同时过滤掉pixnum<20
    mask_ids, mask_nums = torch.unique(masks, return_counts=True)
    valid_mask_ids = mask_ids[mask_nums > min_pixnum]
    valid_semantic_idx = valid_semantic_idx & torch.isin(masks, valid_mask_ids)

    masks = masks[valid_semantic_idx].type(torch.int64)  # 移除了0
    masks = masks - 1
    features = features[valid_semantic_idx, :]  # N,16
    features = features / (torch.norm(features, dim=-1, keepdim=True) + 1e-9).detach()

    mask_ids, mask_nums = torch.unique(masks, return_counts=True)
    if predef_u_list is not None:
        u_list = predef_u_list[mask_ids + 1]
    # remapping the mask index
    label_mapping = torch.zeros(mask_ids.max() + 1, dtype=torch.long).cuda()
    label_mapping[mask_ids] = torch.arange(len(mask_ids)).cuda()
    masks = label_mapping[masks]
    mask_ids, mask_nums = torch.unique(masks, return_counts=True)

    mask_num = mask_ids.shape[0]  # cluster number
    # note: mask=0被移除掉了
    # 计算各个mask的均值和方差
    if predef_u_list is None:
        u_list_sum = torch.zeros(mask_num, features.shape[1]).cuda()
        u_list_sum.scatter_add_(0, masks.unsqueeze(1).expand(-1, features.shape[1]), features)
        u_list = u_list_sum / mask_nums[:, None]  # 均值 N_labels,16

    cluster_diff = features - u_list[masks]  # 每个样本与其所属均值的差
    cluster_diff_norm = torch.norm(cluster_diff, dim=1, keepdim=True)
    phi_list_sum = torch.zeros(mask_num, 1).cuda()
    phi_list_sum.scatter_add_(0, masks.unsqueeze(1), cluster_diff_norm)
    phi_list = phi_list_sum / (mask_nums.unsqueeze(1) * torch.log(mask_nums.unsqueeze(1) + temp_lambda))
    phi_list = torch.clip(phi_list * 10, min=0.5, max=1.0)  # why x10
    phi_list = phi_list.detach()  # 方差

    dist = torch.exp(torch.matmul(features, u_list.T) / phi_list.T)  # [N_pix, N_cluster]
    dist_sum = dist.sum(dim=1, keepdim=True)
    # 计算最终的 ProtoNCE loss
    ProtoNCE = -torch.sum(torch.log(dist[torch.arange(features.shape[0]), masks].unsqueeze(1) / (dist_sum + 1e-9)))

    return ProtoNCE


def consist_3d_feat_loss(sample_feat, sampled_gaussian_xyz, global_feat, global_xyz, topk=5):
    sample_feat = sample_feat / (sample_feat.norm(dim=-1, keepdim=True) + 1e-9)
    global_feat = global_feat / (global_feat.norm(dim=-1, keepdim=True) + 1e-9)

    # dists = sample_feat @ global_feat.T
    dists = torch.cdist(sampled_gaussian_xyz, global_xyz)  # 距离最近的gs
    # dists = torch.cdist(sample_feat, global_feat)  # Compute pairwise distances
    _, neighbor_indices_tensor = dists.topk(topk, largest=False)

    neighbor_feat = global_feat[neighbor_indices_tensor[:, 1:]]
    return (1 - torch.matmul(sample_feat[:, None], neighbor_feat.transpose(1, 2))).mean()


@torch.no_grad()
def reclustering_3d_mask(clusteringSeg3D_labels, global_feat, global_xyz, score_threshold=0.9):
    global_feat = global_feat / (global_feat.norm(dim=-1, keepdim=True) + 1e-9)

    origin2map2update = defaultdict(list)
    clusteringSeg3D_masks_update = [clusteringSeg3D_labels.cpu().numpy() == 0]
    labels_cnts = 1
    for i in range(clusteringSeg3D_labels.max())[1:]:
        current_mask = clusteringSeg3D_labels == i

        selected_pseudo_3d_feat = global_feat[current_mask]
        selected_pseudo_3d_feat_mean = selected_pseudo_3d_feat.mean(0)

        feat_score = selected_pseudo_3d_feat_mean @ global_feat.T
        chosen_gs_mask = feat_score >= score_threshold
        if chosen_gs_mask.sum() == 0:
            clusteringSeg3D_masks_update[0][current_mask] = True
            continue

        selected_global_xyz = global_xyz[chosen_gs_mask]

        pcld = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(selected_global_xyz.detach().cpu().numpy()))
        labels = np.array(pcld.cluster_dbscan(eps=0.3, min_points=4)) + 1  # -1 for noise
        # print()
        for label in np.unique(labels):
            update_mask = np.zeros((len(current_mask),), dtype=bool)  # N_pts
            chosen_label_mask = labels == label  # N_labels

            sys.stdout.write('\r')
            sys.stdout.write(
                f"Loading label {chosen_label_mask.sum()} {label}/{labels.max()} {i}/{clusteringSeg3D_labels.max()}")
            sys.stdout.flush()

            update_mask[chosen_gs_mask.cpu().numpy()] = chosen_label_mask
            if chosen_label_mask.sum() < 20:
                # 这个instance实际上应该被归为0
                clusteringSeg3D_masks_update[0][update_mask] = True
                # clusteringSeg3D_masks_update[0][chosen_gs_mask.cpu().numpy()][chosen_label_mask] = True
            else:
                clusteringSeg3D_masks_update.append(update_mask)
                origin2map2update[i].append(labels_cnts)
                labels_cnts += 1

    clusteringSeg3D_masks_update = np.stack(clusteringSeg3D_masks_update, axis=-1)

    # 重复点
    # clusteringSeg3D_masks_update = filter_repeat_gs_labels(clusteringSeg3D_masks_update, global_feat, global_xyz)

    return clusteringSeg3D_masks_update, origin2map2update


def filter_clustering_3d_loss(selected_pseudo_3d_feat, global_feat, global_xyz, score_threshold=0.9):
    # 基于threshold找到全局global_feat中和selected-feat相似的，然后基于位置做聚类
    global_feat = global_feat / (global_feat.norm(dim=-1, keepdim=True) + 1e-9)
    feat_score = selected_pseudo_3d_feat @ global_feat.T
    selected_global_feat = global_feat[feat_score >= score_threshold]
    selected_global_xyz = global_xyz[feat_score >= score_threshold]

    pcld = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(selected_global_xyz.detach().cpu().numpy()))
    labels = np.array(pcld.cluster_dbscan(eps=0.3, min_points=4)) + 1  # -1 for noise

    print(np.unique(labels))
    pclds = [o3d.io.read_point_cloud("/home/bytedance/Projects/Datasets/ZipNeRF/nyc/meetingroom/sparse/0/points3D.ply")]
    for label in np.unique(labels):
        pcld = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(selected_global_xyz.detach().cpu().numpy()[labels == label]))
        color = np.random.rand(3)
        pcld.paint_uniform_color(color)
        pclds.append(pcld)

        if len(pcld.points) > 3:
            bbox = o3d.geometry.OrientedBoundingBox.create_from_points(pcld.points)
            bbox.color = color
            pclds.append(bbox)
    o3d.visualization.draw_geometries(pclds)

    return contrastive_loss(selected_global_feat, torch.from_numpy(labels + 1).cuda())


def feature_to_rgb(features, pca_proj_mat=None, type="PCA"):
    # Input features shape: (16, H, W)

    # Reshape features for PCA
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T

    sam_norm = features_reshaped / (features_reshaped.norm(dim=1, keepdim=True) + 1e-9)
    features_reshaped = sam_norm  # * 0.5 + 0.5  # [-1,1]->[0,1]

    if pca_proj_mat is not None:
        low_feat = (features_reshaped @ pca_proj_mat).reshape(H, W, 3).cpu().numpy()
    else:
        # Apply PCA and get the first 3 components
        if type == "PCA":
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(features_reshaped.cpu().numpy())

            # Reshape back to (H, W, 3)
            low_feat = pca_result.reshape(H, W, 3)

    # Normalize to [0, 255]
    low_feat = (low_feat * 0.5 + 0.5).clip(0, 1)
    feat_normalized = 255 * (low_feat)  # * 0.5 + 0.5

    rgb_array = feat_normalized.astype('uint8')

    return rgb_array


def feature3d_to_rgb(features):
    sam_norm = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-9)
    features_reshaped = sam_norm  # * 0.5 + 0.5  # [-1,1]

    # Apply PCA and get the first 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped)
    # tsne = TSNE(n_components=3, random_state=42, perplexity=10, n_iter=500)
    # pca_result = tsne.fit_transform(features_reshaped)
    return ((pca_result + 1).clip(0, 2) / 2) * 0.7 + 0.3
    # (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min()) * 0.7 + 0.3


def mask_to_rgb(mask):
    mask_image = mask.detach().cpu().numpy()
    num_classes = np.max(mask_image) + 1
    colors = plt.get_cmap('hsv', num_classes)
    norm = mcolors.Normalize(vmin=0, vmax=num_classes - 1)
    colored_segmentation = colors(norm(mask_image))
    return np.uint8(colored_segmentation[..., :3] * 255.0)


def rearrange_mask(mask_folder, mask_assocation_info):
    mask_files = sorted(glob.glob(f"{mask_folder}/*"))
    save_dir = os.path.join(os.path.dirname(mask_folder), "mask_sorted")
    # if os.path.exists(save_dir):
    #    return
    os.makedirs(save_dir, exist_ok=True)

    masks_origin = []
    for mask_file in tqdm.tqdm(mask_files):
        masks_origin.append(np.array(Image.open(mask_file)))

    masks_origin = np.stack(masks_origin)
    masks_new = np.zeros_like(masks_origin, dtype=np.int16)

    for cluster_id, cluster_info in tqdm.tqdm(clustering_result.items()):
        if cluster_id == 0:
            continue
        for frame_mask_id in cluster_info['mask_list']:
            frame_id, mask_id = frame_mask_id[:2]
            masks_new[frame_id][masks_origin[frame_id] == mask_id] = cluster_id  # 更新为instance

    for mask_id in tqdm.tqdm(range(len(masks_origin))):
        save_path = os.path.join(save_dir, os.path.basename(mask_files[mask_id]))
        Image.fromarray(masks_new[mask_id]).save(save_path)


def filter_3d_mask(clusteringSeg3D_masks, num_threshold=50):
    # 将数量少的点变成了0
    valid_labels = []
    clusteringSeg3D_masks_update = [clusteringSeg3D_masks[:, 0]]
    for i in range(clusteringSeg3D_masks.shape[1])[1:]:
        curr_mask = clusteringSeg3D_masks[:, i]
        if curr_mask.sum() < num_threshold:  # 同时更新[0]
            clusteringSeg3D_masks_update[0] = np.logical_or(clusteringSeg3D_masks[:, 0], curr_mask)
            continue
        else:
            valid_labels.append(i)
            clusteringSeg3D_masks_update.append(curr_mask)
    return np.stack(clusteringSeg3D_masks_update, -1), valid_labels


def filter_2d_mask(clusteringSeg2D_masks, num_threshold=50):
    # 对Mask进行过滤，过滤掉包含了多个instance的mask
    # 将点云投影到2D Mask上，如果点云80%有效投影(可见的)属于该Mask，则将该Mask归属于该Instance
    # 如果一个mask属于多个instance，则丢弃该Mask
    print()


def update_2d_mask(clusteringSeg2D_masks, valid_labels):
    # 过滤掉3D点太少了的instance
    clusteringSeg2D_masks_update = {}
    class_cnts = list(clusteringSeg2D_masks.keys())[0]  # TODO: clusteringSeg2D_masks可能没有1！
    for class_3d, class_2d in clusteringSeg2D_masks.items():
        if class_3d in valid_labels or class_3d == 0:
            clusteringSeg2D_masks_update[class_cnts] = class_2d
            class_cnts += 1
        else:
            # 点太少的instance合并入0
            if 0 not in clusteringSeg2D_masks_update.keys():
                clusteringSeg2D_masks_update[0] = class_2d
            else:
                # 本身已经有0了
                clusteringSeg2D_masks_update[0]["point_ids"] = np.concatenate(
                    [clusteringSeg2D_masks_update[0]["point_ids"], class_2d["point_ids"]]
                )
                clusteringSeg2D_masks_update[0]['mask_list'] += class_2d['mask_list']
                clusteringSeg2D_masks_update[0]['repre_mask_list'] += class_2d['repre_mask_list']

    return clusteringSeg2D_masks_update


def filter_repeat_gs_labels(clusteringSeg3D_masks, global_feat, global_xyz, score_threshold=0.9):
    global_feat = global_feat / (global_feat.norm(dim=-1, keepdim=True) + 1e-9)

    # 找到重复的mask
    gs_repeat_mask = clusteringSeg3D_masks.sum(axis=-1) > 1
    repeat_gs_feats = global_feat[gs_repeat_mask]

    class_average_feats = []
    for i in range(clusteringSeg3D_masks.shape[1])[1:]:
        # bug: 有可能全是重复
        class_average_feats.append(global_feat[(clusteringSeg3D_masks[:, i] & ~gs_repeat_mask)].mean(0))
    class_average_feats = torch.stack(class_average_feats, 0)  # 少了一个0

    repeat_class_score = repeat_gs_feats @ class_average_feats.T
    best_repeat_class_score, best_class_idx = torch.topk(repeat_class_score, k=1)
    best_class_idx = best_class_idx + 1  # 因为没有0
    best_class_idx[best_repeat_class_score <= score_threshold] = 0  # 全部算0

    clusteringSeg3D_masks[gs_repeat_mask] = False
    clusteringSeg3D_masks[
        np.arange(0, len(clusteringSeg3D_masks))[gs_repeat_mask], best_class_idx.squeeze().cpu().numpy()] = True
    '''
    for id in tqdm.tqdm(best_class_idx.unique()):
        pcld = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
            global_xyz[clusteringSeg3D_masks[:, id]].detach().cpu().numpy()))
        pcld.paint_uniform_color([1, 0, 0])
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(pcld.points)
        bbox.color = np.array([1, 0, 0])

        pcld_repeat = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
            global_xyz[gs_repeat_mask][best_class_idx[:, 0] == id].detach().cpu().numpy()))
        pcld_repeat.paint_uniform_color([0, 1, 0])

        o3d.visualization.draw_geometries([pcld, bbox, pcld_repeat, o3d.io.read_point_cloud(
            "/home/bytedance/Projects/Datasets/ZipNeRF/nyc/meetingroom/sparse/0/points3D.ply")])
    '''
    return clusteringSeg3D_masks


@torch.no_grad()
def filter_invalid_mask_segmap(clusteringSeg3D_masks, gaussian, viewcams, render_func, threshold=0.5,
                               save_dir=None):
    '''
    找到当前视角各个有效mask对应的gs，如果和各个instance交集占各自的60%，则保留该mask
    '''
    print("\nFiltering Invalid Segmentation Mask...")
    invalid_frame_mask_list = []

    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    gs_total_ids = np.arange(0, clusteringSeg3D_masks.shape[0])  # 所有gs的idx

    mask_instance_match_list = {}

    for frame_idx, viewcam in tqdm.tqdm(enumerate(viewcams)):
        render_out = render_func(viewcam, gaussian, gaussian.pipelineparams, background)
        gau_related_pixels = render_out['gau_related_pixels']
        # 很有可能和pixel数量不对应
        gaus_ids = gau_related_pixels[:, 0].detach().cpu().numpy()
        pixel_ids = gau_related_pixels[:, 1].detach().cpu().numpy()  # 每一个gs对应的pixel_id

        segmap = viewcam.segmap.numpy().reshape(-1)
        ids = np.unique(segmap)
        ids.sort()

        # 找到每个2d mask对应的gs
        mask3d_gs_relations = np.zeros((clusteringSeg3D_masks.shape[0], len(ids)), dtype=bool)
        for mask_idx in range(len(ids)):
            mask_id = ids[mask_idx]
            if mask_id == 0:
                continue
            segmentation = segmap == mask_id
            segmentation_pixel_idxs = np.where(segmentation)[0]

            gs_related = gaus_ids[np.isin(pixel_ids, segmentation_pixel_idxs)]  # 属于当前mask对应pixel对应的gs
            mask3d_gs_relations[gs_related, mask_idx] = True  # 当前mask和gaussian之间的对应关系
        '''
        pclds = []
        for mask3d in mask3d_gs_relations.T:
            if mask3d.sum() == 0:
                print("!")
            mask_pcld = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gaussian.get_xyz.cpu().numpy()[mask3d]))
            mask_pcld.paint_uniform_color(np.random.rand(3))
            pclds.append(mask_pcld)
        o3d.visualization.draw_geometries(pclds)
        '''

        # 找到各个instance可见的gs
        visibility_filter = np.isin(gs_total_ids, np.unique(gaus_ids))  # render_out['visibility_filter'].cpu().numpy()
        current_3dmask = clusteringSeg3D_masks.copy()
        current_3dmask = current_3dmask[visibility_filter]
        mask3d_gs_relations = mask3d_gs_relations[visibility_filter]  # 当前可见的gaussian
        # 不在当前

        # [N_3d,N_2d]
        # 当前instance3d和mask2d之间对应gs的重叠关系
        interaction_2d_3d = current_3dmask[..., None] & mask3d_gs_relations[:, None]  # 每个3dmask的gs & 每个2dmask的gs
        interaction_2d_3d_sum = interaction_2d_3d.sum(0)  # 相交的总点数
        mask3d_sum = current_3dmask.sum(0)
        mask2d_sum = mask3d_gs_relations.sum(0)  # mask2d的总点数

        # 相交区域占至少一个instance的50%;相交区域占自己区域的50%
        # 找到每个mask相交面积最大的instance
        percent_inter3d = interaction_2d_3d_sum / (mask3d_sum[:, None] + 1e-7)
        percent_inter2d = interaction_2d_3d_sum / (mask2d_sum[None] + 1e-7)  # 各个instance占2d的%

        valid_ids_mask = (percent_inter3d > threshold).any(0) & (percent_inter2d > threshold).any(0)
        invalid_ids = ids[~valid_ids_mask]

        # 每个mask相交的instance
        mask_instance_ids = percent_inter2d.T.argsort(axis=-1)[:, -1]  # [N_label3d,N_mask2d]
        mask_instance_ids[~valid_ids_mask] = 0
        mask_instance_match_list[viewcam.image_name] = mask_instance_ids

        for invalid_id in invalid_ids:
            invalid_frame_mask_list.append([frame_idx, invalid_id])
    if len(invalid_frame_mask_list) > 0:
        invalid_frame_mask_list = np.stack(invalid_frame_mask_list)
    return invalid_frame_mask_list, mask_instance_match_list


def save_enhance_instance_mask(mask_instance_match_list, view_cams, save_dir):
    print("\nSaving Enhanced Instance Mask...")

    os.makedirs(save_dir, exist_ok=True)
    # 重新更新mask
    for cam in tqdm.tqdm(view_cams):
        frame_name = cam.image_name
        segmap = cam.segmap[0].cpu().numpy()

        enhance_mask_ids = mask_instance_match_list[frame_name]
        sorted_segmap = copy.deepcopy(segmap)
        for mask_id, instance_id in enumerate(enhance_mask_ids):
            sorted_segmap[sorted_segmap == mask_id] = instance_id

            Image.fromarray(np.uint16(sorted_segmap)).save(os.path.join(save_dir, frame_name + ".png"))


def save_undersegmask(undersegment_masks, view_cams, save_dir):
    images = defaultdict(list)

    os.makedirs(save_dir, exist_ok=True)

    for frame_mask_id in undersegment_masks:
        images[view_cams[frame_mask_id[0]].image_name].append(frame_mask_id[-1])

    for cam in view_cams:
        frame_name = cam.image_name
        frame_underseg_masks = np.zeros((cam.image_height, cam.image_width))
        segmap = cam.segmap[0].cpu().numpy()

        underseg_mask_ids = images[frame_name]
        for mask_id in underseg_mask_ids:
            frame_underseg_masks[segmap == mask_id] = mask_id

        Image.fromarray(np.uint8(frame_underseg_masks)).save(os.path.join(save_dir, frame_name + ".png"))
    print("Saving undersegment masks...")
