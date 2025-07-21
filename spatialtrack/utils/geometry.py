import numpy as np

def judge_bbox_overlay(bbox_1, bbox_2):
    for i in range(3):
        if bbox_1[0][i] > bbox_2[1][i] or bbox_2[0][i] > bbox_1[1][i]:
            return False
    return True

def denoise(pcd):
    labels = np.array(pcd.cluster_dbscan(eps=0.04, min_points=4)) + 1 # -1 for noise
    mask = np.ones(len(labels), dtype=bool)
    count = np.bincount(labels)

    # remove component with less than 20% points
    for i in range(len(count)):
        if count[i] < 0.2 * len(labels):
            mask[labels == i] = False

    remain_index = np.where(mask)[0]
    pcd = pcd.select_by_index(remain_index)
    
    pcd, index = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    remain_index = remain_index[index]
    return pcd, remain_index

def filter_boundary(depth, delta=0.05):
    remove_mask = np.zeros(depth.shape).astype(bool)
    delta_depth_1 = np.abs(depth[1:, :] - depth[:-1, :])
    delta_depth_2 = np.abs(depth[:, 1:] - depth[:, :-1])
    remove_mask[1:, :] = remove_mask[1:, :] | (delta_depth_1 > delta)
    remove_mask[:-1, :] = remove_mask[:-1, :] | (delta_depth_1 > delta)
    remove_mask[:, 1:] = remove_mask[:, 1:] | (delta_depth_2 > delta)
    remove_mask[:, :-1] = remove_mask[:, :-1] | (delta_depth_2 > delta)
    depth[remove_mask] = 0
    return depth