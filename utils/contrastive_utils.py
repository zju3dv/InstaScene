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


def contrastive_loss(features, masks, predef_u_list=None, min_pixnum=0, temp_lambda=1000,
                     consider_negative=False):
    '''
    :param features:
    :param masks:
    :param predef_u_list:
    :param min_pixnum:
    :param temp_lambda:
    :return:
    '''
    if not consider_negative:
        valid_semantic_idx = masks > 0  # note:已经移除0了
    else:  # 考虑0标签
        valid_semantic_idx = torch.ones_like(masks, dtype=torch.bool).cuda()

    mask_ids, mask_nums = torch.unique(masks, return_counts=True)
    valid_mask_ids = mask_ids[mask_nums > min_pixnum]
    valid_semantic_idx = valid_semantic_idx & torch.isin(masks, valid_mask_ids)

    masks = masks[valid_semantic_idx].type(torch.int64)
    if not consider_negative:
        masks = masks - 1  # from zero
    features = features[valid_semantic_idx, :]  # N,16
    features = features / (torch.norm(features, dim=-1, keepdim=True) + 1e-9).detach()

    mask_ids, mask_nums = torch.unique(masks, return_counts=True)
    if predef_u_list is not None:
        u_list = predef_u_list[mask_ids]
    # remapping the mask index -> continual indices
    label_mapping = torch.zeros(mask_ids.max() + 1, dtype=torch.long).cuda()
    label_mapping[mask_ids] = torch.arange(len(mask_ids)).cuda()
    masks = label_mapping[masks]
    mask_ids, mask_nums = torch.unique(masks, return_counts=True)

    mask_num = mask_ids.shape[0]  # cluster number
    # compute average and variance
    if predef_u_list is None:
        # compute current mask average
        u_list_sum = torch.zeros(mask_num, features.shape[1]).cuda()
        u_list_sum.scatter_add_(0, masks.unsqueeze(1).expand(-1, features.shape[1]), features)
        u_list = u_list_sum / mask_nums[:, None]

    cluster_diff = features - u_list[masks]
    cluster_diff_norm = torch.norm(cluster_diff, dim=1, keepdim=True)
    phi_list_sum = torch.zeros(mask_num, 1).cuda()
    phi_list_sum.scatter_add_(0, masks.unsqueeze(1), cluster_diff_norm)
    phi_list = phi_list_sum / (mask_nums.unsqueeze(1) * torch.log(mask_nums.unsqueeze(1) + temp_lambda))
    phi_list = torch.clip(phi_list * 10, min=0.5, max=1.0)
    phi_list = phi_list.detach()  # variance

    dist = torch.exp(torch.matmul(features, u_list.T) / phi_list.T)  # [N_pix, N_cluster]
    dist_sum = dist.sum(dim=1, keepdim=True)
    # Final ProtoNCE loss
    ProtoNCE = -torch.sum(torch.log(dist[torch.arange(features.shape[0]), masks].unsqueeze(1) / (dist_sum + 1e-9)))

    return ProtoNCE


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
