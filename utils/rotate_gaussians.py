import glob

import open3d as o3d
import numpy as np
import cv2
import json
import os
from collections import defaultdict

from sklearn.cluster import KMeans

from base_utils.colmap_read_write_model import *
from base_utils.colmap_read_model import *
from scipy.spatial.transform import Rotation

from tqdm import tqdm

if __name__ == "__main__":
    scene_dir = "/home/bytedance/Projects/3DGS/FunctionSplatting++/2d-gaussian-splatting/output/DeepBlending/playroom/test++/instances/99/generate_info/generate_gaussian_rotate.ply"
    rotate_colmap = False
    rotate_gs = {
        "xyz": True,
        "rotation": True,
        "sh": True
    }

    rotation_matrix_4x4 = np.eye(4)
    # 将旋转角度转换为旋转矩阵
    rotation_matrix = Rotation.from_euler('xyz', [0, 0, 90], degrees=True).as_matrix()
    # rotation_matrix[:, 1:3] *= -1
    rotation_matrix_4x4[:3, :3] = rotation_matrix

    from scene.gaussian_model import GaussianModel
    import torch

    with torch.no_grad():
        gaussians = GaussianModel(sh_degree=3)
        gaussians.use_seg_feature = True
        gaussians.seg_feat_dim = 16
        gaussians.load_seg_feat = True
        gaussians.load_ply(
            "/home/bytedance/Projects/3DGS/FunctionSplatting++/2d-gaussian-splatting/output/DeepBlending/playroom/test++/instances/99/generate_info/generate_gaussian.ply")

        gaussians_center = gaussians.get_xyz.mean(axis=0)
        gaussians._xyz -= gaussians_center

        rotation_matrix_4x4 = torch.from_numpy(rotation_matrix_4x4).cuda()
        if rotate_gs["xyz"]:
            gaussians._xyz = gaussians._xyz @ rotation_matrix_4x4[:3, :3].float().T

        if rotate_gs["rotation"]:
            quat_r = Rotation.from_matrix(rotation_matrix).as_quat()  # 得到 [x, y, z, w] 四元数
            quat_r = quat_r[[3, 0, 1, 2]]  # 转换为 [w, x, y, z]


            def quaternion_multiply(q1, q2):
                w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
                w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

                w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
                x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
                y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
                z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

                return torch.stack((w, x, y, z), -1)


            # 假设你的四元数数组为 ellipsoid_quats，shape 为 [1000, 4]
            # 旋转四元数 quat_r, shape 为 [4]
            gaussians._rotation = quaternion_multiply(torch.from_numpy(quat_r).float().cuda(),
                                                      gaussians.get_rotation)

        if rotate_gs["sh"]:
            from e3nn import o3
            import einops
            import einsum


            def transform_shs(shs_feat, rotation_matrix):
                ## rotate shs
                P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # switch axes: yzx -> xyz
                permuted_rotation_matrix = np.linalg.inv(P) @ rotation_matrix @ P
                rot_angles = o3._rotation.matrix_to_angles(torch.from_numpy(permuted_rotation_matrix))

                # Construction coefficient
                D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2])
                D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2])
                D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2])

                # rotation of the shs features
                one_degree_shs = shs_feat[:, 0:3]
                one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
                one_degree_shs = np.einsum(
                    "... i j, ... j -> ... i",
                    D_1,
                    one_degree_shs,
                )
                one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
                shs_feat[:, 0:3] = one_degree_shs

                two_degree_shs = shs_feat[:, 3:8]
                two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
                two_degree_shs = np.einsum(
                    "... i j, ... j -> ... i",
                    D_2,
                    two_degree_shs,
                )
                two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
                shs_feat[:, 3:8] = two_degree_shs

                three_degree_shs = shs_feat[:, 8:15]
                three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
                three_degree_shs = np.einsum(
                    "... i j, ... j -> ... i",
                    D_3,
                    three_degree_shs,
                )
                three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
                shs_feat[:, 8:15] = three_degree_shs

                return shs_feat


            sh = gaussians._features_rest.cpu().numpy()
            sh = transform_shs(sh, rotation_matrix)
            gaussians._features_rest = torch.from_numpy(sh).float().cuda()

        gaussians._xyz += gaussians_center
        gaussians._xyz += torch.tensor([4.3, 0.3, 0]).float().cuda()
        if rotate_gs["xyz"] or rotate_gs["rotation"] or rotate_gs["sh"]:
            gaussians.save_ply(
                "/home/bytedance/Projects/3DGS/FunctionSplatting++/2d-gaussian-splatting/output/DeepBlending/playroom/test++/instances/99/generate_info/generate_gaussian_rotate.ply")
