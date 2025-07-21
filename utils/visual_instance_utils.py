import argparse
import glob

import open3d as o3d
import numpy as np
import trimesh
import raytracing
import os
from tqdm import tqdm
import torch
from PIL import Image


def get_camera_frustum(img_size, K, C2W, frustum_length, color, scale_ratio=0.1):
    # pose_scale用于放大位姿
    # [w,h]  [4,4]  [3,4]
    H, W = img_size
    hfov = np.rad2deg(np.arctan(W / 2. / K[0, 0]) * 2.)  # 光心到图像左右两边的角度
    vfov = np.rad2deg(np.arctan(H / 2. / K[1, 1]) * 2.)  # 光心到图像上下两边的角度
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))  # 归一化平面
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))
    # 不就是W,H的一半

    pose = np.eye(4)
    pose[:3] = C2W
    C2W = pose

    # build view frustum for camera (I, 0)
    frustum_points = np.array([[0., 0., 0.],  # frustum origin
                               [-half_w, -half_h, frustum_length],  # 左上角，注意x朝右，y朝上
                               [half_w, -half_h, frustum_length],  # 右上角
                               [half_w, half_h, frustum_length],  # bottom-right image corner
                               [-half_w, half_h, frustum_length],
                               # 坐标轴
                               [0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]  # x轴,y轴,z轴
                               ])  # bottom-left image corner

    frustum_points *= scale_ratio

    frustum_lines = np.array([[0, i] for i in range(1, 5)] +
                             [[i, (i + 1)] for i in range(1, 4)] +
                             [[4, 1], [0, 5], [0, 6], [0, 7]])
    # 平铺
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))
    frustum_colors[-1] = np.array([0, 0, 1])  # Z 蓝色
    frustum_colors[-2] = np.array([0, 1, 0])  # y 绿色
    frustum_colors[-3] = np.array([1, 0, 0])  # x 红色

    # frustum_colors = np.vstack((np.tile(np.array([[1., 0., 0.]]), (4, 1)),
    #                            np.tile(np.array([[0., 1., 0.]]), (4, 1))))

    # transform view frustum from (I, 0) to (R, t)
    frustum_points = np.dot(
        np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))),  # 齐次坐标
        C2W.T)  # 8，4
    # 归一化矩阵乘以C2W.T
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]

    # 将这些点转换成线
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(frustum_points)
    lineset.lines = o3d.utility.Vector2iVector(frustum_lines)
    lineset.colors = o3d.utility.Vector3dVector(frustum_colors)
    return lineset


def filter_occ_cam(raytracer, meshtracer, points, train_cams, intrinsic, img_hw, candidate_cams, use_train_cams=False,
                   valid_threshold=0.75):
    # TODO: 使用mesh来过滤遮挡，而不是直接点云，有些gs可能被自身给挡住
    # 首先选出可见的，再选出和GT最接近的
    points = np.array(points)

    if use_train_cams:
        # closet_train_cams = train_cams[np.linalg.norm(train_cams[:, :3, 3] - points.mean(0), axis=1).argsort()[:30]]
        candidate_cams = train_cams
    '''
    else:
        closet_train_cams = train_cams[np.linalg.norm(train_cams[:, :3, 3] - points.mean(0), axis=1).argsort()[:10]]
        candidate_cams = candidate_cams[
            np.linalg.norm(candidate_cams[:, :3, 3][:, None] - closet_train_cams[:, :3, 3][None],
                           axis=-1).sum(axis=-1).argsort()[:10]]
    '''
    # return candidate_cams
    # occ-mask
    candidate_cams_origin = candidate_cams[:, :3, 3]

    # 只考虑和mesh有交点的rays
    rays_d = (points[:, None] - candidate_cams_origin[None]).reshape(-1, 3)
    rays_d_norm = np.linalg.norm(rays_d, axis=1)
    rays_d = rays_d / rays_d_norm[:, None]
    rays_o = candidate_cams_origin[None].repeat(len(points), 0).reshape(-1, 3)

    line_depth = rays_d_norm.reshape(len(points), len(candidate_cams_origin))  # 各个点到各个cam的距离
    # 首先过滤掉被自己挡住的points

    # 首先找到和mesh有交点的rays，然后这些rays还没被其他挡住，如果没有被挡住的rays占和mesh有交点的rays的75%，则算好rays
    _, _, _, instance_depth = meshtracer.trace(torch.from_numpy(rays_o).cuda(),
                                               torch.from_numpy(rays_d).cuda())
    instance_depth = instance_depth.reshape(len(points), len(candidate_cams_origin)).cpu().numpy()
    valid_gs_points_mask = np.abs(line_depth - instance_depth) < 0.1  # 实际上各个视角可见的points

    intersection, _, _, depth = raytracer.trace(torch.from_numpy(rays_o).cuda(),
                                                torch.from_numpy(rays_d).cuda())
    trace_depth = depth.reshape(len(points), len(candidate_cams_origin)).cpu().numpy()

    occ_mask = (np.abs(line_depth - trace_depth) < 0.25) & valid_gs_points_mask  # 本身没有被自己挡住的点& 本身又没有被场景挡住
    # valid_mask
    height, width = img_hw
    vertex_cam = (np.linalg.inv(candidate_cams)[None] @
                  np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)[:, None, :,
                  None])

    vertex_img = (intrinsic[None, None] @ vertex_cam).squeeze()[..., :3]
    vertex_img_coords = vertex_img[..., :2] / (vertex_img[..., 2:] + 1e-5)
    valid_cam_depth_mask = vertex_img[..., 2] > 0
    valid_img_mask = np.logical_and(
        np.logical_and(vertex_img_coords[..., 0] >= 0, vertex_img_coords[..., 0] < width),
        np.logical_and(vertex_img_coords[..., 1] >= 0, vertex_img_coords[..., 1] < height))

    occ_mask = occ_mask & valid_img_mask & valid_cam_depth_mask  # 先判断可见性，再选最近的
    valid_gs_points_mask = valid_gs_points_mask & valid_img_mask & valid_cam_depth_mask

    # 生成位姿：再选和GT最近的；GT位姿：再选和点云最近的
    if use_train_cams:
        chosen_index = occ_mask.sum(0).argsort()[-10:]
        print("Chosen index:", chosen_index)
        candidate_cams = candidate_cams[chosen_index]  # 最近的10个
        return candidate_cams, chosen_index
    else:
        candidate_cams = candidate_cams[occ_mask.sum(0) / valid_gs_points_mask.sum(0) > valid_threshold]
        # 如果80%都可见，那么返回cam
        return candidate_cams, occ_mask.sum(0) / valid_gs_points_mask.sum(0) > valid_threshold


def create_spheric_poses(radius, origin, n_poses=12,
                         elevation=30,
                         radius_level=[3], up_world=None):
    def pad_camera_extrinsics_4x4(extrinsics):
        if extrinsics.shape[-2] == 4:
            return extrinsics
        padding = np.array([[0, 0, 0, 1]])
        if extrinsics.ndim == 3:
            padding = padding[None].repeat(extrinsics.shape[0], 0)
        extrinsics = np.concatenate([extrinsics, padding], axis=-2)
        return extrinsics

    def center_looking_at_camera_pose(camera_position, look_at=None,
                                      up_world=None):
        """
        Create OpenGL camera extrinsics from camera locations and look-at position.

        camera_position: (M, 3) or (3,)
        look_at: (3)
        up_world: (3)
        return: (M, 3, 4) or (3, 4)
        """
        # by default, looking at the origin and world up is z-axis
        if look_at is None:
            look_at = np.array([0, 0, 0], dtype=np.float32)
        if up_world is None:
            up_world = np.array([0, 0, -1], dtype=np.float32)

        # OpenGL camera: z-backward, x-right, y-up
        # ！ Colmap camera: z-front, x-right, y-down
        z_axis = look_at - camera_position
        z_axis = z_axis / np.linalg.norm(z_axis, axis=-1, keepdims=True)  # F.normalize(z_axis, dim=-1).float()
        x_axis = np.cross(up_world, z_axis, axis=-1)
        x_axis = x_axis / np.linalg.norm(x_axis, axis=-1, keepdims=True)  # F.normalize(x_axis, dim=-1).float()
        y_axis = np.cross(z_axis, x_axis, axis=-1)
        y_axis = y_axis / np.linalg.norm(y_axis, axis=-1, keepdims=True)  # F.normalize(y_axis, dim=-1).float()

        extrinsics = np.stack([x_axis, y_axis, z_axis, camera_position], axis=-1)
        extrinsics = pad_camera_extrinsics_4x4(extrinsics)
        return extrinsics

    def spherical_camera_pose(azimuths: np.ndarray, elevations: np.ndarray, radius=2.5, up_world=None):
        # azimuths = np.deg2rad(azimuths) 360度
        # elevations = np.deg2rad(elevations) # 仰

        xs = radius * np.cos(elevations) * np.cos(azimuths)
        ys = radius * np.cos(elevations) * np.sin(azimuths)
        zs = radius * np.sin(elevations)

        cam_locations = np.array([xs, ys, zs])

        c2ws = center_looking_at_camera_pose(cam_locations, up_world=up_world)
        return c2ws

    spheric_poses = []
    origin_trans = np.eye(4)
    origin_trans[:3, 3] = np.array(origin)
    for level in radius_level:
        for azimuth in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:  # 绕z轴360度
            # for elevation in np.linspace(0, np.pi / 2, 6 + 1)[2:3]:  # 绕x轴90度
            elevations = np.pi * elevation / 180
            spheric_poses += [
                (origin_trans @ spherical_camera_pose(azimuth, elevations,
                                                      radius * level,
                                                      up_world=up_world))]  # 36 degree view downwards
    return np.stack(spheric_poses, 0)


def filter_normal_cam(candidate_cams, instance_normal, instance_bbox, angle_thres=np.pi / 4):
    instance_normal = instance_normal[:2] / (np.linalg.norm(instance_normal[:2]) + 1e-9)
    # 先找到最近的xy轴
    axis_dir = np.array([
        [1, 0], [-1, 0], [0, 1], [0, -1]
    ])
    instance_normal = axis_dir[(instance_normal[None] * axis_dir).sum(-1).argsort()[-1]]  # 余弦越大越相似

    cams_origin = candidate_cams[:, :3, 3]
    instance_center = instance_bbox.center
    cams_dir = cams_origin - instance_center
    cams_dir = cams_dir[:, :2] / np.linalg.norm(cams_dir[:, :2], axis=-1)[:, None]

    cos_cam_instance = (cams_dir * instance_normal[None]).sum(-1)
    valid_cam_mask = np.logical_and(cos_cam_instance >= np.cos(angle_thres) * 0.95, cos_cam_instance <= 1)
    return candidate_cams[valid_cam_mask], instance_normal, valid_cam_mask
