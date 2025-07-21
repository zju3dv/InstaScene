import glob
import os
import numpy as np
import torch
from PIL import Image
from kornia import create_meshgrid
import open3d as o3d
from tqdm import tqdm

import raytracing

import torch.nn.functional as F


def get_ray_directions(H, W, K):
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)  # i->width,j->height
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24json
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    directions = \
        torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], -1)  # (H, W, 3)
    return directions, torch.stack([i, j, torch.ones_like(i)], -1)  # [x,y]


def get_rays(directions, c2w):
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T  # (H, W, 3)
    rays_d_norm = torch.norm(rays_d, dim=-1, keepdim=True)
    rays_d = rays_d / rays_d_norm  # mlp要归一化
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape)  # (H, W, 3)

    return rays_o, rays_d, rays_d_norm  # depth


def warp_known_target_views(known_images, known_masks, pose_deltas, input_mesh, c2ws_all):
    '''

    Args:
        known_images: [N,256,256,3]
        known_masks:
        pose_deltas:
        input_mesh:

    Returns:

    '''

    K = torch.tensor([
        [280, 0, 128, 0],
        [0, 280, 128, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]).float()
    width, height = known_images[0].shape[1], known_images[0].shape[0]
    mesh_tracer = raytracing.RayTracer(np.array(input_mesh.vertices),
                                       np.array(input_mesh.triangles))
    all_target_images_list = []  # 最后各个视角的投影
    all_target_depths_list = []  # 最后各个视角的投影
    all_target_masks_list = []

    for frame_idx in tqdm(range(16)):
        if frame_idx in pose_deltas:
            continue

        novel_image = torch.ones_like(known_images[0]).cuda()
        novel_depth = torch.zeros_like(known_masks[0]).cuda()
        novel_warped_mask = -torch.ones_like(known_masks[0]).cuda()  # 记录当前pixel来自哪些图像

        c2w = c2ws_all[frame_idx]

        directions, pixel_coords = get_ray_directions(height, width,
                                                      K[:3, :3])
        rays_o, rays_d, rays_d_norm = get_rays(directions, c2w[:3])

        rays_o = rays_o.reshape(-1, 3).cuda()
        rays_d = rays_d.reshape(-1, 3).cuda()

        intersections, intersect_normals, _, intersect_depths = mesh_tracer.trace(rays_o, rays_d)

        normal_angle = (intersect_normals * -rays_d).sum(dim=-1)
        normal_mask = torch.logical_and(normal_angle > 0, normal_angle < 1)  # 有交

        valid_mask = (normal_mask & (intersect_depths < intersect_depths.max()))  # 有效的pixel

        K_source = K.cuda()[:3, :3]
        valid_intersections = intersections[valid_mask]  # 将这些点投影到
        # 将其他视角投影到input视角
        target_warp_images_list = []

        # 先从离其最近的
        resort_idxs = np.abs(frame_idx - np.array(pose_deltas)).argsort()  # 从小到大->离得最近
        sorted_pose_deltas = np.array(pose_deltas)[resort_idxs]
        sorted_images = known_images[resort_idxs]
        sorted_masks = known_masks[resort_idxs]
        for target_idx, warp_frame_idx in enumerate(sorted_pose_deltas):  # 根据远近
            target_warp_images = torch.ones_like(known_images[0]).cuda()
            warp_target_c2w = c2ws_all[warp_frame_idx].cuda()
            warp_target_w2c = warp_target_c2w.inverse()
            target_image = sorted_images[target_idx]
            target_image = target_image.permute(2, 0, 1).float().cuda()
            target_mask = sorted_masks[target_idx].cuda()
            # 将intersect投影到当前视角

            mesh_vertices_cam = torch.cat(
                [valid_intersections, torch.ones_like(valid_intersections[:, :1]).cuda()],
                dim=-1) @ warp_target_w2c.T

            vertices_cam_depths = mesh_vertices_cam[:, 2:3] # 这是target view的深度，有问题！
            mesh_vertices_cam = mesh_vertices_cam[:, :3] / (mesh_vertices_cam[:, 2:3] + 1e-8)
            mesh_vertices_img = (mesh_vertices_cam @ K_source.T)[:, :2]

            # 2. 2D采样的同时，判断有无越界
            valid_img_mask = torch.logical_and(
                torch.logical_and(mesh_vertices_img[..., 0] >= 0, mesh_vertices_img[..., 0] < width),
                torch.logical_and(mesh_vertices_img[..., 1] >= 0, mesh_vertices_img[..., 1] < height))

            # 3. 判断遮挡，相机光心到顶点的距离和raytracing的距离是否一样
            rays_d = valid_intersections - warp_target_c2w[:3, 3][None]
            depth_cam2vert_line = rays_d.norm(dim=-1, keepdim=True)  # 点直接到相机的距离
            rays_d = rays_d / (depth_cam2vert_line + 1e-8)
            rays_o = warp_target_c2w[:3, 3][None].expand(rays_d.size(0), -1)
            pts, _, _, depth_cam2vert_cast = mesh_tracer.trace(rays_o, rays_d)
            occ_mask = (depth_cam2vert_line.squeeze() - depth_cam2vert_cast).abs() < 0.1  # Vertex对各个视角的可见性
            # 4. 只保留raytracing不为0的
            intersect_mask = depth_cam2vert_cast < depth_cam2vert_cast.max()

            valid_mesh_mask = occ_mask & intersect_mask & valid_img_mask

            # 5. 进行采样
            mesh_vertices_img_norm = mesh_vertices_img.clone()
            mesh_vertices_img_norm[..., 0] = mesh_vertices_img_norm[..., 0] / width * 2 - 1
            mesh_vertices_img_norm[..., 1] = mesh_vertices_img_norm[..., 1] / height * 2 - 1

            intersections_rgb = F.grid_sample(target_image[None],
                                              mesh_vertices_img_norm[None, :, None].float(),
                                              align_corners=True, mode="nearest").squeeze().permute(1, 0)
            intersections_mask = F.grid_sample(target_mask[None, None].float(),
                                               mesh_vertices_img_norm[None, :, None].float(),
                                               align_corners=True, mode="nearest").squeeze()
            valid_mesh_mask = valid_mesh_mask & (intersections_mask > 0)

            if valid_mesh_mask.any():
                valid_pixel_coords = pixel_coords.cuda().reshape(-1, 3)[:, :2][valid_mask][
                    valid_mesh_mask].long()  # xy

                # 只更新为-1的
                update_coords_mask = novel_warped_mask[valid_pixel_coords[:, 1], valid_pixel_coords[:, 0]] < 0
                novel_warped_mask[valid_pixel_coords[:, 1], valid_pixel_coords[:, 0]] = warp_frame_idx
                valid_pixel_coords = valid_pixel_coords[update_coords_mask]

                valid_pixel_rgb = intersections_rgb[valid_mesh_mask]
                valid_pixel_depth = vertices_cam_depths[valid_mesh_mask].squeeze()

                novel_image[valid_pixel_coords[:, 1], valid_pixel_coords[:, 0]] = valid_pixel_rgb[
                    update_coords_mask]
                novel_depth[valid_pixel_coords[:, 1], valid_pixel_coords[:, 0]] = valid_pixel_depth[
                    update_coords_mask]

                target_warp_images[valid_pixel_coords[:, 1], valid_pixel_coords[:, 0]] = valid_pixel_rgb[
                    update_coords_mask]
                target_warp_images_list.append(target_warp_images)

            # 判断有无越界，

        all_target_images_list.append(novel_image)
        all_target_depths_list.append(((novel_depth - 0.5) / 2).clip(0, 1))  # [1,0]
        all_target_masks_list.append(novel_warped_mask > 0)

    target_depths = np.uint8(torch.stack(all_target_depths_list, dim=0).cpu().numpy() * 255)

    results_images = np.zeros((16, 256, 256, 3))
    results_depths = np.zeros((16, 256, 256))
    target_idxs = np.arange(0, 16)
    target_idxs = target_idxs[~np.isin(target_idxs, pose_deltas)]
    results_images[target_idxs] = np.uint8(torch.stack(all_target_images_list, dim=0).cpu().numpy() * 255.0)
    results_images[pose_deltas] = np.uint8(known_images.cpu().numpy() * 255.0)
    results_depths[target_idxs] = target_depths
    results_image = Image.fromarray(
        np.vstack(np.stack(
            [np.uint8(np.hstack(results_images)), np.uint8(np.hstack(results_depths))[..., None].repeat(3, -1)], 0))
    )

    return torch.stack(all_target_images_list).float().permute(0, 3, 1, 2), torch.stack(
        all_target_depths_list).float(), torch.stack(all_target_masks_list).float(), results_image
