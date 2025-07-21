#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_mesh_utils import get_ray_directions, get_rays


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, segmap=None, sorted_segmap=None,
                 image_name=None, uid=None, normal=None, depth=None, gau_related_pixels=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda",
                 use_train=True
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0) if image is not None else None  # .to(self.data_device)
        if normal is not None:
            self.normal = normal  # .to(self.data_device)
            normal_norm = torch.norm(self.normal, dim=0, keepdim=True)
            self.normal_mask = ~((normal_norm > 1.1) | (normal_norm < 0.9))
            self.normal = self.normal / normal_norm
        else:
            self.normal = None
            self.normal_mask = None

        self.segmap = segmap
        self.sorted_segmap = sorted_segmap

        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask  # .to(self.data_device)
        else:
            # self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.gt_alpha_mask = None

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                     fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.use_train = use_train

    def convert2c2w_intrinsics(self):
        W2C = np.eye(4)
        W2C[:3] = np.concatenate([np.linalg.inv(self.R), self.T[:, None]], -1)  # W2C
        c2w = np.linalg.inv(W2C)

        intrinsic = np.eye(4)
        focal = (self.image_width / 2) / (np.tan(self.FoVx / 2))
        intrinsic[0, 0] = focal
        intrinsic[1, 1] = focal
        intrinsic[0, 2] = self.image_width / 2
        intrinsic[1, 2] = self.image_height / 2
        return c2w, intrinsic

    def get_mesh_normal(self, mesh_tracer):
        # 生成rays_d,rays_o
        c2w, intrinsic = self.convert2c2w_intrinsics()
        Height, Width = self.image_height, self.image_width

        rays_o, rays_d, rays_d_norm = get_rays(
            get_ray_directions(Height, Width, torch.from_numpy(intrinsic[:3, :3]).float())[0],
            torch.from_numpy(c2w[:3]).float())
        rays_o, rays_d = rays_o.reshape(-1, 3).cuda(), rays_d.reshape(-1, 3).cuda()

        positions, face_normals, _, _ = mesh_tracer.trace(rays_o, rays_d)

        normals = face_normals.reshape(Height, Width, 3)
        # from PIL import Image
        # Image.fromarray(np.uint8(((normals + 1) / 2).cpu().detach().numpy() * 255.0)).show()
        self.normal = normals.permute(2, 0, 1)
        normal_norm = torch.norm(self.normal, dim=0, keepdim=True) + 1e-9
        self.normal_mask = ~((normal_norm > 1.1) | (normal_norm < 0.9))
        self.normal = self.normal / normal_norm


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
