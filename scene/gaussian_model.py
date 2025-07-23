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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import open3d as o3d
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.contrastive_utils import feature3d_to_rgb

from scipy.spatial.transform import Rotation
from e3nn import o3
import einops
import einsum


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1),
                                        rotation).permute(0, 2, 1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:, :3, :3] = RS
            trans[:, 3, :3] = center
            trans[:, 3, 3] = 1
            return trans

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        self._seg_feature = None
        self.use_seg_feature = False
        self.seg_feat_dim = 0
        self.load_seg_feat = False

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)  # .clamp(max=1)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_seg_feature(self):
        if self._seg_feature is not None:
            return self._seg_feature / (torch.norm(self._seg_feature, p=2, dim=1, keepdim=True) + 1e-6)
        return self._seg_feature

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def set_segfeat_params(self, modelparams):
        self.use_seg_feature = modelparams.use_seg_feature
        self.seg_feat_dim = modelparams.seg_feat_dim
        self.load_seg_feat = modelparams.load_seg_feat

    def set_3d_feat(self, Seg3D_masks, gram_feat=False):
        '''
        gram_feat: more robust 3D feature as supervision
        :param clustering3d_mask:
        :param gram_feat:
        :return:
        '''
        self.class_feat = None
        if self._seg_feature is None:
            # 开始feature训练的时候，往模型中加入language feature参数
            seg_feature = torch.rand((self._xyz.shape[0], self.seg_feat_dim), device="cuda")
            if gram_feat:
                init_feat = torch.rand((Seg3D_masks.shape[1], self.seg_feat_dim)).cuda()

                def gram_schmidt(vectors):
                    orthogonal_vectors = []
                    for v in vectors:
                        for u in orthogonal_vectors:
                            v = v - torch.dot(v, u) * u
                        orthogonal_vectors.append(v / (torch.norm(v) + 1e-9))
                    return torch.stack(orthogonal_vectors)

                init_feat = gram_schmidt(init_feat)

                for i in range(Seg3D_masks.shape[1]):
                    curr_mask = Seg3D_masks[:, i]
                    gs_ids = torch.from_numpy(np.where(curr_mask)[0]).cuda()
                    seg_feature[gs_ids] = init_feat[i]  # torch.rand((self.seg_feat_dim)).cuda()

                self.class_feat = init_feat

            seg_feature = seg_feature / (seg_feature.norm(dim=1, keepdim=True) + 1e-9)
            self._seg_feature = nn.Parameter(seg_feature.requires_grad_(True))

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, require_grad=True):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(require_grad))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(require_grad))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(require_grad))
        self._scaling = nn.Parameter(scales.requires_grad_(require_grad))
        self._rotation = nn.Parameter(rots.requires_grad_(require_grad))
        self._opacity = nn.Parameter(opacities.requires_grad_(require_grad))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args,
                       optim_seg_feature=True,
                       optim_xyz=True,
                       optim_sh=True,
                       optim_scale=True,
                       optim_rotate=True,
                       optim_opacity=True):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        if self.use_seg_feature and optim_seg_feature:
            if self._seg_feature is None:
                # 开始feature训练的时候，往模型中加入language feature参数
                seg_feature = torch.rand((self._xyz.shape[0], self.seg_feat_dim), device="cuda")  # TODO: 只有3维
                seg_feature = seg_feature / seg_feature.norm(dim=1, keepdim=True)
                self._seg_feature = nn.Parameter(seg_feature.requires_grad_(True))

            l = [
                {'params': [self._seg_feature], 'lr': training_args.seg_feature_lr,
                 "name": "language_feature"},  # TODO: training_args.language_feature_lr
            ]
            self._xyz.requires_grad_(False)
            self._features_dc.requires_grad_(False)
            self._features_rest.requires_grad_(False)
            self._scaling.requires_grad_(False)
            self._rotation.requires_grad_(False)
            self._opacity.requires_grad_(False)
        else:
            self._xyz.requires_grad_(optim_xyz)
            self._features_dc.requires_grad_(optim_sh)
            self._features_rest.requires_grad_(optim_sh)
            self._scaling.requires_grad_(optim_scale)
            self._rotation.requires_grad_(optim_rotate)
            self._opacity.requires_grad_(optim_opacity)
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self, export_as_3dgs=False):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        if not export_as_3dgs:
            for i in range(self._scaling.shape[1]):
                l.append('scale_{}'.format(i))
        else:
            for i in range(self._scaling.shape[1] + 1):
                l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        if self._seg_feature is not None:
            for i in range(self._seg_feature.shape[1]):
                l.append('segfeat_{}'.format(i))
        return l

    def save_ply(self, path, crop_mask=None):
        mkdir_p(os.path.dirname(path))

        if crop_mask is not None:
            valid_mask = crop_mask.detach().cpu()
        else:
            valid_mask = np.ones((len(self.get_xyz)), dtype=bool)

        xyz = self._xyz.detach().cpu().numpy()[valid_mask]
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()[valid_mask]
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()[
            valid_mask]
        opacities = self._opacity.detach().cpu().numpy()[valid_mask]
        scale = self._scaling.detach().cpu().numpy()[valid_mask]
        rotation = self._rotation.detach().cpu().numpy()[valid_mask]
        if self._seg_feature is not None:
            seg_feat = self._seg_feature.detach().cpu().numpy()[valid_mask]

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]
        if self._seg_feature is not None:
            attributes.append(seg_feat)
        attributes = np.concatenate(attributes, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        o3d_pointcloud = o3d.geometry.PointCloud()
        o3d_pointcloud.points = o3d.utility.Vector3dVector(xyz)
        o3d_pointcloud.colors = o3d.utility.Vector3dVector(SH2RGB(f_dc).clip(0., 1.))
        o3d.io.write_point_cloud(path.split(".")[0] + "_color.ply", o3d_pointcloud)
        if self._seg_feature is not None:
            o3d_pointcloud.colors = o3d.utility.Vector3dVector(feature3d_to_rgb(seg_feat))
            o3d.io.write_point_cloud(path.split(".")[0] + "_feat.ply", o3d_pointcloud)

    def save_ply_as_3dgs(self, path):
        print("### Saving PointCloud Params ###")

        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        scale = np.concatenate([scale, np.ones_like(scale[:, :1]) * np.log(1e-6)], axis=-1)
        rotation = self._rotation.detach().cpu().numpy()
        if self._seg_feature is not None:
            seg_feat = self._seg_feature.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(export_as_3dgs=True)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]
        if self._seg_feature is not None:
            attributes.append(seg_feat)
        attributes = np.concatenate(attributes, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        o3d_pointcloud = o3d.geometry.PointCloud()
        o3d_pointcloud.points = o3d.utility.Vector3dVector(xyz)
        o3d_pointcloud.colors = o3d.utility.Vector3dVector(SH2RGB(f_dc).clip(0., 1.))
        o3d.io.write_point_cloud(path.split(".")[0] + "_color.ply", o3d_pointcloud)
        if self._seg_feature is not None:
            o3d_pointcloud.colors = o3d.utility.Vector3dVector(feature3d_to_rgb(seg_feat))
            o3d.io.write_point_cloud(path.split(".")[0] + "_feat.ply", o3d_pointcloud)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        print("### Load the PointCloud Params ###")
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # 有可能是0阶
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))[:2]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # seg_feat
        if self.use_seg_feature and self.load_seg_feat:
            segfeat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("segfeat")]
            if len(segfeat_names) == self.seg_feat_dim:
                seg_feat = np.zeros((xyz.shape[0], self.seg_feat_dim))
                for idx in range(self.seg_feat_dim):
                    seg_feat[:, idx] = np.asarray(plydata.elements[0]["segfeat_" + str(idx)])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        if self.use_seg_feature and self.load_seg_feat:
            if len(segfeat_names) == self.seg_feat_dim:
                self._seg_feature = nn.Parameter(
                    torch.tensor(seg_feat, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def delete_ply(self, refer_ply_path):
        print("### Deleting the PointCloud Params ###")
        refer_points = np.array(o3d.io.read_point_cloud(refer_ply_path).points)
        gs_points = self.get_xyz.detach().cpu().numpy()
        mask = np.isin(gs_points.view([('', gs_points.dtype)] * gs_points.shape[1]),
                       refer_points.view([('', refer_points.dtype)] * refer_points.shape[1]))
        self.crop_mask(torch.from_numpy(mask.squeeze()).cuda())
        self.save_ply("./tmp.ply")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask, optimizer_type=True):
        valid_points_mask = ~mask

        if optimizer_type:
            optimizable_tensors = self._prune_optimizer(valid_points_mask)

            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]

            self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

            self.denom = self.denom[valid_points_mask]
            self.max_radii2D = self.max_radii2D[valid_points_mask]
        else:
            self._xyz = self._xyz[valid_points_mask]
            self._features_dc = self._features_dc[valid_points_mask]
            self._features_rest = self._features_rest[valid_points_mask]
            self._opacity = self._opacity[valid_points_mask]
            self._scaling = self._scaling[valid_points_mask]
            self._rotation = self._rotation[valid_points_mask]

            if self._seg_feature is not None:
                self._seg_feature = self._seg_feature[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:, :1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1

    ############# Instance
    def crop_mask(self, gs_mask, type="save"):
        # type:save delete
        if type == "delete":
            gs_mask = ~gs_mask
        self._xyz = self._xyz[gs_mask]
        self._features_dc = self._features_dc[gs_mask]
        self._features_rest = self._features_rest[gs_mask]
        self._opacity = self._opacity[gs_mask]
        self._scaling = self._scaling[gs_mask]
        self._rotation = self._rotation[gs_mask]
        if self.use_seg_feature:
            self._seg_feature = self._seg_feature[gs_mask]

    def combine_gaussian(self, new_gaussian, load_seg_feat=True):
        self._xyz = torch.cat([self._xyz, new_gaussian._xyz], dim=0)
        self._features_dc = torch.cat([self._features_dc, new_gaussian._features_dc], dim=0)
        self._features_rest = torch.cat([self._features_rest, new_gaussian._features_rest], dim=0)
        self._opacity = torch.cat([self._opacity, new_gaussian._opacity], dim=0)
        self._scaling = torch.cat([self._scaling, new_gaussian._scaling], dim=0)
        self._rotation = torch.cat([self._rotation, new_gaussian._rotation], dim=0)

        self._xyz = nn.Parameter(self._xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(self._features_dc.requires_grad_(True))
        self._features_rest = nn.Parameter(self._features_rest.requires_grad_(True))
        self._opacity = nn.Parameter(self._opacity.requires_grad_(True))
        self._scaling = nn.Parameter(self._scaling.requires_grad_(True))
        self._rotation = nn.Parameter(self._rotation.requires_grad_(True))

        if load_seg_feat and self.use_seg_feature:
            gs_feat_mean = (self._seg_feature / (self._seg_feature.norm(dim=-1, keepdim=True) + 1e-9)).mean(0)
            self._seg_feature = torch.cat([self._seg_feature,
                                           gs_feat_mean * torch.ones((len(new_gaussian.get_xyz), len(gs_feat_mean)),
                                                                     device="cuda")], dim=0)
            self._seg_feature = nn.Parameter(self._seg_feature.requires_grad_(True))

    def crop_pts_with_convexhull(self, pts, type="save", return_bbox=False):
        # 首先计算得到convex hull
        from scipy.spatial import ConvexHull, Delaunay
        delaunay = Delaunay(pts)
        points_inside_hull_mask = delaunay.find_simplex(self.get_xyz.detach().cpu().numpy()) >= 0
        if return_bbox:
            crop_points = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(self.get_xyz[points_inside_hull_mask].detach().cpu().numpy()))
            instance_bbox = o3d.geometry.AxisAlignedBoundingBox().create_from_points(crop_points.points)
            instance_bbox = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(instance_bbox)

            self.crop_mask(points_inside_hull_mask, type=type)

            return instance_bbox
        else:
            self.crop_mask(points_inside_hull_mask, type=type)

    '''
    @torch.no_grad
    def rotate_translate(self, rotation_degrees=np.array([0, 0, 0]), translate=np.array([0, 0, 0])):
        rotation_matrix_4x4 = np.eye(4)
        # 将旋转角度转换为旋转矩阵
        rotation_matrix = Rotation.from_euler('xyz', rotation_degrees, degrees=True).as_matrix()
        # rotation_matrix[:, 1:3] *= -1
        rotation_matrix_4x4[:3, :3] = rotation_matrix
        # rotation_matrix_4x4[:3, 3] = translate
        rotation_matrix_4x4 = torch.from_numpy(rotation_matrix_4x4).cuda()

        xyz_center = self._xyz.mean(dim=0)  # 绕着自身中心

        # xyz
        # 先回归原点
        self._xyz -= xyz_center
        self._xyz = self._xyz @ rotation_matrix_4x4[:3, :3].float().T
        self._xyz += xyz_center

        self._xyz += torch.from_numpy(translate).float().cuda()

        # rotate
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
        self._rotation = quaternion_multiply(torch.from_numpy(quat_r).float().cuda(),
                                             self.get_rotation)

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

        sh = self._features_rest.cpu().numpy()
        sh = transform_shs(sh, rotation_matrix)
        self._features_rest = torch.from_numpy(sh).float().cuda()
    

    @torch.no_grad
    def update_gaussians(self, update_mask, update_gaussian):
        self._xyz[update_mask] = update_gaussian._xyz
        self._features_dc[update_mask] = update_gaussian._features_dc
        self._features_rest[update_mask] = update_gaussian._features_rest
        self._opacity[update_mask] = update_gaussian._opacity
        self._scaling[update_mask] = update_gaussian._scaling
        self._rotation[update_mask] = update_gaussian._rotation
        if self.use_seg_feature and self.load_seg_feat:
            self._seg_feature[update_mask] = update_gaussian._seg_feature
    '''
