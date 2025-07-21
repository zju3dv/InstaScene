import os
from collections import defaultdict

from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
from PIL import Image
import cv2
import open3d as o3d
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from random import randint
import copy
import matplotlib
from scene.gaussian_model import GaussianModel
from utils.instance_utils import *
from gaussian_renderer import render
from arguments import OptimizationParams, PipelineParams
from utils.loss_utils import l1_loss, cos_loss, ssim
from kornia import create_meshgrid
import raytracing

from utils.zero123_utils import get_zero123plus_input_cameras, get_syncdreamer_input_cameras



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


class Renderer():
    def __init__(self, height=480, width=640, mesh=None, light_coff=0.5):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene(ambient_light=[light_coff, light_coff, light_coff])
        self.mesh = self.mesh_opengl(mesh)
        # self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES

    def __call__(self, height, width, intrinsics, pose):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width

        self.scene.clear()
        self.scene.add(self.mesh)

        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene, flags=pyrender.constants.RenderFlags.FLAT)  # , self.render_flags)

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh, smooth=False)

    def delete(self):
        self.renderer.delete()


class InstanceModel:
    def __init__(self, instance_label,
                 gaussian, gaussian_mask,
                 gen_cams_list, valid_gen_cam_masks,  # 记录有效的pose
                 zero123_cams_list, gen360_cams_list, cams_gen360_pose,
                 train_cams_list, train_cams_idx,
                 instance_bbox,
                 instance_mesh,
                 function_type, save_dir):
        '''
        Others:
            instance_bbox:{
                "center":
                "radius":
                "obj_axis":
                "bbox_bound":
            }

        '''
        self.instance_label = instance_label
        self.gaussian = gaussian
        self.gaussian_mask = gaussian_mask
        self.gen_cams_list = gen_cams_list
        self.valid_gen_cam_masks = valid_gen_cam_masks

        self.zero123_cams_list = zero123_cams_list
        self.gen360_cams_list = gen360_cams_list
        self.cams_gen360_pose = cams_gen360_pose
        self.train_cams_list = train_cams_list
        self.train_cams_idx = train_cams_idx
        self.instance_bbox = instance_bbox
        self.instance_mesh = instance_mesh
        self.function_type = function_type
        self.save_dir = save_dir

        self.cams = {
            "train": self.train_cams_list,
            "gen": self.gen_cams_list,
            "zero123": self.zero123_cams_list,
            "gen360": self.gen360_cams_list
        }

        # render
        self.render_cache = {
            "train": defaultdict(lambda: defaultdict(list)),
            "gen": defaultdict(lambda: defaultdict(list)),
            "zero123": defaultdict(lambda: defaultdict(list)),
            "gen360": defaultdict(lambda: defaultdict(list))
        }
        '''
        self.render_train_cache = {
            "rgb": [],
            "depth": [],
            "normal": [],
            "seg_feat": [],
            "instance_mask": []
        }
        self.render_gen_cache = {
            "rgb": [],
            "depth": [],
            "normal": [],
            "seg_feat": [],
            "instance_mask": []
        }
        self.render_zero123_cache = {
            "rgb": [],
            "depth": [],
            "normal": [],
            "seg_feat": [],
            "instance_mask": []
        }
        '''

        # segment
        self.sam_cache = {}

    @torch.no_grad()
    def compute_inpaint_visual_hull(self, scene,
                                    voxel_size=32, scale=1,
                                    n_unseen=3, n_instance=2,  # 见到的次数小于1，则为unseen，越大unseen越多；见到的次数大于2，则为instance
                                    use_mesh_tracer=True,
                                    load_from_cache=True):
        # TODO: bug:如果bbox内有其他物体挡住了，那可能instance_occupancy_grid会出问题
        # 发出射线，看场景哪些区域是空的
        # 根据bbox设定一个occupancy grid，如果rays和bbox有交点，并且该rays的depth大于和bbox的交点，则说明该rays和物体有交点，则将near,depth之间的点全部置0
        # pcld = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.gaussian.get_xyz.detach().cpu().numpy()))
        # bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(pcld.points)
        # bbox = o3d.geometry.AxisAlignedBoundingBox(self.instance_bbox['bbox_bound'][0], self.instance_bbox['bbox_bound'][1])

        bbox = o3d.geometry.AxisAlignedBoundingBox(self.instance_bbox["center"] - self.instance_bbox["radius"],
                                                   self.instance_bbox["center"] + self.instance_bbox["radius"])
        # 半径应该是radius
        # bbox.scale(1.5, (bbox.min_bound + bbox.max_bound) / 2)
        min_bound = torch.from_numpy(bbox.min_bound).cuda().float()
        max_bound = torch.from_numpy(bbox.max_bound).cuda().float()
        aabb_bound = torch.cat((min_bound, max_bound))

        occupancy_grid = OccupancyGrid(  # 全被占据，过滤见过的
            roi_aabb=aabb_bound,
            # x_min,y_min,z_min,x_max,y_max,z_max
            resolution=voxel_size,
            contraction_type=ContractionType.AABB).to("cuda")
        occupancy = torch.zeros_like(occupancy_grid.occs).reshape(voxel_size, voxel_size, voxel_size)
        # 最外圈
        occupancy_grid_instance = OccupancyGrid(
            roi_aabb=aabb_bound,
            # x_min,y_min,z_min,x_max,y_max,z_max
            resolution=voxel_size,
            contraction_type=ContractionType.AABB).to("cuda")  # 一开始全被占据
        occupancy_instance = torch.zeros_like(occupancy_grid_instance.occs).reshape(voxel_size, voxel_size,
                                                                                    voxel_size)  # 全都没被占据

        if os.path.exists(os.path.join(self.save_generate_dir, f"occgrid_unseen_{voxel_size}.pth")) and os.path.exists(
                os.path.join(self.save_generate_dir, f"occgrid_instance_{voxel_size}.pth")) and load_from_cache:
            occupancy_grid_binary = torch.load(os.path.join(self.save_generate_dir, f"occgrid_unseen_{voxel_size}.pth"))
            occupancy_grid_instance_binary = torch.load(
                os.path.join(self.save_generate_dir, f"occgrid_instance_{voxel_size}.pth"))
        else:
            background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
            print("\033[95mComputing Visual Hull...\033[0m")
            pclds_instance = []
            if use_mesh_tracer:
                mesh_tracer = raytracing.RayTracer(np.array(self.instance_mesh.vertices),
                                                   np.array(self.instance_mesh.triangles))
            for camera in tqdm(scene.getTrainCameras()):
                camera.image_height *= scale  # 提高分辨率
                camera.image_width *= scale

                c2w, intrinsic = camera.convert2c2w_intrinsics()
                Height, Width = camera.image_height, camera.image_width

                rays_o, rays_d, rays_d_norm = get_rays(
                    get_ray_directions(Height, Width, torch.from_numpy(intrinsic[:3, :3]).float())[0],
                    torch.from_numpy(c2w[:3]).float())
                rays_o, rays_d = rays_o.reshape(-1, 3).cuda(), rays_d.reshape(-1, 3).cuda()
                rays_d_norm = rays_d_norm.reshape(-1).cuda()
                # 和bbox有交点的就可以留下
                near, _ = ray_aabb_intersect(rays_o, rays_d, aabb_bound)
                # 有交点，但是可能被遮挡，保存和depth有交点，并且

                valid_mask = near < 100  # 和bbox有交点
                if not valid_mask.any():
                    print("Skipping...")
                    continue

                # 求depth
                rays_o = rays_o[valid_mask]
                rays_d = rays_d[valid_mask]
                rays_d_norm = rays_d_norm[valid_mask]
                if use_mesh_tracer:
                    _, _, _, depth_rays = mesh_tracer.trace(rays_o, rays_d)
                else:
                    render_result = render(camera, scene.gaussians, scene.gaussians.pipelineparams, background,
                                           norm_seg_feat=False)
                    depth_rays = render_result['surf_depth'].squeeze()[valid_mask.reshape(Height, Width)] * rays_d_norm

                near, far = ray_aabb_intersect(rays_o, rays_d, aabb_bound)  # 和bbox有交点的rays
                far[depth_rays <= far] = depth_rays[depth_rays <= far]  # 被遮挡
                # depth_rays[depth_rays >= far] = far[depth_rays >= far]
                # far = depth_rays[:, None]
                near = near[:, None]
                far = far[:, None]
                z_vals = near + (far - near) * torch.linspace(0, 1, 64).cuda()
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                pts_norm = ((pts - min_bound) / (max_bound - min_bound) * (voxel_size - 1)).long()
                pts_norm = pts_norm.reshape(-1, 3).clip(0, voxel_size - 1)
                occupancy[pts_norm[:, 0], pts_norm[:, 1], pts_norm[:, 2]] += 1  # 观察过
                # instance_occupancy_grid, depth_rays超过了far
                # 有一种情况，depth_rays比far大，比near小
                pts_instance = (rays_o + rays_d * depth_rays[:, None])[
                    (depth_rays <= far.squeeze()) & (depth_rays >= near.squeeze())]  # 打到物体表面的
                # pclds_instance.append(pts_instance.cpu().numpy())
                pts_instance_norm = ((pts_instance - min_bound) / (max_bound - min_bound) * (voxel_size - 1)).long()
                pts_instance_norm = pts_instance_norm.reshape(-1, 3).clip(0, voxel_size - 1)
                occupancy_instance[
                    pts_instance_norm[:, 0], pts_instance_norm[:, 1], pts_instance_norm[:, 2]] += 1  # 观察过
                torch.cuda.empty_cache()

            occupancy_grid_binary = occupancy_grid.binary.clone()
            occupancy_grid_instance_binary = occupancy_grid_instance.binary.clone()
            occupancy_grid_binary[occupancy < n_unseen] = True  # 见到的次数少于3次，希望是最内圈的
            occupancy_grid_instance_binary[occupancy_instance > n_instance] = True  # 希望是最外围的
            # unseen
            '''
            visual_grids = []
            color = np.random.random(3)
            voxel_size = len(occupancy_grid._binary)
            max_bound = occupancy_grid.roi_aabb[3:].cpu().numpy()
            min_bound = occupancy_grid.roi_aabb[:3].cpu().numpy()
            grid_indices = occupancy_grid.grid_coords
            valid_grid_mask = occupancy_grid_binary.view(-1, 1)[..., 0]
            valid_grid = grid_indices[valid_grid_mask].detach().cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                valid_grid / (voxel_size - 1) * (max_bound - min_bound) + min_bound)
            pcd.paint_uniform_color(color)
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=max(
                (max_bound - min_bound) / (voxel_size - 1)))
            visual_grids.append(voxel_grid)

            color = np.random.random(3)
            valid_grid_mask = occupancy_grid_instance_binary.view(-1, 1)[..., 0]
            valid_grid = grid_indices[valid_grid_mask].detach().cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                valid_grid / (voxel_size - 1) * (max_bound - min_bound) + min_bound)
            pcd.paint_uniform_color(color)
            voxel_grid_instance = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=max(
                (max_bound - min_bound) / (voxel_size - 1)))
            visual_grids.append(voxel_grid_instance)

            o3d.visualization.draw_geometries(visual_grids)
            '''

            # 判断视角
            torch.save(occupancy_grid_binary, os.path.join(self.save_generate_dir, f"occgrid_unseen_{voxel_size}.pth"))
            torch.save(occupancy_grid_instance_binary,
                       os.path.join(self.save_generate_dir, f"occgrid_instance_{voxel_size}.pth"))

        occupancy_grid._binary = occupancy_grid_binary
        occupancy_grid_instance._binary = occupancy_grid_instance_binary

        self.occupancy_grid = {
            "unseen_occupancy_grid": occupancy_grid,
            "instance_occupancy_grid": occupancy_grid_instance
        }
        # self.visual_occupancy_grid()

    def visual_occupancy_grid(self, type="both"):
        def create_occupancy_grid(occupancy_grid, color):
            voxel_size = len(occupancy_grid._binary)
            max_bound = occupancy_grid.roi_aabb[3:].cpu().numpy()
            min_bound = occupancy_grid.roi_aabb[:3].cpu().numpy()
            grid_indices = occupancy_grid.grid_coords
            valid_grid_mask = occupancy_grid._binary.view(-1, 1)[..., 0]
            valid_grid = grid_indices[valid_grid_mask].detach().cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                valid_grid / (voxel_size - 1) * (max_bound - min_bound) + min_bound)
            pcd.paint_uniform_color(color)
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=max(
                (max_bound - min_bound) / (voxel_size - 1)))
            return voxel_grid

        visual_instances = []
        if type in ["both", "unseen"]:
            unseen_occupancy_grid = create_occupancy_grid(self.occupancy_grid["unseen_occupancy_grid"],
                                                          np.array([1, 0, 0]))
            visual_instances.append(unseen_occupancy_grid)
        if type in ["both", "instance"]:
            instance_occupancy_grid = create_occupancy_grid(self.occupancy_grid["instance_occupancy_grid"],
                                                            np.array([0, 1, 0]))
            visual_instances.append(instance_occupancy_grid)
        o3d.visualization.draw_geometries(visual_instances)

        return visual_instances

    def compute_visibility_mask(self, cameras, n_samples=1024):
        unseen_masks = []
        for camera in tqdm(cameras):
            c2w, intrinsic = camera.convert2c2w_intrinsics()
            Height, Width = camera.image_height, camera.image_width

            rays_o, rays_d, rays_d_norm = get_rays(
                get_ray_directions(Height, Width, torch.from_numpy(intrinsic[:3, :3]).float())[0],
                torch.from_numpy(c2w[:3]).float())
            rays_o, rays_d = rays_o.reshape(-1, 3).cuda(), rays_d.reshape(-1, 3).cuda()
            # unseen_occgrid
            grid_nearest_dist = []  # [unseen,instance]
            for occupancy_grid in self.occupancy_grid.values():
                render_step_size = (occupancy_grid.roi_aabb[3:] - occupancy_grid.roi_aabb[:3]
                                    ).norm() / n_samples
                ray_indices, t_starts, t_ends = ray_marching(rays_o, rays_d,
                                                             scene_aabb=occupancy_grid.roi_aabb,
                                                             grid=occupancy_grid,
                                                             alpha_fn=None,
                                                             near_plane=None, far_plane=None,
                                                             render_step_size=render_step_size,
                                                             stratified=False,
                                                             cone_angle=0.0,
                                                             alpha_thre=0.0)

                def find_nearest_dist(ray_indices, t_starts, n_rays):
                    rays_nearest_dist = torch.full((n_rays,), float('inf')).cuda()  # 初始化为一个较大的值
                    rays_nearest_dist = rays_nearest_dist.scatter_reduce(0, ray_indices, t_starts.squeeze(),
                                                                         reduce='amin')
                    rays_nearest_dist[rays_nearest_dist == float('inf')] = 0
                    return rays_nearest_dist

                grid_nearest_dist.append(find_nearest_dist(ray_indices, t_starts, len(rays_o)))

            # unseen_near小于instance_near,则这些区域是unseen_mask
            '''
            pcld_unseen = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
                (rays_o + rays_d * grid_nearest_dist[0][..., None])[::10].reshape(-1, 3).cpu().numpy()))
            pcld_unseen.paint_uniform_color(np.array([1, 0, 0]))

            pcld_instance = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
                (rays_o + rays_d * grid_nearest_dist[1][..., None])[::10].reshape(-1, 3).cpu().numpy()))
            pcld_instance.paint_uniform_color(np.array([0, 0, 1]))
            
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=Width, height=Height)
            vis.add_geometry(pcld_unseen)
            vis.add_geometry(pcld_instance)

            camera = o3d.camera.PinholeCameraParameters()
            camera.intrinsic = o3d.camera.PinholeCameraIntrinsic(Width, Height,
                                                                 intrinsic[0, 0], intrinsic[1, 1],
                                                                 intrinsic[0, 2], intrinsic[1, 2])
            camera.extrinsic = np.linalg.inv(c2w)

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(camera, True)
            vis.update_renderer()

            while True:
                vis.poll_events()
            '''

            # 如果instance_mask为0，但是unseen不为0，则为near;两者都有效的区域，则选unseen小的区域

            def dilate_mask(mask, kernel_size_erode=10, kernel_size_dilate=15, mode="de"):
                kernel_erode = np.ones((kernel_size_erode, kernel_size_erode), np.uint8)  # rubbish -> 15
                kernel_dilate = np.ones((kernel_size_dilate, kernel_size_dilate), np.uint8)  # rubbish -> 15
                mask = np.float32(mask)
                if mode == "de":
                    mask = cv2.dilate(mask, kernel_dilate, iterations=1)
                    mask = cv2.erode(mask, kernel_erode, iterations=1)
                elif mode == "ed":
                    mask = cv2.erode(mask, kernel_erode, iterations=1)
                    mask = cv2.dilate(mask, kernel_dilate, iterations=1)
                return mask > 0

            unseen_mask = ((grid_nearest_dist[0] > 0) & (grid_nearest_dist[1] == 0)) | (
                    (grid_nearest_dist[0] > 0) & (grid_nearest_dist[1] > 0) & (
                    grid_nearest_dist[1] > grid_nearest_dist[0]))  #

            unseen_mask = np.uint8(unseen_mask.reshape(Height, Width).cpu().numpy() * 255.0)
            unseen_mask = dilate_mask(unseen_mask, kernel_size_erode=10, kernel_size_dilate=15, mode="ed")

            # Image.fromarray(unseen_mask).show()

            camera.unseen_mask = unseen_mask

    def align_zero123_mesh(self, generate_mesh, view_idx, type):
        # 首先得到zero123_cam_pose,
        zero123_cam_pose, _ = get_zero123plus_input_cameras(radius=4.0)
        # 归一化generate_mesh
        mesh_scale = np.linalg.norm(np.array(generate_mesh.vertices), axis=-1).max()
        generate_mesh.scale(1 / mesh_scale, np.array([0, 0, 0]))
        # 然后得到colmap_cam_pose
        if type == "train":
            cam_pose = self.train_cams_list[view_idx].convert2c2w_intrinsics()[0]  # 位姿
        else:
            cam_pose = self.gen_cams_list[view_idx].convert2c2w_intrinsics()[0]

        cam_pose[:3, 3] -= self.instance_bbox["center"]  # 归一化到单位空间
        cam_pose[:3, 3] /= self.instance_bbox["radius"]
        # TODO: Bug 两个物体的center不一定一样(cabinet只有最前面)
        mesh = generate_mesh.transform(np.linalg.inv(zero123_cam_pose)).transform(cam_pose)
        # scale+translate
        mesh.scale(self.instance_bbox["radius"], np.array([0, 0, 0]))
        mesh.translate(self.instance_bbox["ce应该nter"])

        # 对齐BBox
        '''
        object_axis = np.array(self.instance_bbox["obj_axis"])
        ## colmap_bbox
        min_bound, max_bound = np.array(self.instance_bbox['bbox_bound'])
        colmap_extent = max_bound - min_bound
        colmap_bbox_axis = object_axis * colmap_extent[:2] / 2 + self.instance_bbox["center"][:2]
        ## zero123_bbox
        zero123_bbox = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(
            o3d.geometry.AxisAlignedBoundingBox.create_from_points(mesh.vertices))
        zero123_bbox_axis = object_axis * zero123_bbox.extent[:2] / 2 + zero123_bbox.center[:2]
        distance_zero2colmap = np.zeros((3))
        distance_zero2colmap[:2] = colmap_bbox_axis - zero123_bbox_axis
        mesh.translate(distance_zero2colmap)
        print()
        '''
        self.generated_mesh = {
            "generated_mesh": mesh
        }
        return mesh

    def align_syncdreamer_mesh(self, generate_mesh, view_idx, elevation=30):
        # 首先得到zero123_cam_pose,
        syncdreamer_cam_pose, _ = get_syncdreamer_input_cameras(radius=3, elevation=elevation)  # 半径为3

        # 然后得到colmap_cam_pose
        cam_pose = self.gen360_cams_list[view_idx].convert2c2w_intrinsics()[0]
        # right
        cam_pose[:3, 3] -= self.instance_bbox["center"]
        cam_pose[:3, 3] /= self.instance_bbox["radius"]
        # TODO: Bug 两个物体的center不一定一样(cabinet只有最前面)
        mesh = generate_mesh.transform(np.linalg.inv(syncdreamer_cam_pose[0])).transform(cam_pose)
        # scale+translate
        mesh.scale(self.instance_bbox["radius"], np.array([0, 0, 0]))
        mesh.translate(self.instance_bbox["center"])

        # 对齐BBox
        '''
        object_axis = np.array(self.instance_bbox["obj_axis"])
        ## colmap_bbox
        min_bound, max_bound = np.array(self.instance_bbox['bbox_bound'])
        colmap_extent = max_bound - min_bound
        colmap_bbox_axis = object_axis * colmap_extent[:2] / 2 + self.instance_bbox["center"][:2]
        ## zero123_bbox
        zero123_bbox = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(
            o3d.geometry.AxisAlignedBoundingBox.create_from_points(mesh.vertices))
        zero123_bbox_axis = object_axis * zero123_bbox.extent[:2] / 2 + zero123_bbox.center[:2]
        distance_zero2colmap = np.zeros((3))
        distance_zero2colmap[:2] = colmap_bbox_axis - zero123_bbox_axis
        mesh.translate(distance_zero2colmap)
        print()
        '''
        self.generated_mesh = {
            "generated_mesh": mesh
        }
        return mesh

    def refineGS(self, optimparams, pipelineparams,
                 use_origin_gs=True,
                 white_bkgd=False,
                 use_normal_consis=True):
        instance_gaussian = self.gaussian
        # 同时用generate_mesh初始化
        generate_mesh = self.generated_mesh["generated_mesh"]
        generate_pointcloud = o3d.geometry.PointCloud(generate_mesh.vertices)
        generate_pointcloud.colors = generate_mesh.vertex_colors

        generate_gaussian = GaussianModel(sh_degree=3)
        ## TODO:直接初始化 or Sugar初始化
        generate_gaussian.create_from_pcd(generate_pointcloud, 4)
        if use_origin_gs:
            instance_gaussian.combine_gaussian(generate_gaussian, load_seg_feat=True)  # 合并mesh的点云
        else:
            # setting feature
            gs_feat_mean = (instance_gaussian._seg_feature / (
                    instance_gaussian._seg_feature.norm(dim=-1, keepdim=True) + 1e-9)).mean(0)
            generate_gaussian._seg_feature = gs_feat_mean * torch.ones(
                (len(generate_gaussian.get_xyz), len(gs_feat_mean)),
                device="cuda")
            generate_gaussian._seg_feature = nn.Parameter(generate_gaussian._seg_feature.requires_grad_(True))

            instance_gaussian = generate_gaussian
        # instance_gaussian.create_from_pcd(generate_pointcloud, 4, require_grad=True)

        if os.path.exists(os.path.join(self.save_generate_dir, "generate_gaussian.ply")):
            instance_gaussian.load_ply(os.path.join(self.save_generate_dir, "generate_gaussian.ply"))
            self.gaussian = instance_gaussian
            return

        optimparams.iterations = 2000
        # instance_gaussian.spatial_lr_scale = 1
        instance_gaussian.training_setup(optimparams, optim_seg_feature=False)

        background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda") if white_bkgd else torch.tensor(
            [0, 0, 0], dtype=torch.float32, device="cuda")
        progress_bar = tqdm(range(1, optimparams.iterations), desc="\033[33m ## Training progress\033[0m")
        for iteration in progress_bar:
            instance_gaussian.update_learning_rate(iteration)

            if iteration % (optimparams.iterations // 5) == 0:
                instance_gaussian.oneupSHdegree()

            viewpoint_cam = self.refine_cams_list[randint(0, len(self.refine_cams_list) - 1)] \
                if iteration % 100 > 0 else self.refine_cams_list[0]

            render_pkg = render(viewpoint_cam, instance_gaussian, pipelineparams, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            gt_image = viewpoint_cam.original_image.cuda()
            gt_mask = viewpoint_cam.gt_alpha_mask.cuda() > 0
            Ll1 = l1_loss(image, gt_image) if viewpoint_cam.gen_type else l1_loss(image[:, gt_mask],
                                                                                  gt_image[:, gt_mask])

            # GT: 只监督Mask部分的RGB+Normal
            # Gen: render_view: 只监督Normal和Mask
            # Gen: generate_view: 监督Normal+RGB+Mask

            if not viewpoint_cam.gen_type:  # gen_type是监督整张图
                image[:, ~gt_mask] = 0
                gt_image[:, ~gt_mask] = 0

            if viewpoint_cam.gen_type:
                # 针对gen物体且只监督normal
                if viewpoint_cam.rgb_supervise:  #
                    loss = (1.0 - optimparams.lambda_dssim) * Ll1 + optimparams.lambda_dssim * (
                            1.0 - ssim(image, gt_image))
                else:
                    loss = 0  # 只监督Normal
            else:
                loss = Ll1  # (1.0 - optimparams.lambda_dssim) * Ll1
                # + optimparams.lambda_dssim * (1.0 - ssim(image, gt_image))

            # render_normal
            rend_normal = render_pkg['rend_normal']  # 世界坐标系下normal
            surf_normal = render_pkg['surf_normal']  # depth转normal
            rend_alpha = render_pkg['rend_alpha']

            lambda_normal = optimparams.lambda_normal if iteration > 0 and use_normal_consis else 0.0
            lambda_normal_prior = optimparams.lambda_normal_prior if iteration > 0 else 0.0
            # normal-consistency
            normal_error = (1 - (rend_normal[:, gt_mask] * surf_normal[:, gt_mask]).sum(dim=0))[None]
            normal_loss = lambda_normal * normal_error.mean()
            if viewpoint_cam.normal is not None:
                if viewpoint_cam.normal_supervise:
                    if viewpoint_cam.gen_type:
                        prior_normal = viewpoint_cam.normal.cuda()
                        prior_normal_mask = viewpoint_cam.normal_mask[0].cuda()

                        normal_prior_error = cos_loss(prior_normal[:, prior_normal_mask],
                                                      rend_normal[:, prior_normal_mask])
                        normal_loss = normal_loss + lambda_normal_prior * normal_prior_error
                    else:
                        prior_normal = viewpoint_cam.normal.cuda()
                        prior_normal_mask = gt_mask & viewpoint_cam.normal_mask[0].cuda()  # gt_mask和normal_mask的交集

                        if prior_normal_mask.sum() > 0:
                            normal_prior_error = cos_loss(prior_normal[:, prior_normal_mask],
                                                          rend_normal[:, prior_normal_mask])
                            normal_loss = normal_loss + lambda_normal_prior * normal_prior_error

            # mask loss
            mask_loss = 0
            if viewpoint_cam.gen_type:
                mask_loss = F.binary_cross_entropy(rend_alpha[0], gt_mask.float()) * optimparams.lambda_mask

            total_loss = loss + normal_loss + mask_loss
            total_loss.backward()

            with torch.no_grad():
                if iteration % 10 == 0:
                    loss_dict = {
                        "total_loss": f"{total_loss:.{5}f}",
                        "loss": f"{loss:.{5}f}"
                    }
                    progress_bar.set_postfix(loss_dict)

                if iteration % 100 == 0:
                    save_rgb = np.uint8(image.clip(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255.0)

                    save_depth = render_pkg['rend_depth'].squeeze().detach().cpu().numpy()
                    cm = matplotlib.colormaps["Spectral"]
                    save_depth = (save_depth - save_depth.min()) / (save_depth.max() - save_depth.min())
                    save_depth = np.uint8(cm(save_depth, bytes=False)[:, :, 0:3] * 255.0)

                    normal = render_pkg['rend_normal']
                    c2w = viewpoint_cam.convert2c2w_intrinsics()[0]
                    c2w = torch.from_numpy(c2w).float().cuda()
                    normal = (c2w[:3, :3].inverse() @ normal.reshape(3, -1)).reshape(3, viewpoint_cam.image_height,
                                                                                     viewpoint_cam.image_width)
                    save_normal = np.uint8(
                        (-normal.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2 * 255.0)

                    save_img = np.concatenate([save_rgb, save_depth, save_normal], axis=1)
                    Image.fromarray(save_img).save(
                        os.path.join(self.save_generate_dir, '{:0>3d}.png'.format(iteration)))

            instance_gaussian.optimizer.step()
            instance_gaussian.optimizer.zero_grad(set_to_none=True)

        instance_gaussian.save_ply(os.path.join(self.save_generate_dir, "generate_gaussian.ply"))

        self.gaussian = instance_gaussian

    def refineBackgroud(self, optimparams,
                        pipelineparams,
                        gaussians,
                        viewcams,
                        white_bkgd=False):
        '''
            * 优化物体的background的distloss和opacity

        '''
        optimparams.iterations = 2000
        gaussians.training_setup(optimparams, optim_seg_feature=False,
                                 optim_sh=False,
                                 optim_scale=False,
                                 optim_rotate=False)  # 只优化opacity

        background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda") if white_bkgd else torch.tensor(
            [0, 0, 0], dtype=torch.float32, device="cuda")

        progress_bar = tqdm(range(1, optimparams.iterations), desc="\033[33m ## Training progress\033[0m")
        for iteration in progress_bar:
            viewpoint_cam = viewcams[randint(0, len(viewcams) - 1)]
            render_pkg = render(viewpoint_cam, gaussians, pipelineparams, background)

            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                "viewspace_points"], \
                render_pkg["visibility_filter"], render_pkg["radii"]

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - optimparams.lambda_dssim) * Ll1 + optimparams.lambda_dssim * (1.0 - ssim(image, gt_image))

            gt_instance_mask = viewpoint_cam.gt_alpha_mask[0].cuda()
            rend_dist = render_pkg["rend_dist"][0]
            dist_loss = (rend_dist[gt_instance_mask]).mean() * 100

            alpha_loss = gaussians.get_opacity[visibility_filter].mean()
            total_loss = loss + dist_loss + alpha_loss * 0.1
            total_loss.backward()

            with torch.no_grad():
                if iteration % 10 == 0:
                    loss_dict = {
                        "total_loss": f"{total_loss:.{5}f}"
                    }
                    progress_bar.set_postfix(loss_dict)

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        return gaussians

    def render_mesh_rgb(self, mesh, view_cam):
        if not hasattr(self, "mesh_render"):
            self.mesh_render = Renderer(view_cam.image_width, view_cam.image_height,
                                        trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                                                        faces=np.asarray(mesh.triangles),
                                                        vertex_colors=np.asarray(mesh.vertex_colors)),
                                        0.5)

        c2w, intrinsic = view_cam.convert2c2w_intrinsics()

        rgb_pred, depth_pred = self.mesh_render(
            view_cam.image_width, view_cam.image_height, intrinsic, c2w)

        return rgb_pred

    def compute_crop_mask(self, scene_mesh, bbox=None):
        # 移除bbox内的mesh，然后计算gen_cam的mask
        if bbox is None:
            bbox = o3d.geometry.AxisAlignedBoundingBox(self.instance_bbox["center"] - self.instance_bbox["radius"],
                                                       self.instance_bbox["center"] + self.instance_bbox["radius"])
            # 半径应该是radius
            # bbox.scale(1.5, (bbox.min_bound + bbox.max_bound) / 2)
            min_bound = torch.from_numpy(bbox.min_bound).cuda().float()
            max_bound = torch.from_numpy(bbox.max_bound).cuda().float()

            aabb_bound = torch.cat((min_bound, max_bound))
        print()
