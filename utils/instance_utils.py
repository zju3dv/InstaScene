import pyrender
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm
import trimesh
from PIL import Image
import cv2
from scipy.spatial import KDTree

from scene.cameras import Camera
from utils.general_utils import PILtoTorch


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
        return self.renderer.render(self.scene)  # , self.render_flags)

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


def create_spheric_poses(radius, origin, n_poses=12, radius_level=3):
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

    def spherical_camera_pose(azimuths: np.ndarray, elevations: np.ndarray, radius=2.5):
        # azimuths = np.deg2rad(azimuths)
        # elevations = np.deg2rad(elevations)

        xs = radius * np.cos(elevations) * np.cos(azimuths)
        ys = radius * np.cos(elevations) * np.sin(azimuths)
        zs = radius * np.sin(elevations)

        cam_locations = np.array([xs, ys, zs])

        c2ws = center_looking_at_camera_pose(cam_locations)
        return c2ws

    spheric_poses = []
    origin_trans = np.eye(4)
    origin_trans[:3, 3] = np.array(origin)
    levels = [4]
    for level in levels:
        for azimuth in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:  # 绕z轴360度
            for elevation in np.linspace(0, np.pi / 2, 6 + 1)[2:4]:  # 绕x轴90度-> 30度，45度
                spheric_poses += [
                    (origin_trans @ spherical_camera_pose(azimuth, elevation,
                                                          radius * level))]  # 36 degree view downwards
    return np.stack(spheric_poses, 0)


def render_multiview_mesh(mesh):
    # 先生成轨迹
    instance_bbox = o3d.geometry.AxisAlignedBoundingBox().create_from_points(mesh.vertices)
    instance_bbox = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(instance_bbox)

    center = instance_bbox.center
    radius = np.linalg.norm(np.array(mesh.vertices) - center, axis=-1).max()
    cams_gen_pose = create_spheric_poses(radius, center)

    # 内参
    intrinsic_gen = np.eye(4)
    fov = 30
    crop_size = 512
    focal = 1 / np.tan(np.deg2rad(fov) / 2)
    intrinsic_gen[:3, :3] = np.array([[focal * crop_size / 2, 0, crop_size / 2],
                                      [0, focal * crop_size / 2, crop_size / 2],
                                      [0, 0, 1]])

    trimesh_mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles),
                                   vertex_colors=np.asarray(mesh.vertex_colors))

    # valid_mask = refuse(trimesh.load(mesh_path), c2ws, intrinsic, [width, height])
    renderer = Renderer(crop_size, crop_size, trimesh_mesh, 0.5)
    render_rgbs = []
    render_masks = []
    for pose in cams_gen_pose:
        intrinsic = intrinsic_gen
        rgb_pred, depth_pred = renderer(crop_size, crop_size, intrinsic, pose)
        render_rgbs.append(rgb_pred)
        render_masks.append(depth_pred > 0)

    return [intrinsic_gen, cams_gen_pose], render_rgbs, render_masks


@torch.no_grad()
def convertCam2Info(cam_intrins, cam_extrins, render_rgbs, render_masks, render_normals=None, render_depths=None):
    cams_info = []
    for cam_idx in range(len(render_rgbs)):
        render_rgb = render_rgbs[cam_idx]
        render_mask = render_masks[cam_idx]
        cam_extrin = cam_extrins[cam_idx]

        resized_image_rgb = PILtoTorch(Image.fromarray(render_rgb), Image.fromarray(render_rgb).size)
        loaded_mask = PILtoTorch(Image.fromarray(render_mask).convert("L"), Image.fromarray(render_mask).size)

        normal = None
        if render_normals is not None:
            normal = render_normals[cam_idx]
            normal = Image.fromarray(normal)
            resized_normal = PILtoTorch(normal, normal.size)
            resized_normal = resized_normal[:3]
            normal = - (resized_normal * 2 - 1)  # ! stable_normal有负号

            normal = normal.permute(1, 2, 0) @ (torch.tensor(np.linalg.inv(cam_extrin[:3, :3])).float())
            normal = normal.permute(2, 0, 1)

        depth = None
        if render_depths is not None:
            depth = render_depths[cam_idx]
            depth = torch.from_numpy(depth).float()[None]

        cams_info.append(Camera(colmap_id=cam_idx,
                                R=cam_extrin[:3, :3], T=np.linalg.inv(cam_extrin)[:3, 3],  # R是C2W,T是W2C
                                FoVx=2 * np.arctan(cam_intrins[0, 2] / cam_intrins[0, 0]),
                                FoVy=2 * np.arctan(cam_intrins[1, 2] / cam_intrins[1, 1]),
                                image=resized_image_rgb, normal=normal, depth=depth, gt_alpha_mask=loaded_mask,
                                image_name=str(cam_idx), uid=id, data_device="cpu"))
    return cams_info


def dilate_mask(mask, dilate_factor=15):
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask


def monodepth_mask_dilation(mask, resize_ratio=0.8):
    mask = np.array(mask, dtype=bool)
    mask_height, mask_width = mask.shape
    alpha = np.where(mask)
    y1, y2, x1, x2 = (
        alpha[0].min(),
        alpha[0].max(),
        alpha[1].min(),
        alpha[1].max(),
    )
    valid_height = int((y2 - y1) / resize_ratio)
    valid_width = int((x2 - x1) / resize_ratio)

    y1_resize = max(0, int((y1 + y2) / 2) - valid_height // 2)
    y2_resize = min(mask_height, int((y1 + y2) / 2) + valid_height // 2)
    x1_resize = max(0, int((x1 + x2) / 2) - valid_width // 2)
    x2_resize = min(mask_width, int((x1 + x2) / 2) + valid_width // 2)

    mask_resize = np.zeros_like(mask, dtype=bool)
    mask_resize[y1_resize:y2_resize, x1_resize:x2_resize] = True

    valid_mask = np.logical_xor(mask_resize, mask)
    return valid_mask


def get_pointcloud(rgb, depth, mask, K, C2W):
    mesh_grids = np.meshgrid(np.arange(rgb.shape[1], dtype=np.float32),  # 宽
                             np.arange(rgb.shape[0], dtype=np.float32),  # 高
                             indexing="xy")
    i_coords = mesh_grids[0]
    j_coords = mesh_grids[1]

    norm_cam_coor = np.stack([(i_coords - K[0][2]) / K[0][0],
                              (j_coords - K[1][2]) / K[1][1],
                              np.ones_like(i_coords)], -1)
    cam_coor = norm_cam_coor * depth[..., None]
    world_coor = cam_coor @ C2W[:3, :3].T + C2W[:3, 3:].T
    world_coor = world_coor[mask].reshape(-1, 3)
    rgb = rgb[mask].reshape(-1, 3)

    pts = o3d.geometry.PointCloud()
    pts.points = o3d.utility.Vector3dVector(world_coor)
    pts.colors = o3d.utility.Vector3dVector(rgb / 255.0)
    return pts


@torch.no_grad()
def K_nearest_neighbors(
        mean: torch.Tensor, K: int, query: None, return_dist: bool = False
):
    mean_np = mean.detach().cpu().numpy()
    query_np = query.detach().cpu().numpy()

    kdtree = KDTree(mean_np)

    nn_dist, nn_idx = kdtree.query(query_np, k=K)  # 在给定mean中找到和query_np最近的

    nn_dist = torch.from_numpy(nn_dist).to(mean)
    nn_idx = torch.from_numpy(nn_idx).to(mean.device).to(torch.long)

    if not return_dist:
        return mean[nn_idx], nn_idx
    else:
        return mean[nn_idx], nn_idx, nn_dist
