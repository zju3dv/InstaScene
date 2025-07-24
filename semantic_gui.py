# Refer from Omniseg3D GUI https://github.com/THU-luvision/OmniSeg3D

import torch
import open3d as o3d
from gaussian_renderer import render
import numpy as np
import cv2
import os

from argparse import ArgumentParser

from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
import dearpygui.dearpygui as dpg
import math
from scene.cameras import Camera
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scipy.spatial.transform import Rotation as R


def depth2img(depth):
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-7)
    depth_img = cv2.applyColorMap((depth * 255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)
    return depth_img


class CONFIG:
    r = 1  # scale ratio
    windows_size = 800
    window_width = int(windows_size / r)
    window_height = int(windows_size / r)

    width = int(windows_size / r)
    height = int(windows_size / r)

    radius = 2

    debug = False
    dt_gamma = 0.2

    # gaussian model
    sh_degree = 3

    convert_SHs_python = False
    compute_cov3D_python = False
    depth_ratio = 0.0

    white_background = False

    # ckpt TODO: load from gui window.

    ply_path = ""
    interactive_note = ""

    use_colmap_camera = True
    source_path = ""
    only_load_camera = True
    resolution = 1
    downscale_ratio = 1
    data_device = "cpu"


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_quat(
            [0, 0, 0, 1]
        )  # init camera matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!
        self.right = np.array([1, 0, 0], dtype=np.float32)  # need to be normalized!
        self.fovy = fovy
        self.translate = np.array([0, 0, self.radius])
        self.scale_f = 1.0

        self.rot_mode = 1  # rotation mode (1: self.pose_movecenter (movable rotation center), 0: self.pose_objcenter (fixed scene center))

    @property
    def pose_movecenter(self):
        # --- first move camera to radius : in world coordinate--- #
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius

        # --- rotate: Rc --- #
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        # --- translate: tc --- #
        res[:3, 3] -= self.center

        # --- Convention Transform --- #
        # now we have got matrix res=c2w=[Rc|tc], but gaussian-splatting requires convention as [Rc|-Rc.T@tc]
        res[:3, 3] = -rot[:3, :3].transpose() @ res[:3, 3]

        return res

    @property
    def pose_objcenter(self):
        res = np.eye(4, dtype=np.float32)

        # --- rotate: Rw --- #
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        # --- translate: tw --- #
        res[2, 3] += self.radius  # camera coordinate z-axis
        res[:3, 3] -= self.center  # camera coordinate x,y-axis

        # --- Convention Transform --- #
        # now we have got matrix res=w2c=[Rw|tw], but gaussian-splatting requires convention as [Rc|-Rc.T@tc]=[Rw.T|tw]
        res[:3, :3] = rot[:3, :3].transpose()

        return res

    @property
    def opt_pose(self):
        # --- deprecated ! Not intuitive implementation --- #
        res = np.eye(4, dtype=np.float32)

        res[:3, :3] = self.rot.as_matrix()

        scale_mat = np.eye(4)
        scale_mat[0, 0] = self.scale_f
        scale_mat[1, 1] = self.scale_f
        scale_mat[2, 2] = self.scale_f

        transl = self.translate - self.center
        transl_mat = np.eye(4)
        transl_mat[:3, 3] = transl

        return transl_mat @ scale_mat @ res

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])

    def orbit(self, dx, dy):
        if self.rot_mode == 1:  # rotate the camera axis, in world coordinate system
            up = self.rot.as_matrix()[:3, 1]
            side = self.rot.as_matrix()[:3, 0]
        elif self.rot_mode == 0:  # rotate in camera coordinate system
            up = -self.up
            side = -self.right
        rotvec_x = up * np.radians(0.01 * dx)
        rotvec_y = side * np.radians(0.01 * dy)

        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        # self.radius *= 1.1 ** (-delta)    # non-linear version
        self.radius -= 0.1 * delta  # linear version

    def pan(self, dx, dy, dz=0):
        if self.rot_mode == 1:
            # pan in camera coordinate system: project from [Coord_c] to [Coord_w]
            self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])
        elif self.rot_mode == 0:
            # pan in world coordinate system: at [Coord_w]
            self.center += 0.0005 * np.array([-dx, dy, dz])


class GaussianSplattingGUI:
    def __init__(self, opt, gaussian_model: GaussianModel) -> None:
        self.opt = opt

        self.known_camera_mode = False
        if opt.use_colmap_camera:
            scene_info = sceneLoadTypeCallbacks["Colmap"](opt.source_path, "images", False)
            self.train_cameras = cameraList_from_camInfos(scene_info.train_cameras, opt.downscale_ratio,
                                                          opt, load_images=False)

            width = self.train_cameras[0].image_width
            height = self.train_cameras[0].image_height
        else:
            width = opt.width
            height = opt.height

        self.width = width  # opt.width
        self.height = height  # opt.height
        self.window_width = width + 100
        self.window_height = height + 200
        self.camera = OrbitCamera(width, height, r=opt.radius)

        bg_color = [1, 1, 1] if opt.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.bg_color = background
        self.render_buffer = np.zeros((self.width, self.height, 3), dtype=np.float32)
        self.update_camera = True
        self.dynamic_resolution = True
        self.debug = opt.debug
        self.engine = gaussian_model

        self.proj_mat = None

        self.load_model = False
        print("\nloading model file...")
        self.engine.use_seg_feature = True
        self.engine.load_seg_feat = True
        self.engine.seg_feat_dim = 16

        self.engine.load_ply(self.opt.ply_path)


        self.do_pca()  # calculate self.proj_mat
        self.load_model = True

        print("loading model file done.")

        self.mode = "image"  # choose from ['image', 'depth']

        dpg.create_context()
        self.register_dpg()

        self.frame_id = 0

        # --- for better operation --- #
        self.moving = False
        self.moving_middle = False
        self.mouse_pos = (0, 0)

        # --- for interactive segmentation --- #
        self.img_mode = 0
        self.clickmode_button = False
        self.clickmode_multi_button = False  # choose multiple object
        self.new_click = False
        self.prompt_num = 0
        self.new_click_xy = []
        self.newest_click_xy = []
        self.click_instance_colors = []
        self.clear_edit = False  # clear all the click prompts
        self.segment3d_flag = False
        self.delete3d_flag = False
        self.reload_flag = False  # reload the whole scene / point cloud
        self.object_seg_id = 0  # to store the segmented object with increasing index order (path at: ./)

    def __del__(self):
        dpg.destroy_context()

    def prepare_buffer(self, outputs):
        if self.model == "images":
            return outputs["render"]
        else:
            return np.expand_dims(outputs["depth"], -1).repeat(3, -1)

    def register_dpg(self):
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.width, self.height, self.render_buffer, format=dpg.mvFormat_Float_rgb,
                                tag="_texture")
        dpg.set_global_font_scale(1.5)
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (30, 30, 30), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Button, (80, 120, 180), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (110, 160, 220), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (160, 200, 250), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (50, 50, 50), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (200, 200, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 4)
        dpg.bind_theme(global_theme)

        ### register window
        with dpg.window(tag="_primary_window", width=self.window_width + 50, height=self.window_height):
            dpg.add_image("_texture")  # add the texture

        dpg.set_primary_window("_primary_window", True)


        def callback_depth(sender, app_data):
            self.img_mode = (self.img_mode + 1) % 2

        def clickmode_callback(sender):
            self.clickmode_button = 1 - self.clickmode_button

        def clickmode_multi_callback(sender):
            self.clickmode_multi_button = dpg.get_value(sender)

        def clear_edit():
            self.clear_edit = True

        @torch.no_grad()
        def callback_segment3d():
            self.segment3d_flag = True

        @torch.no_grad()
        def callback_delete3d():
            self.delete3d_flag = True

        def callback_reload():
            self.reload_flag = True

        def callback_known_camera():
            self.known_camera_mode = ~self.known_camera_mode

        # control window
        with dpg.window(label="Control", tag="_control_window", width=400, height=500, pos=[self.width + 10, 0]):
            dpg.add_text("Mouse position: click anywhere to start.", tag="pos_item")
            dpg.add_spacing(count=2)

            with dpg.group():
                dpg.add_button(label="Render Option", tag="_button_depth", callback=callback_depth)
                dpg.add_spacing(count=1)
                dpg.add_slider_float(label="Score Threshold", default_value=0.0, min_value=0.0, max_value=1.0,
                                     tag="_ScoreThres", width=200)
                dpg.add_spacing(count=1)

            dpg.add_checkbox(label="Click Mode", callback=clickmode_callback, user_data="Some Data")
            dpg.add_spacing(count=1)
            dpg.add_checkbox(label="multi-clickmode", callback=clickmode_multi_callback, user_data="Some Data")
            dpg.add_spacing(count=1)

            dpg.add_separator()

            with dpg.group():
                dpg.add_button(label="clear_edit", callback=clear_edit, user_data="Some Data")
                dpg.add_spacing(count=1)
                dpg.add_button(label="segment_3d", callback=callback_segment3d, user_data="Some Data")
                dpg.add_spacing(count=1)
                dpg.add_button(label="delete_3d", callback=callback_delete3d, user_data="Some Data")
                dpg.add_spacing(count=1)
                dpg.add_button(label="reload_data", callback=callback_reload, user_data="Some Data")
                dpg.add_spacing(count=1)

            if self.opt.use_colmap_camera:
                with dpg.group():
                    dpg.add_button(label="Use Colmap Camera", tag="_button_colmap", callback=callback_known_camera)
                    dpg.add_spacing(count=1)
                    self.known_camera_idx = dpg.add_slider_int(label="Colmap Camera Idx",
                                                               default_value=0,
                                                               max_value=len(self.train_cameras) - 1)
                    dpg.add_spacing(count=1)

            def file_callback(sender, app_data, user_data):
                file_data = app_data["selections"]
                file_names = list(file_data.keys())
                self.opt.ply_file = file_data[file_names[0]]
                print(f"Loading model from {self.opt.ply_file}...")
                self.engine.load_ply(self.opt.ply_file)
                self.do_pca()
                print("Model loaded.")
                self.load_model = True

            with dpg.file_dialog(directory_selector=False, show=False, callback=file_callback,
                                 id="file_dialog_id", width=700, height=400):
                dpg.add_file_extension(".*")
                dpg.add_file_extension("Ply files (*.ply){.ply}", color=(0, 255, 255, 255))
            dpg.add_button(label="Load .ply File", callback=lambda: dpg.show_item("file_dialog_id"))

        if self.debug:
            with dpg.collapsing_header(label="Debug"):
                dpg.add_separator()
                dpg.add_text("Camera Pose:")
                dpg.add_text(str(self.camera.pose), tag="_log_pose")

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            delta = app_data
            self.camera.scale(delta)
            self.update_camera = True
            if self.debug:
                dpg.set_value("_log_pose", str(self.camera.pose))

        def toggle_moving_left():
            self.moving = not self.moving

        def toggle_moving_middle():
            self.moving_middle = not self.moving_middle

        def move_handler(sender, pos, user):
            if self.moving and dpg.is_item_focused("_primary_window"):
                dx = self.mouse_pos[0] - pos[0]
                dy = self.mouse_pos[1] - pos[1]
                if dx != 0.0 or dy != 0.0:
                    self.camera.orbit(-dx * 30, dy * 30)
                    self.update_camera = True

            if self.moving_middle and dpg.is_item_focused("_primary_window"):
                dx = self.mouse_pos[0] - pos[0]
                dy = self.mouse_pos[1] - pos[1]
                if dx != 0.0 or dy != 0.0:
                    self.camera.pan(-dx * 20, dy * 20)
                    self.update_camera = True

            self.mouse_pos = pos

        def change_pos(sender, app_data):
            # if not dpg.is_item_focused("_primary_window"):
            #     return
            xy = dpg.get_mouse_pos(local=False)
            dpg.set_value("pos_item", f"Mouse position = ({xy[0]}, {xy[1]})")
            if self.clickmode_button and app_data == 1:  # in the click mode and right click
                print(xy)
                self.new_click_xy.append(np.array(xy))
                self.new_click = True

                self.click_instance_colors.append(torch.rand(3).cuda() * 0.7 + 0.3)

        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)

            dpg.add_mouse_click_handler(dpg.mvMouseButton_Left, callback=lambda: toggle_moving_left())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Left, callback=lambda: toggle_moving_left())
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Middle, callback=lambda: toggle_moving_middle())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Middle, callback=lambda: toggle_moving_middle())
            dpg.add_mouse_move_handler(callback=lambda s, a, u: move_handler(s, a, u))

            dpg.add_mouse_click_handler(callback=change_pos)

        dpg.create_viewport(title="Gaussian-Splatting-Viewer", width=self.window_width + 320, height=self.window_height,
                            resizable=False)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        dpg.show_viewport()

    def render(self):
        while dpg.is_dearpygui_running():
            # update texture every frame
            # TODO : fetch rgb and depth
            if self.load_model:
                if self.opt.use_colmap_camera and self.known_camera_mode:
                    cam = self.train_cameras[dpg.get_value(self.known_camera_idx)]
                else:
                    cam = self.construct_camera()
                self.fetch_data(cam)
            dpg.render_dearpygui_frame()

    def construct_camera(
            self,
    ) -> Camera:
        if self.camera.rot_mode == 1:
            pose = self.camera.pose_movecenter
        elif self.camera.rot_mode == 0:
            pose = self.camera.pose_objcenter

        R = pose[:3, :3]
        t = pose[:3, 3]

        ss = math.pi / 180.0
        fovy = self.camera.fovy * ss

        fy = fov2focal(fovy, self.height)
        fovx = focal2fov(fy, self.width)

        cam = Camera(
            colmap_id=0,
            R=R,  # C2W
            T=t,  # W2C
            FoVx=fovx,
            FoVy=fovy,
            image=None,
            image_width=self.width,
            image_height=self.height,
            image_name=None,
            uid=0,
        )
        return cam

    def pca(self, X, n_components=3):
        n = X.shape[0]
        mean = torch.mean(X, dim=0)
        X = X - mean
        covariance_matrix = (1 / n) * torch.matmul(X.T, X).float()  # An old torch bug: matmul float32->float16,
        eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix)
        # eigenvalues = torch.norm(eigenvalues, dim=1)
        idx = torch.argsort(-eigenvalues.type(torch.float32))
        eigenvectors = eigenvectors[:, idx]
        proj_mat = eigenvectors[:, 0:n_components]

        return proj_mat.type(torch.float32)

    def do_pca(self):
        sems = self.engine._seg_feature.clone().squeeze()
        N, C = sems.shape
        torch.manual_seed(0)
        randint = torch.randint(0, N, [200_000])
        sems /= (torch.norm(sems, dim=1, keepdim=True) + 1e-6)
        sem_chosen = sems[randint, :]
        self.proj_mat = self.pca(sem_chosen, n_components=3)
        print("project mat initialized !")

    @torch.no_grad()
    def fetch_data(self, view_camera):
        outputs = render(view_camera, self.engine, self.opt, self.bg_color)

        # --- RGB image --- #
        img = outputs["render"].permute(1, 2, 0)  #
        rgb_score = img.clone()
        depth_score = rgb_score.cpu().numpy().reshape(-1)

        # --- semantic image --- #
        sems = outputs["seg_feature"].permute(1, 2, 0)
        H, W, C = sems.shape
        sems /= (torch.norm(sems, dim=-1, keepdim=True) + 1e-6)
        sem_transed = sems @ self.proj_mat
        sem_transed_rgb = torch.clip(sem_transed * 0.5 + 0.5, 0, 1)

        if self.clear_edit:
            self.new_click_xy = []
            self.click_instance_colors = []
            self.clear_edit = False
            self.prompt_num = 0

        if self.reload_flag:
            self.reload_flag = False
            self.engine.load_ply(self.opt.ply_path)
            print("Reload original 3D Gaussians from: ", self.opt.ply_path)

        if len(self.new_click_xy) > 0:
            featmap = sems.reshape(H, W, -1)

            if self.new_click:
                xy = self.new_click_xy[-1]
                new_feat = featmap[int(xy[1]) % H, int(xy[0]) % W, :].reshape(featmap.shape[-1], -1)

                if len(self.new_click_xy) == 1:
                    self.chosen_feature = new_feat
                else:
                    self.chosen_feature = torch.cat([self.chosen_feature, new_feat],
                                                    dim=-1)  # extend to get more prompt features

                self.prompt_num += 1
                self.new_click = False

            score_map = featmap @ self.chosen_feature

            score_map = (score_map + 1.0) / 2  # [-1,1]->[0,1]
            score_binary = score_map > dpg.get_value('_ScoreThres')

            rgb_score = img.clone()
            for click_idx in range(len(self.new_click_xy)):
                click_score_map = score_binary[..., click_idx]
                click_color = self.click_instance_colors[click_idx]
                rgb_score[click_score_map] = img[click_score_map] * 0.3 + click_color * 0.7

            if self.segment3d_flag or self.delete3d_flag:
                feat_pts = self.engine._seg_feature.squeeze()
                feat_pts /= (torch.norm(feat_pts, dim=-1, keepdim=True) + 1e-6)
                score_pts = feat_pts @ self.chosen_feature
                score_pts = (score_pts + 1.0) / 2
                score_pts_binary = (score_pts > dpg.get_value('_ScoreThres')).sum(1) > 0

                if True:
                    print("\033[96m### Filter Noisy with DBscan ###\033[0m")
                    pcld_points = self.engine.get_xyz[score_pts_binary].detach().cpu().numpy()
                    pcld = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcld_points))
                    labels = np.array(pcld.cluster_dbscan(eps=0.15, min_points=10)) + 1
                    label_lists, labels_cnts = np.unique(labels, return_counts=True)

                    suitable_label = label_lists[labels_cnts.argsort()[-1]]
                    score_pts_indices = np.where(score_pts_binary.detach().cpu().numpy())[0][labels == suitable_label]
                    score_pts_binary = torch.zeros_like(score_pts_binary, dtype=torch.bool)
                    score_pts_binary[score_pts_indices] = True

                save_dir = f"interactive_segmentation/{self.opt.interactive_note}"

                self.engine.save_ply(os.path.join(save_dir, f"segment_{self.object_seg_id}.ply"),
                                     crop_mask=score_pts_binary)
                torch.save(score_pts_binary, os.path.join(save_dir, f"segment_{self.object_seg_id}_mask.pt"))

                if self.segment3d_flag:
                    self.segment3d_flag = False
                    self.engine.load_ply(os.path.join(save_dir, f"segment_{self.object_seg_id}.ply"))
                elif self.delete3d_flag:
                    self.delete3d_flag = False
                    self.engine.prune_points(score_pts_binary, optimizer_type=False)
                    self.engine.save_ply(os.path.join(save_dir, f"deleted_{self.object_seg_id}.ply"))
                self.object_seg_id += 1

        if self.img_mode == 0:
            self.render_buffer = rgb_score.cpu().numpy().reshape(-1)
        elif self.img_mode == 1:
            self.render_buffer = sem_transed_rgb.cpu().numpy() * 0.7 + 0.3

        dpg.set_value("_texture", self.render_buffer)
        print(f"Mode:{self.img_mode}")


if __name__ == "__main__":
    opt = CONFIG()
    '''
    python semantic_gui.py --ply_path data/lerf/waldo_kitchen/point_cloud.ply \
                           --interactive_note lerf_waldo_kitchen \
                           --use_colmap_camera \
                           --source_path data/lerf/waldo_kitchen --resolution 1
    '''

    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--ply_path', type=str, default="data/lerf/waldo_kitchen/point_cloud.ply")
    parser.add_argument('--interactive_note', type=str, default="lerf_waldo_kitchen")

    parser.add_argument('--use_colmap_camera', action="store_true")
    parser.add_argument('--source_path', type=str, default="data/lerf/waldo_kitchen")
    parser.add_argument('--resolution', type=int, default=1)
    args = parser.parse_args()

    opt.ply_path = args.ply_path
    opt.interactive_note = args.interactive_note
    opt.use_colmap_camera = args.use_colmap_camera
    opt.source_path = args.source_path
    opt.resolution = args.resolution

    gs_model = GaussianModel(opt.sh_degree)
    gui = GaussianSplattingGUI(opt, gs_model)

    gui.render()
