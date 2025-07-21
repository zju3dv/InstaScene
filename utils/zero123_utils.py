import numpy as np


def FOV_to_intrinsics(fov, focal_length=None, device='cpu'):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """
    focal_length = 0.5 / np.tan(np.deg2rad(fov) * 0.5) if focal_length is None else focal_length
    intrinsics = np.array([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]])
    return intrinsics


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
    azimuths = np.deg2rad(azimuths)
    elevations = np.deg2rad(elevations)

    xs = radius * np.cos(elevations) * np.cos(azimuths)
    ys = radius * np.cos(elevations) * np.sin(azimuths)
    zs = radius * np.sin(elevations)

    cam_locations = np.concatenate([xs, ys, zs], axis=-1)

    c2ws = center_looking_at_camera_pose(cam_locations)
    return c2ws


def get_zero123plus_input_cameras(radius=4.0, fov=30.0,
                                  azimuths=[0, 30, 90, 150, 210, 270, 330],
                                  elevations=[30, 30, -20, 30, -20, 30, -20]):
    """
    Get the input camera parameters.
    """
    azimuths = np.array(azimuths).astype(float)[..., None]
    elevations = np.array(elevations).astype(float)[..., None]

    c2w = spherical_camera_pose(azimuths, elevations, radius)

    Ks = FOV_to_intrinsics(fov) * 512

    return c2w, [512, 512, Ks[0, 0]]


def get_syncdreamer_input_cameras(radius=3, fov=np.arctan(128 / 280) * 2 * 180 / np.pi, elevation=30):
    """
    Get the input camera parameters.
    """
    azimuths = np.linspace(0, 360, 17)[:16].astype(float)[..., None]
    # np.array([30, 90, 150, 210, 270, 330]).astype(float)
    elevations = elevation * np.ones_like(azimuths)

    c2w = spherical_camera_pose(azimuths, elevations, radius)

    Ks = FOV_to_intrinsics(fov) * 256

    return c2w, [256, 256, Ks[0, 0]]


if __name__ == '__main__':
    cams_zero123_pose = get_zero123plus_input_cameras(1.0 * 4)[0]
