"""Cameras/Perspective.py: Implementation of a perspective camera model."""

from dataclasses import dataclass

import torch

import Framework
from Cameras.Base import BaseCamera
from Cameras.utils import BaseDistortion, fov_to_focal


DEFAULT_VERTICAL_FOV = 45.0


@dataclass(kw_only=True)
class PerspectiveCamera(BaseCamera):
    """Defines a perspective camera model."""
    focal_x: float = None
    focal_y: float = None
    center_x: float = None
    center_y: float = None
    distortion: BaseDistortion = None

    def __post_init__(self) -> None:
        # focal length
        if self.focal_x is None and self.focal_y is None:
            focal = fov_to_focal(DEFAULT_VERTICAL_FOV, degrees=True) * self.height
            self.focal_x = self.focal_y = focal
        elif self.focal_x is None:
            self.focal_x = self.focal_y
        elif self.focal_y is None:
            self.focal_y = self.focal_x
        # principal point
        if self.center_x is None:
            self.center_x = self.width / 2
        if self.center_y is None:
            self.center_y = self.height / 2

    def cam_to_screen(self, xyz_cam: torch.Tensor, z_culling: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Projects points (Nx3) onto image plane. Returns pixel coordinates, depth values, and mask of points in view."""
        focals = torch.tensor((self.focal_x, self.focal_y), device=xyz_cam.device, dtype=xyz_cam.dtype)
        screen_size = torch.tensor((self.width, self.height), device=xyz_cam.device, dtype=xyz_cam.dtype)
        center = torch.tensor((self.center_x, self.center_y), device=xyz_cam.device, dtype=xyz_cam.dtype)
        depth = xyz_cam[:, 2]
        xy_screen = xyz_cam[:, :2] / depth.clamp_min(1.0e-8)[:, None]
        if self.distortion is not None:
            xy_screen = self.distortion.distort(xy_screen)
        xy_screen = xy_screen * focals + center
        in_frustum = ((xy_screen >= 0) & (xy_screen < screen_size)).all(dim=-1)
        if z_culling:
            in_frustum &= (depth > self.near_plane) & (depth < self.far_plane)
        return xy_screen, depth, in_frustum

    def screen_to_cam(self, xy_screen: torch.Tensor) -> torch.Tensor:
        """Unprojects pixels (Nx2) from screen space to camera space."""
        focals_rcp = torch.tensor((1 / self.focal_x, 1 / self.focal_y), device=xy_screen.device, dtype=xy_screen.dtype)
        center = torch.tensor((self.center_x, self.center_y), device=xy_screen.device, dtype=xy_screen.dtype)
        xy_cam = (xy_screen - center) * focals_rcp
        if self.distortion is not None:
            xy_cam = self.distortion.undistort(xy_cam)
        z_cam = torch.ones((1, 1), device=xy_cam.device, dtype=xy_cam.dtype).expand(xy_cam.shape[0], 1)
        return torch.cat((xy_cam, z_cam), dim=1)

    def compute_local_ray_directions(self, through_pixel_center: bool = True, enable_cache: bool = True) -> torch.Tensor:
        """Returns the direction of each camera ray in local coordinates."""
        current_key = (self.width, self.height, self.focal_x, self.focal_y, self.center_x, self.center_y, self.distortion, through_pixel_center)
        if enable_cache and self._local_ray_directions_cache is not None:
            cache_key, local_ray_directions = self._local_ray_directions_cache
            if cache_key == current_key:
                return local_ray_directions

        # calculate initial directions
        pixel_offset = 0.5 if through_pixel_center else 0.0
        min_x = (pixel_offset - self.center_x) / self.focal_x
        min_y = (pixel_offset - self.center_y) / self.focal_y
        max_x = (self.width - 1 + pixel_offset - self.center_x) / self.focal_x
        max_y = (self.height - 1 + pixel_offset - self.center_y) / self.focal_y
        xs = torch.linspace(min_x, max_x, self.width)
        ys = torch.linspace(min_y, max_y, self.height)
        local_directions = torch.empty((self.height, self.width, 3), dtype=xs.dtype, device=xs.device)
        local_directions[..., 0] = xs[None, :]
        local_directions[..., 1] = ys[:, None]
        local_directions[..., 2] = 1.0
        local_directions = local_directions.reshape(-1, 3)  # TODO: should return HxWx3 instead

        # undistort if needed
        if self.distortion is not None:
            local_directions[:, :2] = self.distortion.undistort(local_directions[:, :2])

        if enable_cache:
            # cache result
            self._local_ray_directions_cache = (current_key, local_directions)

        return local_directions

    def get_projection_matrix(self, invert_z: bool = False) -> torch.Tensor:
        """
            Returns the projection matrix for this camera.
            After perspective division, x, y, and z will be in [-1, 1] (OpenGL convention).
            The z-axis can be inverted for camera coordinate systems where the camera looks along the negative z-axis.

            Args:
                invert_z: If True, the z-axis will be inverted.

            Returns:
                The projection matrix from camera space to clip space.
        """
        half_width = self.width * 0.5
        half_height = self.height * 0.5
        offset_x = self.center_x - half_width
        offset_y = self.center_y - half_height
        z_sign = -1.0 if invert_z else 1.0
        projection_matrix = torch.tensor([
            [self.focal_x / half_width, 0.0, z_sign * offset_x / half_width, 0.0],
            [0.0, self.focal_y / half_height, z_sign * offset_y / half_height, 0.0],
            [0.0, 0.0, z_sign * (self.far_plane + self.near_plane) / (self.far_plane - self.near_plane), -2.0 * self.far_plane * self.near_plane / (self.far_plane - self.near_plane)],
            [0.0, 0.0, z_sign, 0.0]
        ], dtype=torch.float32, device=Framework.config.GLOBAL.DEFAULT_DEVICE)
        return projection_matrix

    def get_viewport_transform(self, pixel_centers_at_integer_coordinates: bool = True) -> torch.Tensor:
        """
            Returns the transformation matrix from NDC to screen space.
            if pixel_centers_at_integer_coordinates:
                [-1, 1]^3 -> [-0.5, width - 0.5] x [-0.5, height - 0.5] x [near_plane, far_plane]
            else:
                [-1, 1]^3 -> [0, width] x [0, height] x [near_plane, far_plane]

            Args:
                pixel_centers_at_integer_coordinates: If True, pixel centers will be at integer coordinates.

            Returns:
                The transformation matrix from NDC to screen space.
        """
        offset = 0.5 if pixel_centers_at_integer_coordinates else 0.0
        center_x = self.width * 0.5
        center_y = self.height * 0.5
        viewport_transform = torch.tensor([
            [center_x, 0.0, 0.0, center_x - offset],
            [0.0, center_y, 0.0, center_y - offset],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=torch.float32, device=Framework.config.GLOBAL.DEFAULT_DEVICE)
        # the following lines add a nonlinear mapping of z from [-1, 1] to [near_plane, far_plane]
        # viewport_transform[2, 2] = (self.far_plane - self.near_plane) * 0.5
        # viewport_transform[2, 3] = (self.far_plane + self.near_plane) * 0.5
        return viewport_transform
