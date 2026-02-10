"""Cameras/Equirectangular.py: Implements a 360-degree panorama camera model."""

import math
from dataclasses import dataclass

import torch

from Cameras.Base import BaseCamera
from Cameras.utils import directions_to_equirectangular_grid_coords, equirectangular_grid_coords_to_directions


@dataclass(kw_only=True)
class EquirectangularCamera(BaseCamera):
    """Defines a 360-degree panorama camera model."""

    def cam_to_screen(self, xyz_cam: torch.Tensor, z_culling: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Projects points (Nx3) onto image plane. Returns pixel coordinates, depth values, and mask of points in view."""
        half_screen_size = torch.tensor((self.width / 2, self.height / 2), device=xyz_cam.device, dtype=xyz_cam.dtype)
        depth = torch.norm(xyz_cam, dim=1)
        in_view = (depth > self.near_plane) & (depth < self.far_plane) if z_culling else torch.ones_like(depth, dtype=torch.bool)
        directions = xyz_cam / depth.clamp_min(1.0e-8)[:, None]
        grid_coords = directions_to_equirectangular_grid_coords(directions)
        xy_screen = (grid_coords + 1.0) * half_screen_size
        return xy_screen, depth, in_view

    def screen_to_cam(self, xy_screen: torch.Tensor) -> torch.Tensor:
        """Unprojects pixels (Nx2) from screen space to camera space."""
        half_screen_size_rcp = torch.tensor((2 / self.width, 2 / self.height), device=xy_screen.device, dtype=xy_screen.dtype)
        grid_coords = xy_screen * half_screen_size_rcp - 1.0
        return equirectangular_grid_coords_to_directions(grid_coords)

    def compute_local_ray_directions(self, through_pixel_center: bool = True, enable_cache: bool = True) -> torch.Tensor:
        """Returns the direction of each camera ray in local coordinates."""
        current_key = (self.width, self.height, through_pixel_center)
        if enable_cache and self._local_ray_directions_cache is not None:
            cache_key, local_ray_directions = self._local_ray_directions_cache
            if cache_key == current_key:
                return local_ray_directions

        pixel_offset = 0.5 if through_pixel_center else 0.0
        min_x = pixel_offset
        min_y = pixel_offset
        max_x = self.width - 1 + pixel_offset
        max_y = self.height - 1 + pixel_offset
        min_azimuth = min_x / self.width * 2 * math.pi - math.pi
        max_azimuth = max_x / self.width * 2 * math.pi - math.pi
        min_elevation = min_y / self.height * math.pi - math.pi / 2
        max_elevation = max_y / self.height * math.pi - math.pi / 2
        azimuth = torch.linspace(min_azimuth, max_azimuth, self.width)
        elevation = torch.linspace(min_elevation, max_elevation, self.height)
        sin_azimuth = torch.sin(azimuth)
        cos_azimuth = torch.cos(azimuth)
        sin_elevation = torch.sin(elevation)
        cos_elevation = torch.cos(elevation)
        local_directions = torch.empty((self.height, self.width, 3), dtype=azimuth.dtype, device=azimuth.device)
        torch.outer(cos_elevation, sin_azimuth, out=local_directions[..., 0])
        local_directions[..., 1] = sin_elevation[:, None].expand(self.height, self.width)
        torch.outer(cos_elevation, cos_azimuth, out=local_directions[..., 2])
        local_directions = local_directions.reshape(-1, 3)  # TODO: should return HxWx3 instead

        if enable_cache:
            # cache result
            self._local_ray_directions_cache = (current_key, local_directions)

        return local_directions
