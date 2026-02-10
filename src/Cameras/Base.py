"""Cameras/Base.py: Implementation of the basic camera model used for ray generation and scene rendering options."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

import Framework
from Cameras.utils import SharedCameraSettings


@dataclass(kw_only=True)
class BaseCamera(ABC):
    """Defines the basic camera template."""
    shared_settings: SharedCameraSettings
    width: int
    height: int

    _local_ray_directions_cache: object = field(init=False, default=None, repr=False)

    @property
    def background_color(self) -> torch.Tensor:
        """Returns the background color of the camera."""
        return self.shared_settings.background_color

    @background_color.setter
    def background_color(self, color: torch.Tensor) -> None:
        """Sets the background color of the camera."""
        if color.shape != self.shared_settings.background_color.shape:
            raise Framework.CameraError(f'Invalid background color shape: expected {self.shared_settings.background_color.shape}, got {color.shape}')
        self.shared_settings.background_color = color.to(self.shared_settings.background_color)

    @property
    def near_plane(self) -> float:
        """Returns the near plane distance."""
        return self.shared_settings.near_plane

    @near_plane.setter
    def near_plane(self, distance: float) -> None:
        """Sets the near plane distance."""
        if distance <= 0.0 or distance >= self.shared_settings.far_plane:
            raise Framework.CameraError(f'Invalid near plane distance: expected 0.0 < near_plane < far_plane ({self.shared_settings.far_plane}), got {distance}')
        self.shared_settings.near_plane = distance

    @property
    def far_plane(self) -> float:
        """Returns the far plane distance."""
        return self.shared_settings.far_plane

    @far_plane.setter
    def far_plane(self, distance: float) -> None:
        """Sets the far plane distance."""
        if distance <= self.shared_settings.near_plane:
            raise Framework.CameraError(f'Invalid far plane distance: expected far_plane > near_plane ({self.shared_settings.near_plane}), got {distance}')
        self.shared_settings.far_plane = distance

    @abstractmethod
    def cam_to_screen(self, xyz_cam: torch.Tensor, z_culling: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Projects points (Nx3) onto image plane. Returns pixel coordinates, depth values, and mask of points in view."""
        pass

    @abstractmethod
    def screen_to_cam(self, xy_screen: torch.Tensor) -> torch.Tensor:
        """Unprojects pixels (Nx2) from screen space to camera space."""
        pass

    @abstractmethod
    def compute_local_ray_directions(self, through_pixel_center: bool = True, enable_cache: bool = True) -> torch.Tensor:
        """Returns the direction of each camera ray in local coordinates."""
        pass

    def get_pixel_coordinates(self) -> tuple[torch.Tensor, torch.Tensor]:
        y_direction, x_direction = torch.meshgrid(
            torch.linspace(0, self.height - 1, self.height),
            torch.linspace(0, self.width - 1, self.width),
            indexing="ij"
        )
        return x_direction, y_direction
