# -- coding: utf-8 --

"""Cameras/Equirectangular.py: Implements a 360-degree panorama camera model."""

import math
from torch import Tensor
import torch

from Cameras.Base import BaseCamera


class EquirectangularCamera(BaseCamera):
    """Defines a 360-degree panorama camera model for ray generation."""

    def __init__(self, near_plane: float, far_plane: float) -> None:
        super(EquirectangularCamera, self).__init__(near_plane, far_plane)

    def getRayOrigins(self) -> Tensor:
        """Returns a tensor containing the origin of each ray."""
        return self.properties.c2w[:3, -1].expand((self.properties.width * self.properties.height, 3))

    def _getLocalRayDirections(self) -> Tensor:
        """Returns a tensor containing the direction of each ray."""
        azimuth: Tensor = torch.linspace(
            start=(-math.pi + (math.pi / self.properties.width)),
            end=(math.pi - (math.pi / self.properties.width)),
            steps=self.properties.width
        )[None, :].expand((self.properties.height, self.properties.width))
        inclination: Tensor = torch.linspace(
            start=((-math.pi / 2.0) + (0.5 * math.pi / self.properties.height)),
            end=((math.pi / 2.0) - (0.5 * math.pi / self.properties.height)),
            steps=self.properties.height
        )[:, None].expand((self.properties.height, self.properties.width))
        x_direction: Tensor = torch.sin(azimuth) * torch.cos(inclination)
        y_direction: Tensor = torch.sin(inclination)
        z_direction: Tensor = -torch.cos(azimuth) * torch.cos(inclination)
        directions_camera_space: Tensor = torch.stack(
            (x_direction, y_direction, z_direction), dim=-1
        ).reshape(-1, 3)
        return directions_camera_space

    def projectPoints(self, points: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError('point projection not yet implemented for panorama camera')

    def getProjectionMatrix(self, invert_z: bool = False) -> Tensor:
        raise NotImplementedError('projection matrix not yet implemented for panorama camera')
