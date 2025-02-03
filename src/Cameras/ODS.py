# -- coding: utf-8 --

"""Cameras/ODS.py: An omni-directional stereo panorama camera model."""

import math
from torch import Tensor
import torch

from Cameras.Equirectangular import EquirectangularCamera


class ODSCamera(EquirectangularCamera):
    """
        Defines an omnidirectional stereo panorama camera model for ray generation.
        Expects the stereo images to be vertically concatenated.
    """

    def __init__(self, near_plane: float, far_plane: float, baseline: float = 0.065) -> None:
        super(ODSCamera, self).__init__(near_plane, far_plane)
        self.half_baseline: float = baseline / 2

    def getRayOrigins(self) -> Tensor:
        """Returns a tensor containing the origin of each ray."""
        rotation_angle: Tensor = torch.linspace(
            start=(-math.pi + (math.pi / self.properties.width)),
            end=(math.pi - (math.pi / self.properties.width)),
            steps=self.properties.width
        )[None, :].expand((self.properties.height // 2, self.properties.width))
        baseline_vector: Tensor = torch.stack(
            (torch.cos(rotation_angle), torch.zeros_like(rotation_angle), torch.sin(rotation_angle)), dim=-1
        ).reshape(-1, 3)
        origins_camera_space: Tensor = torch.cat([
            -self.half_baseline * baseline_vector,
            self.half_baseline * baseline_vector
        ], dim=0)
        origins_world_space: Tensor = torch.matmul(
            self.properties.c2w,
            torch.cat([origins_camera_space, torch.ones((origins_camera_space.shape[0], 1))], dim=-1)[:, :, None]
        ).squeeze()
        return origins_world_space[:, :3]

    def _getLocalRayDirections(self) -> Tensor:
        """Returns a tensor containing the direction of each ray."""
        # adjust height for vertical concatenation
        self.properties.height = self.properties.height // 2
        directions: Tensor = super().getLocalRayDirections()
        self.properties.height = self.properties.height * 2
        return torch.cat([directions, directions], dim=0)

    @property
    def baseline(self) -> float:
        """Returns the baseline of the stereo camera."""
        return self.half_baseline * 2.0

    @baseline.setter
    def baseline(self, value: float) -> None:
        self.half_baseline = value / 2.0

    def projectPoints(self, points: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError('point projection not yet implemented for ODS camera')

    def getProjectionMatrix(self, invert_z: bool = False) -> Tensor:
        raise NotImplementedError('projection matrix not yet implemented for ODS camera')
