# -- coding: utf-8 --

"""
Visual/Trajectories/SpiralPath.py: A spiral camera trajectory for forward facing scenes.
Used by the original NeRF for LLFF visualizations.
"""

import math
import torch

import Framework
from Cameras.Base import BaseCamera
from Cameras.utils import CameraProperties, createCameraMatrix, normalizeRays
from Datasets.utils import getAveragePose
from Visual.Trajectories.utils import CameraTrajectory


class spiral_path(CameraTrajectory):
    """A spiral camera trajectory for forward facing scenes."""

    def __init__(self, num_views: int = 120, num_rotations: int = 2, ) -> None:
        super().__init__()
        self.num_views: int = num_views
        self.num_rotations: int = num_rotations

    def _generate(self, reference_camera: BaseCamera, reference_poses: list[CameraProperties]) -> list[CameraProperties]:
        """Generates the camera trajectory using a list of reference poses."""
        mean_focal_x: float = torch.tensor([c.focal_x for c in reference_poses]).mean().item()
        mean_focal_y: float = torch.tensor([c.focal_y for c in reference_poses]).mean().item()
        c2ws = torch.stack([c.c2w for c in reference_poses], dim=0).to(Framework.config.GLOBAL.DEFAULT_DEVICE)
        return createSpiralPath(
            view_matrices=c2ws,
            near_plane=reference_camera.near_plane,
            far_plane=reference_camera.far_plane,
            image_shape=(3, reference_poses[0].height, reference_poses[0].width),
            focal_x=mean_focal_x,
            focal_y=mean_focal_y,
            n_views=self.num_views,
            n_rots=self.num_rotations
        )


def createSpiralPath(view_matrices: torch.Tensor, near_plane: float, far_plane: float, image_shape: tuple[int, int, int],
                     focal_x: float, focal_y: float, n_views: int, n_rots: int) -> list[CameraProperties]:
    """Creates views on spiral path, adapted from the original NeRF implementation."""
    average_pose: torch.Tensor = getAveragePose(view_matrices)
    up: torch.Tensor = -normalizeRays(view_matrices[:, :3, 1].sum(0))[0]
    close_depth: float = near_plane * 0.9
    inf_depth: float = far_plane * 1.0
    dt: float = 0.75
    focal: float = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)
    rads: torch.Tensor = 0.01 * torch.quantile(torch.abs(view_matrices[:, :3, 3]), q=0.9, dim=0)
    view_matrices_spiral: list[torch.Tensor] = []
    rads: torch.Tensor = torch.tensor(list(rads) + [1.])
    for theta in torch.linspace(0.0, 2.0 * math.pi * n_rots, n_views + 1)[:-1]:
        c: torch.Tensor = torch.mm(
            average_pose[:3, :4],
            (torch.tensor([torch.cos(theta), torch.sin(theta), torch.sin(theta * 0.5), 1.]) * rads)[:, None]
        ).squeeze()
        z: torch.Tensor = -normalizeRays(c - torch.mm(average_pose[:3, :4], torch.tensor([[0], [0], [-focal], [1.]])).squeeze())[0]
        view_matrices_spiral.append(createCameraMatrix(z, up, c))
    return [
        CameraProperties(
            width=image_shape[2],
            height=image_shape[1],
            rgb=None,
            alpha=None,
            c2w=c2w.float(),
            focal_x=focal_x,
            focal_y=focal_y
        )
        for c2w in view_matrices_spiral
    ]
