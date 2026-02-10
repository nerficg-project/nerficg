"""
Visual/Trajectories/SpiralPath.py: A spiral camera trajectory for forward facing scenes.
Used by the original NeRF for LLFF visualizations.
"""

import numpy as np

import Framework
from Cameras.Base import BaseCamera
from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import look_at
from Datasets.utils import View, get_average_pose
from Visual.Trajectories.utils import CameraTrajectory


class spiral_path(CameraTrajectory):
    """A spiral camera trajectory for forward facing scenes."""

    def __init__(self, n_views: int = 120, n_rotations: int = 2) -> None:
        super().__init__()
        self.n_views = n_views
        self.n_rotations = n_rotations

    def _generate(self, reference_camera: BaseCamera, reference_views: list[View]) -> list[View]:
        """Generates the camera trajectory using a list of reference views."""
        if not isinstance(reference_camera, PerspectiveCamera):
            raise Framework.VisualizationError('reference_camera must be an instance of PerspectiveCamera')
        reference_poses = np.stack([view.c2w_numpy for view in reference_views])
        spiral_path_poses = create_spiral_path(
            poses=reference_poses,
            camera=reference_camera,
            n_views=self.n_views,
            n_rotations=self.n_rotations
        )
        views = []
        for frame_idx, pose in enumerate(spiral_path_poses):
            views.append(View(
                camera=reference_camera,
                camera_index=0,
                frame_idx=frame_idx,
                global_frame_idx=0,
                c2w=pose,
            ))
        return views


def create_spiral_path(poses: np.ndarray, camera: PerspectiveCamera, n_views: int, n_rotations: int) -> list[np.ndarray]:
    """Creates views on spiral path, adapted from the original NeRF implementation."""
    average_pose = get_average_pose(poses)
    down = poses[:, :3, 1].sum(axis=0)
    close_depth = camera.near_plane * 0.9
    inf_depth = camera.far_plane * 1.0
    dt = 0.75
    focal = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)
    rads = 0.01 * np.quantile(np.abs(poses[:, :3, 3]), q=0.9, axis=0)
    rads = np.concatenate([rads, [1.0]])
    thetas = np.linspace(0.0, 2.0 * np.pi * n_rotations, n_views, endpoint=False)
    c2w = average_pose[:3, :4]
    spiral_path_poses = []
    for theta in thetas:
        spiral_offset = np.array([np.cos(theta), np.sin(theta), np.sin(theta * 0.5), 1.0]) * rads
        lookat = c2w @ spiral_offset
        eye = c2w @ np.array([0.0, 0.0, -focal, 1.0])
        pose = look_at(eye, lookat, -down)
        spiral_path_poses.append(pose)
    return spiral_path_poses
