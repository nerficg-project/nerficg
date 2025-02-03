# -- coding: utf-8 --

"""
Visual/Trajectories/BulletTime.py: A visualization for dynamic scenes, following a lemniscate trajectory while replaying time.
Adapted from DyCheck (iPhone dataset) by Gao et al. 2022 (https://github.com/KAIR-BAIR/dycheck/blob/main).
"""

import torch

import Framework
from Cameras.Base import BaseCamera
from Cameras.utils import CameraProperties
from Visual.Trajectories.utils import getLemniscateTrajectory, CameraTrajectory


class bullet_time(CameraTrajectory):
    """A visualization for dynamic scenes, following a lemniscate trajectory while replaying time."""

    def __init__(self, reference_pose_rel_id: float = 0.5, custom_lookat: torch.Tensor | None = None,
                 custom_up: torch.Tensor | None = None, num_frames_per_rotation: float = 90, degree: float = 10, num_repeats: int = 2) -> None:
        super().__init__()
        self.reference_pose_rel_id = reference_pose_rel_id
        self.custom_lookat = custom_lookat
        self.custom_up = custom_up
        self.num_frames_per_rotation = num_frames_per_rotation
        self.degree = degree
        self.num_repeats = num_repeats

    def _generate(self, reference_camera: BaseCamera, reference_poses: list[CameraProperties]) -> list[CameraProperties]:
        """A visualization for dynamic scenes, following a lemniscate trajectory while freezing time at a reference frame."""
        data: list[CameraProperties] = []
        reference_pose = reference_poses[int(min(1.0, max(0.0, self.reference_pose_rel_id)) * len(reference_poses))]
        reference_camera.setProperties(reference_pose)
        lookat = self.custom_lookat
        if lookat is None:
            pos_viewdir = reference_camera.getPositionAndViewdir()[..., :3]
            lookat = pos_viewdir[0] + (((reference_camera.near_plane + reference_camera.far_plane) / 2) * pos_viewdir[1])
        up = self.custom_up if self.custom_up is not None else reference_camera.getUpVector()[:3]
        lemniscate_trajectory_c2ws = getLemniscateTrajectory(
            reference_pose,
            lookat=lookat.to(Framework.config.GLOBAL.DEFAULT_DEVICE),
            up=up.to(Framework.config.GLOBAL.DEFAULT_DEVICE),
            num_frames=self.num_frames_per_rotation,
            degree=self.degree,
        )
        num_frames = self.num_frames_per_rotation * self.num_repeats
        for frame_idx in range(num_frames):
            data.append(CameraProperties(
                    width=reference_pose.width,
                    height=reference_pose.height,
                    rgb=None,
                    alpha=None,
                    c2w=lemniscate_trajectory_c2ws[frame_idx % self.num_frames_per_rotation].clone(),
                    focal_x=reference_pose.focal_x,
                    focal_y=reference_pose.focal_y,
                    distortion_parameters=reference_pose.distortion_parameters,
                    principal_offset_x=0.0,
                    principal_offset_y=0.0,
                    timestamp=frame_idx / (num_frames - 1)
                ))
        return data
