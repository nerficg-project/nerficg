# -- coding: utf-8 --

"""Visual/Trajectories/FancyZoom.py: A camera trajectory following the training poses, interrupted by zooms and spiral movements."""

import math
import torch

import Framework
from Cameras.Base import BaseCamera
from Cameras.utils import CameraProperties
from Visual.Trajectories.utils import getLemniscateTrajectory, CameraTrajectory


class fancy_zoom(CameraTrajectory):
    """A camera trajectory following the training poses, interrupted by zooms and spiral movements."""

    def __init__(self, num_breaks: int = 2, num_zoom_frames: int = 90, zoom_factor: float = 0.2, lemni_frames_per_rot: int = 60, lemni_degree: int = 3) -> None:
        super().__init__()
        self.num_breaks = num_breaks
        self.num_zoom_frames = num_zoom_frames
        self.zoom_factor = zoom_factor
        self.lemni_frames_per_rot = lemni_frames_per_rot
        self.lemni_degree = lemni_degree

    def _generate(self, reference_camera: BaseCamera, reference_poses: list[CameraProperties]) -> list[CameraProperties]:
        """Generates the camera trajectory using a list of reference poses."""
        data: list[CameraProperties] = []
        num_reference_frames = len(reference_poses)
        break_indices = torch.linspace(0, len(reference_poses), self.num_breaks + 2)[1:-1].int().tolist()
        for i in range(break_indices[0]):
            data.append(reference_poses[i].toSimple())
        for j in range(len(break_indices)):
            reference = reference_poses[break_indices[j]]
            for i in range(self.num_zoom_frames):
                new = reference.toSimple()
                new.focal_x = new.focal_x + (new.focal_x * self.zoom_factor * math.sin((i / (self.num_zoom_frames - 1)) * 2 * math.pi))
                new.focal_y = new.focal_y + (new.focal_y * self.zoom_factor * math.sin((i / (self.num_zoom_frames - 1)) * 2 * math.pi))
                data.append(new)
            reference_camera.setProperties(reference)
            pos_viewdir = reference_camera.getPositionAndViewdir()[..., :3]
            lookat = (pos_viewdir[0] + (((reference_camera.near_plane + reference_camera.far_plane) / 2) * pos_viewdir[1])).to(Framework.config.GLOBAL.DEFAULT_DEVICE)
            up = reference_camera.getUpVector().to(Framework.config.GLOBAL.DEFAULT_DEVICE)
            lemniscate_trajectory_c2ws = getLemniscateTrajectory(reference, lookat=lookat, up=up, num_frames=self.lemni_frames_per_rot, degree=self.lemni_degree)
            for i in lemniscate_trajectory_c2ws:
                new = reference.toSimple()
                new.c2w = i
                data.append(new)
            if j < len(break_indices) - 1:
                for i in range(break_indices[j], break_indices[j + 1]):
                    data.append(reference_poses[i].toSimple())
        for i in range(break_indices[-1], num_reference_frames):
            data.append(reference_poses[i].toSimple())
        return data
