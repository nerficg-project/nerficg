# -- coding: utf-8 --

"""Visual/Trajectories/StabilizedTrain.py: A camera trajectory following the training poses, stabilizing poses over a window."""

import torch

import Framework
from Cameras.Base import BaseCamera
from Cameras.utils import CameraProperties
from Datasets.utils import getAveragePose
from Visual.Trajectories.utils import CameraTrajectory


class stabilized_train(CameraTrajectory):
    """A camera trajectory following the training poses, stabilizing poses over a window."""

    def __init__(self, window: int = 5) -> None:
        super().__init__()
        if window % 2 == 0:
            raise Framework.VisualizationError('Window size must be an odd number.')
        self.half_window = window // 2

    def _generate(self, _: BaseCamera, reference_poses: list[CameraProperties]) -> list[CameraProperties]:
        """Generates the camera trajectory using a list of reference poses."""
        data: list[CameraProperties] = []
        poses_all = torch.stack([c.c2w for c in reference_poses], dim=0)
        for i, c in enumerate(reference_poses):
            c2w = getAveragePose(poses_all[max(0, i - self.half_window):min(len(reference_poses), i + self.half_window)])
            prop = c.toSimple()
            prop.c2w = c2w
            data.append(prop)
        return data
