# -- coding: utf-8 --

"""Visual/Trajectories/utils.py: Utilities for visualization tasks."""

from abc import ABC, abstractmethod
import math
from typing import Type

import torch

import Framework
from Logging import Logger
from Cameras.Base import BaseCamera
from Cameras.utils import CameraProperties, createLookAtMatrix
from Datasets.Base import BaseDataset


class CameraTrajectory(ABC):

    _options: list[str] = []

    def __init__(self) -> None:
        super().__init__()
        self._trajectory: list[CameraProperties] = []
        self.name: str = self.__class__.__name__

    @classmethod
    def listOptions(cls) -> list[str]:
        """Lists all available camera trajectories."""
        if not cls._options:
            cls._options = [cls.__name__ for cls in CameraTrajectory.__subclasses__()]
        return cls._options

    @classmethod
    def get(cls, trajectory_name: str) -> Type['CameraTrajectory']:
        """Returns a camera trajectory class by its name."""
        options = cls.listOptions()
        if trajectory_name not in options:
            raise Framework.VisualizationError(f'Unknown camera trajectory type: {trajectory_name}.\nAvailable options are: {options}')
        for trajectory in cls.__subclasses__():
            if trajectory.__name__ == trajectory_name:
                return trajectory

    @abstractmethod
    def _generate(self, reference_camera: BaseCamera, reference_poses: list[CameraProperties]) -> list[CameraProperties]:
        """Generates the camera trajectory using a list of reference poses."""
        pass

    def generate(self, reference_camera: BaseCamera, reference_poses: list[CameraProperties]) -> None:
        """Generates the camera trajectory using a list of reference poses."""
        Logger.logInfo(f'generating {self.name} trajectory...')
        self._trajectory = self._generate(reference_camera, reference_poses)

    def addTo(self, dataset: BaseDataset, reference_set: str | None = 'train') -> BaseDataset:
        """Adds the camera trajectory to a dataset."""
        if self.name in dataset.subsets:
            Logger.logInfo(f'{self.name} trajectory already exists in dataset.')
            return dataset
        if not self._trajectory:
            if reference_set is None:
                reference_poses = [*dataset.data['train'], *dataset.data['val'], *dataset.data['test']]
            else:
                reference_poses = dataset.data[reference_set]
            self.generate(reference_camera=dataset.camera, reference_poses=reference_poses)
        dataset.subsets.append(self.name)
        dataset.data[self.name] = self._trajectory
        return dataset


def getLemniscateTrajectory(
    reference_camera: CameraProperties,
    lookat: torch.Tensor,
    up: torch.Tensor,
    num_frames: int,
    degree: float,
) -> list[torch.Tensor]:
    reference_camera = reference_camera.toDefaultDevice()
    camera_position = reference_camera.T
    a = torch.norm(camera_position - lookat) * math.tan(degree / 360 * math.pi)
    # Lemniscate curve in camera space. Starting at the origin.
    positions = torch.stack([
        torch.tensor([
            a * math.cos(t) / (1 + math.sin(t) ** 2),
            a * math.cos(t) * math.sin(t) / (1 + math.sin(t) ** 2),
            0,
        ]) for t in (torch.linspace(0, 2 * math.pi, num_frames) + math.pi / 2)
    ], dim=0)
    # Transform to world space.
    positions = torch.matmul(reference_camera.R.T, positions[..., None])[..., 0] + camera_position
    cameras = [createLookAtMatrix(p, lookat, up) for p in positions]
    return cameras
