"""Visual/Trajectories/utils.py: Utilities for visualization tasks."""

from abc import ABC, abstractmethod

import numpy as np

import Framework
from Logging import Logger
from Cameras.Base import BaseCamera
from Cameras.utils import look_at
from Datasets.Base import BaseDataset
from Datasets.utils import View


class CameraTrajectory(ABC):

    _options: list[str] = []

    def __init__(self) -> None:
        super().__init__()
        self._trajectory: list[View] = []
        self.name: str = self.__class__.__name__

    @classmethod
    def list_options(cls) -> list[str]:
        """Lists all available camera trajectories."""
        if not cls._options:
            cls._options = [cls.__name__ for cls in CameraTrajectory.__subclasses__()]
        return cls._options

    @classmethod
    def get(cls, trajectory_name: str) -> type['CameraTrajectory']:
        """Returns a camera trajectory class by its name."""
        for trajectory in cls.__subclasses__():
            if trajectory.__name__ == trajectory_name:
                return trajectory
        raise Framework.VisualizationError(f'Unknown camera trajectory type: {trajectory_name}.\nAvailable options are: {cls.list_options()}')

    @abstractmethod
    def _generate(self, default_camera: BaseCamera, reference_views: list[View]) -> list[View]:
        """Generates the camera trajectory using a list of reference views."""
        pass

    def generate(self, default_camera: BaseCamera, reference_views: list[View]) -> None:
        """Generates the camera trajectory using a list of reference views."""
        Logger.log_info(f'generating {self.name} trajectory...')
        self._trajectory = self._generate(default_camera, reference_views)

    def add_to_dataset(self, dataset: BaseDataset, reference_set: str | None = 'train') -> BaseDataset:
        """Adds the camera trajectory to a dataset."""
        if self.name in dataset.subsets:
            Logger.log_info(f'{self.name} trajectory already exists in dataset.')
            return dataset
        if not self._trajectory:
            if reference_set is None:
                reference_views = [*dataset.data['train'], *dataset.data['val'], *dataset.data['test']]
            else:
                reference_views = dataset.data[reference_set]
            self.generate(default_camera=dataset.default_camera, reference_views=reference_views)
        dataset.subsets.append(self.name)
        dataset.data[self.name] = self._trajectory
        return dataset


def get_lemniscate_trajectory(
    reference_view: View,
    lookat: np.ndarray,
    up: np.ndarray,
    n_views: int,
    degree: float,
) -> list[np.ndarray]:
    # Reference camera position.
    reference_eye = reference_view.position_numpy

    # Lemniscate curve scale.
    a = np.linalg.norm(reference_eye - lookat) * np.tan(degree / 360 * np.pi)

    # Lemniscate curve in camera space (starting at origin).
    ts = np.linspace(0, 2 * np.pi, n_views) + np.pi / 2
    cos_t = np.cos(ts)
    sin_t = np.sin(ts)
    denom = 1 + sin_t ** 2
    positions_camera = np.stack([
        a * cos_t / denom,
        a * cos_t * sin_t / denom,
        np.zeros_like(ts),
        np.ones_like(ts)
    ], axis=1)

    # Transform to world space.
    positions_world = (reference_view.c2w_numpy @ positions_camera.T).T[:, :3]

    # Create poses.
    lemniscate_trajectory_poses = [look_at(eye, lookat, up) for eye in positions_world]

    return lemniscate_trajectory_poses
