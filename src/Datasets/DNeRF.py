# -- coding: utf-8 --

"""
Datasets/DNerf.py: Provides a dataset class for D-NeRF scenes.
Data available at https://github.com/albertpumarola/D-NeRF (last accessed 2023-05-25).
"""

import json
import math
from pathlib import Path
from typing import Any

import torch

import Framework
from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import CameraProperties
from Datasets.Base import BaseDataset
from Datasets.utils import applyBGColor, loadImagesParallel, \
                           CameraCoordinateSystemsTransformations, WorldCoordinateSystemTransformations


@Framework.Configurable.configure(
    PATH='dataset/d-nerf/standup',
    IMAGE_SCALE_FACTOR=0.5,
    NORMALIZE_CUBE=1.5,
)
class CustomDataset(BaseDataset):
    """Dataset class for D-NeRF scenes."""

    def __init__(self, path: str) -> None:
        super().__init__(
            path=path,
            camera=PerspectiveCamera(2.0, 6.0),
            camera_system=CameraCoordinateSystemsTransformations.RIGHT_HAND,
            world_system=WorldCoordinateSystemTransformations.XnYnZ
        )

    def load(self) -> dict[str, list[CameraProperties]]:
        """Loads the dataset into a dict containing lists of CameraProperties for training, evaluation, and testing."""
        # set bb
        self._bounding_box = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]]).cpu()
        # load data
        data: dict[str, list[CameraProperties]] = {subset: [] for subset in self.subsets}
        for subset in self.subsets:
            metadata_filepath: Path = self.dataset_path / f'transforms_{subset}.json'
            try:
                with open(metadata_filepath, 'r') as f:
                    metadata_file: dict[str, Any] = json.load(f)
            except IOError:
                raise Framework.DatasetError(f'Invalid dataset metadata file path "{metadata_filepath}"')
            opening_angle: float = float(metadata_file['camera_angle_x'])
            # load images
            image_filenames = [str(self.dataset_path / (frame['file_path'] + '.png')) for frame in metadata_file['frames']]
            rgbs, alphas = loadImagesParallel(image_filenames, self.IMAGE_SCALE_FACTOR, num_threads=4, desc=subset)
            # create split CameraProperties objects
            for frame, rgb, alpha in zip(metadata_file['frames'], rgbs, alphas):
                # apply background color where alpha < 1
                rgb = applyBGColor(rgb, alpha, self.camera.background_color)
                # load camera extrinsics
                c2w = torch.as_tensor(frame['transform_matrix'], dtype=torch.float32)
                # load camera intrinsics
                focal_x: float = 0.5 * rgb.shape[2] / math.tan(0.5 * opening_angle)
                focal_y: float = 0.5 * rgb.shape[1] / math.tan(0.5 * opening_angle)
                # insert loaded values
                data[subset].append(CameraProperties(
                    width=rgb.shape[2],
                    height=rgb.shape[1],
                    rgb=rgb,
                    alpha=alpha,
                    c2w=c2w,
                    focal_x=focal_x,
                    focal_y=focal_y,
                    timestamp=frame['time']
                ))
        return data
