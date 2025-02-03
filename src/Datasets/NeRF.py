# -- coding: utf-8 --

"""
Datasets/NeRF.py: Provides a dataset class for NeRF scenes.
Data available at https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1 (last accessed 2023-05-25).
"""

import json
import math
from pathlib import Path
from typing import Any

import torch
from torchvision import io

import Framework
from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import CameraProperties
from Datasets.Base import BaseDataset
from Datasets.utils import applyBGColor, loadImagesParallel, getParallelLoadIterator, \
    applyImageScaleFactor, CameraCoordinateSystemsTransformations, WorldCoordinateSystemTransformations
from Logging import Logger


@Framework.Configurable.configure(
    PATH='dataset/nerf_synthetic/lego',
    LOAD_TESTSET_DEPTHS=False,
)
class CustomDataset(BaseDataset):
    """Dataset class for NeRF scenes."""

    def __init__(self, path: str) -> None:
        super().__init__(
            path,
            PerspectiveCamera(2.0, 6.0),
            CameraCoordinateSystemsTransformations.RIGHT_HAND,
            WorldCoordinateSystemTransformations.XnYnZ
        )

    def load(self) -> dict[str, list[CameraProperties]]:
        """Loads the dataset into a dict containing lists of CameraProperties for training, evaluation, and testing."""
        # set bb
        self._bounding_box = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32, device='cpu')
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
            if subset == 'test' and self.LOAD_TESTSET_DEPTHS:
                # the synthetic NeRF dataset's test set includes depth maps
                depths = self.loadTestsetDepthsParallel(metadata_file['frames'], num_threads=4)
            else:
                depths = [None] * len(rgbs)
            # create split CameraProperties objects
            for frame, rgb, alpha, depth in zip(metadata_file['frames'], rgbs, alphas, depths):
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
                    depth=depth,
                    c2w=c2w,
                    focal_x=focal_x,
                    focal_y=focal_y
                ))
        # return the dataset
        return data

    def loadTestsetDepthsParallel(self, frames: list[dict[str, Any]], num_threads: int) -> list[torch.Tensor]:
        """Loads a multiple depth maps in parallel."""
        filenames = [str(next(self.dataset_path.glob(f'{frame["file_path"]}_depth_*.png'))) for frame in frames]
        iterator, pool = getParallelLoadIterator(filenames, self.IMAGE_SCALE_FACTOR, num_threads, load_function=self.loadNeRFDepth)
        depths = []
        for depth in Logger.logProgressBar(iterator, desc='test depth', leave=False, total=len(filenames)):
            # clone tensor to extract it from shared memory (/dev/shm), otherwise we can not use all RAM
            depths.append(depth.clone())
        pool.close()
        pool.join()
        return depths

    @staticmethod
    def loadNeRFDepth(filename: str, scale_factor: float | None) -> torch.Tensor:
        """Loads a depth map from the test set of a NeRF scene."""
        try:
            depth_raw: torch.Tensor = io.read_image(path=filename, mode=io.ImageReadMode.UNCHANGED)
        except Exception:
            raise Framework.DatasetError(f'Failed to load image file: "{filename}"')
        # convert image to the format used by the framework
        depth_raw = depth_raw.float() / 255
        # apply scaling factor
        if scale_factor is not None:
            depth_raw = applyImageScaleFactor(depth_raw, scale_factor)
        # see depth map creation in blender files of original NeRF codebase
        depth = -(depth_raw[:1] - 1) * 8
        return depth
