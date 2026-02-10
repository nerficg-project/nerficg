"""
Datasets/NeRF.py: Provides a dataset class for NeRF scenes.
Data available at https://www.kaggle.com/datasets/nguyenhung1903/nerf-synthetic-dataset (last accessed 2025-09-17).
"""

import json
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision import io

import Framework
from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import fov_to_focal
from Datasets.Base import BaseDataset
from Datasets.utils import View, apply_image_scale_factor, compute_scaled_image_size, read_image_size, ImageData
from Logging import Logger


def load_nerf_depth(path: Path) -> torch.Tensor:
    """Loads a depth map from the test set of a NeRF scene."""
    try:
        depth_raw: torch.Tensor = io.decode_image(input=str(path), mode=io.ImageReadMode.UNCHANGED)
    except Exception:
        raise Framework.DatasetError(f'Failed to load image file: "{path}"')
    # convert to [0, 1]
    depth_raw = depth_raw.float() / 255.0
    # see depth map creation in blender files of original NeRF codebase
    depth = -(depth_raw[:1] - 1.0) * 8.0
    return depth


@Framework.Configurable.configure(
    PATH='dataset/nerf_synthetic/lego',
    NORMALIZE_CUBE=4.0 / 1.5,  # cameras are inside [-4, 4]^3, geometry is only in [-1.5, 1.5]^3
    NEAR_PLANE=2.0,
    FAR_PLANE=6.0,
)
class CustomDataset(BaseDataset):
    """Dataset class for NeRF scenes."""

    def load(self) -> tuple[list[PerspectiveCamera], dict[str, list[View]]]:
        """Loads the dataset into a dict containing lists of views for training, evaluation, and testing."""
        camera = None
        # define coordinate system transformations
        cam_transform = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenGL to Colmap
        world_transform = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])  # Blender to Colmap
        self.bounding_box = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32, device='cpu')
        data: dict[str, list[View]] = {subset: [] for subset in self.subsets}
        global_frame_idx = 0
        for subset in self.subsets:
            metadata_filepath: Path = self.dataset_path / f'transforms_{subset}.json'
            try:
                with open(metadata_filepath, 'r') as f:
                    metadata_file: dict[str, Any] = json.load(f)
            except IOError:
                raise Framework.DatasetError(f'Invalid dataset metadata file path "{metadata_filepath}"')

            # create View objects for this subset
            for frame_idx, frame in Logger.log_progress(enumerate(metadata_file['frames']), desc=subset, leave=False, total=len(metadata_file['frames'])):
                rgba_path = self.dataset_path / f'{frame["file_path"]}.png'

                # set up or validate camera intrinsics
                width, height = compute_scaled_image_size(read_image_size(rgba_path), self.IMAGE_SCALE_FACTOR)
                focal = fov_to_focal(float(metadata_file['camera_angle_x'])) * width
                if camera is None:
                    camera = PerspectiveCamera(
                        shared_settings=self._camera_settings, width=width, height=height, focal_x=focal, focal_y=focal,
                    )
                elif camera.focal_x != focal or camera.width != width or height != camera.height:
                    raise Framework.DatasetError('The NeRF loader requires all views to have the same image size and focal length.')

                # load camera extrinsics
                c2w = world_transform @ frame['transform_matrix'] @ cam_transform.T
                # setup image data
                rgb = ImageData(rgba_path, n_channels=3, scale_factor=self.IMAGE_SCALE_FACTOR)
                alpha = ImageData(rgba_path, n_channels=1, channel_offset=3, scale_factor=self.IMAGE_SCALE_FACTOR)
                if subset == 'test':
                    # the synthetic NeRF dataset includes depth for the test set
                    depth = ImageData(
                        next(self.dataset_path.glob(f'{frame["file_path"]}_depth_*.png')),
                        n_channels=1, scale_factor=self.IMAGE_SCALE_FACTOR,
                        load_fn=load_nerf_depth, resize_fn=partial(apply_image_scale_factor, mode='nearest')
                    )
                else:
                    depth = None
                # insert loaded values
                data[subset].append(View(
                    camera=camera,
                    camera_index=0,
                    frame_idx=frame_idx,
                    global_frame_idx=global_frame_idx,
                    c2w=c2w,
                    rgb=rgb,
                    alpha=alpha,
                    depth=depth,
                ))
                global_frame_idx += 1
        return [camera], data
