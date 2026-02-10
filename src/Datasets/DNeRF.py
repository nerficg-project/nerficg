"""
Datasets/DNeRF.py: Provides a dataset class for D-NeRF scenes.
Data available at https://github.com/albertpumarola/D-NeRF (last accessed 2023-05-25).
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import Framework
from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import fov_to_focal
from Datasets.Base import BaseDataset
from Datasets.utils import View, compute_scaled_image_size, read_image_size, ImageData
from Logging import Logger


@Framework.Configurable.configure(
    PATH='dataset/dnerf/standup',
    IMAGE_SCALE_FACTOR=0.5,
    NORMALIZE_CUBE=4.0 / 1.5,  # cameras are inside [-4, 4]^3, geometry is only in [-1.5, 1.5]^3
    NEAR_PLANE=2.0,
    FAR_PLANE=6.0,
)
class CustomDataset(BaseDataset):
    """Dataset class for D-NeRF scenes."""

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
        # self.bounding_box = torch.tensor([[-1.3, -1.3, -1.3], [1.3, 1.3, 1.3]], dtype=torch.float32, device='cpu')  # used by 4DGS
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
                    raise Framework.DatasetError('The DNeRF loader requires all views to have the same image size and focal length.')

                # load camera extrinsics
                c2w = world_transform @ frame['transform_matrix'] @ cam_transform.T
                # setup image data
                rgb = ImageData(rgba_path, n_channels=3, scale_factor=self.IMAGE_SCALE_FACTOR)
                alpha = ImageData(rgba_path, n_channels=1, channel_offset=3, scale_factor=self.IMAGE_SCALE_FACTOR)
                # insert loaded values
                data[subset].append(View(
                    camera=camera,
                    camera_index=0,
                    frame_idx=frame_idx,
                    global_frame_idx=global_frame_idx,
                    c2w=c2w,
                    timestamp=frame['time'],
                    rgb=rgb,
                    alpha=alpha,
                ))
                global_frame_idx += 1
        if self.dataset_path.name == 'lego':
            # in the original test set of the lego scene, the shovel has a different orientation compared to the
            # training data. Similar to other implementations, we use the validation set for testing instead.
            data['test'], data['val'] = data['val'], data['test']
        return [camera], data
