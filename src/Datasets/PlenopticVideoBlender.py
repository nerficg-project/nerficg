"""
Datasets/PlenopticVideoBlender.py: Provides a dataset class for PlenopticVideo scenes in D-NeRF format.
See https://github.com/fudan-zvg/4d-gaussian-splatting/tree/main?tab=readme-ov-file#data-preparation (last accessed 2026-02-03).
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

import Framework
from Cameras.Perspective import PerspectiveCamera
from Datasets.Base import BaseDataset
from Datasets.utils import ImageData, View, compute_scaled_image_size, read_image_size, BasicPointCloud


@Framework.Configurable.configure(
    PATH='dataset/plenoptic_video/coffee_martini',
    IMAGE_SCALE_FACTOR=0.5,
    NEAR_PLANE=0.2,
    FAR_PLANE=100.0,
    MAX_TIMESTAMP=10.0,
)
class CustomDataset(BaseDataset):
    """Dataset class for PlenopticVideo scenes in D-NeRF format."""

    def load(self) -> tuple[list[PerspectiveCamera], dict[str, list[View]]]:
        """Loads the dataset into a dict containing lists of CameraProperties for training, evaluation, and testing."""

        self._camera_settings.near_plane = self.NEAR_PLANE
        self._camera_settings.far_plane = self.FAR_PLANE

        camera = None

        # define coordinate system transformations
        cam_transform = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenGL to Colmap
        world_transform = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])  # Blender to Colmap

        # load data
        data: dict[str, list[View]] = {subset: [] for subset in self.subsets}
        global_frame_idx = 0
        for subset in ['train', 'test']:

            metadata_filepath: Path = self.dataset_path / f'transforms_{subset}.json'
            try:
                with open(metadata_filepath, 'r') as f:
                    metadata_file: dict[str, Any] = json.load(f)
            except IOError:
                raise Framework.DatasetError(f'Invalid dataset metadata file path "{metadata_filepath}"')

            for frame_idx, frame in Framework.Logger.log_progress(enumerate(metadata_file['frames']), desc=subset, leave=False, total=len(metadata_file['frames'])):
                if frame['time'] >= self.MAX_TIMESTAMP:
                    continue
                rgb_path = self.dataset_path / f'{frame["file_path"]}.png'

                # setup camera
                if camera is None:
                    width, height = compute_scaled_image_size(read_image_size(rgb_path), self.IMAGE_SCALE_FACTOR)
                    scale_factor_intrinsic_x = width / int(metadata_file['w'])
                    scale_factor_intrinsic_y = height / int(metadata_file['h'])
                    focal_x = float(metadata_file['fl_x']) * scale_factor_intrinsic_x
                    focal_y = float(metadata_file['fl_y']) * scale_factor_intrinsic_y
                    center_x = float(metadata_file['cx']) * scale_factor_intrinsic_x
                    center_y = float(metadata_file['cy']) * scale_factor_intrinsic_y
                    camera = PerspectiveCamera(
                        shared_settings=self._camera_settings, width=width, height=height,
                        focal_x=focal_x, focal_y=focal_y, center_x=center_x, center_y=center_y,
                    )

                # load camera extrinsics
                c2w = world_transform @ frame['transform_matrix'] @ cam_transform.T
                # setup image data
                rgb = ImageData(rgb_path, n_channels=3, scale_factor=self.IMAGE_SCALE_FACTOR)
                # create and append view object
                data[subset].append(View(
                    camera=camera,
                    camera_index=0,
                    frame_idx=frame_idx,
                    global_frame_idx=global_frame_idx,
                    c2w=c2w,
                    timestamp=frame['time'],
                    rgb=rgb,
                ))
                global_frame_idx += 1

        min_timestamp = min(view.timestamp for view in data['train'] + data['test'])
        max_timestamp = max(view.timestamp for view in data['train'] + data['test'])
        for view in data['train'] + data['test']:
            view.timestamp = (view.timestamp - min_timestamp) / (max_timestamp - min_timestamp)

        # load point cloud
        self.point_cloud = BasicPointCloud.from_ply(self.dataset_path / 'points3d.ply')
        self.point_cloud.transform(world_transform)

        return [camera], data