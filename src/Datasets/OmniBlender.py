"""
Datasets/OmniBlender.py: Provides a dataset class for scenes from EgoNeRF's OmniBlender dataset.
Data available at https://drive.google.com/drive/folders/1kqLAATjSSDwfLHI5O7RTfM9NOUi7PvcK (last accessed 2026-01-21).
"""

import json

import numpy as np
from natsort import natsorted

import Framework
from Cameras.Equirectangular import EquirectangularCamera
from Datasets.Base import BaseDataset
from Datasets.utils import read_image_size, compute_scaled_image_size, View, ImageData, BasicPointCloud
from Logging import Logger


@Framework.Configurable.configure(
    PATH='dataset/OmniBlender/barbershop',
    NEAR_PLANE=0.1,
    FAR_PLANE=1000.0,
)
class CustomDataset(BaseDataset):
    """Dataset class for OmniBlender scenes."""

    def load(self) -> tuple[list[EquirectangularCamera], dict[str, list[View]]]:
        """Loads the dataset into a dict containing lists of CameraProperties for training and testing."""
        camera = None
        # load data
        data: dict[str, list[View]] = {subset: [] for subset in self.subsets}
        global_frame_idx = 0
        for subset in self.subsets:
            if subset == 'val':
                continue

            openmvg_filepath = self.dataset_path / 'openMVG' / f'data_openmvg_{subset}.json'
            try:
                with open(openmvg_filepath) as f:
                    openmvg_data = json.load(f)
            except IOError:
                raise Framework.DatasetError(f'Invalid dataset metadata file path "{openmvg_filepath}"')

            # sort views by image filename
            sorted_views = natsorted(openmvg_data['views'], key=lambda view: view['value']['ptr_wrapper']['data']['filename'])

            # create View objects for this subset
            for frame_idx, view in Logger.log_progress(enumerate(sorted_views), desc=subset, leave=False, total=len(sorted_views)):
                view_info = view['value']['ptr_wrapper']['data']
                rgb_path = self.dataset_path / 'images' / view_info['filename']

                # set up or validate camera intrinsics
                width, height = compute_scaled_image_size(read_image_size(rgb_path), self.IMAGE_SCALE_FACTOR)
                if camera is None:
                    camera = EquirectangularCamera(shared_settings=self._camera_settings, width=width, height=height)
                elif camera.width != width or height != camera.height:
                    raise Framework.DatasetError('The OmniBlender loader requires all views to have the same image size.')

                # load camera extrinsics
                pose = openmvg_data['extrinsics'][view_info['id_pose']]['value']
                c2w = np.eye(4)
                c2w[:3, :3] = np.array(pose['rotation']).T
                c2w[:3, 3] = pose['center']

                # insert loaded values
                data[subset].append(View(
                    camera=camera,
                    camera_index=0,
                    frame_idx=frame_idx,
                    global_frame_idx=global_frame_idx,
                    c2w=c2w,
                    rgb=ImageData(rgb_path, n_channels=3, scale_factor=self.IMAGE_SCALE_FACTOR),
                ))
                global_frame_idx += 1

        # load point cloud
        self.point_cloud = BasicPointCloud.from_ply(self.dataset_path / 'openMVG'/ 'reconstruction' / 'colorized.ply')

        return [camera], data
