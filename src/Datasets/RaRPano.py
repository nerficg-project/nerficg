"""
Datasets/RaRPano.py: Provides a dataset class for panorama scenes from the Rounding and Roaming dataset.
Data available at https://www.cg.cs.tu-bs.de/upload/projects/drone/datasets/RaR_pano.zip (last accessed 2026-01-21).
"""

import json
import math

import numpy as np

import Framework
from Cameras.Equirectangular import EquirectangularCamera
from Cameras.utils import invert_3d_affine, quaternion_to_rotation_matrix
from Datasets.Base import BaseDataset
from Datasets.utils import View, ImageData, read_image_size, compute_scaled_image_size, BasicPointCloud


def axis_angle_to_quaternion(axis_angle: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(axis_angle)

    x = axis_angle[0] / angle
    y = axis_angle[1] / angle
    z = axis_angle[2] / angle

    qw = math.cos(angle / 2)
    factor = math.sqrt(1 - qw * qw)
    qx = x * factor
    qy = y * factor
    qz = z * factor

    return np.array([qw, qx, qy, qz])


@Framework.Configurable.configure(
    PATH='dataset/RaR/pano/O_lion',
    TEST_STEP=8,
    NEAR_PLANE=0.2,
    FAR_PLANE=1000.0,
)
class CustomDataset(BaseDataset):
    """Dataset class for panorama scenes from the Roaming and Rounding dataset."""

    def load(self) -> tuple[list[EquirectangularCamera], dict[str, list[View]]]:
        """Loads the dataset into a dict containing lists of views for training and testing."""
        # define world coordinate system transformation
        world_transform = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])  # Blender to Colmap
        # load opensfm data
        reconstruction_file = self.dataset_path / 'reconstruction.json'
        with open(reconstruction_file) as f:
            reconstruction = json.load(f)

        # validate reconstruction
        if len(reconstruction) != 1:
            raise Framework.DatasetError('The RaRPano loader only supports datasets with a single reconstruction.')
        reconstruction = reconstruction[0]

        # set up cameras
        cameras: list[EquirectangularCamera] = []
        camera_helpers = {}
        for cam_idx, (cam_name, cam_data) in enumerate(sorted(reconstruction['cameras'].items())):
            if cam_data['projection_type'] not in ['spherical', 'equirectangular']:
                raise NotImplementedError(f'{cam_data["projection_type"]} camera model from OpenSfM data format is not implemented')
            width = cam_data['width']
            height = cam_data['height']
            cameras.append(EquirectangularCamera(shared_settings=self._camera_settings, width=width, height=height))
            camera_helpers[cam_name] = {'camera_idx': cam_idx, 'resized': False, 'n_views': 0}

        # use pre-downscaled images if possible
        image_directory_name = 'images'
        image_scale_factor = self.IMAGE_SCALE_FACTOR
        match self.IMAGE_SCALE_FACTOR:
            case 0.5:
                image_directory_name += '_2'
                image_scale_factor = None
            case _:
                pass

        # load views
        data: list[View] = []
        for global_frame_idx, (image_name, shot) in enumerate(sorted(reconstruction['shots'].items())):
            # image path
            rgb_path = self.dataset_path / image_directory_name / image_name

            # get camera
            camera_info = camera_helpers[shot['camera']]
            camera_idx = camera_info['camera_idx']
            camera = cameras[camera_idx]

            # determine and update/validate camera intrinsics
            width, height = read_image_size(self.dataset_path / image_directory_name / rgb_path)
            if image_scale_factor is not None:  # determine resolution after manual downscaling
                width, height = compute_scaled_image_size((width, height), image_scale_factor)
            invalid_intrinsics = camera.width != width or camera.height != height
            if not camera_info['resized'] and invalid_intrinsics:
                camera.width = width
                camera.height = height
                camera_info['resized'] = True
            elif invalid_intrinsics:
                raise Framework.DatasetError('Detected invalid OpenSfM data with inconsistent image sizes for the same camera.')

            # create w2c transform
            w2c = np.eye(4)
            w2c[:3, :3] = quaternion_to_rotation_matrix(axis_angle_to_quaternion(shot['rotation']))
            w2c[:3, 3] = shot['translation']
            c2w = invert_3d_affine(w2c)
            c2w = world_transform @ c2w

            # insert loaded values
            data.append(View(
                camera=camera,
                camera_index=camera_idx,
                frame_idx=camera_info['n_views'],
                global_frame_idx=global_frame_idx,
                c2w=c2w,
                rgb=ImageData(rgb_path, n_channels=3, scale_factor=image_scale_factor),
            ))
            camera_info['n_views'] += 1

        # load point cloud
        self.point_cloud = BasicPointCloud.from_opensfm(reconstruction)
        self.point_cloud.transform(world_transform)

        # create splits
        dataset: dict[str, list[View]] = {subset: [] for subset in self.subsets}
        if self.TEST_STEP > 0:
            for i in range(len(data)):
                if i % self.TEST_STEP == 0:
                    dataset['test'].append(data[i])
                else:
                    dataset['train'].append(data[i])
        else:
            dataset['train'] = data

        # return the dataset
        return cameras, dataset
