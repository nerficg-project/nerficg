# -- coding: utf-8 --

"""
Datasets/MipNeRF360.py: Provides a dataset class for scenes from the Mip-NeRF 360 dataset.
Data available at https://storage.googleapis.com/gresearch/refraw360/360_v2.zip (last accessed 2024-02-01) and
https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip (last accessed 2024-02-01).
Will also work for any other scene in the same format as the Mip-NeRF 360 dataset.
"""

import os
import torch

import Framework
from Logging import Logger
from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import CameraProperties
from Datasets.Base import BaseDataset
from Datasets.utils import CameraCoordinateSystemsTransformations, loadImagesParallel, WorldCoordinateSystemTransformations
from Datasets.Colmap import quaternion_to_R, read_points3D_binary, storePly, fetchPly, read_extrinsics_binary, read_intrinsics_binary, transformPosesPCA

@Framework.Configurable.configure(
    PATH='dataset/mipnerf360/garden',
    IMAGE_SCALE_FACTOR=0.25,
    BACKGROUND_COLOR=[0.0, 0.0, 0.0],
    NEAR_PLANE=0.01,
    FAR_PLANE=100.0,
    TEST_STEP=8,
    APPLY_PCA=True,
    APPLY_PCA_RESCALE=True,
    USE_PRECOMPUTED_DOWNSCALING=True,
)
class CustomDataset(BaseDataset):
    """Dataset class for MipNeRF360 scenes."""

    def __init__(self, path: str) -> None:
        super().__init__(
            path,
            PerspectiveCamera(0.01, 100.0),  # mipnerf360 itself uses 0.2, 1e6
            CameraCoordinateSystemsTransformations.LEFT_HAND,
            WorldCoordinateSystemTransformations.XnZY,
        )

    def load(self) -> dict[str, list[CameraProperties] | None]:
        """Loads the dataset into a dict containing lists of CameraProperties for training and testing."""
        # set near and far plane to values from config
        self.camera.near_plane = self.NEAR_PLANE
        self.camera.far_plane = self.FAR_PLANE

        # load colmap data
        cam_extrinsics = read_extrinsics_binary(self.dataset_path / 'sparse' / '0' / 'images.bin')
        cam_intrinsics = read_intrinsics_binary(self.dataset_path / 'sparse' / '0' / 'cameras.bin')

        # create camera properties
        data: list[CameraProperties] = []
        for cam_idx, cam_data in enumerate(cam_intrinsics.values()):
            # load images
            images = [data for data in cam_extrinsics.values() if data.camera_id == cam_data.id]
            images = sorted(images, key=lambda data: data.name)
            image_directory_name = 'images'
            image_scale_factor = self.IMAGE_SCALE_FACTOR
            # optionally use pre-downscaled images
            if self.USE_PRECOMPUTED_DOWNSCALING:
                match self.IMAGE_SCALE_FACTOR:
                    case 0.5:
                        image_directory_name = 'images_2'
                        image_scale_factor = None
                    case 0.25:
                        image_directory_name = 'images_4'
                        image_scale_factor = None
                    case 0.125:
                        image_directory_name = 'images_8'
                        image_scale_factor = None
                    case _:
                        pass
            image_filenames = [str(self.dataset_path / image_directory_name / image.name) for image in images]
            rgbs, _ = loadImagesParallel(image_filenames, image_scale_factor, num_threads=4, desc=f'camera {cam_data.id}')
            for idx, (image, rgb) in enumerate(zip(images, rgbs)):
                # extract w2c matrix
                rotation_matrix = torch.from_numpy(quaternion_to_R(image.qvec)).float()
                translation_vector = torch.from_numpy(image.tvec).float()
                w2c = torch.eye(4, device=torch.device('cpu'))
                w2c[:3, :3] = rotation_matrix
                w2c[:3, 3] = translation_vector
                # intrinsics
                focal_x = cam_data.params[0]
                focal_y = cam_data.params[1]
                principal_offset_x = cam_data.params[2] - cam_data.width / 2
                principal_offset_y = cam_data.params[3] - cam_data.height / 2
                if self.IMAGE_SCALE_FACTOR is not None:
                    scale_factor_intrinsics_x = rgb.shape[2] / cam_data.width
                    scale_factor_intrinsics_y = rgb.shape[1] / cam_data.height
                    focal_x *= scale_factor_intrinsics_x
                    focal_y *= scale_factor_intrinsics_y
                    principal_offset_x *= scale_factor_intrinsics_x
                    principal_offset_y *= scale_factor_intrinsics_y
                # create and append camera properties object
                camera_properties = CameraProperties(
                    width=rgb.shape[2],
                    height=rgb.shape[1],
                    rgb=rgb,
                    focal_x=focal_x,
                    focal_y=focal_y,
                    principal_offset_x=principal_offset_x,
                    principal_offset_y=principal_offset_y,
                    timestamp=idx / (len(images) - 1),  # TODO: rename this to id
                )
                camera_properties.w2c = w2c
                data.append(camera_properties)

        # load point cloud
        ply_path = self.dataset_path / 'sparse' / '0' / 'points3D.ply'
        if not os.path.exists(ply_path):
            Logger.logInfo('Found new scene. Converting sparse SfM points to .ply format.')
            xyz, rgb, _ = read_points3D_binary(self.dataset_path / 'sparse' / '0' / 'points3D.bin')
            storePly(ply_path, xyz, rgb)
        try:
            self.point_cloud = fetchPly(ply_path)
        except Exception:
            raise Framework.DatasetError(f'Failed to load SfM point cloud')

        # rotate/scale poses to align ground with xy plane and optionally fit to [-1, 1]^3 cube
        if self.APPLY_PCA:
            c2ws = torch.stack([camera.c2w for camera in data])
            c2ws, transformation = transformPosesPCA(c2ws, rescale=self.APPLY_PCA_RESCALE)
            for camera_properties, c2w in zip(data, c2ws):
                camera_properties.c2w = c2w
            self.point_cloud.transform(transformation)
            self.world_coordinate_system = None

        # create splits
        dataset: dict[str, list[CameraProperties]] = {subset: [] for subset in self.subsets}
        if self.TEST_STEP > 0:
            for i in range(len(data)):
                if i % self.TEST_STEP == 0:
                    dataset['test'].append(data[i])
                else:
                    dataset['train'].append(data[i])
        else:
            dataset['train'] = data

        # return the dataset
        return dataset
