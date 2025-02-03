# -- coding: utf-8 --

"""
Datasets/VGGSfM.py: Dataset for image sequences posed via VGGSfM.
You can use our scripts/runVGGSfM.py script to generate a dataset in this format.
(see https://github.com/facebookresearch/vggsfm)
"""

import os
from pathlib import Path

import numpy as np
import torch

import Framework
from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import CameraProperties, RadialTangentialDistortion
from Datasets.Base import BaseDataset
from Datasets.Colmap import fetchPly, quaternion_to_R, read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, storePly
from Datasets.utils import CameraCoordinateSystemsTransformations, WorldCoordinateSystemTransformations, applyImageScaleFactor, getNearFarFromPointCloud, loadImagesParallel, \
                           loadOpticalFlowParallel, loadImage, transformPosesPCA
from Logging import Logger


@Framework.Configurable.configure(
    PATH='dataset/vggsfm/train',
    BACKGROUND_COLOR=[0.0, 0.0, 0.0],
    TEST_STEP=0,
    APPLY_PCA=False,
)
class CustomDataset(BaseDataset):
    """Dataset class for Colmap-calibrated scenes."""

    def __init__(self, path: str) -> None:
        super().__init__(
            path,
            PerspectiveCamera(0.01, 1.0),  # will be updated according to sfm pointcloud
            CameraCoordinateSystemsTransformations.LEFT_HAND,
            WorldCoordinateSystemTransformations.XnZY
        )
        self.camera.near_plane, self.camera.far_plane = getNearFarFromPointCloud(self.camera, self.point_cloud, self.data['train'], 0.1)

    def load(self) -> dict[str, list[CameraProperties] | None]:
        """Loads the dataset into a dict containing lists of CameraProperties for training, evaluation, and testing."""
        dataset: dict[str, list[CameraProperties]] = {subset: [] for subset in self.subsets}
        load_segmentation = Path(self.dataset_path / 'masks').exists()
        load_flow = Path(self.dataset_path / 'flow').exists()
        load_disp = Path(self.dataset_path / 'monoc_depth').exists()
        # load colmap data
        cameras_extrinsic_file = self.dataset_path / 'images.bin'
        cameras_intrinsic_file = self.dataset_path / 'cameras.bin'
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

        # create camera properties
        scale_factor_intrinsics = self.IMAGE_SCALE_FACTOR if self.IMAGE_SCALE_FACTOR is not None else 1.0
        data: list[CameraProperties] = []
        for cam_data in cam_intrinsics.values():
            images = [data for data in cam_extrinsics.values() if data.camera_id == cam_data.id]
            images = sorted(images, key=lambda data: data.name)
            # load images
            image_filenames = [str(self.dataset_path / 'images' / image.name) for image in images]
            rgbs, alphas = loadImagesParallel(image_filenames, self.IMAGE_SCALE_FACTOR, num_threads=12, desc=f'camera {cam_data.id}')
            forward_flows, backward_flows = [None] * len(rgbs), [None] * len(rgbs)
            if load_flow:
                # load optical flow
                forward_flows, backward_flows = loadOpticalFlowParallel(self.dataset_path / 'flow', [image.name for image in images],
                                                                        num_threads=4, image_scale_factor=self.IMAGE_SCALE_FACTOR)
            for idx, (image, rgb, alpha) in enumerate(zip(images, rgbs, alphas)):
                # extract c2w matrix
                rotation_matrix = torch.from_numpy(quaternion_to_R(image.qvec)).float()
                translation_vector = torch.from_numpy(image.tvec).float()
                w2c = torch.eye(4, device=torch.device('cpu'))
                w2c[:3, :3] = rotation_matrix
                w2c[:3, 3] = translation_vector
                c2w = torch.linalg.inv(w2c)
                # intrinsics
                focal_x = cam_data.params[0] * scale_factor_intrinsics
                focal_y = cam_data.params[0] * scale_factor_intrinsics
                principal_offset_x = (cam_data.params[1] - cam_data.width / 2) * scale_factor_intrinsics
                principal_offset_y = (cam_data.params[2] - cam_data.height / 2) * scale_factor_intrinsics
                distortion_parameters: RadialTangentialDistortion | None = None
                match cam_data.model:
                    case 'SIMPLE_PINHOLE':
                        distortion_parameters = None
                    case 'PINHOLE':
                        distortion_parameters = None
                    case 'SIMPLE_RADIAL':
                        distortion_parameters = RadialTangentialDistortion(
                            k1=cam_data.params[4]
                        )
                    case 'RADIAL':
                        distortion_parameters = RadialTangentialDistortion(
                            k1=cam_data.params[4],
                            k2=cam_data.params[5],
                        )
                    case 'OPENCV':
                        distortion_parameters = RadialTangentialDistortion(
                            k1=cam_data.params[4],
                            k2=cam_data.params[5],
                            p1=cam_data.params[6],
                            p2=cam_data.params[7],
                        )
                    case 'OPENCV_FISHEYE':
                        distortion_parameters = RadialTangentialDistortion(
                            k1=cam_data.params[4],
                            k2=cam_data.params[5],
                            k3=cam_data.params[6],
                            # k4=cam_data.params[7],
                        )
                    case '_':
                        raise Framework.DatasetError(f'Unknown camera model "{cam_data.model}"')
                disp = segmentation = None
                if load_disp:
                    disp = torch.from_numpy(np.load(self.dataset_path / 'monoc_depth' / f'{Path(image_filenames[idx]).name}.npy'))
                    if self.IMAGE_SCALE_FACTOR is not None:
                        disp = applyImageScaleFactor(disp, self.IMAGE_SCALE_FACTOR, 'nearest')
                if load_segmentation:
                    segmentation, _ = loadImage(self.dataset_path / 'masks' / Path(image_filenames[idx]).name, scale_factor=self.IMAGE_SCALE_FACTOR)
                # create camera properties and subsets
                data.append(CameraProperties(
                    width=rgb.shape[2],
                    height=rgb.shape[1],
                    rgb=rgb,
                    _misc=disp,
                    segmentation=segmentation,
                    alpha=alpha,
                    c2w=c2w,
                    focal_x=focal_x,
                    focal_y=focal_y,
                    principal_offset_x=principal_offset_x,
                    principal_offset_y=principal_offset_y,
                    distortion_parameters=distortion_parameters,
                    forward_flow=forward_flows[idx],
                    backward_flow=backward_flows[idx],
                    timestamp=idx / (len(images) - 1),
                ))

        # load point cloud
        point_path_bin = self.dataset_path / 'points3D.bin'
        point_path_ply = self.dataset_path / 'points3D.ply'
        if not os.path.exists(point_path_ply):
            Logger.logInfo('Found new scene. Converting sparse SfM points to .ply format.')
            xyz, rgb, _ = read_points3D_binary(point_path_bin)
            storePly(point_path_ply, xyz, rgb)
        try:
            self.point_cloud = fetchPly(str(point_path_ply))
        except Exception:
            raise Framework.DatasetError(f'Could not load point cloud from "{point_path_ply}"')

        if self.APPLY_PCA:
            # rotate/scale poses to align ground with xy plane and fit to [-1, 1]^3 cube
            c2ws = torch.stack([camera.c2w for camera in data])
            c2ws, transformation = transformPosesPCA(c2ws)
            for camera_properties, c2w in zip(data, c2ws):
                camera_properties.c2w = c2w
            self.point_cloud.transform(transformation)
            self.world_coordinate_system = None

        # extract bounding box
        self.point_cloud.filterOutliers(filter_percentage=0.95)
        self._bounding_box = self.point_cloud.getBoundingBox(tolerance_factor=0.05)

        # perform test split
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
