# -- coding: utf-8 --

"""
Datasets/iPhone.py: Provides a dataset class for DyCheck iPhone scenes.
Data available at https://github.com/KAIR-BAIR/dycheck/blob/main/docs/DATASETS.md (last accessed 2023-05-25).
"""

import json
from pathlib import Path
from typing import Any

import kornia
import numpy as np
import torch

import Framework
from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import CameraProperties
from Datasets.Base import BaseDataset
from Datasets.utils import BasicPointCloud, WorldCoordinateSystemTransformations, applyImageScaleFactor, loadImage, loadImagesParallel, \
    CameraCoordinateSystemsTransformations, loadOpticalFlowParallel
from Visual.Trajectories.BulletTime import bullet_time
from Visual.Trajectories.StabilizedView import stabilized_view
from Visual.Trajectories.NovelView import novel_view
from Logging import Logger


@Framework.Configurable.configure(
    PATH='dataset/iphone/paper-windmill',
    BACKGROUND_COLOR=[0.0, 0.0, 0.0],
    LOAD_OPTICAL_FLOW=True,
    LOAD_LIDAR=True,
)
class CustomDataset(BaseDataset):
    """Dataset class forDyCheck iPhone scenes."""

    def __init__(self, path: str) -> None:
        super().__init__(
            path,
            PerspectiveCamera(0.01, 2.0),
            CameraCoordinateSystemsTransformations.LEFT_HAND,
            WorldCoordinateSystemTransformations.XZY
        )
        # correct lidar, if loaded
        if self.LOAD_LIDAR:
            for i in Logger.logProgressBar(range(len(self.data['train'])), leave=False, desc='correcting lidar data'):
                properties = self.data['train'][i].toDefaultDevice()
                self.camera.setProperties(properties)
                # correct depth
                directions = self.camera.getGlobalRayDirections()
                unprojection_factor = torch.norm(directions, p=2, dim=-1, keepdim=True)
                depth_corrected = properties.depth / unprojection_factor.reshape(properties.depth.shape)
                # fill borders
                depth_edges = depth_corrected < 1e-6
                values = -depth_corrected.clone()
                values[depth_edges] = -self.camera.far_plane
                # fill via maxpool
                while depth_edges.sum().item() > 0:
                    depth_edges_eroded = kornia.morphology.erosion(depth_edges[None].float(), torch.ones(3, 3))[0].bool()
                    depth_edges_diff = depth_edges_eroded ^ depth_edges  # xor
                    values_filled = torch.nn.functional.max_pool2d(values[None], kernel_size=3, stride=1, padding=1)[0]
                    values[depth_edges_diff] = values_filled[depth_edges_diff]
                    depth_edges = depth_edges_eroded
                depth_corrected = -values
                # resize depth
                if self.IMAGE_SCALE_FACTOR is not None:
                    depth_corrected = applyImageScaleFactor(depth_corrected, self.IMAGE_SCALE_FACTOR, 'nearest')
                target_device = self.data['train'][i].depth.device
                # apply correction
                self.data['train'][i].depth = depth_corrected.to(target_device)

    def load(self) -> dict[str, list[CameraProperties]]:
        """Loads the dataset into a dict containing lists of CameraProperties for training and testing."""
        data: dict[str, list[CameraProperties]] = {subset: [] for subset in self.subsets}
        # load scene info
        scene_info_filepath: Path = self.dataset_path / 'scene.json'
        try:
            with open(scene_info_filepath, 'r') as f:
                scene_info_dict: dict[str, Any] = json.load(f)
        except IOError:
            raise Framework.DatasetError(f'Invalid scene info file path "{scene_info_filepath}"')
        center = torch.as_tensor(scene_info_dict['center'], dtype=torch.float32)
        scale = scene_info_dict['scale']
        self.camera.near_plane = scene_info_dict['near']
        self.camera.far_plane = scene_info_dict['far']

        # load dataset info
        dataset_info_filepath: Path = self.dataset_path / 'dataset.json'
        try:
            with open(dataset_info_filepath, 'r') as f:
                dataset_info_dict: dict[str, Any] = json.load(f)
        except IOError:
            raise Framework.DatasetError(f'Invalid dataset info file path "{dataset_info_filepath}"')
        max_time_id = dataset_info_dict['num_exemplars'] - 1

        # load extra info
        extra_info_filepath: Path = self.dataset_path / 'extra.json'
        try:
            with open(extra_info_filepath, 'r') as f:
                extra_info_dict: dict[str, Any] = json.load(f)
        except IOError:
            raise Framework.DatasetError(f'Invalid extra info file path "{extra_info_filepath}"')
        factor = extra_info_dict['factor']
        self._bounding_box = torch.as_tensor(extra_info_dict['bbox'], dtype=torch.float32)
        # fps = extra_info_dict['fps']

        # load pointcloud
        points_data = torch.from_numpy(np.load(self.dataset_path / 'points.npy'))
        self.point_cloud = BasicPointCloud(positions=(points_data - center.cpu()) * scale)

        # load images and cameras
        image_directory_path = self.dataset_path / 'rgb' / f'{factor}x'
        for split in ['train', 'val']:
            # load split info
            split_info_filepath: Path = self.dataset_path / 'splits' / f'{split}.json'
            try:
                with open(split_info_filepath, 'r') as f:
                    split_info_dict: dict[str, Any] = json.load(f)
            except IOError:
                raise Framework.DatasetError(f'Invalid {split} split info file path "{split_info_filepath}"')
            # load the authors' val split into the test set of our dataset
            split = split if split == 'train' else 'test'
            # load images
            image_filenames = [str(image_directory_path / f'{frame}.png') for frame in split_info_dict['frame_names']]
            if len(image_filenames) == 0:
                Logger.logWarning(f'No images found for split "{split}"')
                continue
            else:
                Logger.logInfo(f'Found {len(image_filenames)} images for split "{split}"')
            rgbs, _ = loadImagesParallel(image_filenames, self.IMAGE_SCALE_FACTOR, num_threads=4, desc=split)
            # load optical flow
            if self.LOAD_OPTICAL_FLOW:
                forward_flows, backward_flows = loadOpticalFlowParallel(self.dataset_path / 'flow' / f'{factor}x', split_info_dict['frame_names'],
                                                                        num_threads=4, image_scale_factor=self.IMAGE_SCALE_FACTOR)
            else:
                forward_flows, backward_flows = [None] * len(rgbs), [None] * len(rgbs)
            # log lidar
            if self.LOAD_LIDAR:
                Logger.logInfo('Loading lidar data instead of monocular depth')
            # create split CameraProperties objects
            for rgb, frame_name, time_id, forward_flow, backward_flow in zip(rgbs, split_info_dict['frame_names'], split_info_dict['time_ids'], forward_flows, backward_flows):
                # load camera info
                camera_info_filepath: Path = self.dataset_path / 'camera' / f'{frame_name}.json'
                try:
                    with open(camera_info_filepath, 'r') as f:
                        camera_info_dict: dict[str, Any] = json.load(f)
                except IOError:
                    raise Framework.DatasetError(f'Invalid camera info file path "{camera_info_filepath}"')
                # make sure there is no skew or distortion
                if camera_info_dict['skew'] != 0.0:
                    raise Framework.DatasetError('Camera axis skew not supported')
                if torch.count_nonzero(torch.as_tensor(camera_info_dict['radial_distortion'])).item() > 0:
                    raise Framework.DatasetError('Radial camera distortion not supported')
                if torch.count_nonzero(torch.as_tensor(camera_info_dict['tangential_distortion'])).item() > 0:
                    raise Framework.DatasetError('Tangential camera distortion not supported')
                # focal length
                focal_length = camera_info_dict['focal_length'] / factor
                pixel_aspect_ratio = camera_info_dict['pixel_aspect_ratio']
                focal_x = focal_length
                focal_y = focal_length * pixel_aspect_ratio
                # principal point
                principal_point_x, principal_point_y = torch.as_tensor(camera_info_dict['principal_point'], dtype=torch.float32) / factor
                # adjust intrinsics when images are resized
                if self.IMAGE_SCALE_FACTOR is not None:
                    focal_x *= self.IMAGE_SCALE_FACTOR
                    focal_y *= self.IMAGE_SCALE_FACTOR
                    principal_point_x *= self.IMAGE_SCALE_FACTOR
                    principal_point_y *= self.IMAGE_SCALE_FACTOR
                # c2w matrix
                rotation = torch.linalg.inv(torch.as_tensor(camera_info_dict['orientation'], dtype=torch.float32))
                translation = (torch.as_tensor(camera_info_dict['position'], dtype=torch.float32) - center) * scale
                c2w = torch.cat([
                    torch.cat([rotation, translation[..., None]], dim=-1),
                    torch.tensor([[0, 0, 0, 1]], dtype=torch.float32),
                ], dim=-2)
                segmentation, _ = loadImage(self.dataset_path / 'sfm_masks' / f'{factor}x' / f'{frame_name}.png.png', scale_factor=self.IMAGE_SCALE_FACTOR)
                # load lidar
                monoc_depth = depth = None
                if split == 'train' and self.LOAD_LIDAR:
                    depth = torch.from_numpy(np.load(self.dataset_path / 'depth' / f'{factor}x' / f'{frame_name}.npy'))
                    depth = depth.reshape((1, depth.shape[0], depth.shape[1]))
                    depth *= scale
                else:
                    monoc_depth = torch.from_numpy(np.load(self.dataset_path / 'monoc_depth' / f'{factor}x' / f'{frame_name}.png.npy'))
                    if self.IMAGE_SCALE_FACTOR is not None:
                        monoc_depth = applyImageScaleFactor(monoc_depth, self.IMAGE_SCALE_FACTOR, 'nearest')
                # create property
                data[split].append(CameraProperties(
                    width=rgb.shape[2],
                    height=rgb.shape[1],
                    rgb=rgb,
                    alpha=None,
                    _misc=monoc_depth,
                    depth=depth,
                    segmentation=1.0 - segmentation,
                    c2w=c2w,
                    focal_x=focal_x,
                    focal_y=focal_y,
                    forward_flow=forward_flow,
                    backward_flow=backward_flow,
                    principal_offset_x=(principal_point_x - (rgb.shape[2] * 0.5)).item(),
                    principal_offset_y=(principal_point_y - (rgb.shape[1] * 0.5)).item(),
                    timestamp=time_id / max_time_id
                ))
        # create val set
        if not data['val']:
            for i in data['train']:
                data['val'].append(CameraProperties(
                    width=data['train'][0].width,
                    height=data['train'][0].height,
                    rgb=data['train'][0].rgb.clone(),
                    alpha=None,
                    c2w=data['train'][0].c2w.clone(),
                    focal_x=data['train'][0].focal_x,
                    focal_y=data['train'][0].focal_y,
                    principal_offset_x=data['train'][0].principal_offset_x,
                    principal_offset_y=data['train'][0].principal_offset_y,
                    timestamp=i.timestamp
                ))
        # save params for visualization trajectories
        self.reference_pose_rel_id = 0.0
        self.custom_lookat = torch.as_tensor(extra_info_dict['lookat'], dtype=torch.float32)
        self.custom_up = torch.as_tensor(extra_info_dict['up'], dtype=torch.float32)
        self.num_frames_per_rotation = 90
        self.degree = 30
        self.num_repeats = round(dataset_info_dict['num_exemplars'] / self.num_frames_per_rotation)
        self.subsets += ['novel_view', 'stabilized_view', 'bullet_time']
        data['novel_view'] = novel_view(reference_pose_rel_id=self.reference_pose_rel_id, custom_lookat=self.custom_lookat, custom_up=self.custom_up,
                                        num_frames_per_rotation=self.num_frames_per_rotation, degree=self.degree)._generate(self.camera, data['train'])
        data['stabilized_view'] = stabilized_view(reference_pose_rel_id=self.reference_pose_rel_id)._generate(self.camera, data['train'])
        data['bullet_time'] = bullet_time(reference_pose_rel_id=self.reference_pose_rel_id, custom_lookat=self.custom_lookat, custom_up=self.custom_up,
                                          num_frames_per_rotation=self.num_frames_per_rotation, degree=self.degree, num_repeats=self.num_repeats)._generate(self.camera, data['train'])
        for s in data['novel_view'] + data['bullet_time']:
            # flip z axis
            s.c2w[:3, 2] *= -1
        # return dataset
        return data
