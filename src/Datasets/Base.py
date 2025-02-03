# -- coding: utf-8 --

"""Datasets/Base.py: Basic dataset class features."""

from abc import ABC, abstractmethod
import dataclasses
from pathlib import Path
from typing import Any, Callable

import torch

import Framework
from Cameras.Base import BaseCamera
from Cameras.utils import CameraProperties
from Datasets.utils import BasicPointCloud, tensor_to_string
from Methods.Base.utils import CallbackTimer
from Logging import Logger


@Framework.Configurable.configure(
    PATH='path/to/dataset/directory',
    IMAGE_SCALE_FACTOR=None,
    NORMALIZE_CUBE=None,
    NORMALIZE_RECENTER=False,
    PRECOMPUTE_RAYS=False,
    TO_DEVICE=False,
    BACKGROUND_COLOR=[1.0, 1.0, 1.0],
)
class BaseDataset(Framework.Configurable, ABC, torch.utils.data.Dataset):
    """Implements common functionalities of all datasets."""

    def __init__(self,
                 path: str,
                 camera: 'BaseCamera',
                 camera_system: Callable | None = None,
                 world_system: Callable | None = None
                 ) -> None:
        Framework.Configurable.__init__(self, 'DATASET')
        ABC.__init__(self)
        torch.utils.data.Dataset.__init__(self)
        # check dataset path
        self.dataset_path: Path = Path(path)
        Logger.log(f'loading dataset: {self.dataset_path}')
        self.load_timer: CallbackTimer = CallbackTimer()
        with self.load_timer:
            # define subsets and load data
            self.subsets: list[str] = ['train', 'test', 'val']
            self.camera: 'BaseCamera' = camera
            self.camera.setBackgroundColor(*self.BACKGROUND_COLOR)
            self.camera_coordinate_system: Callable | None = camera_system
            self.world_coordinate_system: Callable | None = world_system
            self._bounding_box: torch.Tensor | None = None  # 2x3 tensor containing the min/max values of the dataset's bounding box
            self.point_cloud: BasicPointCloud | None = None
            self.num_training_cameras: int = 1
            self.mode: str = 'train'
            self.data: dict[str, list[CameraProperties]] = self.load()
            if self._bounding_box is not None:
                self._bounding_box = self._bounding_box.cpu()
            self.convertToInternalCoordinateSystem(camera_system=self.camera_coordinate_system, world_system=self.world_coordinate_system)
            self.normalizePoses('train', cube_side=self.NORMALIZE_CUBE, recenter=self.NORMALIZE_RECENTER)
            self.ray_collection: dict[str, torch.Tensor | None] = {subset: None for subset in self.subsets}
            self.on_device = False
            if self.TO_DEVICE:
                self.toDefaultDevice(['train'])
            if self.PRECOMPUTE_RAYS:
                self.precomputeRays(['train'])

    def setMode(self, mode: str) -> 'BaseDataset':
        """Sets the dataset's mode to a given string."""
        self.mode = mode
        if self.mode not in self.subsets:
            raise Framework.DatasetError(f'requested invalid dataset mode: "{mode}"\n'
                               f'available option are: {self.subsets}')
        return self

    def train(self) -> 'BaseDataset':
        """Sets the dataset's mode to training."""
        return self.setMode('train')

    def test(self) -> 'BaseDataset':
        """Sets the dataset's mode to testing."""
        self.mode = 'test'
        return self.setMode('test')

    def eval(self) -> 'BaseDataset':
        """Sets the dataset's mode to validation."""
        return self.setMode('val')

    @abstractmethod
    def load(self) -> dict[str, list[CameraProperties] | None]:
        """Loads the dataset into a dict containing lists of CameraProperties for training, evaluation, and testing."""
        return {}

    def __len__(self) -> int:
        """Returns the size of the dataset depending on its current mode."""
        return len(self.data[self.mode])

    def __getitem__(self, index: int) -> CameraProperties:
        """Fetch specified item(s) from dataset."""
        element: CameraProperties = self.data[self.mode][index]
        return element.toDefaultDevice()

    def toDefaultDevice(self, subsets: list[str] | None = None) -> None:
        """Moves the specified dataset subsets to the default device."""
        if subsets is None:
            subsets = self.data.keys()
        for subset in subsets:
            self.data[subset] = [i.toDefaultDevice() for i in self.data[subset]]
        if self._bounding_box is not None:
            self._bounding_box = self._bounding_box.to(Framework.config.GLOBAL.DEFAULT_DEVICE)
        if self.point_cloud is not None:
            self.point_cloud.toDevice(Framework.config.GLOBAL.DEFAULT_DEVICE)
        self.on_device = True

    def precomputeRays(self, subsets: list[str] | None = None) -> None:
        """Precomputes the rays for the specified dataset subsets."""
        if subsets is None:
            subsets = self.data.keys()
        for subset in subsets:
            self.setMode(subset)
            self.getAllRays()

    def getAllRays(self) -> torch.Tensor:
        """Returns all rays of the current dataset mode."""
        if self.ray_collection[self.mode] is None:
            # generate rays for all data points
            cp = self.camera.properties
            self.ray_collection[self.mode] = torch.cat([self.camera.setProperties(i).generateRays().cpu() for i in self], dim=0)
            if self.on_device:
                self.ray_collection[self.mode] = self.ray_collection[self.mode].to(Framework.config.GLOBAL.DEFAULT_DEVICE)
            last_index = 0
            for properties in self.data[self.mode]:
                num_pixels = properties.width * properties.height
                if properties._precomputed_rays is None:
                    properties._precomputed_rays = self.ray_collection[self.mode][last_index:last_index + num_pixels]
                last_index += num_pixels
            self.camera.setProperties(cp)
        return self.ray_collection[self.mode]

    def normalizePoses(self, reference_set: str = None, cube_side: float = None, recenter: bool = True) -> None:
        """Recenters and/or scales the dataset camera poses to fit into a cube of a given side."""
        if cube_side is not None or recenter:
            # get min/max for each axis over all data points in reference set (or all data points if no reference set is given)
            reference_samples = []
            for subset_key in self.data.keys():
                if reference_set is None or subset_key == reference_set:
                    reference_samples += self.data[subset_key]
            reference_positions = torch.stack([sample.c2w[:3, -1] for sample in reference_samples if sample.c2w is not None], dim=0)
            min_position = reference_positions.min(dim=0, keepdim=True).values
            max_position = reference_positions.max(dim=0, keepdim=True).values
            # compute center and scale factor
            center = ((min_position + max_position) * 0.5) if recenter else torch.zeros((1, 3), device=torch.device('cpu'))
            scale = (cube_side / (max_position - min_position).max()).item() if cube_side is not None and cube_side > 0.0 else 1.0
            # apply centering and scaling to all data points in all subsets
            for subset in self.data.values():
                for sample in subset:
                    if sample.c2w is not None:
                        c2w = sample.c2w.clone()
                        c2w[:3, -1] = (c2w[:3, -1] - center) * scale
                        sample.c2w = c2w
                        # update depth
                        if sample.depth is not None:
                            sample.depth *= scale
            # update camera near/far planes
            self.camera.near_plane *= scale
            self.camera.far_plane *= scale
            # update bounding box and point cloud
            if self._bounding_box is not None:
                self._bounding_box = (self._bounding_box - center) * scale
            if self.point_cloud is not None:
                self.point_cloud.normalize(center, scale)

    def convertToInternalCoordinateSystem(self,
                                          camera_system: Callable | None,
                                          world_system: Callable | None) -> None:
        """Converts the dataset's camera and world coordinate systems to the framework's internal representation."""
        if camera_system is not None or world_system is not None:
            # iter over subsets
            for subset in self.data.values():
                # iter over all samples
                for sample in subset:
                    if (c2w := sample.c2w) is not None:
                        c2w = c2w.clone()
                        # apply camera coordinate system conversion
                        if camera_system is not None:
                            c2w[:3, :3] = torch.cat(camera_system(*torch.split(c2w[:3, :3], split_size_or_sections=1, dim=1)), dim=1)
                        # apply world coordinate system conversion
                        if world_system is not None:
                            c2w[:3, :] = torch.cat(world_system(*torch.split(c2w[:3, :], split_size_or_sections=1, dim=0)), dim=0)
                        # set updated transformation
                        sample.c2w = c2w
            # update bounding box and point cloud
            if self._bounding_box is not None and world_system is not None:
                self._bounding_box = torch.cat(world_system(*torch.split(self._bounding_box, split_size_or_sections=1, dim=1)), dim=1).sort(dim=0).values
            if self.point_cloud is not None and world_system is not None:
                self.point_cloud.convert(world_system)

    @torch.no_grad()
    def getBoundingBox(self) -> torch.Tensor:
        if self._bounding_box is None:
            Logger.logInfo('calculating dataset bounding box')
            if self.point_cloud is not None:
                # calculate bounding box from point cloud
                self._bounding_box = self.point_cloud.getBoundingBox().cpu()
                Logger.logInfo(f'bounding box estimated from point cloud:'
                               f' {tensor_to_string(self._bounding_box[0])} (min),'
                               f' {tensor_to_string(self._bounding_box[1])} (max)')
            else:
                # calculate bounding box from camera positions and near/far planes
                positions = []
                for sample in self.train():
                    self.camera.setProperties(sample)
                    cam_position = sample.c2w[:3, 3]
                    view_dirs = self.camera.getGlobalRayDirections()
                    positions.append(cam_position + (view_dirs * self.camera.near_plane))
                    positions.append(cam_position + (view_dirs * self.camera.far_plane))
                positions = torch.cat(positions, dim=0)
                self._bounding_box = torch.stack([positions.min(dim=0).values, positions.max(dim=0).values], dim=0).cpu()
                Logger.logInfo(f'bounding box estimated from dataset camera poses:'
                               f' {tensor_to_string(self._bounding_box[0])} (min),'
                               f' {tensor_to_string(self._bounding_box[1])} (max)')
        return self._bounding_box

    def addCameraPropertyFields(self, fields: list[tuple[str, type, Any]], new_class_name: str = 'CustomCameraProperties') -> None:
        """Adds a new data field to the dataset."""
        new_class = dataclasses.make_dataclass(new_class_name, fields=[(f[0], f[1], dataclasses.field(default=f[2])) for f in fields], bases=(CameraProperties,))
        for subset in self.data.values():
            for sample in subset:
                sample.__class__ = new_class
