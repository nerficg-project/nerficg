"""Datasets/Base.py: Basic dataset class features."""

from typing import Iterator, Iterable
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

import Framework
from Cameras.Base import BaseCamera
from Cameras.utils import SharedCameraSettings, look_at
from Datasets.utils import BasicPointCloud, AxisAlignedBox, View, RayCollection, RayBatch
from Methods.Base.utils import CallbackTimer
from Logging import Logger

DEFAULT_CAMERA_INDEX = 0
DEFAULT_VIEW_INDEX = 0

@Framework.Configurable.configure(
    PATH='path/to/dataset/directory',
    IMAGE_SCALE_FACTOR=None,
    NORMALIZE_CUBE=None,
    NORMALIZE_RECENTER=False,
    BACKGROUND_COLOR=[0.0, 0.0, 0.0],
    NEAR_PLANE=0.01,
    FAR_PLANE=1000.0,
)
class BaseDataset(Framework.Configurable, ABC, Iterable[View]):
    """Implements common functionalities of all datasets."""

    def __init__(self, path: str) -> None:
        Framework.Configurable.__init__(self, 'DATASET')
        ABC.__init__(self)
        # data model
        self.subsets = ['train', 'test', 'val']
        self.mode = 'train'
        self._bounding_box: AxisAlignedBox | None = None
        self._point_cloud: BasicPointCloud | None = None  # TODO: only load on demand
        self._camera_settings = SharedCameraSettings(
            background_color=torch.tensor(self.BACKGROUND_COLOR, dtype=torch.float32, device=Framework.config.GLOBAL.DEFAULT_DEVICE),
            near_plane=float(self.NEAR_PLANE),
            far_plane=float(self.FAR_PLANE)
        )
        # check dataset path
        self.dataset_path = Path(path)
        Logger.log(f'loading dataset: {self.dataset_path}')
        self.load_timer = CallbackTimer()
        with self.load_timer:
            # load and process dataset
            self.cameras, self.data = self.load()
            self.ray_collection: dict[str, RayCollection | None] = {subset: None for subset in self.subsets}
            if self.NORMALIZE_CUBE is not None or self.NORMALIZE_RECENTER:
                self.normalize('train', self.NORMALIZE_CUBE, self.NORMALIZE_RECENTER)

    def set_mode(self, mode: str) -> 'BaseDataset':
        """Sets the dataset's mode to a given string."""
        self.mode = mode
        if self.mode not in self.subsets:
            raise Framework.DatasetError(f'requested invalid dataset mode: "{mode}"\n'
                                         f'available option are: {self.subsets}')
        return self

    def train(self) -> 'BaseDataset':
        """Sets the dataset's mode to training."""
        return self.set_mode('train')

    def test(self) -> 'BaseDataset':
        """Sets the dataset's mode to testing."""
        return self.set_mode('test')

    def eval(self) -> 'BaseDataset':
        """Sets the dataset's mode to validation."""
        return self.set_mode('val')

    @abstractmethod
    def load(self) -> tuple[list[BaseCamera], dict[str, list[View] | None]]:
        """Parses the dataset-specific format into the data model."""
        pass

    def __len__(self) -> int:
        """Returns the size of the dataset depending on its current mode."""
        return len(self.data[self.mode])

    def __getitem__(self, index: int) -> View:
        """Fetch specified item(s) from dataset."""
        return self.data[self.mode][index]

    def __iter__(self) -> Iterator[View]:
        return iter(self.data[self.mode])

    @property
    def default_camera(self) -> BaseCamera:
        """Returns the dataset's default camera."""
        return self.cameras[DEFAULT_CAMERA_INDEX]

    @property
    def default_view(self) -> View:
        """Returns the dataset's default view."""
        for subset in self.subsets:
            if len(self.data[subset]) > 0:
                return self.data[subset][DEFAULT_VIEW_INDEX]
        cam_position = np.array([0.0, -1.0, -2.0])
        lookat_point = np.array([0.0, -1.0, 0.0])
        up_axis = np.array([0.0, -1.0, 0.0])
        default_c2w = look_at(cam_position, lookat_point, up_axis)
        return View(self.default_camera, DEFAULT_CAMERA_INDEX, 0, 0, default_c2w)

    @property
    def point_cloud(self) -> BasicPointCloud | None:
        """Returns the dataset's point cloud."""
        return self._point_cloud

    @point_cloud.setter
    def point_cloud(self, new_point_cloud: BasicPointCloud) -> None:
        """Sets the dataset's point cloud."""
        if not isinstance(new_point_cloud, BasicPointCloud):
            raise Framework.DatasetError(f'point cloud must be specified as BasicPointCloud, got {type(new_point_cloud)}')
        if self._point_cloud is not None:
            Logger.log_warning(f'overwriting existing point cloud: {self._point_cloud}')
        self._point_cloud = new_point_cloud
        Logger.log_info(f'point cloud set to: {self._point_cloud}')

    @property
    def bounding_box(self) -> AxisAlignedBox:
        """Returns the dataset's bounding box."""
        if self._bounding_box is None:
            Logger.log_info('bounding box not set, estimating from dataset')
            self.estimate_bounding_box()
        return self._bounding_box

    @bounding_box.setter
    def bounding_box(self, new_bounding_box: torch.Tensor | AxisAlignedBox) -> None:
        """Sets the dataset's bounding box."""
        if isinstance(new_bounding_box, torch.Tensor):
            new_bounding_box = AxisAlignedBox(new_bounding_box)
        if not isinstance(new_bounding_box, AxisAlignedBox):
            raise Framework.DatasetError(f'bounding box must be specified as torch.Tensor or AxisAlignedBox, got {type(new_bounding_box)}')
        if self._bounding_box is not None:
            Logger.log_warning(f'overwriting existing bounding box: {self._bounding_box}')
        self._bounding_box = new_bounding_box
        Logger.log_info(f'bounding box set to: {self._bounding_box}')

    def estimate_bounding_box(self) -> None:
        """Estimates the dataset's bounding box from the available data."""
        if self._point_cloud is not None:
            Logger.log_info(f'using axis-aligned bounding box of point cloud')
            self.bounding_box = self._point_cloud.get_aabb()
        elif len(self.train()) > 0:
            # TODO: extract into util function
            Logger.log_info(f'estimating axis-aligned bounding box from near and far planes of training views')
            min_max = torch.tensor([[torch.inf, torch.inf, torch.inf], [-torch.inf, -torch.inf, -torch.inf]])
            for view in self.train():
                # TODO: use numpy here?
                # FIXME: if there is distortion, this is not fully correct
                xy_screen_bounds = torch.tensor([
                    [0.0, 0.0],
                    [0.0, view.camera.height],
                    [view.camera.width, view.camera.height],
                    [view.camera.width, 0.0]
                ])[None, :, :].expand(2, -1, -1).reshape(-1, 2)  # [0,0], [0,h], [w,h], [w,0], [0,0], [0,h], [w,h], [w,0]
                depths = torch.tensor([
                    view.camera.near_plane, view.camera.far_plane
                ])[:, None].expand(2, 4).reshape(-1, 1)  # [near, near, near, near, far, far, far, far]
                frustum_bounds_world = view.unproject_points(xy_screen_bounds, depths)
                min_max[0] = torch.min(min_max[0], frustum_bounds_world.min(dim=0).values)
                min_max[1] = torch.max(min_max[1], frustum_bounds_world.max(dim=0).values)
            self.bounding_box = AxisAlignedBox(min_max)
        else:
            raise Framework.DatasetError('cannot estimate bounding box, neither point cloud nor training views available')

    def precompute_rays(self, subsets: str | list[str] | None = None, store_on_cpu: bool = False) -> None:
        """Precomputes the rays for the specified dataset subsets."""
        if subsets is None:
            subsets = self.data.keys()
        elif isinstance(subsets, str):
            subsets = [subsets]

        old_mode = self.mode
        for subset in subsets:
            self.set_mode(subset)
            if self.ray_collection[self.mode] is None:
                self.ray_collection[self.mode] = self.compute_all_rays(store_on_cpu=store_on_cpu, as_ray_collection=True)
        self.set_mode(old_mode)

    def get_total_ray_count(self) -> int:
        """Returns the total number of rays for the current dataset mode."""
        if self.ray_collection[self.mode] is not None:
            return len(self.ray_collection[self.mode])
        return sum(view.camera.width * view.camera.height for view in self.data[self.mode])

    def get_all_rays(self) -> RayBatch:
        """Returns all rays for the current dataset mode."""
        # TODO: this currently is only used by RayPoolSampler for randomly sampling all images in train/val iterations
        #  we should rework this function and the RayPoolSampler to support the following three workflows:
        #     1. rays fit into VRAM -> precompute before training and store in VRAM
        #     2. rays fit into RAM but not VRAM -> precompute before training, store in RAM, upload batch-wise to VRAM
        #     3. (new) sample pixels from all images, compute rays dynamically on CPU/GPU depending on required memory
        if self.ray_collection[self.mode] is not None:
            return self.ray_collection[self.mode].all_rays
        return self.compute_all_rays(store_on_cpu=True)  # assume memory is an issue if rays are not precomputed

    def compute_all_rays(self, store_on_cpu: bool = False, as_ray_collection: bool = False) -> RayBatch | RayCollection:
        """Computes the rays for the current dataset mode."""
        subset_rays = []
        subset_camera_slices = []
        start_idx = 0
        for view in self:
            ray_batch = view.get_rays()
            subset_rays.append(ray_batch.cpu() if store_on_cpu else ray_batch)
            if as_ray_collection:
                n_rays = len(ray_batch)
                subset_camera_slices.append(slice(start_idx, start_idx + n_rays))
                start_idx += n_rays
        subset_rays = RayBatch.cat(subset_rays)
        return RayCollection(subset_rays, subset_camera_slices) if as_ray_collection else subset_rays

    def normalize(self, reference_set: str = None, cube_side: float = None, recenter: bool = True) -> None:
        """Recenters and/or scales the dataset so that the camera poses to fit into a cube with the given side length."""
        # get min/max for each axis over all data points in reference set (or all data points if no reference set is given)
        # FIXME: cpu vs gpu and numpy vs torch is a mess here
        reference_views = []
        for subset_key in self.data.keys():
            if reference_set is None or subset_key == reference_set:
                reference_views += self.data[subset_key]
        reference_positions = torch.stack([view.position.cpu() for view in reference_views])
        min_position = reference_positions.min(dim=0).values
        max_position = reference_positions.max(dim=0).values
        # compute center and scale factor
        center = ((min_position + max_position) * 0.5) if recenter else torch.zeros(3, dtype=torch.float32, device=torch.device('cpu'))
        scale = (cube_side / (max_position - min_position).max()).item() if cube_side is not None and cube_side > 0.0 else 1.0
        # apply centering and scaling to all data points in all subsets
        for subset in self.data.values():
            for view in subset:
                view.recenter_and_scale(center.cpu().numpy().astype(np.float64), scale)
        # update camera near/far planes
        self._camera_settings.near_plane *= scale
        self._camera_settings.far_plane *= scale
        # update bounding box and point cloud
        if self._bounding_box is not None:
            self._bounding_box.normalize(center, scale)
        if self._point_cloud is not None:
            self._point_cloud.normalize(center, scale)
        Logger.log_info(f'normalized cameras to fit into {self._bounding_box} with center at {center.tolist()}')  # FIXME: self._bounding_box can be None
