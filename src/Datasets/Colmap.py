"""Datasets/Colmap.py: Provides a dataset class for scenes in COLMAP format."""

from functools import partial
from pathlib import Path

import numpy as np
import pycolmap

import Framework
from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import RadialTangentialDistortion
from Datasets.Base import BaseDataset
from Datasets.utils import compute_scaled_image_size, View, ImageData, transform_poses_pca, BasicPointCloud, \
    load_inverted_segmentation_mask, load_disparity, apply_image_scale_factor, load_optical_flow, \
    apply_image_scale_factor_optical_flow, estimate_near_far
from Logging import Logger


@Framework.Configurable.configure(
    PATH='dataset/colmap/myscene',
    TEST_STEP=0,
    APPLY_PCA=False,
    SFM_POINTS_FILTER_RATIO=1.0,  # 0.95 works well in practice
    AABB_TOLERANCE_FACTOR=0.05,  # framework default is 0.1
    ESTIMATE_NEAR_FAR_FROM_SFM_POINTS=False,  # works well with methods that rely on tight near and far bounds
)
class CustomDataset(BaseDataset):
    """Dataset class for scenes in COLMAP format."""

    def __init__(self, path: str) -> None:
        super().__init__(path)

    def load(self) -> tuple[list[PerspectiveCamera], dict[str, list[View]]]:
        """Loads the dataset into a dict containing lists of views for training and testing."""
        # load colmap data
        reconstruction = pycolmap.Reconstruction(self.dataset_path / 'sparse' / '0')
        Logger.log_debug(reconstruction.summary())

        has_segmentation = Path(self.dataset_path / 'sfm_masks').exists()
        has_flow = Path(self.dataset_path / 'flow').exists()
        has_disp = Path(self.dataset_path / 'monoc_depth').exists()

        # load cameras and views
        cameras: list[PerspectiveCamera] = []
        data: list[View] = []
        global_frame_idx = 0
        for camera_idx, colmap_camera in Logger.log_progress(enumerate(reconstruction.cameras.values()), desc=f'loading camera views', leave=False, total=len(cameras)):
            # load intrinsics
            match colmap_camera.model:
                case pycolmap.CameraModelId.SIMPLE_PINHOLE:
                    focal_x = focal_y = colmap_camera.params[0]
                    center_x = colmap_camera.params[1]
                    center_y = colmap_camera.params[2]
                    distortion = None
                case pycolmap.CameraModelId.PINHOLE:
                    focal_x = colmap_camera.params[0]
                    focal_y = colmap_camera.params[1]
                    center_x = colmap_camera.params[2]
                    center_y = colmap_camera.params[3]
                    distortion = None
                case pycolmap.CameraModelId.SIMPLE_RADIAL:
                    focal_x = focal_y = colmap_camera.params[0]
                    center_x = colmap_camera.params[1]
                    center_y = colmap_camera.params[2]
                    distortion = RadialTangentialDistortion(k1=colmap_camera.params[3])
                case pycolmap.CameraModelId.RADIAL:
                    focal_x = focal_y = colmap_camera.params[0]
                    center_x = colmap_camera.params[1]
                    center_y = colmap_camera.params[2]
                    distortion = RadialTangentialDistortion(k1=colmap_camera.params[3], k2=colmap_camera.params[4])
                case pycolmap.CameraModelId.OPENCV:
                    focal_x = colmap_camera.params[0]
                    focal_y = colmap_camera.params[1]
                    center_x = colmap_camera.params[2]
                    center_y = colmap_camera.params[3]
                    distortion = RadialTangentialDistortion(
                        k1=colmap_camera.params[4],
                        k2=colmap_camera.params[5],
                        p1=colmap_camera.params[6],
                        p2=colmap_camera.params[7],
                    )
                case _:
                    raise Framework.DatasetError(f'Camera model {colmap_camera.model} from COLMAP is not yet supported.')
            # rescale intrinsics
            width, height = compute_scaled_image_size((colmap_camera.width, colmap_camera.height), self.IMAGE_SCALE_FACTOR)
            scale_factor_intrinsics_x = width / colmap_camera.width
            scale_factor_intrinsics_y = height / colmap_camera.height
            focal_x *= scale_factor_intrinsics_x
            focal_y *= scale_factor_intrinsics_y
            center_x *= scale_factor_intrinsics_x
            center_y *= scale_factor_intrinsics_y
            camera = PerspectiveCamera(
                shared_settings=self._camera_settings, width=width, height=height,
                focal_x=focal_x, focal_y=focal_y, center_x=center_x, center_y=center_y,
                distortion=distortion,
            )
            cameras.append(camera)
            # sort images belonging to this camera
            images = [image for image in reconstruction.images.values() if image.camera.camera_id == colmap_camera.camera_id]
            images = sorted(images, key=lambda image: image.name)
            # create View instances
            n_views = len(images)
            last_view_idx = n_views - 1
            idx2timestamp = 1 / last_view_idx
            for frame_idx, image in enumerate(images):
                data.append(View(
                    camera=camera,
                    camera_index=camera_idx,
                    frame_idx=frame_idx,
                    global_frame_idx=global_frame_idx,
                    c2w=image.cam_from_world().inverse().matrix(),
                    timestamp=frame_idx * idx2timestamp,  # significant assumption about how images were taken
                    rgb=ImageData(
                        self.dataset_path / 'images' / image.name,
                        n_channels=3, scale_factor=self.IMAGE_SCALE_FACTOR
                    ),
                    segmentation=ImageData(
                        self.dataset_path / 'sfm_masks' / f'{image.name}.png',
                        n_channels=1, scale_factor=self.IMAGE_SCALE_FACTOR,
                        load_fn=load_inverted_segmentation_mask
                    ) if has_segmentation else None,
                    forward_flow=ImageData(
                        self.dataset_path / 'flow' / f'{image.name.split(".")[0]}_forward.flo',
                        n_channels=2, scale_factor=self.IMAGE_SCALE_FACTOR,
                        load_fn=load_optical_flow, resize_fn=apply_image_scale_factor_optical_flow
                    ) if has_flow and frame_idx < last_view_idx else None,
                    backward_flow=ImageData(
                        self.dataset_path / 'flow' / f'{image.name.split(".")[0]}_backward.flo',
                        n_channels=2, scale_factor=self.IMAGE_SCALE_FACTOR,
                        load_fn=load_optical_flow, resize_fn=apply_image_scale_factor_optical_flow
                    ) if has_flow and frame_idx > 0 else None,
                    misc=ImageData(
                        self.dataset_path / 'monoc_depth' / f'{image.name}.npy',
                        n_channels=1, load_fn=load_disparity, resize_fn=partial(apply_image_scale_factor, mode='nearest')
                    ) if has_disp else None,
                ))
                global_frame_idx += 1

        # load point cloud
        self.point_cloud = BasicPointCloud.from_colmap(reconstruction)

        # rotate poses to align ground with xz plane
        if self.APPLY_PCA:
            c2ws = np.stack([view.c2w_numpy for view in data])
            c2ws, transformation = transform_poses_pca(c2ws, rescale=False)
            for view, c2w in zip(data, c2ws):
                view.c2w = c2w
            self.point_cloud.transform(transformation)

        # filter point cloud outliers
        filter_ratio = 1.0 if self.SFM_POINTS_FILTER_RATIO is None else self.SFM_POINTS_FILTER_RATIO
        if filter_ratio != 1.0:
            self.point_cloud.filter_outliers(filter_ratio)

        # extract bounding box from point cloud
        self.bounding_box = self.point_cloud.get_aabb(tolerance_factor=self.AABB_TOLERANCE_FACTOR)

        # estimate near and far plane from point cloud
        if self.ESTIMATE_NEAR_FAR_FROM_SFM_POINTS:
            self._camera_settings.near_plane, self._camera_settings.far_plane = estimate_near_far(data, self.point_cloud)

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
