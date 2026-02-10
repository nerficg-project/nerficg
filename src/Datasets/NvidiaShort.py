"""
Datasets/NvidiaShort.py: Short monocularized version of the Nvidia Dynamic Scenes Dataset as described in NSFF.
"""

import numpy as np
import pycolmap

import Framework
from Logging import Logger
from Cameras.Perspective import PerspectiveCamera
from Datasets.Base import BaseDataset
from Datasets.utils import list_sorted_files, View, read_image_size, ImageData, load_optical_flow, \
    load_inverted_segmentation_mask, load_disparity, BasicPointCloud, estimate_near_far


@Framework.Configurable.configure(
    PATH='dataset/nds_preprocessed/Skating',
    WORLD_SCALING=None,
)
class CustomDataset(BaseDataset):
    """Dataset class for short Nvidia Dynamic Scenes Dataset."""

    def load(self) -> tuple[list[PerspectiveCamera], dict[str, list[View]]]:
        """Loads the dataset into a dict containing lists of views for training and testing."""
        if self.IMAGE_SCALE_FACTOR is not None:
            raise Framework.DatasetError('The NvidiaShort loader does not support image resizing.')

        # load camera information
        poses_bounds = np.load(self.dataset_path / 'poses_bounds.npy')
        if poses_bounds.shape[1] != 17:
            raise Framework.DatasetError(f'Invalid poses_bounds.npy file with shape {poses_bounds.shape}')
        extrinsics_intrinsics = poses_bounds[:, :15].reshape(-1, 3, 5)
        extrinsics = extrinsics_intrinsics[..., :4]
        intrinsics = extrinsics_intrinsics[..., 4]
        depth_min_max = poses_bounds[:, 15:]
        heights, widths, focals = intrinsics[:, 0], intrinsics[:, 1], intrinsics[:, 2]
        if np.any(widths != widths[0]) or np.any(heights != heights[0]) or np.any(focals != focals[0]):
            raise Framework.DatasetError('Intrinsics must be the same for all images')

        # load training image paths
        train_images_path = self.dataset_path / 'images_2'
        train_image_paths = [train_images_path / file for file in list_sorted_files(train_images_path)]

        # set up intrinsics
        original_width, original_height, original_focal = round(widths[0]), round(heights[0]), float(focals[0])
        width, height = read_image_size(train_image_paths[0])
        scale_factor_intrinsics_x = width / original_width
        scale_factor_intrinsics_y = height / original_height
        focal_x = original_focal * scale_factor_intrinsics_x
        focal_y = original_focal * scale_factor_intrinsics_y

        # set up extrinsics
        c2ws = np.concatenate([extrinsics, np.broadcast_to([0, 0, 0, 1], (extrinsics.shape[0], 1, 4))], axis=1)
        cam_transform = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])  # LLFF to Colmap
        c2ws = c2ws @ cam_transform.T

        # rescale coordinates
        if self.WORLD_SCALING is not None:
            scaling = 1.0 / (depth_min_max.min() * self.WORLD_SCALING)
            c2ws[:, :3, 3] *= scaling
            depth_min_max *= scaling

        # set up camera
        self._camera_settings.near_plane = float(depth_min_max.min()) * 0.9
        self._camera_settings.far_plane = float(depth_min_max.max())
        camera = PerspectiveCamera(
            shared_settings=self._camera_settings, width=width, height=height, focal_x=focal_x, focal_y=focal_y,
        )

        # load point cloud
        reconstruction = pycolmap.Reconstruction(self.dataset_path / 'sparse' / '0')
        Logger.log_debug(reconstruction.summary())
        self.point_cloud = BasicPointCloud.from_colmap(reconstruction)

        # create bounding box
        self.point_cloud.filter_outliers(filter_ratio=0.90)
        self.bounding_box = self.point_cloud.get_aabb(tolerance_factor=0.05)

        # load segmentation mask paths
        segmentation_path = self.dataset_path / 'motion_masks_dnpc'
        segmentation_paths = [segmentation_path / file for file in list_sorted_files(segmentation_path)]

        # initialize dataset
        dataset: dict[str, list[View]] = {subset: [] for subset in self.subsets}

        # load training views
        n_cameras = len(train_image_paths)
        last_camera_idx = n_cameras - 1
        idx2timestamp = 1 / last_camera_idx
        for idx, rgb_path in Logger.log_progress(enumerate(train_image_paths), desc=f'training views', leave=False):
            flow_path = self.dataset_path / 'flow_ours'
            fw_flow_path = flow_path / f'{idx:03d}_forward.flo'
            bw_flow_path = flow_path / f'{idx:03d}_backward.flo'
            segmentation_path = segmentation_paths[idx]
            disparity_path = self.dataset_path / 'disp_dnpc' / f'{idx:03d}.png.npy'
            dataset['train'].append(View(
                camera=camera,
                camera_index=idx,
                frame_idx=idx,
                global_frame_idx=idx * n_cameras + idx,
                c2w=c2ws[idx],
                timestamp=idx * idx2timestamp,
                rgb=ImageData(rgb_path, n_channels=3),
                segmentation=ImageData(segmentation_path, n_channels=1, load_fn=load_inverted_segmentation_mask),
                forward_flow=ImageData(fw_flow_path, n_channels=2, load_fn=load_optical_flow) if idx < last_camera_idx else None,
                backward_flow=ImageData(bw_flow_path, n_channels=2, load_fn=load_optical_flow) if idx > 0 else None,
                misc=ImageData(disparity_path, n_channels=1, load_fn=load_disparity),
            ))

        # load testing views
        test_images_path = self.dataset_path / 'gt_2'
        if test_images_path.exists():
            test_c2w = c2ws[0]
            for idx, image_filename in Logger.log_progress(enumerate(list_sorted_files(test_images_path)), desc=f'testing views', leave=False):
                dataset['test'].append(View(
                    camera=camera,
                    camera_index=0,
                    frame_idx=idx,
                    global_frame_idx=idx,
                    c2w=test_c2w,
                    rgb=ImageData(test_images_path / image_filename, n_channels=3),
                    timestamp=idx * idx2timestamp,
                ))
        else:
            Logger.log_warning(f'No test images found in {test_images_path}.')

        # estimate near and far plane
        self._camera_settings.near_plane, self._camera_settings.far_plane = estimate_near_far(dataset['train'], self.point_cloud, min_near_plane=1e-4)

        # return the dataset
        return [camera], dataset
