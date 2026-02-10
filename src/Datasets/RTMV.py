"""
Datasets/RTMV.py: Provides a dataset class for RTMV scenes.
Data available at https://huggingface.co/datasets/TontonTremblay/RTMV (last accessed 2025-09-17).
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import cv2

import Framework
from Cameras.Perspective import PerspectiveCamera
from Datasets.Base import BaseDataset
from Datasets.utils import View, ImageData, list_sorted_files, linear_to_srgb, compute_scaled_image_size
from Logging import Logger


def load_rtmv_rgba_exr(path: Path) -> torch.Tensor:
    """Loads an RTMV .exr image using OpenCV."""
    # TODO: is there a better way for loading or at least setting the environment variable?
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'  # enable OpenEXR support in OpenCV
    try:
        bgra_linear = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    except Exception:
        raise Framework.DatasetError(f'Failed to load image file: "{path}"')
    rgba = cv2.cvtColor(bgra_linear, cv2.COLOR_BGRA2RGBA)
    # FIXME: the .exr file contains linear rgb that can have values > 1.0, which should be handled properly
    rgba[..., :3] = linear_to_srgb(rgba[..., :3])
    rgba = rgba.clip(0.0, 1.0)
    return torch.as_tensor(rgba, dtype=torch.float32, device='cpu').permute(2, 0, 1)


@Framework.Configurable.configure(
    PATH='dataset/rtmv/bricks/Bonsai_Tree',
    NEAR_PLANE=0.01,
    FAR_PLANE=10.0,
)
class CustomDataset(BaseDataset):
    """Dataset class for RTMV scenes."""

    def load(self) -> tuple[list[PerspectiveCamera], dict[str, list[View]]]:
        """Loads the dataset into a dict containing lists of views for training, evaluation, and testing."""
        camera = None

        # list all view info files
        view_info_filenames = list_sorted_files(self.dataset_path, '.json')
        n_views = len(view_info_filenames)

        # set up bounding box with center at (0, 0, 0) from first view info file
        first_view_info_path = self.dataset_path / view_info_filenames[0]
        try:
            with open(first_view_info_path, 'r') as f:
                view_info = json.load(f)
        except IOError:
            raise Framework.DatasetError(f'Invalid dataset view info file path "{first_view_info_path}"')
        camera_data = view_info['camera_data']
        center = np.array(camera_data['scene_center_3d_box'])
        min_bounds = np.array(camera_data['scene_min_3d_box']) - center
        max_bounds = np.array(camera_data['scene_max_3d_box']) - center
        self.bounding_box = torch.tensor([min_bounds, max_bounds], dtype=torch.float32, device='cpu')

        # Define coordinate system transformations
        cam_transform = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenGL to Colmap
        world_transform = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])  # Blender to Colmap
        data: dict[str, list[View]] = {subset: [] for subset in self.subsets}
        for frame_idx, view_info_filename in Logger.log_progress(enumerate(view_info_filenames), desc='views', leave=False, total=n_views):
            # load view info
            view_info_path = self.dataset_path / view_info_filename
            try:
                with open(view_info_path, 'r') as f:
                    view_info = json.load(f)
            except IOError:
                raise Framework.DatasetError(f'Invalid dataset view info file path "{view_info_path}"')

            # set up or validate camera intrinsics
            camera_data = view_info['camera_data']
            intrinsics = camera_data['intrinsics']
            width, height = compute_scaled_image_size((camera_data['width'], camera_data['height']), self.IMAGE_SCALE_FACTOR)
            scale_factor_intrinsics_x = width / camera_data['width']
            scale_factor_intrinsics_y = height / camera_data['height']
            focal_x = intrinsics['fx'] * scale_factor_intrinsics_x
            focal_y = intrinsics['fy'] * scale_factor_intrinsics_y
            center_x = intrinsics['cx'] * scale_factor_intrinsics_x
            center_y = intrinsics['cy'] * scale_factor_intrinsics_y
            if camera is None:
                camera = PerspectiveCamera(
                    shared_settings=self._camera_settings, width=width, height=height,
                    focal_x=focal_x, focal_y=focal_y, center_x=center_x, center_y=center_y,
                )
            elif camera.width != width or camera.height != height or camera.focal_x != focal_x or camera.focal_y != focal_y or camera.center_x != center_x or camera.center_y != center_y:
                raise Framework.DatasetError('the RTMV loader requires shared intrinsics across all views')

            # load c2w
            c2w = np.array(camera_data['cam2world']).T
            c2w[:3, 3] -= center
            c2w = world_transform @ c2w @ cam_transform.T
            # setup image data
            rgba_path = view_info_path.with_suffix('.exr')
            rgb = ImageData(rgba_path, n_channels=3, scale_factor=self.IMAGE_SCALE_FACTOR, load_fn=load_rtmv_rgba_exr)
            alpha = ImageData(rgba_path, n_channels=1, channel_offset=3, scale_factor=self.IMAGE_SCALE_FACTOR, load_fn=load_rtmv_rgba_exr)
            # TODO: add loading of depth and segmentation mask
            data['train'].append(View(
                camera=camera,
                camera_index=0,
                frame_idx=frame_idx,
                global_frame_idx=frame_idx,
                c2w=c2w,
                rgb=rgb,
                alpha=alpha,
            ))

        return [camera], data
