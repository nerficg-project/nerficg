# -- coding: utf-8 --

"""
Datasets/LLFF.py: Provides a dataset class for Local Light Field Fusion (LLFF) scenes.
Data available at https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1 (last accessed 2023-05-25).
"""

import numpy as np
import torch

import Framework
from Cameras.NDC import NDCCamera
from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import CameraProperties
from Datasets.Base import BaseDataset
from Datasets.utils import list_sorted_files, recenterPoses, loadImagesParallel, \
                            CameraCoordinateSystemsTransformations


@Framework.Configurable.configure(
    PATH='dataset/nerf_llff_data/fern',
    IMAGE_SCALE_FACTOR=0.25,
    DISABLE_NDC=False,
    TEST_STEP=8,
    WORLD_SCALING=0.75
)
class CustomDataset(BaseDataset):
    """Dataset class for Local Light Field Fusion (LLFF) scenes."""

    def __init__(self, path: str) -> None:
        super().__init__(
            path,
            NDCCamera(),
            CameraCoordinateSystemsTransformations.RIGHT_HAND
        )

    def load(self) -> dict[str, list[CameraProperties] | None]:
        """Loads the dataset into a dict containing lists of CameraProperties for training, evaluation, and testing."""
        # TODO K-planes uses this bounding box for llff scenes!
        # torch.tensor([[-3.0, -1.67, -1.2], [3.0, 1.67, 1.2]])

        # load images
        images_path = self.dataset_path / 'images'
        image_filenames = [str(images_path / file) for file in list_sorted_files(images_path)]
        rgbs, alphas = loadImagesParallel(image_filenames, self.IMAGE_SCALE_FACTOR, num_threads=4, desc='images')
        # load intrinsics and extrinsics
        colmap_poses = torch.as_tensor(np.load(str(self.dataset_path / 'poses_bounds.npy')))
        view_matrices = colmap_poses[:, :-2].reshape([-1, 3, 5])
        intrinsics = view_matrices[:, :, 4]
        focals = intrinsics[:, 2:3]
        if self.IMAGE_SCALE_FACTOR is not None:
            focals = focals * self.IMAGE_SCALE_FACTOR
        view_matrices = view_matrices[:, :, :-1]
        view_matrices = torch.cat(
            [view_matrices[:, :, 1:2], -view_matrices[:, :, 0:1], view_matrices[:, :, 2:]], dim=2
        )
        view_matrices = torch.cat(
            (view_matrices, torch.broadcast_to(torch.tensor([0, 0, 0, 1]), (view_matrices.shape[0], 1, 4))), dim=1
        )
        depth_min_max = colmap_poses[:, -2:]
        # rescale coordinates
        if self.WORLD_SCALING is not None:
            scaling = 1.0 / (depth_min_max.min() * self.WORLD_SCALING)
            view_matrices[:, :3, 3] *= scaling
            depth_min_max *= scaling
        # disable normalized device coordinates, if enabled
        if self.DISABLE_NDC:
            view_matrices = view_matrices.cpu()
            self.camera = PerspectiveCamera(
                near_plane=depth_min_max.min().item() * 0.9,
                far_plane=depth_min_max.max().item()
            )
            self.camera.setBackgroundColor(*self.BACKGROUND_COLOR)
        else:
            # recenter coordinates cameras (only makes sense for forward facing scenes)
            view_matrices = recenterPoses(view_matrices).cpu()
        # insert data into target data structure
        data: list[CameraProperties] = [
            CameraProperties(
                width=rgb.shape[2],
                height=rgb.shape[1],
                rgb=rgb,
                alpha=alpha,
                c2w=c2w,
                focal_x=focal.item(),
                focal_y=focal.item()
            )
            for rgb, alpha, c2w, focal in zip(rgbs, alphas, view_matrices, focals)
        ]
        # perform test split
        train_data: list[CameraProperties] = []
        test_data: list[CameraProperties] = []
        if self.TEST_STEP > 0:
            for i in range(len(data)):
                if i % self.TEST_STEP == 0:
                    test_data.append(data[i])
                else:
                    train_data.append(data[i])
        else:
            train_data: list[CameraProperties] = data
        # return the dataset
        return {
            'train': train_data,
            'test': test_data,
            'val': []
        }
