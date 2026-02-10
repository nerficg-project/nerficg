"""
Datasets/TanksAndTemples_3DGS.py: A dataset class for the Truck and Train scenes as provided by the 3DGS authors.
This loader only exists because these specific calibrations are not provided in standard Colmap format.
However, supporting them is important as they have become a standard benchmark due to the popularity of 3DGS.
The issue is that the provided images do not match the camera data in the Colmap files, i.e., images have been
downscaled to half their original resolution, but the cameras.bin still state the intrinsics for the original
resolution. While the official 3DGS code handles this, most principled Colmap loaders do not.
Data available at https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip (last accessed 2026-01-22).
"""

import numpy as np
import pycolmap

import Framework
from Logging import Logger
from Cameras.Perspective import PerspectiveCamera
from Datasets.Base import BaseDataset
from Datasets.utils import read_image_size, View, ImageData, transform_poses_pca, BasicPointCloud


@Framework.Configurable.configure(
    PATH='dataset/gs_data/truck',
    TEST_STEP=8,
    APPLY_PCA=True,
    APPLY_PCA_RESCALE=True,
    NEAR_PLANE=0.01,  # mipnerf360 uses 0.2 before APPLY_PCA_RESCALE
    FAR_PLANE=100.0,  # mipnerf360 uses 1e6 before APPLY_PCA_RESCALE
)
class CustomDataset(BaseDataset):
    """Dataset class for the truck and train scenes from the Tanks & Temples datasets as provided by the 3DGS authors."""

    def load(self) -> tuple[list[PerspectiveCamera], dict[str, list[View]]]:
        """Loads the dataset into a dict containing lists of views for training and testing."""
        if self.IMAGE_SCALE_FACTOR is not None:
            raise Framework.DatasetError('The TanksAndTemples_3DGS loader does not support image resizing.')

        # load colmap data
        reconstruction = pycolmap.Reconstruction(self.dataset_path / 'sparse' / '0')
        Logger.log_debug(reconstruction.summary())

        # this is a specialized loader, so let's make sure all assumptions are met
        if len(reconstruction.cameras) != 1:
            raise Framework.DatasetError(
                'The TanksAndTemples_3DGS loader only supports COLMAP calibrations with a single camera.'
                'Please use the Colmap loader instead.'
            )
        else:
            colmap_camera = reconstruction.camera(1)  # COLMAP camera IDs start at 1
        if colmap_camera.model != pycolmap.CameraModelId.PINHOLE:
            raise Framework.DatasetError(
                f'The TanksAndTemples_3DGS loader only supports the PINHOLE camera model from COLMAP but found {colmap_camera.model} instead.'
                f'Please use the Colmap loader instead.'
            )
        if colmap_camera.params[2] != colmap_camera.width / 2 or colmap_camera.params[3] != colmap_camera.height / 2:
            raise Framework.DatasetError(
                f'The TanksAndTemples_3DGS loader only supports centered principal points.'
                f'Please use the Colmap loader instead.'
            )

        # load image file names
        images = [image for image in reconstruction.images.values() if image.camera.camera_id == colmap_camera.camera_id]
        images = sorted(images, key=lambda image: image.name)

        # set up intrinsics
        focal_x, focal_y, center_x, center_y = colmap_camera.params
        width, height = read_image_size(self.dataset_path / 'images' / images[0].name)
        scale_factor_intrinsics_x = width / colmap_camera.width
        scale_factor_intrinsics_y = height / colmap_camera.height
        focal_x *= scale_factor_intrinsics_x
        focal_y *= scale_factor_intrinsics_y
        center_x *= scale_factor_intrinsics_x
        center_y *= scale_factor_intrinsics_y

        # setup camera
        camera = PerspectiveCamera(
            shared_settings=self._camera_settings, width=width, height=height,
            focal_x=focal_x, focal_y=focal_y, center_x=center_x, center_y=center_y,
        )

        # load dataset items
        data: list[View] = []
        for idx, image in Logger.log_progress(enumerate(images), desc='loading views', leave=False, total=len(images)):
            data.append(View(
                camera=camera,
                camera_index=0,
                frame_idx=idx,
                global_frame_idx=idx,
                c2w=image.cam_from_world().inverse().matrix(),
                rgb=ImageData(self.dataset_path / 'images' / image.name, n_channels=3),
            ))

        # load point cloud
        self.point_cloud = BasicPointCloud.from_colmap(reconstruction)

        # rotate/scale poses to align ground with xz plane and optionally fit to [-1, 1]^3 cube
        if self.APPLY_PCA:
            c2ws = np.stack([view.c2w_numpy for view in data])
            c2ws, transformation = transform_poses_pca(c2ws, rescale=self.APPLY_PCA_RESCALE)
            for view, c2w in zip(data, c2ws):
                view.c2w = c2w
            self.point_cloud.transform(transformation)

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
        return [camera], dataset
