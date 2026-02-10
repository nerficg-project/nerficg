"""
Datasets/TanksAndTemples.py: Provides a dataset class for scenes from the Tanks & Temples dataset calibrated with `scripts/colmap.py`.
Data available at https://www.tanksandtemples.org/download/ (last accessed 2024-02-01).
"""

import numpy as np
import pycolmap

import Framework
from Logging import Logger
from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import RadialTangentialDistortion
from Datasets.Base import BaseDataset
from Datasets.utils import read_image_size, compute_scaled_image_size, View, ImageData, transform_poses_pca, BasicPointCloud

@Framework.Configurable.configure(
    PATH='dataset/tanks_and_temples/training_data/truck',
    IMAGE_SCALE_FACTOR=0.5,
    LOAD_UNDISTORTED=True,
    TEST_STEP=8,
    APPLY_PCA=True,
    APPLY_PCA_RESCALE=True,
    NEAR_PLANE=0.01,  # for when APPLY_PCA_RESCALE is True
    FAR_PLANE=100.0,  # for when APPLY_PCA_RESCALE is True
)
class CustomDataset(BaseDataset):
    """Dataset class for Tanks & Temples scenes."""

    def load(self) -> tuple[list[PerspectiveCamera], dict[str, list[View]]]:
        """Loads the dataset into a dict containing lists of views for training and testing."""
        # original distorted data is stored in directories with '_distorted' suffix
        directory_suffix = '' if self.LOAD_UNDISTORTED else '_distorted'

        # load colmap data
        reconstruction = pycolmap.Reconstruction(self.dataset_path / 'sparse' / ('0' + directory_suffix))
        Logger.log_debug(reconstruction.summary())

        # this is a specialized loader, so let's make sure all assumptions are met
        if len(reconstruction.cameras) != 1:
            raise Framework.DatasetError(
                'The TanksAndTemples loader only supports COLMAP calibrations with a single camera.'
                'Please use the Colmap loader instead.'
            )
        else:
            colmap_camera = reconstruction.camera(1)  # COLMAP camera IDs start at 1
        if self.LOAD_UNDISTORTED:
            if colmap_camera.model != pycolmap.CameraModelId.PINHOLE:
                raise Framework.DatasetError(
                    f'The TanksAndTemples loader only supports the PINHOLE camera model from COLMAP when loading undistorted images but found {colmap_camera.model} instead.'
                    f'Please use the Colmap loader instead.'
                )
            if colmap_camera.params[2] != colmap_camera.width / 2 or colmap_camera.params[3] != colmap_camera.height / 2:
                raise Framework.DatasetError(
                    f'The TanksAndTemples loader only supports centered principal points when loading undistorted images.'
                    f'Please use the Colmap loader instead.'
                )
        else:
            if colmap_camera.model != pycolmap.CameraModelId.OPENCV:
                raise Framework.DatasetError(
                    f'The TanksAndTemples loader only supports the OPENCV camera model from COLMAP when loading distorted images but found {colmap_camera.model} instead.'
                    f'Please use the Colmap loader instead.'
                )

        # load image file names
        images = [image for image in reconstruction.images.values() if image.camera.camera_id == colmap_camera.camera_id]
        images = sorted(images, key=lambda image: image.name)
        image_directory_name = 'images' + directory_suffix
        image_scale_factor = self.IMAGE_SCALE_FACTOR
        # use pre-downscaled images if possible
        match self.IMAGE_SCALE_FACTOR:
            case 0.5:
                image_directory_name += '_2'
                image_scale_factor = None
            case _:
                pass

        # set up intrinsics
        width, height = colmap_camera.width, colmap_camera.height
        focal_x, focal_y, center_x, center_y = colmap_camera.params[:4]
        if self.IMAGE_SCALE_FACTOR is not None:
            if image_scale_factor is None:  # using pre-downscaled images
                width, height = read_image_size(self.dataset_path / image_directory_name / images[0].name)
            else:
                width, height = compute_scaled_image_size((colmap_camera.width, colmap_camera.height), image_scale_factor)
            scale_factor_intrinsics_x = width / colmap_camera.width
            scale_factor_intrinsics_y = height / colmap_camera.height
            focal_x *= scale_factor_intrinsics_x
            focal_y *= scale_factor_intrinsics_y
            center_x *= scale_factor_intrinsics_x
            center_y *= scale_factor_intrinsics_y

        # set up distortion
        if self.LOAD_UNDISTORTED:
            distortion = None
        else:
            k1, k2, p1, p2 = colmap_camera.params[4:8]
            distortion = RadialTangentialDistortion(k1=k1, k2=k2, p1=p1, p2=p2)

        # setup camera
        camera = PerspectiveCamera(
            shared_settings=self._camera_settings, width=width, height=height,
            focal_x=focal_x, focal_y=focal_y, center_x=center_x, center_y=center_y, distortion=distortion,
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
                rgb=ImageData(
                    self.dataset_path / image_directory_name / image.name, n_channels=3, scale_factor=image_scale_factor
                ),
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
