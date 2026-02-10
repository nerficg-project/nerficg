"""
Datasets/Empty.py: A dataset class that provides only a perspective camera.
"""

import numpy as np

import Framework
from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import fov_to_focal
from Datasets.Base import BaseDataset
from Datasets.utils import View

@Framework.Configurable.configure(
    PATH='',
    DEFAULT_WIDTH=1920,
    DEFAULT_HEIGHT=1080,
    VERTICAL_FOV=60.0,
)
class CustomDataset(BaseDataset):
    """Dataset class for methods that want to use the GUI without a dataset."""

    def load(self) -> tuple[list[PerspectiveCamera], dict[str, list[View]]]:
        """Create a single camera properties object inside the test subset."""
        # compute focal length from vertical field of view
        width, height = self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT
        focal = height * fov_to_focal(self.VERTICAL_FOV, degrees=True)
        # setup camera
        camera = PerspectiveCamera(
            shared_settings=self._camera_settings, width=width, height=height, focal_x=focal, focal_y=focal,
        )
        # setup dataset
        dataset: dict[str, list[View]] = {subset: [] for subset in self.subsets}
        dataset['train'] = [View(camera, 0, 0, 0, np.eye(4))]
        return [camera], dataset
