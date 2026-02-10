"""Visual/Trajectories/FancyZoom.py: A trajectory following a set of reference views interrupted by zooming and lemniscate trajectories."""

from copy import deepcopy

import numpy as np

from Cameras.Base import BaseCamera
from Cameras.Perspective import PerspectiveCamera
from Datasets.utils import View
from Visual.Trajectories.utils import get_lemniscate_trajectory, CameraTrajectory


class fancy_zoom(CameraTrajectory):
    """A trajectory following a set of reference views interrupted by zooming and lemniscate trajectories."""

    def __init__(
        self,
        n_breaks: int = 2,
        zoom_n_views: int = 90,
        zoom_factor: float = 0.2,
        lemniscate_n_views_per_rotation: int = 60,
        lemniscate_degree: int = 3,
    ) -> None:
        super().__init__()
        self.n_breaks = n_breaks
        self.zoom_n_views = zoom_n_views
        self.zoom_factor = zoom_factor
        self.lemniscate_n_views_per_rotation = lemniscate_n_views_per_rotation
        self.lemniscate_degree = lemniscate_degree

    def _generate(self, reference_camera: BaseCamera, reference_views: list[View]) -> list[View]:
        """A trajectory following a set of reference views interrupted by zooming and lemniscate trajectories."""
        break_indices = np.rint(np.linspace(0, len(reference_views) - 1, self.n_breaks + 2)).astype(np.intp)[1:-1]
        views = []
        for view in reference_views[:break_indices[0]]:
            views.append(view.to_simple())
        for break_idx in range(self.n_breaks):
            break_view_idx = break_indices[break_idx]
            break_view = reference_views[break_view_idx]
            # zoom in and out
            for zoom_frame_idx in range(self.zoom_n_views):
                focal_scale = 1 + self.zoom_factor * np.sin(zoom_frame_idx / (self.zoom_n_views - 1) * 2 * np.pi)
                camera = PerspectiveCamera(
                    shared_settings=break_view.camera.shared_settings,
                    width=break_view.camera.width, height=break_view.camera.height,
                    focal_x=break_view.camera.focal_x * focal_scale, focal_y=break_view.camera.focal_y * focal_scale,
                    center_x=break_view.camera.center_x, center_y=break_view.camera.center_y,
                    distortion=deepcopy(break_view.camera.distortion),
                )
                break_view_zoomed = break_view.to_simple()
                break_view_zoomed.camera = camera
                views.append(break_view_zoomed)
            # lemniscate trajectory
            lookat = break_view.position_numpy + (break_view.camera.near_plane + break_view.camera.far_plane) / 2 * break_view.forward_numpy
            up = break_view.up_numpy
            lemniscate_trajectory_poses = get_lemniscate_trajectory(
                break_view, lookat, up, self.lemniscate_n_views_per_rotation, self.lemniscate_degree
            )
            for pose in lemniscate_trajectory_poses:
                view = break_view.to_simple()
                view.c2w = pose
                views.append(view)
            # if this is not the last break, add views until the next break
            if break_idx < self.n_breaks - 1:
                start = break_view_idx + 1
                stop = break_indices[break_idx + 1]
                for view in reference_views[start:stop]:
                    views.append(view.to_simple())
        for view in reference_views[break_indices[-1]:]:
            views.append(view.to_simple())
        return views
