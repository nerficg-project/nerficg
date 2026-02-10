"""Visual/Trajectories/Stabilized.py: Generates a stabilized path from a set of reference poses by using a sliding window."""

import numpy as np

import Framework
from Datasets.utils import View, get_average_pose
from Visual.Trajectories.utils import CameraTrajectory


class stabilized_path(CameraTrajectory):
    """Generates a stabilized path from a set of reference poses by using a sliding window."""

    def __init__(self, window: int = 5) -> None:
        super().__init__()
        if window % 2 == 0 or window < 3:
            raise Framework.VisualizationError('Window size must be an odd number >= 3.')
        self.half_window = window // 2

    def _generate(self, _, reference_views: list[View]) -> list[View]:
        """Generates the camera trajectory using a list of reference views."""
        reference_poses = np.stack([view.c2w_numpy for view in reference_views])
        n_views = len(reference_poses)
        views = []
        for view_idx, view in enumerate(reference_views):
            start = max(0, view_idx - self.half_window)
            stop = min(n_views, view_idx + self.half_window + 1)
            smoothed_pose = get_average_pose(reference_poses[start:stop])
            view = view.to_simple()
            view.c2w = smoothed_pose
            views.append(view)
        return views
