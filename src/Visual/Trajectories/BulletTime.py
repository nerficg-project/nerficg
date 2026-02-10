"""
Visual/Trajectories/BulletTime.py: A trajectory for dynamic scenes following a lemniscate trajectory while replaying time at a reference view.
Adapted from DyCheck (iPhone dataset) by Gao et al. 2022 (https://github.com/KAIR-BAIR/dycheck/blob/main).
"""

import numpy as np

from Datasets.utils import View
from Visual.Trajectories.utils import get_lemniscate_trajectory, CameraTrajectory


class bullet_time(CameraTrajectory):
    """A trajectory for dynamic scenes following a lemniscate trajectory while replaying time at a reference view."""

    def __init__(
        self, reference_view_rel_id: float = 0.5,
        custom_lookat: np.ndarray | None = None,
        custom_up: np.ndarray | None = None,
        n_views_per_rotation: int = 90,
        degree: float = 10,
        n_repeats: int = 2,
    ) -> None:
        super().__init__()
        self.reference_view_rel_id = reference_view_rel_id
        self.custom_lookat = custom_lookat
        self.custom_up = custom_up
        self.n_views_per_rotation = n_views_per_rotation
        self.degree = degree
        self.n_repeats = n_repeats

    def _generate(self, _, reference_views: list[View]) -> list[View]:
        """A trajectory for dynamic scenes following a lemniscate trajectory while replaying time at a reference view."""
        target_view_idx = int(min(1.0, max(0.0, self.reference_view_rel_id)) * (len(reference_views) - 1))
        target_view = reference_views[target_view_idx].to_simple()
        lookat = self.custom_lookat
        if lookat is None:
            lookat = target_view.position_numpy + (target_view.camera.near_plane + target_view.camera.far_plane) / 2 * target_view.forward_numpy
        up = target_view.up_numpy if self.custom_up is None else self.custom_up
        lemniscate_trajectory_poses = get_lemniscate_trajectory(
            target_view, lookat, up, self.n_views_per_rotation, self.degree
        )
        n_views = self.n_views_per_rotation * self.n_repeats
        views = []
        for view_idx in range(n_views):
            views.append(View(
                camera=target_view.camera,
                camera_index=target_view.camera_index,
                frame_idx=target_view.frame_idx,
                global_frame_idx=target_view.global_frame_idx,
                c2w=lemniscate_trajectory_poses[view_idx % self.n_views_per_rotation].copy(),
                timestamp=view_idx / (n_views - 1),
                exif=target_view.exif,
            ))
        return views
