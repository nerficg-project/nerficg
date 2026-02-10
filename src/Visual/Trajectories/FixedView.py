"""
Visual/Trajectories/StabilizedView.py: A trajectory for dynamic scenes, replaying time for a single, fixed view.
Adapted from DyCheck (iPhone dataset) by Gao et al. 2022 (https://github.com/KAIR-BAIR/dycheck/blob/main).
"""

from Datasets.utils import View
from Visual.Trajectories.utils import CameraTrajectory


class fixed_view(CameraTrajectory):
    """A trajectory for dynamic scenes, replaying time for a single, fixed view."""

    def __init__(self, reference_view_rel_id: float = 0.5) -> None:
        super().__init__()
        self.reference_view_rel_id = reference_view_rel_id

    def _generate(self, _, reference_views: list[View]) -> list[View]:
        """Generates the camera trajectory using a list of reference views."""
        target_view_idx = int(min(1.0, max(0.0, self.reference_view_rel_id)) * (len(reference_views) - 1))
        target_view = reference_views[target_view_idx].to_simple()
        views = []
        for view in reference_views:
            views.append(View(
                camera=target_view.camera,
                camera_index=target_view.camera_index,
                frame_idx=target_view.frame_idx,
                global_frame_idx=target_view.global_frame_idx,
                c2w=target_view.c2w_numpy,
                timestamp=view.timestamp,
                exif=target_view.exif,
            ))
        return views
