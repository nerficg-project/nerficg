# -- coding: utf-8 --

"""
Visual/Trajectories/StabilizedView.py: A trajectory for dynamic scenes, replaying time for a single, fixed view.
Adapted from DyCheck (iPhone dataset) by Gao et al. 2022 (https://github.com/KAIR-BAIR/dycheck/blob/main).
"""

from Cameras.Base import BaseCamera
from Cameras.utils import CameraProperties
from Visual.Trajectories.utils import CameraTrajectory


class stabilized_view(CameraTrajectory):
    """A trajectory for dynamic scenes, replaying time for a single, fixed view."""

    def __init__(self, reference_pose_rel_id: float = 0.5) -> None:
        super().__init__()
        self.reference_pose_rel_id = reference_pose_rel_id

    def _generate(self, reference_camera: BaseCamera, reference_poses: list[CameraProperties]) -> list[CameraProperties]:
        """Generates the camera trajectory using a list of reference poses."""
        data: list[CameraProperties] = []
        reference_pose = reference_poses[int(min(1.0, max(0.0, self.reference_pose_rel_id)) * len(reference_poses))]
        for camera_properties in reference_poses:
            data.append(CameraProperties(
                width=reference_pose.width,
                height=reference_pose.height,
                rgb=None,
                alpha=None,
                c2w=reference_pose.c2w.clone(),
                focal_x=reference_pose.focal_x,
                focal_y=reference_pose.focal_y,
                principal_offset_x=0.0,
                principal_offset_y=0.0,
                distortion_parameters=reference_pose.distortion_parameters,
                timestamp=camera_properties.timestamp
            ))
        return data
