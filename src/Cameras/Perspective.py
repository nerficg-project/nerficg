# -- coding: utf-8 --

"""Cameras/Perspective.py: Implementation of a perspective RGB camera model."""

from torch import Tensor
import torch

from Cameras.Base import BaseCamera


class PerspectiveCamera(BaseCamera):
    """Defines a perspective RGB camera model for ray generation."""

    def __init__(self, near_plane: float, far_plane: float) -> None:
        super(PerspectiveCamera, self).__init__(near_plane, far_plane)

    def getRayOrigins(self) -> Tensor:
        """Returns a tensor containing the origin of each ray."""
        return self.properties.c2w[:3, -1].expand((self.properties.width * self.properties.height, 3))

    def _getLocalRayDirections(self) -> Tensor:
        """Returns a tensor containing the direction of each ray."""
        # calculate initial directions
        x_direction, y_direction = self.getPixelCoordinates()
        x_direction: Tensor = ((x_direction + 0.5) - (0.5 * self.properties.width + self.properties.principal_offset_x)) / self.properties.focal_x
        y_direction: Tensor = ((y_direction + 0.5) - (0.5 * self.properties.height + self.properties.principal_offset_y)) / self.properties.focal_y
        z_direction: Tensor = torch.full((self.properties.height, self.properties.width), fill_value=-1)
        directions_camera_space: Tensor = torch.stack(
            (x_direction, y_direction, z_direction), dim=-1
        ).reshape(-1, 3)
        return directions_camera_space

    def projectPoints(self, points: Tensor, eps: float = 1.0e-8) -> tuple[Tensor, Tensor, Tensor]:
        """projects points (Nx3) to image plane. returns xy image plane pixel coordinates,
        mask of points that hit sensor, and depth values."""
        # points = torch.cat((points, torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)), dim=1)
        # points = points @ self.properties.w2c.T
        points = self.pointsToCameraSpace(points)
        depths = -points[:, 2]
        valid_mask = ((depths > self.near_plane) & (depths < self.far_plane))
        focal = torch.tensor((self.properties.focal_x, self.properties.focal_y), device=points.device, dtype=points.dtype)
        screen_size = torch.tensor((self.properties.width, self.properties.height), device=points.device, dtype=points.dtype)
        offset = screen_size * 0.5 + torch.tensor((self.properties.principal_offset_x, self.properties.principal_offset_y), device=points.device, dtype=points.dtype)
        points = self.properties.distortion_parameters.distort(points[:, :2] / (depths[:, None] + eps)) * focal + offset
        valid_mask &= ((points >= 0) & (points < screen_size - 1)).all(dim=-1)
        return points, valid_mask, depths

    def getProjectionMatrix(self, invert_z: bool = False) -> Tensor:
        """
            Returns the projection matrix for this camera.
            After perspective division, x, y, and z will be in [-1, 1] (OpenGL convention).
            The z-axis can be inverted for camera coordinate systems where the camera looks along the negative z-axis.

            Args:
                invert_z: If True, the z-axis will be inverted.

            Returns:
                The projection matrix from camera space to clip space.
        """
        half_width = self.properties.width * 0.5
        half_height = self.properties.height * 0.5
        z_sign = -1.0 if invert_z else 1.0
        projection_matrix = torch.tensor([
            [self.properties.focal_x / half_width, 0.0, z_sign * self.properties.principal_offset_x / half_width, 0.0],
            [0.0, self.properties.focal_y / half_height, z_sign * self.properties.principal_offset_y / half_height, 0.0],
            [0.0, 0.0, z_sign * (self.far_plane + self.near_plane) / (self.far_plane - self.near_plane), -2.0 * self.far_plane * self.near_plane / (self.far_plane - self.near_plane)],
            [0.0, 0.0, z_sign, 0.0]
        ], dtype=torch.float32, device=self.properties.c2w.device)
        return projection_matrix

    def getViewportTransform(self, pixel_centers_at_integer_coordinates: bool = True) -> Tensor:
        """
            Returns the transformation matrix from NDC to screen space.
            if pixel_centers_at_integer_coordinates:
                [-1, 1]^3 -> [-0.5, width - 0.5] x [-0.5, height - 0.5] x [near_plane, far_plane]
            else:
                [-1, 1]^3 -> [0, width] x [0, height] x [near_plane, far_plane]

            Args:
                pixel_centers_at_integer_coordinates: If True, pixel centers will be at integer coordinates.

            Returns:
                The transformation matrix from NDC to screen space.
        """
        offset = 0.5 if pixel_centers_at_integer_coordinates else 0.0
        center_x = self.properties.width * 0.5
        center_y = self.properties.height * 0.5
        viewport_transform = torch.tensor([
            [center_x, 0.0, 0.0, center_x - offset],
            [0.0, center_y, 0.0, center_y - offset],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=torch.float32, device=self.properties.c2w.device)
        # the following lines add a nonlinear mapping of z from [-1, 1] to [near_plane, far_plane]
        # viewport_transform[2, 2] = (self.far_plane - self.near_plane) * 0.5
        # viewport_transform[2, 3] = (self.far_plane + self.near_plane) * 0.5
        return viewport_transform
