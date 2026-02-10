"""Cameras/utils.py: Contains utility functions used for the implementation of the available camera models."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

import Framework


@dataclass
class BaseDistortion(ABC):
    """Base class for storing and accessing distortion coefficients."""
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    undistortion_eps: float = 1e-9
    undistortion_iterations: int = 10

    @abstractmethod
    def distort(self, positions_camera_space: torch.Tensor) -> torch.Tensor:
        """Abstract method applying distortion to the given image coordinates.
        Args:
            positions_camera_space (torch.Tensor): normalized 2D positions (x,y) in camera space (ndc), shape (N, 2).

        Returns:
            torch.Tensor: Distorted normalized 2D positions (x,y) in camera space (ndc), shape (N, 2).
        """
        pass

    @abstractmethod
    def undistort(self, positions_camera_space: torch.Tensor) -> torch.Tensor:
        """Abstract method removing distortion from the given image coordinates.

        Args:
            positions_camera_space (torch.Tensor): Distorted 2D positions (x,y) in normalized camera space (ndc), shape (N, 2).

        Returns:
            torch.Tensor: Undistorted 2D positions (x,y) in normalized camera space (ndc), shape (N, 2).
        """
        pass


@dataclass
class RadialTangentialDistortion(BaseDistortion):
    """
    A Class for storing and applying radial and tangential distortion parameters.
    Adapted from:
    https://github.com/google-research/multinerf/blob/main/internal/camera_utils.py
    and
    From https://github.com/google/nerfies/blob/main/nerfies/camera.py
    """

    def _compute_residual_and_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        xd: torch.Tensor,
        yd: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # FIXME: probably only partially correct
        r = x * x + y * y
        # d = 1.0 + r * (self.k1 + r * (self.k2 + r * (self.k3 + r * self.k4)))
        d = 1.0 + r * (self.k1 + r * (self.k2 + self.k3 * r))
        fx = d * x + 2 * self.p1 * x * y + self.p2 * (r + 2 * x * x) - xd
        fy = d * y + 2 * self.p2 * x * y + self.p1 * (r + 2 * y * y) - yd

        # d_r = (self.k1 + r * (2.0 * self.k2 + r * (3.0 * self.k3 + r * 4.0 * self.k4)))
        d_r = (self.k1 + r * (2.0 * self.k2 + 3.0 * self.k3 * r))
        d_x = 2.0 * x * d_r
        d_y = 2.0 * y * d_r

        fx_x = d + d_x * x + 2.0 * self.p1 * y + 6.0 * self.p2 * x
        fx_y = d_y * x + 2.0 * self.p1 * x + 2.0 * self.p2 * y

        fy_x = d_x * y + 2.0 * self.p2 * y + 2.0 * self.p1 * x
        fy_y = d + d_y * y + 2.0 * self.p2 * x + 6.0 * self.p1 * y

        return fx, fy, fx_x, fx_y, fy_x, fy_y

    def undistort(self, positions_camera_space: torch.Tensor):
        x_dir, y_dir = positions_camera_space.split(1, dim=-1)
        x = x_dir.clone()
        y = y_dir.clone()
        for _ in range(self.undistortion_iterations):
            fx, fy, fx_x, fx_y, fy_x, fy_y = self._compute_residual_and_jacobian(x=x, y=y, xd=x_dir, yd=y_dir)
            denominator = fy_x * fx_y - fx_x * fy_y
            x_numerator = fx * fy_y - fy * fx_y
            y_numerator = fy * fx_x - fx * fy_x
            step_x = torch.where(
                torch.abs(denominator) > self.undistortion_eps, x_numerator / denominator,
                torch.zeros_like(denominator))
            step_y = torch.where(
                torch.abs(denominator) > self.undistortion_eps, y_numerator / denominator,
                torch.zeros_like(denominator))
            x = x + step_x
            y = y + step_y
        return torch.cat([x, y], dim=-1)

    def distort(self, positions_camera_space: torch.Tensor):
        # adapted from ADOP: https://github.com/darglein/saiga/blob/master/shader/vision/distortion.glsl
        x_dir, y_dir = positions_camera_space[..., 0], positions_camera_space[..., 1]
        x2 = torch.square(x_dir)
        y2 = torch.square(y_dir)
        r2 = x2 + y2
        # get valid points
        mask = r2 < 2.0
        x_dir = x_dir[mask]
        y_dir = y_dir[mask]
        x2 = x2[mask]
        y2 = y2[mask]
        r2 = r2[mask]
        # distort
        _2xy = 2.0 * x_dir * y_dir
        radial = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3))
        tangential_x = self.p1 * _2xy + self.p2 * (r2 + 2.0 * x2)
        tangential_y = self.p1 * (r2 + 2.0 * y2) + self.p2 * _2xy
        positions_undistorted = positions_camera_space.clone()
        positions_undistorted[mask] = positions_undistorted[mask] * radial[:, None] + torch.stack((tangential_x, tangential_y), dim=-1)
        return positions_undistorted


# class FisheyeLens(BaseDistortion):

#     def distort(self, positions_camera_space: torch.Tensor) -> torch.Tensor:
#         theta = torch.sqrt(torch.sum(torch.square(positions_camera_space), axis=-1))
#         theta.clamp_max_(torch.pi)
#         sin_theta_over_theta = torch.sin(theta) / theta
#         z = -torch.cos(theta).clamp_min_(1e-6)
#         return torch.cat([positions_camera_space[..., 0] * sin_theta_over_theta / z, -positions_camera_space[..., 1] * sin_theta_over_theta / z], axis=-1)

#     def undistort(self, positions_camera_space: torch.Tensor) -> torch.Tensor:
#         raise NotImplementedError
#         # theta = torch.acos(-directions_camera_space[..., 2])
#         # return torch.cat([directions_camera_space[..., 0] / torch.sin(theta), -directions_camera_space[..., 1] / torch.sin(theta)], axis=-1)


def look_at(eye: np.ndarray, lookat: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Creates a camera-to-world matrix looking from eye to position with the given up vector."""
    forward = lookat - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)

    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w


@dataclass
class SharedCameraSettings:
    """A container class for virtual camera settings."""
    background_color: torch.Tensor
    near_plane: float
    far_plane: float

    def __post_init__(self):
        if self.background_color.shape != (3,):
            raise Framework.CameraError(
                f'background_color must be a torch tensor of shape (3,), but got {self.background_color.shape}'
            )
        if self.near_plane <= 0 or self.far_plane <= self.near_plane:
            raise Framework.CameraError(
                f'invalid near and far plane values (near_plane={self.near_plane}, far_plane={self.far_plane}). Must be: 0 < near_plane < far_plane'
            )


def quaternion_to_rotation_matrix(quaternions: np.ndarray | torch.Tensor, normalize: bool = True) -> np.ndarray | torch.Tensor:
    """Converts quaternions to 3x3 rotation matrices."""
    # add batch dimension if not present
    if batch_dim_added := quaternions.ndim == 1:
        quaternions = quaternions[None]
    # optionally take care of normalization and allocate memory for the matrices
    if isinstance(quaternions, torch.Tensor):
        if normalize:
            quaternions = torch.nn.functional.normalize(quaternions)  # uses 1e-12 as epsilon to prevent zero divisions
        rotation_matrix = torch.empty((quaternions.shape[0], 3, 3), dtype=quaternions.dtype, device=quaternions.device)
    else:
        if normalize:
            quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
        rotation_matrix = np.empty((quaternions.shape[0], 3, 3), dtype=quaternions.dtype)
    # conversion
    r, i, j, k = quaternions.T
    ii2, jj2, kk2 = i * i * 2, j * j * 2, k * k * 2
    ij2, ik2, jk2 = i * j * 2, i * k * 2, j * k * 2
    ri2, rj2, rk2 = r * i * 2, r * j * 2, r * k * 2
    rotation_matrix[:, 0, 0] = 1 - (jj2 + kk2)
    rotation_matrix[:, 0, 1] = ij2 - rk2
    rotation_matrix[:, 0, 2] = ik2 + rj2
    rotation_matrix[:, 1, 0] = ij2 + rk2
    rotation_matrix[:, 1, 1] = 1 - (ii2 + kk2)
    rotation_matrix[:, 1, 2] = jk2 - ri2
    rotation_matrix[:, 2, 0] = ik2 - rj2
    rotation_matrix[:, 2, 1] = jk2 + ri2
    rotation_matrix[:, 2, 2] = 1 - (ii2 + jj2)
    return rotation_matrix[0] if batch_dim_added else rotation_matrix


def invert_3d_affine(transform: np.ndarray | torch.Tensor, is_rigid: bool = True) -> np.ndarray | torch.Tensor:
    """Inverts an 3D affine transformation matrix (shape 4x4). Assumes a rigid transformation by default."""
    if isinstance(transform, torch.Tensor):
        inverted_upper_3x3 = transform[:3, :3].T if is_rigid else torch.linalg.inv(transform[:3, :3])
        inverted_matrix = torch.eye(4, device=transform.device, dtype=transform.dtype)
    else:
        inverted_upper_3x3 = transform[:3, :3].T if is_rigid else np.linalg.inv(transform[:3, :3])
        inverted_matrix = np.eye(4, dtype=transform.dtype)
    inverted_translation = inverted_upper_3x3 @ -transform[:3, 3]
    inverted_matrix[:3, :3] = inverted_upper_3x3
    inverted_matrix[:3, 3] = inverted_translation
    return inverted_matrix


def focal_to_fov(focal: float, degrees: bool = False) -> float:
    """Converts (normalized) focal length to field of view."""
    fov_radians = 2 * math.atan(0.5 / focal)
    return math.degrees(fov_radians) if degrees else fov_radians


def fov_to_focal(fov: float, degrees: bool = False) -> float:
    """Converts field of view to (normalized) focal length."""
    fov_radians = math.radians(fov) if degrees else fov
    return 0.5 / math.tan(0.5 * fov_radians)


def directions_to_equirectangular_grid_coords(directions: torch.Tensor) -> torch.Tensor:
    """Converts unit directions to grid coordinates in [-1, 1]^2 using equirectangular projection."""
    x, y, z = directions.unbind(dim=-1)
    azimuth = torch.atan2(x, z)
    elevation = torch.asin(y.clamp(-1.0, 1.0))
    return torch.stack([azimuth / math.pi, elevation / (0.5 * math.pi)], dim=-1)

def equirectangular_grid_coords_to_directions(grid_coords: torch.Tensor) -> torch.Tensor:
    """Converts grid coordinates in [-1, 1]^2 to unit directions using equirectangular projection."""
    u, v = grid_coords.unbind(dim=-1)
    azimuth = u * math.pi
    elevation = v * (0.5 * math.pi)
    sin_azimuth = torch.sin(azimuth)
    cos_azimuth = torch.cos(azimuth)
    sin_elevation = torch.sin(elevation)
    cos_elevation = torch.cos(elevation)
    return torch.stack([cos_elevation * sin_azimuth, sin_elevation, cos_elevation * cos_azimuth], dim=-1)
