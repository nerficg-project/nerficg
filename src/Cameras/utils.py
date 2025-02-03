# -- coding: utf-8 --

"""Cameras/utils.py: Contains utility functions used for the implementation of the available camera models."""

from typing import ClassVar
from abc import ABC, abstractmethod
import torch
from torch import Tensor
from dataclasses import dataclass, field, fields, replace

import Framework


@dataclass
class DistortionParameters(ABC):
    """Base Class for storing and applying camera distortion parameters."""

    @abstractmethod
    def distort(self, positions_camera_space: Tensor) -> Tensor:
        """Abstract method applying distortion to the given image coordinates.
        Args:
            positions_camera_space (Tensor): normalized 2D positions (x,y) in camera space (ndc), shape (N, 2).

        Returns:
            Tensor: Distorted normalized 2D positions (x,y) in camera space (ndc), shape (N, 2).
        """
        pass

    @abstractmethod
    def undistort(self, positions_camera_space: Tensor) -> Tensor:
        """Abstract method removing distortion from the given image coordinates.

        Args:
            positions_camera_space (Tensor): Distorted 2D positions (x,y) in normalized camera space (ndc), shape (N, 2).

        Returns:
            Tensor: Undistorted 2D positions (x,y) in normalized camera space (ndc), shape (N, 2).
        """
        pass


class IdentityDistortion(DistortionParameters):
    """apply indentity as distortion"""

    def distort(self, positions_camera_space: Tensor) -> Tensor:
        return positions_camera_space

    def undistort(self, positions_camera_space: Tensor) -> Tensor:
        return positions_camera_space


@dataclass
class RadialTangentialDistortion(DistortionParameters):
    """
    A Class for storing and applying radial and tangential distortion parameters.
    Adapted from:
    https://github.com/google-research/multinerf/blob/main/internal/camera_utils.py
    and
    From https://github.com/google/nerfies/blob/main/nerfies/camera.py
    """
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    # k4: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    eps: float = 1e-9
    num_iter: int = 10

    def _compute_residual_and_jacobian(
        self,
        x: Tensor,
        y: Tensor,
        xd: Tensor,
        yd: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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

    def undistort(self, positions_camera_space: Tensor):
        x_dir, y_dir = positions_camera_space.split(1, dim=-1)
        x = x_dir.clone()
        y = y_dir.clone()
        for _ in range(self.num_iter):
            fx, fy, fx_x, fx_y, fy_x, fy_y = self._compute_residual_and_jacobian(x=x, y=y, xd=x_dir, yd=y_dir)
            denominator = fy_x * fx_y - fx_x * fy_y
            x_numerator = fx * fy_y - fy * fx_y
            y_numerator = fy * fx_x - fx * fy_x
            step_x = torch.where(
                torch.abs(denominator) > self.eps, x_numerator / denominator,
                torch.zeros_like(denominator))
            step_y = torch.where(
                torch.abs(denominator) > self.eps, y_numerator / denominator,
                torch.zeros_like(denominator))
            x = x + step_x
            y = y + step_y
        return torch.cat([x, y], axis=-1)

    def distort(self, positions_camera_space: Tensor):
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


# class FisheyeLens(DistortionParameters):

#     def distort(self, positions_camera_space: Tensor) -> Tensor:
#         theta = torch.sqrt(torch.sum(torch.square(positions_camera_space), axis=-1))
#         theta.clamp_max_(torch.pi)
#         sin_theta_over_theta = torch.sin(theta) / theta
#         z = -torch.cos(theta).clamp_min_(1e-6)
#         return torch.cat([positions_camera_space[..., 0] * sin_theta_over_theta / z, -positions_camera_space[..., 1] * sin_theta_over_theta / z], axis=-1)

#     def undistort(self, positions_camera_space: Tensor) -> Tensor:
#         raise NotImplementedError
#         # theta = torch.acos(-directions_camera_space[..., 2])
#         # return torch.cat([directions_camera_space[..., 0] / torch.sin(theta), -directions_camera_space[..., 1] / torch.sin(theta)], axis=-1)


def transformPoints(points: Tensor, transform: Tensor) -> Tensor:
    """
    Transforms the given points by the given homogeneous transformation matrix.

    Args:
        points:     Torch tensor of shape (N, 3) containing the points to transform.
        transform:  Torch tensor of shape (4, 4) containing the transformation matrix.

    Returns:
        Torch tensor of shape (N, 3) containing the transformed points.
    """
    return torch.matmul(transform[:3, :3], points[..., None])[..., 0] + transform[:3, 3]


def normalizeRays(ray: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalizes a vector."""
    norm = torch.linalg.norm(ray, dim=-1, keepdim=True)
    return ray / norm, norm


def createCameraMatrix(view_dir: torch.Tensor, up_dir: torch.Tensor, position: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Creates a view matrix."""
    forward, forward_norm = normalizeRays(view_dir)
    if forward_norm < eps:
        raise Framework.CameraError("The view direction vector is too small.")
    right, right_norm = normalizeRays(torch.cross(forward, up_dir))
    # right, right_norm = normalizeRays(torch.cross(up_dir, forward))
    if right_norm < eps:
        raise Framework.CameraError("The up-vector is colinear to the view direction.")
    # up, _ = normalizeRays(torch.cross(forward, right))
    up, _ = normalizeRays(torch.cross(right, forward))
    return torch.cat([torch.stack([right, -up, -forward, position], dim=1),
                      torch.tensor([[0, 0, 0, 1]], dtype=forward.dtype, device=forward.device)], dim=0)


def createLookAtMatrix(position: torch.Tensor, lookat: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Returns a c2w matrix looking at a point from a given position and up vector."""
    return createCameraMatrix(lookat - position, up, position)


@dataclass
class CameraProperties:
    """
    A Class for all kinds of image sensor data.

    Attributes:
        width                 Integer representing the image's width.
        height                Integer representing the image's height.
        rgb                   Torch tensor of shape (3, H, W) containing the RGB image.
        alpha                 Torch tensor of shape (1, H, W) containing the foreground mask.
        depth                 Torch tensor of shape (1, H, W) containing the depth image.
        c2w                   Torch tensor of shape (4, 4) containing the camera to world matrix.
        principal_offset_x    Float stating the principal point's offset from the image center in pixels (positive value -> right)
        principal_offset_y    Float stating the principal point's offset from the image center in pixels (positive value -> down)
        focal_x               Float representing the image sensor's focal length (in pixels) in x direction.
        focal_y               Float representing the image sensor's focal length (in pixels) in y direction.
        timestamp             Float stating normalized chronological timestamp at which the sample was recorded.
        camera_index          The camera index during capturing (for multi-view recordings).
        exposure_value        Float representing the cameras exposure value.
        distortion_parameters DistortionParameters object containing the camera's distortion parameters.
        forward_flow          Torch tensor of shape (2, H, W) containing the forward optical flow.
        backward_flow         Torch tensor of shape (2, H, W) containing the backward optical flow.
    """
    width: int | None = None
    height: int | None = None
    rgb: Tensor | None = None
    alpha: Tensor | None = None
    depth: Tensor | None = None
    segmentation: Tensor | None = None
    principal_offset_x: float = 0.0
    principal_offset_y: float = 0.0
    focal_x: float | None = None
    focal_y: float | None = None
    timestamp: float = 0.0
    camera_index: int = 0
    _default_exposure_value: float = 0.0
    exposure_value: float = field(default_factory=lambda: CameraProperties._default_exposure_value)
    distortion_parameters: DistortionParameters | None = None
    forward_flow: Tensor | None = None
    backward_flow: Tensor | None = None
    c2w: Tensor = torch.eye(4, device=torch.device('cpu'))  # camera to world matrix (inverse of view matrix)
    _precomputed_rays: Tensor | None = field(init=False, repr=False, default=None)  # precomputed rays for faster ray sampling
    _is_device_copy: bool = False
    _misc: Tensor | None = None

    def __post_init__(self):
        if self.distortion_parameters is None:
            self.distortion_parameters = IdentityDistortion()
        if not self._is_device_copy:
            # convert all tensors to cpu
            for member in fields(self):
                val = getattr(self, member.name)
                if isinstance(val, Tensor):
                    setattr(self, member.name, val.cpu())

    @classmethod
    def set_default_exposure(cls, value: float) -> None:
        cls._default_exposure_value = value

    @property
    def w2c(self) -> Tensor:
        return torch.linalg.inv(self.c2w)

    @w2c.setter
    def w2c(self, value: Tensor) -> None:
        self.c2w = torch.linalg.inv(value)

    @property
    def R(self) -> Tensor:
        return self.c2w[:3, :3]

    @R.setter
    def R(self, value: Tensor) -> None:
        if value.shape != (3, 3):
            raise Framework.CameraError('R must be a 3x3 matrix')
        self.c2w[:3, :3] = value

    @property
    def T(self) -> Tensor:
        return self.c2w[:3, 3]

    @T.setter
    def T(self, value: Tensor) -> None:
        if value.shape != (3,):
            raise Framework.CameraError('T must be a 3D vector')
        self.c2w[:3, 3] = value

    def toDefaultDevice(self) -> 'CameraProperties':
        """returns a shallow copy where all torch Tensors are cast to the default device"""
        return replace(self, _is_device_copy=True, **{field.name: getattr(self, field.name).to(Framework.config.GLOBAL.DEFAULT_DEVICE, copy=True)
                       for field in fields(self) if isinstance(getattr(self, field.name), Tensor) and not field.name.startswith('_')})

    def toSimple(self):
        return replace(self, **{field.name: (None if field.name != 'c2w' else getattr(self, field.name).cpu().clone()) for field in fields(self) if isinstance(getattr(self, field.name), Tensor)})


@dataclass(frozen=True)
class RayPropertySlice:
    """Stores slice objects used to access specific elements within ray tensors."""
    origin: ClassVar[slice] = slice(0, 3)
    origin_xy: ClassVar[slice] = slice(0, 2)
    origin_xz: ClassVar[slice] = slice(0, 3, 2)
    origin_yz: ClassVar[slice] = slice(1, 3)
    direction: ClassVar[slice] = slice(3, 6)
    direction_xy: ClassVar[slice] = slice(3, 5)
    direction_xz: ClassVar[slice] = slice(3, 6, 2)
    direction_yz: ClassVar[slice] = slice(4, 6)
    view_direction: ClassVar[slice] = slice(6, 9)
    rgb: ClassVar[slice] = slice(9, 12)
    alpha: ClassVar[slice] = slice(12, 13)
    rgba: ClassVar[slice] = slice(9, 13)
    depth: ClassVar[slice] = slice(13, 14)
    timestamp: ClassVar[slice] = slice(14, 15)
    xy_coordinates: ClassVar[slice] = slice(15, 17)
    x_coordinate: ClassVar[slice] = slice(15, 16)
    y_coordinate: ClassVar[slice] = slice(16, 17)
    all_annotations: ClassVar[slice] = slice(9, 17)  # rgb, alpha, depth, timestamp, xy_coordinates
    all_annotations_ndc: ClassVar[slice] = slice(6, 17)  # also contains view direction


def quaternion_to_rotation_matrix(quaternions: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Converts quaternions to 3x3 rotation matrices."""
    # add batch dimension if not present
    batch_dim_added = quaternions.dim() == 1
    if batch_dim_added:
        quaternions = quaternions[None]
    # optionally also take care of normalization
    if normalize:
        quaternions = torch.nn.functional.normalize(quaternions)
    # allocate memory for the matrices
    R = torch.empty((quaternions.shape[0], 3, 3), dtype=quaternions.dtype, device=quaternions.device)
    # conversion
    r, i, j, k = torch.unbind(quaternions, dim=1)
    R[:, 0, 0] = 1.0 - 2.0 * (j * j + k * k)
    R[:, 0, 1] = 2.0 * (i * j - r * k)
    R[:, 0, 2] = 2.0 * (i * k + r * j)
    R[:, 1, 0] = 2.0 * (i * j + r * k)
    R[:, 1, 1] = 1.0 - 2.0 * (i * i + k * k)
    R[:, 1, 2] = 2.0 * (j * k - r * i)
    R[:, 2, 0] = 2.0 * (i * k - r * j)
    R[:, 2, 1] = 2.0 * (j * k + r * i)
    R[:, 2, 2] = 1.0 - 2.0 * (i * i + j * j)
    return R[0] if batch_dim_added else R


def invert_camera_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Inverts a camera matrix with shape (4, 4) consisting of a rotation and translation."""
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverted_rotation = rotation.T
    inverted_translation = inverted_rotation @ -translation
    inverted_matrix = torch.eye(4, device=matrix.device, dtype=matrix.dtype)
    inverted_matrix[:3, :3] = inverted_rotation
    inverted_matrix[:3, 3] = inverted_translation
    return inverted_matrix


def invert_affine_3d(matrix: torch.Tensor) -> torch.Tensor:
    """Inverts a 3d affine transformation matrix with shape (4, 4)."""
    matrix3 = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverted_matrix3 = torch.linalg.inv(matrix3)
    inverted_translation = inverted_matrix3 @ -translation
    inverted_matrix = torch.eye(4, device=matrix.device, dtype=matrix.dtype)
    inverted_matrix[:3, :3] = inverted_matrix3
    inverted_matrix[:3, 3] = inverted_translation
    return inverted_matrix
