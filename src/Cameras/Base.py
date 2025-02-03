# -- coding: utf-8 --

"""Cameras/Base.py: Implementation of the basic camera model used for ray generation and scene rendering options."""

from abc import ABC, abstractmethod
import torch
from torch import Tensor

import Framework
from Cameras.utils import CameraProperties, createLookAtMatrix, transformPoints


class BaseCamera(ABC):
    """Defines the basic camera template for ray generation."""

    def __init__(self, near_plane: float, far_plane: float) -> None:
        super().__init__()
        self.near_plane: float = near_plane
        self.far_plane: float = far_plane
        self.background_color: Tensor = torch.tensor([1.0, 1.0, 1.0])
        self.properties: CameraProperties = CameraProperties()

    def setBackgroundColor(self, r: float = 1.0, g: float = 1.0, b: float = 1.0):
        self.background_color = torch.tensor([r, g, b])

    def setProperties(self, properties: CameraProperties) -> 'BaseCamera':
        """Sets the given sensor data sample for the camera."""
        self.properties = properties
        return self

    def generateRays(self) -> Tensor:
        """Generates tensor containing all rays and their properties according to camera model."""
        # check if precomputed rays exist
        if self.properties._precomputed_rays is not None:
            return self.properties._precomputed_rays
        # get directions
        directions: Tensor = self.getGlobalRayDirections()
        view_directions: Tensor = directions / torch.norm(directions, p=2, dim=-1, keepdim=True)
        # get origins
        origins: Tensor = self.getRayOrigins()
        # get annotations
        annotations: Tensor = self.getRayAnnotations()
        # build rays tensor
        rays: Tensor = torch.cat([origins, directions, view_directions, annotations], dim=-1).to(Framework.config.GLOBAL.DEFAULT_DEVICE)
        return rays

    def getRayAnnotations(self) -> Tensor:
        """Returns a tensor containing the annotations (e.g. color, alpha, and depth) of each ray."""
        # add color if ground truth rgb image is available
        if self.properties.rgb is not None:
            img_flat: Tensor = self.properties.rgb.permute(1, 2, 0).reshape(-1, 3)
        else:
            # add empty color if no ground truth view is available
            img_flat: Tensor = torch.zeros((self.properties.height * self.properties.width, 3))
        # add alpha if ground truth alpha mask is available
        if self.properties.alpha is not None:
            alpha_flat: Tensor = self.properties.alpha.permute(1, 2, 0).reshape(-1, 1)
        else:
            # assume every pixel is valid
            alpha_flat: Tensor = torch.ones((self.properties.height * self.properties.width, 1))
        # add depth if ground truth depth mask is available
        if self.properties.depth is not None:
            depth_flat: Tensor = self.properties.depth.permute(1, 2, 0).reshape(-1, 1)
        else:
            # set depth to -1 for every pixel
            depth_flat: Tensor = torch.full((self.properties.height * self.properties.width, 1), fill_value=-1)
        # add timestamp
        if self.properties.timestamp is not None:
            timestamps_flat = torch.full(
                (self.properties.height * self.properties.width, 1), fill_value=float(self.properties.timestamp)
            )
        else:
            # set to 0
            timestamps_flat = torch.zeros((self.properties.height * self.properties.width, 1))
        # add xy image coordinates
        x_coord, y_coord = self.getPixelCoordinates()
        x_coord = x_coord[:, None].reshape(-1, 1)
        y_coord = y_coord[:, None].reshape(-1, 1)
        # combine and return
        return torch.cat([img_flat, alpha_flat, depth_flat, timestamps_flat, x_coord, y_coord], dim=-1)

    @abstractmethod
    def getRayOrigins(self) -> Tensor:
        """Returns a tensor containing the origin of each ray."""
        return Tensor()

    @abstractmethod
    def _getLocalRayDirections(self) -> Tensor:
        """Returns a tensor containing the direction of each ray in camera coordinate sytem."""
        pass

    def getLocalRayDirections(self) -> Tensor:
        """Returns a tensor containing the direction of each ray in camera coordinate sytem."""
        # calculate initial directions
        local_directions = self._getLocalRayDirections()
        # undistort x and y directions
        undistorted_xy = self.properties.distortion_parameters.undistort(local_directions[:, :2])
        local_directions[:, :2] = undistorted_xy
        # return
        return local_directions

    def pointsToWorldSpace(self, points: Tensor) -> Tensor:
        """Transforms the given points from camera to world space."""
        return transformPoints(points, self.properties.c2w)

    def pointsToCameraSpace(self, points: Tensor) -> Tensor:
        """Transforms the given points from world to camera space."""
        return transformPoints(points, self.properties.w2c)

    def lookAt(self, lookat: torch.Tensor, up: torch.Tensor) -> 'BaseCamera':
        self.properties.c2w = createLookAtMatrix(self.properties.T, lookat, up)
        return self

    def teleport(self, position: torch.Tensor) -> 'BaseCamera':
        self.properties.T = position
        return self

    def getGlobalRayDirections(self) -> Tensor:
        """Returns a tensor containing the direction of each ray in world coordinate system."""
        # get rays in local system
        local_directions = self.getLocalRayDirections()
        # transform directions into world
        directions_world_space: Tensor = torch.matmul(
            self.properties.R,
            local_directions[:, :, None]
        ).squeeze()
        return directions_world_space

    def getPositionAndViewdir(self) -> Tensor | None:
        """Returns the current camera position and direction in world coordinates."""
        if self.properties is None:
            return None
        data: Tensor = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -1.0, 0.0]], device=self.properties.c2w.device)
        return torch.matmul(self.properties.c2w, data[:, :, None]).squeeze()

    def getUpVector(self) -> Tensor:
        """Returns the current up vector in world coordinates."""
        if self.properties is None:
            return None
        data: Tensor = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=self.properties.c2w.device)
        return torch.matmul(self.properties.c2w, data[:, :, None]).squeeze()[:3]

    def getRightVector(self) -> Tensor:
        """Returns the current right vector in world coordinates."""
        if self.properties is None:
            return None
        data: Tensor = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.properties.c2w.device)
        return torch.matmul(self.properties.c2w, data[:, :, None]).squeeze()[:3]

    def getPixelCoordinates(self) -> tuple[Tensor, Tensor]:
        y_direction, x_direction = torch.meshgrid(
            torch.linspace(0, self.properties.height - 1, self.properties.height),
            torch.linspace(0, self.properties.width - 1, self.properties.width),
            indexing="ij"
        )
        return x_direction, y_direction

    @abstractmethod
    def projectPoints(self, points: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """projects points (Nx3) to image plane. returns xy image plane pixel coordinates,
        mask of points that hit sensor, and depth values."""
        pass

    @abstractmethod
    def getProjectionMatrix(self, invert_z: bool = False) -> Tensor:
        """Returns the projection matrix of the camera."""
        pass
