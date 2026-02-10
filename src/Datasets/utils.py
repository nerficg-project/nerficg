"""Datasets/utils.py: Contains utility functions used for the implementation of the available dataset classes."""

import os
from copy import deepcopy
from pathlib import Path
from typing import Iterator, Callable, Any
import natsort
from dataclasses import dataclass, fields, field
import math

import numpy as np
import torch
from torch.multiprocessing import Pool
from torchvision import io
from torchvision.utils import _normalized_flow_to_image
from PIL import Image
from plyfile import PlyData

from Cameras.Base import BaseCamera
import Framework
from Cameras.utils import invert_3d_affine, look_at
from Logging import Logger


def list_sorted_files(path: Path, pattern: str = None) -> list[str]:
    """Returns a naturally sorted list of files in the given directory."""
    file_list = [i.name for i in path.iterdir() if i.is_file()]
    if pattern is not None:
        file_list = [f for f in file_list if pattern in f]
    return natsort.natsorted(file_list)


def list_sorted_directories(path: Path) -> list[str]:
    """Returns a naturally sorted list of subdirectories in the given directory."""
    return natsort.natsorted([i.name for i in path.iterdir() if i.is_dir()])


def srgb_to_linear(image: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert sRGB to linear RGB."""
    where_fn = torch.where if isinstance(image, torch.Tensor) else np.where
    return where_fn(image > 0.04045, ((image + 0.055) / 1.055) ** 2.4, image / 12.92)


def linear_to_srgb(image: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert linear RGB to sRGB."""
    where_fn = torch.where if isinstance(image, torch.Tensor) else np.where
    return where_fn(image > 0.0031308, image ** (1.0 / 2.4) * 1.055 - 0.055, 12.92 * image)


def load_image(filename: str, scale_factor: float | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Loads an image from the specified file."""
    try:
        image: torch.Tensor = io.read_image(path=str(filename), mode=io.ImageReadMode.UNCHANGED)
    except Exception:
        raise Framework.DatasetError(f'Failed to load image file: "{filename}"')
    # convert image to the format used by the framework
    image = image.float() / (65535 if image.dtype == torch.uint16 else 255)
    # apply scaling factor to image
    if scale_factor is not None:
        image = apply_image_scale_factor(image, scale_factor)
    # extract alpha channel if available
    rgb, alpha = image.split([3, 1]) if image.shape[0] == 4 else (image, None)
    return rgb, alpha


def load_image_simple(path: Path) -> torch.Tensor:
    """Loads an image from the specified file."""
    try:
        image: torch.Tensor = io.decode_image(input=str(path), mode=io.ImageReadMode.UNCHANGED)
    except Exception:
        raise Framework.DatasetError(f'Failed to load image file: "{path}"')
    # convert to [0, 1]
    image = image.float() / (65535 if image.dtype == torch.uint16 else 255)
    return image


def load_inverted_segmentation_mask(path: Path) -> torch.Tensor:
    """Loads a segmentation mask from the specified file and inverts it."""
    return 1.0 - load_image_simple(path)


def load_optical_flow(path: Path, check_validity: bool = False) -> torch.Tensor | None:
    """Reads optical flow from Middlebury .flo file."""
    flow = None
    try:
        with open(path, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if magic != 202021.25:
                raise Framework.DatasetError(f'invalid .flo file: {path} (magic number incorrect)')
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            flow = np.fromfile(f, np.float32, count=2 * w * h)
            flow = np.resize(flow, (h, w, 2))
            flow = torch.from_numpy(flow)
            flow = flow.permute(2, 0, 1)
    except FileNotFoundError:
        if check_validity:
            raise Framework.DatasetError(f'invalid flow file: {path}')
    return flow


def load_disparity(path: Path) -> torch.Tensor:
    """Loads disparities output by, e.g., DepthAnything"""
    return torch.from_numpy(np.load(path))


def parallel_load_fn(args: dict[str, Any]) -> Any:
    """Function executed by each thread when loading in parallel."""
    torch.set_num_threads(1)
    load_function = args['load_function']
    del args['load_function']
    return load_function(**args)


def get_parallel_load_iterator(
    filenames: list[str],
    scale_factor: float | None,
    num_threads: int,
    load_function: Callable,
) -> tuple[Iterator, Pool]:  # type: ignore
    """Returns iterator for parallel image loading."""
    # create thread pool
    if num_threads < 1:
        num_threads = os.cpu_count() - 1
    pool = Pool(min(num_threads, len(filenames)))
    # create and return the iterator
    return pool.imap(
        func=parallel_load_fn,
        iterable=[{'load_function': load_function, 'filename': filenames[i], 'scale_factor': scale_factor} for i in range(len(filenames))],
        chunksize=1,
    ), pool


def load_images(
    filenames: list[str],
    scale_factor: float | None,
    num_threads: int,
    desc="",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Loads multiple images in parallel."""
    iterator, pool = get_parallel_load_iterator(filenames, scale_factor, num_threads, load_function=load_image)
    rgbs, alphas = [], []
    for rgb, alpha in Logger.log_progress(iterator, desc=desc, leave=False, total=len(filenames)):
        # clone tensors to extract them from shared memory (/dev/shm), otherwise we can not use all RAM
        rgbs.append(rgb.clone())
        alphas.append(alpha.clone() if alpha is not None else None)
    pool.close()
    pool.join()
    return rgbs, alphas


def read_image_size(path):
    """Reads the size of an image file without loading it into memory."""
    try:
        with Image.open(path) as img:
            return img.width, img.height
    except Exception:
        raise Framework.DatasetError(f'Failed to read size of image at "{path}"')


def compute_scaled_image_size(size: tuple[int, int], scale_factor: float | None) -> tuple[int, int]:
    """Computes the scaled image size based on the original size and scale factor."""
    if scale_factor is None or scale_factor == 1.0:
        return size
    # Python rounds to nearest even (1.5 -> 2; 2.5 -> 2). GPU code prefers even numbers, so this should be beneficial.
    return round(size[0] * scale_factor), round(size[1] * scale_factor)


def apply_image_scale_factor(image: torch.Tensor | None, scale_factor: float, mode: str = 'area') -> torch.Tensor:
    """Scales the image by the specified factor."""
    return torch.nn.functional.interpolate(
        input=image[None],
        size=compute_scaled_image_size(image.shape[1:], scale_factor),
        mode=mode,
    )[0].clamp(0.0, 1.0)


def apply_image_scale_factor_optical_flow(flow: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Scales the optical flow by the specified factor."""
    flow = apply_image_scale_factor(flow, scale_factor, mode='nearest')
    flow *= scale_factor
    return flow


def apply_background_color(raw_rgb: torch.Tensor, alpha: torch.Tensor, background_color: torch.Tensor, is_chw: bool = True) -> torch.Tensor:
    """Alpha composites the raw rgb values with the given background color using the provided alpha values."""
    if is_chw:
        background_color = background_color[:, None, None]
    return torch.lerp(background_color.to(raw_rgb.device), raw_rgb, alpha).clamp(0, 1)


def get_average_pose(poses: np.ndarray) -> np.ndarray:
    """Computes the "average" pose of the inputs."""
    mean_translation = poses[:, :3, 3].mean(axis=0)
    forward_sum = poses[:, :3, 2].sum(axis=0)
    down_sum = poses[:, :3, 1].sum(axis=0)
    return look_at(mean_translation, mean_translation + forward_sum, -down_sum)


def recenter_poses(poses: np.ndarray) -> np.ndarray:
    """Recenters the scene coordinate system."""
    # create inverse average transformation
    center_transform = np.linalg.inv(get_average_pose(poses))
    return center_transform @ poses


def save_image(filepath: Path, image: torch.Tensor) -> None:
    """Writes the input image tensor to the file given by filepath."""
    image = image.clamp(0.0, 1.0).mul(255.0).add(0.5).byte().cpu()
    filename, filetype = os.path.splitext(filepath)
    match filetype.lower():
        case '.png' | '':
            io.write_png(
                input=image,
                filename=f'{filename}.png',
                compression_level=6,  # opencv uses 3
            )
        case '.jpg' | '.jpeg':
            io.write_jpeg(
                input=image,
                filename=f'{filename}.jpg',
                quality=75,  # opencv uses 95
            )
        case _:
            raise Framework.DatasetError(f'Invalid file type specified "{filetype}"')


def loadOpticalFlowFile(filename: str, scale_factor: float | None, check_validity: bool = False) -> torch.Tensor | None:
    """Reads optical flow from Middlebury .flo file."""
    try:
        with open(filename, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if magic != 202021.25:
                raise Framework.DatasetError(f'invalid .flo file: {filename} (magic number incorrect)')
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            flow = np.fromfile(f, np.float32, count=2 * w * h)
            flow = np.resize(flow, (h, w, 2))
            flow = torch.from_numpy(flow)
            flow = flow.permute(2, 0, 1)
            # apply scaling factor to image
            if scale_factor is not None:
                flow = apply_image_scale_factor(flow, scale_factor, mode='nearest')
                flow *= scale_factor
    except FileNotFoundError:
        flow = None
        if check_validity:
            raise Framework.DatasetError(f'invalid flow file: {filename}')
    return flow


def loadOpticalFlowParallel(path: Path, frame_names: list[str], num_threads: int,
                            image_scale_factor: float = None) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Loads multiple forward/backward flow maps in parallel."""
    flow_filenames = [str(path / f'{frame_name.split(".")[0]}_{kind}.flo') for frame_name in frame_names for kind in ['forward', 'backward']]
    iterator_flow, pool = get_parallel_load_iterator(flow_filenames, image_scale_factor, num_threads, load_function=loadOpticalFlowFile)
    forward_flows, backward_flows = [], []
    for _ in Logger.log_progress(range(len(frame_names)), desc='flow', leave=False):
        forward_flow, backward_flow = next(iterator_flow), next(iterator_flow)
        # clone tensors to extract them from shared memory (/dev/shm), otherwise we can not use all RAM
        forward_flows.append(forward_flow.clone() if forward_flow is not None else None)
        backward_flows.append(backward_flow.clone() if backward_flow is not None else None)
    pool.close()
    pool.join()
    return forward_flows, backward_flows


def save_optical_flow(filename: Path, optical_flow: torch.Tensor) -> None:
    """Writes optical flow to Middlebury .flo file."""
    if optical_flow is None or optical_flow.shape[0] != 2:
        raise Framework.DatasetError('Invalid optical flow tensor given')
    if '.flo' not in str(filename).lower():
        raise Framework.DatasetError('expected .flo extension in optical flow filename')
    optical_flow = optical_flow.permute(1, 2, 0).cpu().numpy()
    with open(filename, 'wb') as f:
        np.array([202021.25], dtype=np.float32).tofile(f)
        np.array([optical_flow.shape[1], optical_flow.shape[0]], dtype=np.int32).tofile(f)
        optical_flow.astype(np.float32).tofile(f)


@torch.no_grad()
def flow_to_image(optical_flow: torch.Tensor, max_norm: float = None) -> torch.Tensor:
    # adapted from https://pytorch.org/vision/stable/_modules/torchvision/utils.html#flow_to_image
    orig_shape = optical_flow.shape
    if optical_flow.ndim == 3:
        optical_flow = optical_flow[None]  # Add batch dim
    if optical_flow.ndim != 4 or optical_flow.shape[1] != 2:
        raise Framework.DatasetError(f"Input flow should have shape (2, H, W) or (N, 2, H, W), got {orig_shape}.")
    norms = torch.linalg.norm(optical_flow, ord=None, dim=1, keepdim=True)
    if max_norm is None or max_norm <= 0:
        max_norm = norms.max().item()
    norms = norms.clamp_min_(min=max_norm + torch.finfo(optical_flow.dtype).eps)
    normalized_flow = optical_flow / norms
    img = _normalized_flow_to_image(normalized_flow)
    if len(orig_shape) == 3:
        img = img[0]  # Remove batch dim
    return img.float() / 255.0


@dataclass
class BasicPointCloud:
    positions: torch.Tensor  # (N, 3) tensor containing the point positions in world space
    colors: torch.Tensor | None = None  # (N, 3) tensor containing the point colors (optional)

    def __post_init__(self):
        """Validates the input data and moves it to RAM as float32."""
        if self.positions.shape[1] != 3:
            raise Framework.DatasetError(f'positions must have shape (N, 3), got {self.positions.shape}')
        if self.colors is not None and self.colors.shape != self.positions.shape:
            raise Framework.DatasetError(f'colors must have the same shape as positions, got {self.colors.shape} vs {self.positions.shape}')
        self.to(dtype=torch.float32, device=torch.device('cpu'))

    def __repr__(self):
        return f'BasicPointCloud with {self.n_points:,}{" colored" if self.colors is not None else ""} points'

    @property
    def n_points(self) -> int:
        """Returns the number of points in the point cloud."""
        return self.positions.shape[0]

    def to(self, dtype: torch.dtype = None, device: torch.device = None) -> None:
        """Moves the point cloud data to the specified device and data type."""
        for member in fields(self):
            val = getattr(self, member.name)
            if isinstance(val, torch.Tensor):
                setattr(self, member.name, val.to(dtype=dtype, device=device))

    def transform(self, transform: torch.Tensor | np.ndarray) -> None:
        """Applies a rigid transformation to the point cloud."""
        if isinstance(transform, np.ndarray):
            transform = torch.as_tensor(transform, dtype=torch.float32, device=self.positions.device)
        self.positions = self.positions @ transform[:3, :3].T + transform[:3, 3]

    def normalize(self, center: torch.Tensor, scale: float | torch.Tensor) -> None:
        """Centers and scales the point cloud."""
        if scale <= 0.0:
            raise Framework.DatasetError(f'scale must be > 0, got {scale}')
        self.positions -= center
        self.positions *= scale

    def filter_outliers(self, filter_ratio: float) -> None:
        """Removes outlier points based on their distance to the mean position."""
        if filter_ratio <= 0.0 or filter_ratio > 1.0:
            raise Framework.DatasetError(f'filter_ratio must be in (0, 1], got {filter_ratio}')
        if filter_ratio == 1.0:
            return
        mean = self.positions.mean(dim=0, keepdim=True)
        dists = torch.linalg.norm((self.positions - mean), dim=1)
        quantile = torch.quantile(dists, filter_ratio, interpolation='midpoint')
        valid_mask = dists < quantile
        self.positions = self.positions[valid_mask]
        if self.colors is not None:
            self.colors = self.colors[valid_mask]

    def get_aabb(self, tolerance_factor: float = 0.1, filter_outliers_percentage: float | None = None) -> 'AxisAlignedBox':
        """Calculates the axis-aligned bounding box (AABB) of the point cloud."""
        positions = self.positions
        if filter_outliers_percentage is not None:
            mean = positions.mean(dim=0, keepdim=True)
            dists = torch.linalg.norm(positions - mean, dim=1)
            quantile = torch.quantile(dists, filter_outliers_percentage, interpolation='midpoint')
            positions = positions[dists < quantile]
        min_position = torch.min(positions, dim=0)[0]
        max_position = torch.max(positions, dim=0)[0]
        center = ((min_position + max_position) * 0.5)[None]
        aabb = ((torch.stack([min_position, max_position], dim=0) - center) * (1.0 + tolerance_factor)) + center
        if filter_outliers_percentage is not None:
            valid_mask = (self.positions > aabb[0]) & (self.positions < aabb[1])
            valid_mask = valid_mask.all(dim=1)
            self.positions = self.positions[valid_mask]
        return AxisAlignedBox(aabb)

    @classmethod
    def from_colmap(cls, reconstruction) -> 'BasicPointCloud':
        """Creates a BasicPointCloud from a Colmap Reconstruction object."""
        num_points = len(reconstruction.points3D)
        positions = np.empty((num_points, 3), dtype=np.float32)
        colors = np.empty((num_points, 3), dtype=np.float32)
        for i, point3D in enumerate(reconstruction.points3D.values()):
            positions[i] = point3D.xyz
            colors[i] = point3D.color / 255
        return cls(positions=torch.from_numpy(positions), colors=torch.from_numpy(colors))

    @classmethod
    def from_ply(cls, path: Path) -> 'BasicPointCloud':
        """Creates a BasicPointCloud from a PLY file."""
        plydata = PlyData.read(path)
        vertices = plydata['vertex']
        positions = np.column_stack((vertices['x'], vertices['y'], vertices['z'])).astype(np.float32)
        colors = np.column_stack((vertices['red'], vertices['green'], vertices['blue'])).astype(np.float32) / 255
        return cls(positions=torch.from_numpy(positions), colors=torch.from_numpy(colors))

    @classmethod
    def from_opensfm(cls, reconstruction: dict[str, dict]) -> 'BasicPointCloud':
        """Creates a BasicPointCloud from an OpenSFM reconstruction dict."""
        points = reconstruction['points']
        num_points = len(points)
        positions = np.empty((num_points, 3), dtype=np.float32)
        colors = np.empty((num_points, 3), dtype=np.float32)
        for i, point in enumerate(points.values()):
            positions[i] = point['coordinates']
            colors[i] = np.array(point['color']) / 255
        return cls(positions=torch.from_numpy(positions), colors=torch.from_numpy(colors))


@dataclass
class AxisAlignedBox:
    data: torch.Tensor  # (2, 3) tensor containing the min and max corners

    def __post_init__(self) -> None:
        """Validates the input data and moves it to RAM as float32."""
        if self.data.shape != (2, 3):
            raise Framework.DatasetError(f'data must have shape (2, 3), got {self.data.shape}')
        self.to(dtype=torch.float32, device=torch.device('cpu'))

    def __repr__(self):
        return f'{tensor_to_string(self.data[0])} (min), {tensor_to_string(self.data[1])} (max)'

    @property
    def center(self) -> torch.Tensor:
        """Returns the center of the bounding box."""
        return ((self.data[0] + self.data[1]) * 0.5).to(Framework.config.GLOBAL.DEFAULT_DEVICE)

    @property
    def size(self) -> torch.Tensor:
        """Returns the size of the bounding box."""
        return (self.data[1] - self.data[0]).to(Framework.config.GLOBAL.DEFAULT_DEVICE)

    @property
    def min(self) -> torch.Tensor:
        """Returns the minimum corner of the bounding box."""
        return self.data[0].to(Framework.config.GLOBAL.DEFAULT_DEVICE)

    @property
    def max(self) -> torch.Tensor:
        """Returns the maximum corner of the bounding box."""
        return self.data[1].to(Framework.config.GLOBAL.DEFAULT_DEVICE)

    @property
    def min_max(self) -> torch.Tensor:
        """Returns the min and max corners of the bounding box."""
        return self.data.to(Framework.config.GLOBAL.DEFAULT_DEVICE)

    def to(self, dtype: torch.dtype = None, device: torch.device = None) -> None:
        """Moves the bounding box data to the specified device and data type."""
        self.data = self.data.to(dtype=dtype, device=device)

    def convert(self, conversion: Callable) -> None:
        """Converts the bounding box to a different coordinate system using the provided conversion function."""
        self.data = torch.cat(conversion(*torch.split(self.data, split_size_or_sections=1, dim=1)), dim=1).sort(dim=0).values

    def normalize(self, center: torch.Tensor, scale: float | torch.Tensor) -> None:
        """Centers and scales the bounding box."""
        if scale <= 0.0:
            raise Framework.DatasetError(f'scale must be > 0, got {scale}')
        self.data -= center
        self.data *= scale


def tensor_to_string(tensor: torch.Tensor, precision: int = 2) -> str:
    """Converts a tensor to a string. Allows for specifying a custom precision."""
    return f'[{", ".join([f"{i:.{precision}f}" for i in tensor])}]'


def rescale_poses_to_unit_cube(poses: np.ndarray, transform: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rescale poses so they fit into the [-1, 1] cube."""
    scale_factor = 1.0 / np.max(np.abs(poses[:, :3, 3]))
    poses[:, :3, 3] *= scale_factor
    scaling = np.diag([scale_factor, scale_factor, scale_factor, 1.0])
    transform = scaling if transform is None else scaling @ transform
    return poses, transform


def transform_poses_pca(poses: np.ndarray, rescale: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Aligns the scene by assuming that most movement happened parallel to the ground plane during capture.
    Adapted from Zip-NeRF (https://github.com/jonbarron/camp_zipnerf)
    """
    # convert to OpenGL/NeRF camera coordinate system for direct compatibility with Zip-NeRF utils
    colmap2opengl = np.diag([1, -1, -1, 1])
    poses = poses @ colmap2opengl

    positions = poses[:, :3, 3]
    mean_position = positions.mean(axis=0)
    displacements = positions - mean_position

    # compute the compute eigenvalues and eigenvectors of the displacements' covariance
    cov = displacements.T @ displacements
    eigvals, eigvecs = np.linalg.eig(cov)

    # sort eigenvectors in order of largest to smallest eigenvalue
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, sorted_indices]

    # set up rotation from eigenvectors
    rotation = eigvecs.T

    # ensure the transform will not change handedness
    if np.linalg.det(rotation) < 0:
        rotation = np.diag([1, 1, -1]) @ rotation

    # create the transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = rotation @ -mean_position

    # apply the transformation to the poses
    poses = transform @ poses

    # flip coordinate system if z component of y-axis is negative across all poses
    if poses.mean(axis=0)[2, 1] < 0:
        flip = np.diag([1, -1, -1, 1])
        poses = flip @ poses
        transform = flip @ transform

    # optionally rescale poses so they fit into the [-1, 1] cube
    if rescale:
        poses, transform = rescale_poses_to_unit_cube(poses, transform)

    # swap y and z axis so that y points down with x and z spanning the ground plane
    aligned2colmap = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]]
    )
    poses = aligned2colmap @ poses
    transform = aligned2colmap @ transform

    # transform back to Colmap camera coordinate system
    poses = poses @ np.linalg.inv(colmap2opengl)

    return poses, transform


@dataclass(frozen=True)
class RayBatch:
    origin: torch.Tensor
    direction: torch.Tensor
    view_direction: torch.Tensor | None = None  # stored separately for NeRF-based methods
    rgb: torch.Tensor | None = None
    alpha: torch.Tensor | None = None
    depth: torch.Tensor | None = None
    timestamp: torch.Tensor | None = None
    _skip_post_init: bool = False

    def __post_init__(self):
        """Validates the input data."""
        if self._skip_post_init:
            return
        n_rays = self.origin.shape[0]
        dtype = self.origin.dtype
        device = self.origin.device
        for member in fields(self):
            value = getattr(self, member.name)
            if isinstance(value, torch.Tensor):
                if value.shape[0] != n_rays:
                    raise Framework.DatasetError(f'All annotations must have the same number of rays, but {member.name} has {value.shape[0]} while origin has {n_rays}')
                if value.dtype != dtype:
                    raise Framework.DatasetError(f'All annotations must have the same data type, but {member.name} is of type {value.dtype} while origin is of type {dtype}')
                if value.device != device:
                    raise Framework.DatasetError(f'All annotations must be on the same device, but {member.name} is on {value.device} while origin is on {device}')

    def __len__(self):
        """Returns the number of rays in the batch."""
        return self.origin.shape[0]

    @property
    def dtype(self) -> torch.dtype:
        """Returns the data type of the ray batch."""
        return self.origin.dtype

    @property
    def device(self) -> torch.device:
        """Returns the device of the ray batch."""
        return self.origin.device

    @property
    def has_annotations(self) -> bool:
        """Returns whether the ray batch contains any annotations."""
        return any(x is not None for x in (self.view_direction, self.rgb, self.alpha, self.depth, self.timestamp))

    @property
    def annotations(self) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Returns all ray annotations."""
        return self.view_direction, self.rgb, self.alpha, self.depth, self.timestamp

    @property
    def stacked_annotations(self) -> torch.Tensor | None:
        """Returns all available ray annotations concatenated into a single tensor."""
        return torch.cat([x for x in self.annotations if x is not None], dim=-1) if self.has_annotations else None

    @property
    def as_tensor(self) -> torch.Tensor:
        """Returns all data in the ray batch concatenated into a single tensor."""
        return torch.cat([self.origin, self.direction] + [x for x in self.annotations if x is not None], dim=-1)

    def __getitem__(self, idx: int | slice | torch.Tensor) -> 'RayBatch':
        """Returns a subset of the ray batch."""
        if idx is Ellipsis or idx == slice(None):
            return self
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        return RayBatch(
            origin=self.origin[idx],
            direction=self.direction[idx],
            view_direction=None if self.view_direction is None else self.view_direction[idx],
            rgb=None if self.rgb is None else self.rgb[idx],
            alpha=None if self.alpha is None else self.alpha[idx],
            depth=None if self.depth is None else self.depth[idx],
            timestamp=None if self.timestamp is None else self.timestamp[idx],
            _skip_post_init=True,
        )

    def to(self, dtype: torch.dtype = None, device: torch.device = None, non_blocking: bool = False) -> 'RayBatch':
        """Moves the ray batch to the specified device and data type."""
        if (dtype is None or dtype == self.dtype) and (device is None or device == self.device):
            return self
        return RayBatch(
            origin=self.origin.to(dtype=dtype, device=device, non_blocking=non_blocking),
            direction=self.direction.to(dtype=dtype, device=device, non_blocking=non_blocking),
            view_direction=None if self.view_direction is None else self.view_direction.to(dtype=dtype, device=device, non_blocking=non_blocking),
            rgb=None if self.rgb is None else self.rgb.to(dtype=dtype, device=device, non_blocking=non_blocking),
            alpha=None if self.alpha is None else self.alpha.to(dtype=dtype, device=device, non_blocking=non_blocking),
            depth=None if self.depth is None else self.depth.to(dtype=dtype, device=device, non_blocking=non_blocking),
            timestamp=None if self.timestamp is None else self.timestamp.to(dtype=dtype, device=device, non_blocking=non_blocking),
            _skip_post_init=True,
        )

    def cpu(self, non_blocking: bool = False) -> 'RayBatch':
        """Moves the ray batch to CPU."""
        return self.to(device=torch.device('cpu'), non_blocking=non_blocking)

    def cuda(self, non_blocking: bool = False) -> 'RayBatch':
        """Moves the ray batch to the default CUDA device."""
        return self.to(device=Framework.config.GLOBAL.DEFAULT_DEVICE, non_blocking=non_blocking)

    def split(self, chunk_size: int) -> list['RayBatch']:
        """Splits the RayBatch into smaller RayBatches of size chunk_size."""
        n_rays = len(self)
        return [self[i:i+chunk_size] for i in range(0, n_rays, chunk_size)]

    @classmethod
    def cat(cls, batches: list['RayBatch']) -> 'RayBatch':
        """Concatenates a sequence of RayBatch instances along the ray dimension."""
        if not batches:
            raise Framework.DatasetError('no RayBatch instances to concatenate')

        origin = torch.cat([b.origin for b in batches], dim=0)
        direction = torch.cat([b.direction for b in batches], dim=0)

        def strict_cat(attr_name: str):
            # Check that all batches either have it or all don't
            has_values = [getattr(b, attr_name) is not None for b in batches]
            if any(has_values) and not all(has_values):
                raise Framework.DatasetError(f'RayBatch field "{attr_name}" is not present in some batches')
            if all(has_values):
                return torch.cat([getattr(b, attr_name) for b in batches], dim=0)
            return None

        return cls(
            origin=origin,
            direction=direction,
            view_direction=strict_cat('view_direction'),
            rgb=strict_cat('rgb'),
            alpha=strict_cat('alpha'),
            depth=strict_cat('depth'),
            timestamp=strict_cat('timestamp'),
            _skip_post_init=True,
        )


@dataclass(frozen=True)
class RayCollection:
    """A collection of rays with camera-specific subsections."""
    rays: RayBatch
    camera_slices: list[slice]

    def __len__(self) -> int:
        """Returns the number of rays in the collection."""
        return len(self.rays)

    def __getitem__(self, index: int) -> RayBatch:
        """Returns the rays for the requested camera."""
        return self.rays[self.camera_slices[index]]

    @property
    def all_rays(self) -> RayBatch:
        """Returns all rays in the collection."""
        return self.rays


@dataclass
class ImageData:
    path: Path
    n_channels: int
    channel_offset: int = 0
    scale_factor: float | None = None
    data_scale: float | None = None
    load_fn: Callable = load_image_simple
    resize_fn: Callable = apply_image_scale_factor
    _data: torch.Tensor | None = field(init=False, default=None)

    def __post_init__(self):
        if not self.path.exists():
            raise FileNotFoundError(f"Image file {self.path} does not exist.")
        if self.scale_factor == 1:
            self.scale_factor = None
        if self.data_scale == 1:
            self.data_scale = None

    @property
    def image(self) -> torch.Tensor:
        return self._load() if self._data is None else self._data

    def prefetch(self, to_default_device: bool) -> None:
        """Loads the image into memory."""
        self._data = self.image.to(Framework.config.GLOBAL.DEFAULT_DEVICE) if to_default_device else self.image.cpu()

    def update_data_scale(self, factor: float) -> None:
        """Updates the data scaling."""
        if factor == 1:
            return
        self.data_scale = factor if self.data_scale is None else factor * self.data_scale
        if self._data is not None:
            self._data = self._load().to(self._data.device)

    def _load(self) -> torch.Tensor:
        """Loads the image from the file."""
        image = self.load_fn(self.path)
        image = image[self.channel_offset:self.channel_offset + self.n_channels]
        if self.data_scale is not None:
            image = image * self.data_scale
        if self.scale_factor is not None:
            image = self.resize_fn(image, self.scale_factor)
        return image.contiguous()

    @staticmethod
    def load_from_worker(args: dict[str, Any]) -> torch.Tensor:
        """Worker function that replicates _load logic with given arguments."""
        torch.set_num_threads(1)

        path: Path = args['path']
        n_channels: int = args['n_channels']
        channel_offset: int = args['channel_offset']
        scale_factor: float | None = args['scale_factor']
        data_scale: float | None = args['data_scale']
        load_fn: Callable = args['load_fn']
        resize_fn: Callable = args['resize_fn']

        image = load_fn(path)
        image = image[channel_offset:channel_offset + n_channels]
        if data_scale is not None:
            image = image * data_scale
        if scale_factor is not None:
            image = resize_fn(image, scale_factor)
        return image.contiguous()

    def set_data(self, data: torch.Tensor, to_default_device: bool) -> None:
        """Sets the image data directly."""
        if data.shape[0] != self.n_channels:
            raise Framework.DatasetError(f'Expected image with {self.n_channels} channels, got {data.shape[0]}')
        self._data = data.to(Framework.config.GLOBAL.DEFAULT_DEVICE) if to_default_device else data.cpu()


class View:
    """Stores information for a view of a scene."""
    def __init__(
        self,
        camera: 'BaseCamera',
        camera_index: int,
        frame_idx: int,
        global_frame_idx: int,
        c2w: np.ndarray,
        timestamp: float = 0.0,
        exif: dict | None = None,
        rgb: ImageData | None = None,
        alpha: ImageData | None = None,
        depth: ImageData | None = None,
        segmentation: ImageData | None = None,
        forward_flow: ImageData | None = None,
        backward_flow: ImageData | None = None,
        misc: ImageData | None = None,
    ) -> None:
        self.camera = camera
        self.camera_index = camera_index
        self.frame_idx = frame_idx
        self.global_frame_idx = global_frame_idx
        self.c2w = c2w
        self.timestamp = timestamp
        self.exif = exif if exif is not None else {}
        self._rgb = rgb
        self._alpha = alpha
        self._depth = depth
        self._segmentation = segmentation
        self._forward_flow = forward_flow
        self._backward_flow = backward_flow
        self._misc = misc

    @property
    def c2w(self) -> torch.Tensor:
        """Returns the camera to world transformation."""
        return torch.as_tensor(self._c2w, dtype=torch.float32, device=Framework.config.GLOBAL.DEFAULT_DEVICE)

    @c2w.setter
    def c2w(self, c2w: np.ndarray) -> None:
        """Sets the camera to world transformation."""
        if c2w.dtype != np.float64:
            raise Framework.DatasetError(f'c2w must be a np.array of type np.float64 but got {c2w.dtype}')
        elif c2w.shape == (3, 4):
            c2w = np.vstack([c2w, np.array([0, 0, 0, 1], dtype=c2w.dtype)])
        elif c2w.shape != (4, 4):
            raise Framework.DatasetError(f'c2w must be a np.array of shape (4, 4) or (3, 4) but got {c2w.shape}')
        self._c2w = c2w

    @property
    def c2w_numpy(self) -> np.ndarray:
        """Returns the camera to world transformation."""
        return self._c2w.copy()

    @property
    def w2c(self) -> torch.Tensor:
        """Returns the world to camera transformation."""
        return torch.as_tensor(invert_3d_affine(self._c2w), dtype=torch.float32, device=Framework.config.GLOBAL.DEFAULT_DEVICE)

    @w2c.setter
    def w2c(self, w2c: np.ndarray) -> None:
        """Sets the camera to world transformation according to the given world to camera transformation."""
        if w2c.dtype != np.float64:
            raise Framework.DatasetError(f'w2c must be a np.array of type np.float64 but got {w2c.dtype}')
        elif w2c.shape == (3, 4):
            w2c = np.vstack([w2c, np.array([0, 0, 0, 1], dtype=w2c.dtype)])
        elif w2c.shape != (4, 4):
            raise Framework.DatasetError(f'w2c must be a np.array of shape (4, 4) or (3, 4) but got {w2c.shape}')
        self._c2w = invert_3d_affine(w2c)

    @property
    def w2c_numpy(self) -> np.ndarray:
        """Returns the world to camera transformation."""
        return invert_3d_affine(self._c2w)

    @property
    def rotation(self) -> torch.Tensor:
        """Returns the orientation of the camera in world space."""
        return torch.as_tensor(self._c2w[:3, :3], dtype=torch.float32, device=Framework.config.GLOBAL.DEFAULT_DEVICE)

    @rotation.setter
    def rotation(self, rotation: np.ndarray) -> None:
        """Sets the orientation of the camera in world space."""
        if rotation.shape != (3, 3) or rotation.dtype != np.float64:
            raise Framework.DatasetError('rotation must be a np.array of shape (3, 3) and type np.float64')
        self._c2w[:3, :3] = rotation

    @property
    def position(self) -> torch.Tensor:
        """Returns the camera position in world space."""
        return torch.as_tensor(self._c2w[:3, 3], dtype=torch.float32, device=Framework.config.GLOBAL.DEFAULT_DEVICE)

    @position.setter
    def position(self, position: np.ndarray) -> None:
        """Sets the camera position in world space."""
        if position.shape != (3,) or position.dtype != np.float64:
            raise Framework.DatasetError('position must be a np.array of shape (3,) and type np.float64')
        self._c2w[:3, 3] = position

    @property
    def position_numpy(self) -> np.ndarray:
        """Returns the camera position in world space."""
        return self._c2w[:3, 3]

    @property
    def forward(self) -> torch.Tensor:
        """Returns the forward direction of the camera in world space."""
        return torch.as_tensor(self._c2w[:3, 2], dtype=torch.float32, device=Framework.config.GLOBAL.DEFAULT_DEVICE)

    @property
    def forward_numpy(self) -> np.ndarray:
        """Returns the forward direction of the camera in world space."""
        return self._c2w[:3, 2]

    @property
    def right(self) -> torch.Tensor:
        """Returns the right direction of the camera in world space."""
        return torch.as_tensor(self._c2w[:3, 0], dtype=torch.float32, device=Framework.config.GLOBAL.DEFAULT_DEVICE)

    @property
    def right_numpy(self) -> np.ndarray:
        """Returns the right direction of the camera in world space."""
        return self._c2w[:3, 0]

    @property
    def up(self) -> torch.Tensor:
        """Returns the up direction of the camera in world space."""
        return torch.as_tensor(-self._c2w[:3, 1], dtype=torch.float32, device=Framework.config.GLOBAL.DEFAULT_DEVICE)

    @property
    def up_numpy(self) -> np.ndarray:
        """Returns the up direction of the camera in world space."""
        return -self._c2w[:3, 1]

    @property
    def rgb(self) -> torch.Tensor | None:
        if self._rgb is None:
            return None
        return self._rgb.image.to(Framework.config.GLOBAL.DEFAULT_DEVICE)

    @rgb.setter
    def rgb(self, data: ImageData) -> None:
        if not isinstance(data, ImageData):
            raise Framework.DatasetError('rgb must be of type ImageData')
        if data.n_channels != 3:
            raise Framework.DatasetError('rgb must have 3 channels')
        self._rgb = data

    @property
    def alpha(self) -> torch.Tensor | None:
        if self._alpha is None:
            return None
        return self._alpha.image.to(Framework.config.GLOBAL.DEFAULT_DEVICE)

    @alpha.setter
    def alpha(self, data: ImageData) -> None:
        if not isinstance(data, ImageData):
            raise Framework.DatasetError('alpha must be of type ImageData')
        if data.n_channels != 1:
            raise Framework.DatasetError('alpha must have 1 channel')
        self._alpha = data

    @property
    def depth(self) -> torch.Tensor | None:
        if self._depth is None:
            return None
        return self._depth.image.to(Framework.config.GLOBAL.DEFAULT_DEVICE)

    @depth.setter
    def depth(self, data: ImageData) -> None:
        if not isinstance(data, ImageData):
            raise Framework.DatasetError('depth must be of type ImageData')
        if data.n_channels != 1:
            raise Framework.DatasetError('depth must have 1 channel')
        self._depth = data

    @property
    def forward_flow(self) -> torch.Tensor | None:
        if self._forward_flow is None:
            return None
        return self._forward_flow.image.to(Framework.config.GLOBAL.DEFAULT_DEVICE)

    @forward_flow.setter
    def forward_flow(self, data: ImageData) -> None:
        if not isinstance(data, ImageData):
            raise Framework.DatasetError('forward flow must be of type ImageData')
        if data.n_channels != 2:
            raise Framework.DatasetError('forward flow must have 2 channels')
        self._forward_flow = data

    @property
    def backward_flow(self) -> torch.Tensor | None:
        if self._backward_flow is None:
            return None
        return self._backward_flow.image.to(Framework.config.GLOBAL.DEFAULT_DEVICE)

    @backward_flow.setter
    def backward_flow(self, data: ImageData) -> None:
        if not isinstance(data, ImageData):
            raise Framework.DatasetError('backward flow must be of type ImageData')
        if data.n_channels != 2:
            raise Framework.DatasetError('backward flow must have 2 channels')
        self._backward_flow = data

    @property
    def segmentation(self) -> torch.Tensor | None:
        if self._segmentation is None:
            return None
        return self._segmentation.image.to(Framework.config.GLOBAL.DEFAULT_DEVICE)

    @segmentation.setter
    def segmentation(self, data: ImageData) -> None:
        if not isinstance(data, ImageData):
            raise Framework.DatasetError('segmentation must be of type ImageData')
        if data.n_channels != 1:
            raise Framework.DatasetError('segmentation must have 1 channel')
        self._segmentation = data

    @property
    def misc(self) -> torch.Tensor | None:
        if self._misc is None:
            return None
        return self._misc.image.to(Framework.config.GLOBAL.DEFAULT_DEVICE)

    @misc.setter
    def misc(self, data: ImageData) -> None:
        if not isinstance(data, ImageData):
            raise Framework.DatasetError('misc must be of type ImageData')
        self._misc = data

    @property
    def available_image_data(self) -> list[str]:
        """Returns a list of available image data fields."""
        return [field[1:] for field in vars(self) if isinstance(getattr(self, field), ImageData)]

    def get_parallel_load_helpers(self, field: str) -> tuple[dict[str, Any], Callable]:
        """Returns the load parameters and callback function for parallel loading of image data."""
        image_data = getattr(self, f'_{field}', None)
        if image_data is None or not isinstance(image_data, ImageData):
            raise Framework.DatasetError(f'ImageData field "{field}" is not available')
        task = {
            'path': image_data.path,
            'n_channels': image_data.n_channels,
            'channel_offset': image_data.channel_offset,
            'scale_factor': image_data.scale_factor,
            'data_scale': image_data.data_scale,
            'load_fn': image_data.load_fn,
            'resize_fn': image_data.resize_fn,
        }
        callback = image_data.set_data
        return task, callback

    def recenter_and_scale(self, center: np.ndarray, scale: float) -> None:
        """Centers and scales the camera properties."""
        if center.shape != (3,) or center.dtype != np.float64:
            raise Framework.DatasetError('center must be a np.array of shape (3,) and type np.float64')
        self.position = (self.position_numpy - center) * scale
        if self._depth is not None:
            self._depth.update_data_scale(scale)

    def world_to_cam(self, xyz: torch.Tensor, is_point: bool = True) -> torch.Tensor:
        """Transforms points or vectors from world space to camera space."""
        if is_point:
            xyz = xyz - self.position
        return xyz @ self.rotation

    def cam_to_world(self, xyz: torch.Tensor, is_point: bool = True) -> torch.Tensor:
        """Transforms points or vectors from camera space to world space."""
        xyz = xyz @ self.rotation.T
        if is_point:
            xyz = xyz + self.position
        return xyz

    def project_points(self, xyz_world: torch.Tensor, z_culling: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Projects 3D points in world space to 2D screen space."""
        xyz_cam = self.world_to_cam(xyz_world)
        xy_screen, depth, in_frustum = self.camera.cam_to_screen(xyz_cam, z_culling)
        return xy_screen, depth, in_frustum

    def unproject_points(self, xy_screen: torch.Tensor, depth: torch.Tensor | float | None = None) -> torch.Tensor:
        """Unprojects 2D screen points to 3D points in world space."""
        xyz_cam = self.camera.screen_to_cam(xy_screen)
        if depth is not None:
            xyz_cam = xyz_cam * depth
        return self.cam_to_world(xyz_cam)

    def get_rays(self) -> RayBatch:
        """Generates rays for all pixels in the image."""
        # base ray properties
        direction = self.cam_to_world(self.camera.compute_local_ray_directions(), is_point=False)
        origin = self.position.expand_as(direction)
        # annotations
        # TODO: ideally view_direction would be None if method does not need it
        view_direction = torch.nn.functional.normalize(direction, dim=-1)
        rgb = None if self._rgb is None else self.rgb.permute(1, 2, 0).reshape(-1, 3)
        alpha = None if self._alpha is None else self.alpha.permute(1, 2, 0).reshape(-1, 1)
        depth = None if self._depth is None else self.depth.permute(1, 2, 0).reshape(-1, 1)
        # TODO: ideally timestamp would be None if scene is static
        timestamp = torch.tensor([self.timestamp], dtype=direction.dtype, device=direction.device).expand(direction.shape[0], 1)
        return RayBatch(
            origin=origin,
            direction=direction,
            view_direction=view_direction,
            rgb=rgb,
            alpha=alpha,
            depth=depth,
            timestamp=timestamp,
        ).to(device=Framework.config.GLOBAL.DEFAULT_DEVICE)

    def to_simple(self) -> 'View':
        """Returns a simplified version of the view without image data."""
        return View(
            camera=deepcopy(self.camera),
            camera_index=self.camera_index,
            frame_idx=self.frame_idx,
            global_frame_idx=self.global_frame_idx,
            c2w=self._c2w.copy(),
            timestamp=self.timestamp,
            exif=deepcopy(self.exif),
        )


def estimate_near_far(
    views: list[View],
    point_cloud: BasicPointCloud,
    tolerance: float = 0.1,
    min_near_plane: float = 0.01
) -> tuple[float, float]:
    """Estimates near and far plane distances for the given views and point cloud."""
    points = point_cloud.positions.cuda()
    min_depth = math.inf
    max_depth = 0.0
    for view in views:
        _, depths, in_frustum = view.project_points(points, z_culling=False)
        valid_mask = in_frustum & (depths > 0.0)
        depths = depths[valid_mask]
        min_depth = min(min_depth, depths.min().item())
        max_depth = max(max_depth, depths.max().item())
    return max(min_near_plane, min_depth * (1.0 - tolerance)), max_depth * (1.0 + tolerance)
