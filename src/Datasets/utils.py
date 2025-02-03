# -- coding: utf-8 --

"""Datasets/utils.py: Contains utility functions used for the implementation of the available dataset classes."""


import os
from pathlib import Path
from typing import Iterator, Callable, Any
import natsort
from enum import Enum
from dataclasses import dataclass, fields

import numpy as np
import torch
from torch.multiprocessing import Pool
from torchvision import io
from torchvision.utils import _normalized_flow_to_image, draw_segmentation_masks

from Cameras.Base import BaseCamera
import Framework
from Cameras.utils import CameraProperties, createCameraMatrix
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


def loadImage(filename: str, scale_factor: float | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Loads an image from the specified file."""
    try:
        image: torch.Tensor = io.read_image(path=str(filename), mode=io.ImageReadMode.UNCHANGED)
    except Exception:
        raise Framework.DatasetError(f'Failed to load image file: "{filename}"')
    # convert image to the format used by the framework
    image = image.float() / 255
    # apply scaling factor to image
    if scale_factor is not None:
        image = applyImageScaleFactor(image, scale_factor)
    # extract alpha channel if available
    rgb, alpha = image.split([3, 1]) if image.shape[0] == 4 else (image, None)
    return rgb, alpha


def parallelLoadFN(args: dict[str, Any]) -> Any:
    """Function executed by each thread when loading in parallel."""
    torch.set_num_threads(1)
    load_function = args['load_function']
    del args['load_function']
    return load_function(**args)


def getParallelLoadIterator(filenames: list[str], scale_factor: float | None, num_threads: int,
                            load_function: Callable) -> tuple[Iterator, Pool]:  # type: ignore
    """Returns iterator for parallel image loading."""
    # create thread pool
    if num_threads < 1:
        num_threads = os.cpu_count() - 1
    pool = Pool(min(num_threads, len(filenames)))
    # create and return the iterator
    return pool.imap(
        func=parallelLoadFN,
        iterable=[{'load_function': load_function, 'filename': filenames[i], 'scale_factor': scale_factor} for i in range(len(filenames))],
        chunksize=1,
    ), pool


def loadImagesParallel(filenames: list[str], scale_factor: float | None, num_threads: int, desc="") -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Loads multiple images in parallel."""
    iterator, pool = getParallelLoadIterator(filenames, scale_factor, num_threads, load_function=loadImage)
    rgbs, alphas = [], []
    for rgb, alpha in Logger.logProgressBar(iterator, desc=desc, leave=False, total=len(filenames)):
        # clone tensors to extract them from shared memory (/dev/shm), otherwise we can not use all RAM
        rgbs.append(rgb.clone())
        alphas.append(alpha.clone() if alpha is not None else None)
    pool.close()
    pool.join()
    return rgbs, alphas


def applyImageScaleFactor(image: torch.Tensor, scale_factor: float, mode: str = 'area') -> torch.Tensor:
    """Scales the image by the specified factor."""
    return torch.nn.functional.interpolate(
        input=image[None],
        scale_factor=scale_factor,
        mode=mode,
    )[0]


def applyBGColor(rgb: torch.Tensor, alpha: torch.Tensor | None, bg_color: torch.Tensor | None) -> torch.Tensor:
    """Applies the given color to the image according to its alpha values."""
    if bg_color is not None and alpha is not None:
        rgb *= alpha
        rgb += (1 - alpha) * bg_color[:, None, None].to(alpha.device)
    return rgb


def getAveragePose(view_matrices: torch.Tensor) -> torch.Tensor:
    """Creates an average view matrix."""
    avg_position: torch.Tensor = view_matrices[:, :3, -1].mean(0)
    view_dir: torch.Tensor = view_matrices[:, :3, 2].sum(0)
    up_dir: torch.Tensor = view_matrices[:, :3, 1].sum(0)
    return createCameraMatrix(-view_dir, -up_dir, avg_position)


def recenterPoses(view_matrices: torch.Tensor) -> torch.Tensor:
    """Recenters the scene coordinate system."""
    # create inverse average transformation
    center_transform: torch.Tensor = torch.linalg.inv(getAveragePose(view_matrices))
    # apply transformation
    return center_transform @ view_matrices


def saveImage(filepath: Path, image: torch.Tensor) -> None:
    """Writes the input image tensor to the file given by filepath."""
    image = image.clamp(0.0, 1.0).mul(255.0).round().byte().cpu()
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


def saveSegmentation(filename: Path, segmentation: torch.Tensor) -> None:
    """Writes semantic segmentation labels to png file."""
    io.write_png(
        input=segmentation.byte().cpu(),
        filename=f'{filename}.png',
        compression_level=6,  # opencv uses 3
    )


def loadSegmentation(filepath: str) -> torch.Tensor | None:
    try:
        segmentation: torch.Tensor = io.read_image(path=filepath, mode=io.ImageReadMode.UNCHANGED)
    except Exception:
        raise Framework.DatasetError(f'Failed to load segmentation image file: "{filepath}"')
    return segmentation


def segmentationToImage(image: torch.Tensor, segmentation: torch.Tensor, num_classes: int, alpha: float = 0.7) -> torch.Tensor:
    # expects segmentation in shape (1, H, W)
    masks = torch.nn.functional.one_hot(segmentation.long()[0], num_classes=num_classes).bool().permute(2, 0, 1)
    color = draw_segmentation_masks((image * 255).byte(), masks, alpha)
    return color.float() / 255.0


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
                flow = applyImageScaleFactor(flow, scale_factor, mode='nearest')
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
    iterator_flow, pool = getParallelLoadIterator(flow_filenames, image_scale_factor, num_threads, load_function=loadOpticalFlowFile)
    forward_flows, backward_flows = [], []
    for _ in Logger.logProgressBar(range(len(frame_names)), desc='flow', leave=False):
        forward_flow, backward_flow = next(iterator_flow), next(iterator_flow)
        # clone tensors to extract them from shared memory (/dev/shm), otherwise we can not use all RAM
        forward_flows.append(forward_flow.clone() if forward_flow is not None else None)
        backward_flows.append(backward_flow.clone() if backward_flow is not None else None)
    pool.close()
    pool.join()
    return forward_flows, backward_flows


def saveOpticalFlowFile(filename: Path, optical_flow: torch.Tensor) -> None:
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
def flowToImage(optical_flow: torch.Tensor, max_norm: float = None) -> torch.Tensor:
    # adapted from https://pytorch.org/vision/stable/_modules/torchvision/utils.html#flow_to_image
    orig_shape = optical_flow.shape
    if optical_flow.ndim == 3:
        optical_flow = optical_flow[None]  # Add batch dim
    if optical_flow.ndim != 4 or optical_flow.shape[1] != 2:
        raise Framework.DatasetError(f"Input flow should have shape (2, H, W) or (N, 2, H, W), got {orig_shape}.")
    norms = torch.linalg.norm(optical_flow, ord=None, dim=1, keepdim=True)
    if max_norm is None or max_norm <= 0:
        max_norm = norms.max().item()
    norms = norms.clamp_min_(min=max_norm + torch.finfo((optical_flow).dtype).eps)
    normalized_flow = optical_flow / norms
    img = _normalized_flow_to_image(normalized_flow)
    if len(orig_shape) == 3:
        img = img[0]  # Remove batch dim
    return img.float() / 255.0


class CameraCoordinateSystemsTransformations(Enum):
    """Provides transformation functions for common camera coordinate systems relative to the framework's internal representation."""
    def OPENGL(x, y, z): return [x, y, z]  # internal framework representation: right/down/backward (left-handed)  # TODO: this should not be called opengl as opengl ndc (i.e. camera) convention is right/up/forward (left-handed)
    def RIGHT_HAND(x, y, z): return [x, -y, z]  # right/up/backward (right-handed)  # TODO: this is, e.g., the OpenGL world coordinate system
    def LEFT_HAND(x, y, z): return [x, y, -z]  # right/down/forward (right-handed)  # TODO: rename and probably make this the default; this is, e.g., used by Colmap
    def PYTORCH3D(x, y, z): return [-x, -y, -z]  # left/up/forward (right-handed)


class WorldCoordinateSystemTransformations(Enum):
    """Provides transformation functions for world coordinate systems."""
    def XYZ(x, y, z): return [x, y, z]  # internal framework representation: right/forward/down (left-handed)  # TODO: double-check this asap
    def XnYZ(x, y, z): return [x, -y, z]
    def XYnZ(x, y, z): return [x, y, -z]
    def XnYnZ(x, y, z): return [x, -y, -z]
    def XZY(x, y, z): return [x, z, y]
    def XnZY(x, y, z): return [x, -z, y]


@dataclass
class BasicPointCloud:
    """
    A class for basic point cloud data, e.g. those produced by Colmap during sparse reconstruction.

    Attributes:
        positions  Torch tensor of shape (N, 3) containing the point positions in world space.
        colors     Torch tensor of shape (N, 3) containing the point colors in RGB (optional).
    """
    positions: torch.Tensor
    colors: torch.Tensor | None = None

    def __post_init__(self):
        self.cast(dtype=torch.float32)
        self.toDevice(device=torch.device('cpu'))

    def cast(self, dtype: torch.dtype):
        for member in fields(self):
            val = getattr(self, member.name)
            if isinstance(val, torch.Tensor):
                setattr(self, member.name, val.to(dtype))

    def toDevice(self, device: torch.device):
        for member in fields(self):
            val = getattr(self, member.name)
            if isinstance(val, torch.Tensor):
                setattr(self, member.name, val.to(device))

    def transform(self, transformation_matrix: torch.Tensor) -> None:
        rotation_matrix = transformation_matrix[:3, :3]
        translation_vector = transformation_matrix[:3, 3]
        self.positions = self.positions @ rotation_matrix.T + translation_vector

    def convert(self, conversion: Callable) -> None:
        self.positions = torch.cat(conversion(*torch.split(self.positions, split_size_or_sections=1, dim=1)), dim=1)

    def normalize(self, center: torch.Tensor, scale: torch.Tensor) -> None:
        self.positions -= center
        self.positions *= scale

    def filterOutliers(self, filter_percentage: float = 0.95) -> None:
        mean = self.positions.mean(dim=0, keepdim=True)
        dists = torch.linalg.norm((self.positions - mean), ord=None, dim=1)
        quantile = torch.quantile(dists, filter_percentage, interpolation='midpoint')
        valid_mask = dists < quantile
        self.positions = self.positions[valid_mask]
        if self.colors is not None:
            self.colors = self.colors[valid_mask]

    def getBoundingBox(self, tolerance_factor: float = 0.1, filter_outliers_percentage: float | None = None) -> torch.Tensor:
        positions = self.positions
        if filter_outliers_percentage is not None:
            mean = positions.mean(dim=0, keepdim=True)
            dists = torch.linalg.norm(positions - mean, ord=None, dim=1)
            quantile = torch.quantile(dists, filter_outliers_percentage, interpolation='midpoint')
            positions = positions[dists < quantile]
        min = torch.min(positions, dim=0)[0]
        max = torch.max(positions, dim=0)[0]
        center = ((min + max) / 2.0)[None]
        bounding_box = ((torch.stack([min, max], dim=0) - center) * (1.0 + tolerance_factor)) + center
        if filter_outliers_percentage is not None:
            valid_mask = (self.positions > bounding_box[0]) & (self.positions < bounding_box[1])
            valid_mask = valid_mask.all(dim=1)
            self.positions = self.positions[valid_mask]
        return bounding_box


def getNearFarFromPointCloud(camera: BaseCamera, point_cloud: BasicPointCloud, camera_properties: list[CameraProperties], tolerance: float = 0.1) -> tuple[float, float]:
    # recalculate near and far plane
    d_min = 1e8
    d_max = 0.0
    for i in camera_properties:
        camera.setProperties(i)
        depths = camera.projectPoints(point_cloud.positions)[-1]
        depths = depths[depths > 0.0]
        d_min = min(d_min, depths.min().item())
        d_max = max(d_max, depths.max().item())
    return max(1e-2, d_min * (1.0 - tolerance)), d_max * (1.0 + tolerance)


def tensor_to_string(tensor: torch.Tensor, precision: int = 2) -> str:
    """Converts a tensor to a string. Allows for specifying a custom precision."""
    return f'[{", ".join([f"{i:.{precision}f}" for i in tensor])}]'


def transformPosesPCA(poses: torch.Tensor, rescale: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    poses = poses.clone().cpu().numpy()
    poses_ = poses.copy()
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    bottom = np.broadcast_to([0, 0, 0, 1.], poses[..., :1, :4].shape)
    pad_poses = np.concatenate([poses[..., :3, :4], bottom], axis=-2)
    poses_recentered = transform @ pad_poses
    poses_recentered = poses_recentered[..., :3, :4]
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    if rescale:
        # Just make sure it's it in the [-1, 1]^3 cube
        scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
        poses_recentered[:, :3, 3] *= scale_factor
        transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    poses_[:, :3, :4] = poses_recentered[:, :3, :4]
    poses_recentered = poses_
    return torch.from_numpy(poses_recentered), torch.from_numpy(transform).float()
