"""Utils/MortonEncoding: CUDA-accelerated morton code computation for 3d points."""

import torch
from MortonEncoding import _C  # noqa

def morton_encode(positions: torch.Tensor) -> torch.Tensor:
    """Computes the morton codes for a set of 3D positions."""
    if not (positions.dtype == torch.float32 and positions.is_contiguous() and positions.is_cuda and positions.shape[1] == 3):
        raise ValueError('positions must be a contiguous CUDA float32 tensor of shape (N, 3)')
    minimum_coordinates = positions.min(dim=0).values
    cube_size = (positions.max(dim=0).values - minimum_coordinates).max()
    return _C.morton_encode_cuda(positions, minimum_coordinates, cube_size)