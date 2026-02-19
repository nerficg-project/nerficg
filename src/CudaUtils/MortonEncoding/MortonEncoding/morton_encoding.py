import torch
from MortonEncoding import _C  # noqa

def morton_encode(positions: torch.Tensor) -> torch.Tensor:
    """Computes the morton codes for a set of 3D positions."""
    return _C.morton_encode(positions)
