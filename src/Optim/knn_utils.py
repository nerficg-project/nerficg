"""Optim/knn_utils.py: K nearest neighbour utilities."""

import warnings

import torch
from sklearn.neighbors import NearestNeighbors

_ALLOW_PROPRIETARY_LICENSED_CODE = True

_FUSED_KNN_AVAILABLE = False
if _ALLOW_PROPRIETARY_LICENSED_CODE:
    try:
        from Thirdparty.SimpleKNN import compute_mean_squared_knn_distances as compute_mean_squared_knn_distances_fused
        _FUSED_KNN_AVAILABLE = True
        warnings.warn(
            'Using proprietary SimpleKNN extension for KNN distance computation. Ensure you comply with its license.',
            RuntimeWarning,
            stacklevel=2,
        )
    except ImportError:
        compute_mean_squared_knn_distances_fused = None


def compute_knn_distances(points: torch.Tensor, n_neighbors: int) -> torch.Tensor:
    """Return Euclidean distances to k nearest neighbors, shape [N, k]."""
    knn_distances, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(points.cpu().numpy()).kneighbors()
    return torch.from_numpy(knn_distances).to(device=points.device, dtype=points.dtype)

def compute_root_mean_squared_knn_distances(points: torch.Tensor, k: int = 3, eps: float = 1e-7) -> torch.Tensor:
    """
    Return per-point RMS distance to k nearest neighbors, shape [N].
    Defaults match the implementation used by the original 3DGS implementation, the initial use case for this function.
    """
    if _FUSED_KNN_AVAILABLE and k == 3:
        # use proprietary CUDA implementation from 3DGS for better performance on large point clouds
        mean_squared_knn_distances = compute_mean_squared_knn_distances_fused(points.cuda())  # noqa
    else:
        mean_squared_knn_distances = compute_knn_distances(points, n_neighbors=k).square().mean(dim=-1)

    return mean_squared_knn_distances.clamp_min(eps).sqrt().to(device=points.device, dtype=points.dtype)
