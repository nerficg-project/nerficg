"""
GaussianSplatting/utils.py: Utility functions for GaussianSplatting.
"""

import torch

from Cameras.utils import quaternion_to_rotation_matrix


def build_covariances(scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    R = quaternion_to_rotation_matrix(rotations, normalize=False)
    # add batch dimension if necessary
    if batch_dim_added := scales.dim() == 1:
        scales = scales[None]
    S = torch.diag_embed(scales)
    RS = R @ S
    RSSR = RS @ RS.transpose(-2, -1)
    return RSSR[0] if batch_dim_added else RSSR


def convert_sh_features(sh_features: torch.Tensor, view_directions: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Convert spherical harmonics features to RGB.
    As in 3DGS, we do not use Sigmoid but instead add 0.5 and clamp with 0 from below.

    adapted from multiple sources:
    1. https://www.ppsloan.org/publications/StupidSH36.pdf
    2. https://github.com/sxyu/svox2/blob/59984d6c4fd3d713353bafdcb011646e64647cc7/svox2/utils.py#L115
    3. https://github.com/NVlabs/tiny-cuda-nn/blob/212104156403bd87616c1a4f73a1c5f2c2e172a9/include/tiny-cuda-nn/common_device.h#L340
    4. https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d/cuda_rasterizer/forward.cu#L20
    """
    result = 0.5 + 0.28209479177387814 * sh_features[..., 0]
    if degree == 0:
        return result.clamp_min(0.0)
    x = view_directions[..., 0:1]
    y = view_directions[..., 1:2]
    z = view_directions[..., 2:3]
    result += -0.48860251190291987 * y * sh_features[..., 1]
    result += 0.48860251190291987 * z * sh_features[..., 2]
    result += -0.48860251190291987 * x * sh_features[..., 3]
    if degree == 1:
        return result.clamp_min(0.0)
    x2, y2, z2 = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    result += 1.0925484305920792 * xy * sh_features[..., 4]
    result += -1.0925484305920792 * yz * sh_features[..., 5]
    result += (0.94617469575755997 * z2 - 0.31539156525251999) * sh_features[..., 6]
    result += -1.0925484305920792 * xz * sh_features[..., 7]
    result += 0.54627421529603959 * (x2 - y2) * sh_features[..., 8]
    if degree == 2:
        return result.clamp_min(0.0)
    result += 0.59004358992664352 * y * (-3.0 * x2 + y2) * sh_features[..., 9]
    result += 2.8906114426405538 * xy * z * sh_features[..., 10]
    result += 0.45704579946446572 * y * (1.0 - 5.0 * z2) * sh_features[..., 11]
    result += 0.3731763325901154 * z * (5.0 * z2 - 3.0) * sh_features[..., 12]
    result += 0.45704579946446572 * x * (1.0 - 5.0 * z2) * sh_features[..., 13]
    result += 1.4453057213202769 * z * (x2 - y2) * sh_features[..., 14]
    result += 0.59004358992664352 * x * (-x2 + 3.0 * y2) * sh_features[..., 15]
    return result.clamp_min(0.0)


def rgb_to_sh0(rgb: torch.Tensor | float) -> torch.Tensor | float:
    return (rgb - 0.5) / 0.28209479177387814


def sh0_to_rgb(sh: torch.Tensor | float) -> torch.Tensor | float:
    return sh * 0.28209479177387814 + 0.5


def extract_upper_triangular_matrix(matrix: torch.Tensor) -> torch.Tensor:
    upper_triangular_indices = torch.triu_indices(matrix.shape[-2], matrix.shape[-1])
    upper_triangular_matrix = matrix[..., upper_triangular_indices[0], upper_triangular_indices[1]]
    return upper_triangular_matrix
