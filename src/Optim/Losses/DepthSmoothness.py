"""
Losses/DepthSmoothness.py: Smooth depth loss.
Adapted from https://kornia.readthedocs.io/en/v0.2.1/_modules/kornia/losses/depth_smooth.html.
"""

import torch


def _gradient_x(img: torch.Tensor) -> torch.Tensor:
    return img[:, :, :, 1:-1] - img[:, :, :, 0:-2]


def _gradient_y(img: torch.Tensor) -> torch.Tensor:
    return img[:, :, 1:-1, :] - img[:, :, 0:-2, :]


def _laplace_x(img: torch.Tensor) -> torch.Tensor:
    mi = img[:, :, :, 1:-1]
    le = img[:, :, :, :-2]
    ri = img[:, :, :, 2:]
    return le + ri - (2 * mi)


def _laplace_y(img: torch.Tensor) -> torch.Tensor:
    mi = img[:, :, 1:-1, :]
    le = img[:, :, :-2, :]
    ri = img[:, :, 2:, :]
    return le + ri - (2 * mi)


def depth_smoothness_loss(depth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    # compute the gradients
    idepth_dx = _laplace_x(depth)
    idepth_dy = _laplace_y(depth)
    image_dx = _gradient_x(image)
    image_dy = _gradient_y(image)
    # compute image weights
    weights_x = torch.exp(-torch.mean(image_dx.abs(), dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(image_dy.abs(), dim=1, keepdim=True))
    # apply image weights to depth
    smoothness_x = (idepth_dx * weights_x).abs()
    smoothness_y = (idepth_dy * weights_y).abs()
    return smoothness_x.mean() + smoothness_y.mean()
