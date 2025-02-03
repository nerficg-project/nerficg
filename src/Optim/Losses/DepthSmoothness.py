# -- coding: utf-8 --

"""Optim/DepthSmoothness.py: smooth depth loss. adapted from https://kornia.readthedocs.io/en/v0.2.1/_modules/kornia/losses/depth_smooth.html."""

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


def depthSmoothnessLoss(
        depth: torch.Tensor,
        image: torch.Tensor) -> torch.Tensor:

    # compute the gradients
    idepth_dx: torch.Tensor = _laplace_x(depth)
    idepth_dy: torch.Tensor = _laplace_y(depth)
    image_dx: torch.Tensor = _gradient_x(image)
    image_dy: torch.Tensor = _gradient_y(image)

    # compute image weights
    weights_x: torch.Tensor = torch.exp(
        -torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
    weights_y: torch.Tensor = torch.exp(
        -torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

    # apply image weights to depth
    smoothness_x: torch.Tensor = torch.abs(idepth_dx * weights_x)
    smoothness_y: torch.Tensor = torch.abs(idepth_dy * weights_y)
    return torch.mean(smoothness_x) + torch.mean(smoothness_y)
