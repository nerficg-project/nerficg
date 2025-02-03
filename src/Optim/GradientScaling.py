# -- coding: utf-8 --

"""Optim/GradientScaling.py: Gradient scaling routines."""

from typing import Any
import torch


class _GradientScaler(torch.autograd.Function):
    """
    Utility Autograd function scaling gradients by a given factor.
    Adapted from NerfStudio: https://docs.nerf.studio/en/latest/_modules/nerfstudio/model_components/losses.html
    """

    @staticmethod
    def forward(ctx, value: torch.Tensor, scaling: Any) -> tuple[torch.Tensor, Any]:
        ctx.save_for_backward(scaling)
        return value, scaling

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor, grad_scaling: Any) -> tuple[torch.Tensor, Any]:
        (scaling,) = ctx.saved_tensors
        return output_grad * scaling, grad_scaling


def scaleGradient(*args: torch.Tensor, scaling: Any) -> tuple[torch.Tensor, ...] | torch.Tensor:
    """
    Scale the gradient of the given tensors.
    """
    output: list[torch.Tensor] = [_GradientScaler.apply(value, scaling)[0] for value in args]
    return tuple(output) if len(output) > 1 else output[0]


def scaleGradientByDistance(*args: torch.Tensor, distances: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """
    Scale the gradient of the given tensor based on the normalized ray distance.
    See: Radiance Field Gradient Scaling for Improved Near-Camera Training (https://gradient-scaling.github.io)
    """
    return scaleGradient(*args, scaling=torch.square(distances).clamp(0, 1))
