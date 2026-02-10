"""Optim/gradient_scaling.py: Gradient scaling routines based on https://gradient-scaling.github.io."""

import torch


class _GradientScaler(torch.autograd.Function):
    """Scales gradients by a given factor during backward computation."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scaling: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.save_for_backward(scaling)
        return x, scaling

    @staticmethod
    def backward(ctx, x_grad: torch.Tensor, grad_scaling: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scaling, = ctx.saved_tensors
        return x_grad * scaling, grad_scaling


def scale_gradient(*args: torch.Tensor, scaling: torch.Tensor) -> tuple[torch.Tensor, ...] | torch.Tensor:
    """Scale the gradient of the given tensors."""
    output = tuple(_GradientScaler.apply(value, scaling)[0] for value in args)
    return tuple(output) if len(output) > 1 else output[0]


def scale_gradient_by_distance(*args: torch.Tensor, distances: torch.Tensor) -> tuple[torch.Tensor, ...] | torch.Tensor:
    """Scale the gradient of the given tensor based on the normalized ray distance."""
    return scale_gradient(*args, scaling=distances.square().clamp(0, 1))
