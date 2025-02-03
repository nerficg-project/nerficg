# -- coding: utf-8 --

"""
Optim/Losses/VGG.py: Perceptual loss based on VGG features.
See https://arxiv.org/abs/1603.08155 or https://arxiv.org/abs/1609.04802 for details.
"""

from dataclasses import dataclass
from typing import Callable
import torch
from torchvision.models import VGG, vgg19, VGG16_Weights, VGG19_Weights


@dataclass(frozen=True)
class VGGLossConfig:
    """Configuration for the VGG loss."""
    model_class: 'VGG' = vgg19
    used_weights: VGG16_Weights | VGG19_Weights = VGG19_Weights.IMAGENET1K_V1
    # used_blocks: tuple[slice] = (slice(0, 4), slice(4, 9), slice(9, 16), slice(16, 23))  # vgg16
    used_blocks: tuple[slice] = (slice(0, 4), slice(4, 9), slice(9, 18), slice(18, 27), slice(27, 36))  # vgg19
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.nn.functional.l1_loss


class VGGLoss:
    """Perceptual loss based on VGG features."""

    def __init__(self, config: VGGLossConfig = VGGLossConfig()) -> None:
        """Initialize the VGG loss."""
        model = config.model_class(weights=config.used_weights).features.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        self.blocks = torch.nn.ModuleList([model[block] for block in config.used_blocks])
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.loss_function = config.loss_function

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the VGG loss. Both input and target are expected to have shape (C, H, W) with C being RGB."""
        input = (input[None] - self.mean) / self.std
        target = (target[None] - self.mean) / self.std
        loss = torch.tensor(0.0, requires_grad=False)
        for block in self.blocks:
            input = block(input)
            target = block(target)
            loss += self.loss_function(input, target)
        return loss
