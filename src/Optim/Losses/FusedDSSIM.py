# -- coding: utf-8 --

"""
Optim/Losses/FusedDSSIM.py: Loss based on the structural (dis-)similarity index measure (DSSIM = 1 - SSIM).
"""

import torch

from Thirdparty.FusedSSIM import fused_ssim


def fused_dssim(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate loss."""
    return 1.0 - fused_ssim(input[None], target[None])  # should be (1.0 - SSIM) / 2.0
