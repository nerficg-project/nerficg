"""Losses/Distortion.py: Distortion loss as introduced in MipNeRF360 / DVGOv2."""

import torch

from Methods.InstantNGP.VolumeRenderingV2 import DistortionLoss


def distortion_loss(ws: torch.Tensor, deltas: torch.Tensor, ts: torch.Tensor, rays_a: torch.Tensor) -> torch.Tensor:
    return DistortionLoss.apply(ws, deltas, ts, rays_a).mean()
