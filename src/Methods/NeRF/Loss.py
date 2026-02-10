"""NeRF/Loss.py: Loss implementation for the NeRF method."""

import torch
import torchmetrics

from Datasets.utils import apply_background_color, RayBatch
from Optim.Losses.Base import BaseLoss


class NeRFLoss(BaseLoss):
    """Defines a class for all sub-losses of the NeRF method."""

    def __init__(self, lambda_color: float, lambda_alpha: float, requires_coarse_losses: bool) -> None:
        super().__init__()
        self.coarse_losses = False
        # base losses
        self.add_loss_metric('L2_Color', torch.nn.functional.mse_loss, lambda_color)
        self.add_loss_metric('L2_Alpha', torch.nn.functional.mse_loss, lambda_alpha)
        self.add_quality_metric('PSNR', torchmetrics.functional.image.peak_signal_noise_ratio)
        if requires_coarse_losses:
            self.coarse_losses = True
            self.add_loss_metric('L2_Color_Coarse', torch.nn.functional.mse_loss, lambda_color)
            self.add_loss_metric('L2_Alpha_Coarse', torch.nn.functional.mse_loss, lambda_alpha)
            self.add_quality_metric('PSNR_Coarse', torchmetrics.functional.image.peak_signal_noise_ratio)

    def forward(self, outputs: dict[str, torch.Tensor | None], rays: RayBatch, bg_color: torch.Tensor) -> torch.Tensor:
        """Defines loss calculation."""
        # FIXME: integrate bg color into ray generation
        color_gt = rays.rgb
        alpha_gt = torch.ones_like(outputs['alpha'], requires_grad=False) if rays.alpha is None else rays.alpha
        color_gt = apply_background_color(color_gt, alpha_gt, bg_color, is_chw=False)
        losses = {
            'L2_Color': {'input': outputs['rgb'], 'target': color_gt},
            'L2_Alpha': {'input': outputs['alpha'], 'target': alpha_gt},
            'PSNR': {'preds': outputs['rgb'], 'target': color_gt, 'data_range': 1.0},
        }
        if self.coarse_losses:
            losses |= {
                'L2_Color_Coarse': {'input': outputs['rgb_coarse'], 'target': color_gt},
                'L2_Alpha_Coarse': {'input': outputs['alpha_coarse'], 'target': alpha_gt},
                'PSNR_Coarse': {'preds': outputs['rgb_coarse'], 'target': color_gt, 'data_range': 1.0},
            }
        return super().forward(losses)
