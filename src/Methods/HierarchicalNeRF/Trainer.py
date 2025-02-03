# -- coding: utf-8 --

"""HierarchicalNeRF/Trainer.py: Implementation of the trainer for the hierarchical NeRF method."""

import torch

from Methods.HierarchicalNeRF.Loss import HierarchicalNeRFLoss
from Methods.NeRF.Trainer import NeRFTrainer


class HierarchicalNeRFTrainer(NeRFTrainer):
    """Defines the trainer for the hierarchical NeRF method."""

    def __init__(self, **kwargs) -> None:
        super(HierarchicalNeRFTrainer, self).__init__(**kwargs)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.LEARNINGRATE,
            betas=(self.ADAM_BETA_1, self.ADAM_BETA_2)
            # eps=Framework.config.GLOBAL.EPS
        )
        for param_group in self.optimizer.param_groups:
            param_group['capturable'] = True  # Hacky fix for PT 1.12 bug
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self.LRDecayPolicy(self.LEARNINGRATE_DECAY_RATE, self.LEARNINGRATE_DECAY_STEPS),
            last_epoch=self.model.num_iterations_trained - 1
        )
        self.loss = HierarchicalNeRFLoss(self.LAMBDA_COLOR_LOSS, self.LAMBDA_ALPHA_LOSS)
        self.renderer.RENDER_COARSE = True
