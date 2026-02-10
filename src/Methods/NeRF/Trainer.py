"""NeRF/Trainer.py: Implementation of the trainer for the original NeRF method."""

from functools import partial

import torch

import Framework
from Datasets.Base import BaseDataset
from Methods.Base.Trainer import BaseTrainer
from Methods.Base.utils import pre_training_callback, training_callback
from Methods.NeRF.Loss import NeRFLoss
from Optim.Samplers.DatasetSamplers import DatasetSampler, RayPoolSampler
from Optim.Samplers.ImageSamplers import RandomImageSampler
from Optim.lr_utils import LRDecayPolicy


@Framework.Configurable.configure(
    NUM_ITERATIONS=500000,
    BATCH_SIZE=1024,
    SAMPLE_SINGLE_IMAGE=True,
    DENSITY_RANDOM_NOISE_STD=0.0,
    LR_INIT=5e-04,
    LR_FINAL=5e-05,
    LAMBDA_COLOR_LOSS=1.0,
    LAMBDA_ALPHA_LOSS=0.0,
)
class NeRFTrainer(BaseTrainer):
    """Defines the trainer for the original NeRF method."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1.0)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=LRDecayPolicy(lr_init=self.LR_INIT, lr_final=self.LR_FINAL, max_steps=self.NUM_ITERATIONS),
            last_epoch=self.model.num_iterations_trained - 1,
        )
        self.loss = NeRFLoss(self.LAMBDA_COLOR_LOSS, self.LAMBDA_ALPHA_LOSS, self.model.coarse_nerf is not None)
        self.sampler_train = None
        self.sampler_val = None

    @pre_training_callback(priority=1000)
    @torch.no_grad()
    def init_samplers(self, _, dataset: 'BaseDataset') -> None:
        """Initializes dataset samplers for training and validation."""
        sampler_cls = partial(DatasetSampler, random=True) if self.SAMPLE_SINGLE_IMAGE else RayPoolSampler
        self.sampler_train = sampler_cls(dataset=dataset.train(), img_sampler_cls=RandomImageSampler)
        if self.RUN_VALIDATION:
            self.sampler_val = sampler_cls(dataset=dataset.eval(), img_sampler_cls=RandomImageSampler)

    @training_callback(priority=50)
    def training_iteration(self, _, dataset: 'BaseDataset') -> None:
        """Process a ray batch for training."""
        self.model.train()
        self.loss.train()
        dataset.train()
        ray_batch = self.sampler_train.get(dataset=dataset, ray_batch_size=self.BATCH_SIZE)['ray_batch']
        outputs = self.renderer.render_rays(ray_batch, dataset.default_camera, randomize_samples=True, random_noise_density=self.DENSITY_RANDOM_NOISE_STD)
        loss = self.loss(outputs, ray_batch, dataset.default_camera.background_color)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()

    @training_callback(active='RUN_VALIDATION', priority=100)
    @torch.no_grad()
    def validation_iteration(self, _, dataset: 'BaseDataset') -> None:
        """Process a ray batch for validation."""
        self.model.eval()
        self.loss.eval()
        dataset.eval()
        ray_batch = self.sampler_val.get(dataset=dataset, ray_batch_size=self.BATCH_SIZE)['ray_batch']
        outputs = self.renderer.render_rays(ray_batch, dataset.default_camera)
        self.loss(outputs, ray_batch, dataset.default_camera.background_color)
