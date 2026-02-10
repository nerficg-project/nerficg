"""
GaussianSplatting/Trainer.py: Implementation of the trainer for the GaussianSplatting method.
Callback schedule is slightly modified from the original implementation. They count iterations starting at 1 instead of 0.
"""

import torch

import Framework
from Datasets.Base import BaseDataset
from Datasets.utils import BasicPointCloud, apply_background_color
from Logging import Logger
from Methods.Base.GuiTrainer import GuiTrainer
from Methods.Base.utils import pre_training_callback, training_callback, post_training_callback
from Methods.GaussianSplatting.Loss import GaussianSplattingLoss
from Optim.Samplers.DatasetSamplers import DatasetSampler


@Framework.Configurable.configure(
    NUM_ITERATIONS=30_000,
    LEARNING_RATE_POSITION_INIT=0.00016,
    LEARNING_RATE_POSITION_FINAL=0.0000016,
    LEARNING_RATE_POSITION_MAX_STEPS=30_000,
    LEARNING_RATE_FEATURE=0.0025,
    LEARNING_RATE_OPACITY=0.025,  # the 3dgs authors used 0.05 for the results in the paper
    LEARNING_RATE_SCALING=0.005,
    LEARNING_RATE_ROTATION=0.001,
    PERCENT_DENSE=0.01,
    OPACITY_RESET_INTERVAL=3_000,
    DENSIFY_START_ITERATION=500,
    DENSIFY_END_ITERATION=15_000,
    DENSIFICATION_INTERVAL=100,
    DENSIFY_GRAD_THRESHOLD=0.0002,
    LOSS=Framework.ConfigParameterList(
        LAMBDA_L1=0.8,
        LAMBDA_DSSIM=0.2,
    ),
)
class GaussianSplattingTrainer(GuiTrainer):
    """Defines the trainer for the GaussianSplatting variant."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.train_sampler = None
        self.loss = GaussianSplattingLoss(loss_config=self.LOSS)

    @pre_training_callback(priority=50)
    @torch.no_grad()
    def create_sampler(self, _, dataset: 'BaseDataset') -> None:
        """Creates the sampler."""
        self.train_sampler = DatasetSampler(dataset=dataset.train(), random=True)

    @pre_training_callback(priority=40)
    @torch.no_grad()
    def setup_gaussians(self, _, dataset: 'BaseDataset') -> None:
        """Sets up the model."""
        camera_centers = torch.stack([view.position for view in dataset.train()])
        radius = (1.1 * torch.max(torch.linalg.norm(camera_centers - torch.mean(camera_centers, dim=0), dim=1))).item()
        Logger.log_info(f'Training cameras extent: {radius:.2f}')

        if dataset.point_cloud is not None:
            point_cloud = dataset.point_cloud
        else:
            n_random_points = 100_000
            min_bounds, max_bounds = dataset.bounding_box.min_max
            extent = max_bounds - min_bounds
            point_cloud = BasicPointCloud(torch.rand(n_random_points, 3, dtype=torch.float32, device=min_bounds.device) * extent + min_bounds)
        self.model.gaussians.initialize_from_point_cloud(point_cloud, radius)
        self.model.gaussians.training_setup(self)

    @training_callback(priority=110, start_iteration=1000, iteration_stride=1000)
    @torch.no_grad()
    def increase_sh_degree(self, *_) -> None:
        """Increase the number of used SH coefficients up to a maximum degree."""
        self.model.gaussians.increase_used_sh_degree()

    @training_callback(priority=100)
    def training_iteration(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Performs a training step without actually doing the optimizer step."""
        # init modes
        self.model.train()
        dataset.train()
        self.loss.train()
        # update learning rate
        self.model.gaussians.update_learning_rate(iteration + 1)
        # get random view
        view = self.train_sampler.get(dataset=dataset)['view']
        # render
        outputs = self.renderer.render_image_training(view=view)
        # calculate loss
        # compose gt with background color if needed  # FIXME: integrate into data model
        rgb_gt = view.rgb
        if (alpha_gt := view.alpha) is not None:
            rgb_gt = apply_background_color(rgb_gt, alpha_gt, view.camera.background_color)
        loss = self.loss(outputs['rgb'], rgb_gt)
        loss.backward()
        # track values for pruning and densification
        if iteration < self.DENSIFY_END_ITERATION:
            self.model.gaussians.add_densification_stats(outputs['viewspace_points'], outputs['visibility_mask'])

    @training_callback(priority=90, start_iteration='DENSIFY_START_ITERATION', end_iteration='DENSIFY_END_ITERATION', iteration_stride='DENSIFICATION_INTERVAL')
    @torch.no_grad()
    def densify(self, iteration: int, _) -> None:
        """Apply densification."""
        if iteration == self.DENSIFY_START_ITERATION or iteration == self.DENSIFY_END_ITERATION:  # matches behavior of official 3dgs codebase
            return
        self.model.gaussians.densify_and_prune(self.DENSIFY_GRAD_THRESHOLD, 0.005, iteration > self.OPACITY_RESET_INTERVAL)

    @training_callback(priority=80, start_iteration='OPACITY_RESET_INTERVAL', end_iteration='DENSIFY_END_ITERATION', iteration_stride='OPACITY_RESET_INTERVAL')
    @torch.no_grad()
    def reset_opacities(self, iteration: int, _) -> None:
        """Reset opacities."""
        if iteration == self.DENSIFY_END_ITERATION:  # matches behavior of official 3dgs codebase
            return
        self.model.gaussians.reset_opacities()

    @training_callback(priority=80, start_iteration='DENSIFY_START_ITERATION', iteration_stride='NUM_ITERATIONS')
    @torch.no_grad()
    def reset_opacities_white_background(self, _, dataset: 'BaseDataset') -> None:
        """Reset opacities one additional time when using a white background."""
        # original implementation only supports black or white background, this is an attempt to make it work with any color
        if (dataset.default_camera.background_color > 0.0).any():
            self.model.gaussians.reset_opacities()

    @training_callback(priority=70)
    @torch.no_grad()
    def perform_optimizer_step(self, *_) -> None:
        """Update parameters."""
        self.model.gaussians.optimizer.step()
        self.model.gaussians.optimizer.zero_grad()

    @training_callback(active='WANDB.ACTIVATE', priority=10, iteration_stride='WANDB.INTERVAL')
    @torch.no_grad()
    def log_wandb(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Adds primitive count to default Weights & Biases logging."""
        Framework.wandb.log({
            'n_primitives': self.model.gaussians.get_positions.shape[0]
        }, step=iteration)
        # default logging
        super().log_wandb(iteration, dataset)

    @post_training_callback(priority=1000)
    @torch.no_grad()
    def bake_activations(self, *_) -> None:
        """Bake relevant activation functions after training."""
        self.model.gaussians.bake_activations()
        Logger.log_info(f'final number of primitives: {self.model.gaussians.get_positions.shape[0]:,}')
        # delete optimizer to save memory
        self.model.gaussians.optimizer = None
