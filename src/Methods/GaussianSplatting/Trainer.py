# -- coding: utf-8 --

"""GaussianSplatting/Trainer.py: Implementation of the trainer for the GaussianSplatting method.
Callback schedule is slightly modified from the original implementation. They count iterations starting at 1 instead of 0."""

import torch

import Framework
from Datasets.Base import BaseDataset
from Datasets.utils import BasicPointCloud
from Logging import Logger
from Methods.Base.GuiTrainer import GuiTrainer
from Methods.Base.utils import preTrainingCallback, trainingCallback, postTrainingCallback
from Methods.GaussianSplatting.Loss import GaussianSplattingLoss
from Optim.Samplers.DatasetSamplers import DatasetSampler


@Framework.Configurable.configure(
    NUM_ITERATIONS=30_000,
    LEARNING_RATE_POSITION_INIT=0.00016,
    LEARNING_RATE_POSITION_FINAL=0.0000016,
    LEARNING_RATE_POSITION_DELAY_MULT=0.01,
    LEARNING_RATE_POSITION_MAX_STEPS=30_000,
    LEARNING_RATE_FEATURE=0.0025,
    LEARNING_RATE_OPACITY=0.05,
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
        super(GaussianSplattingTrainer, self).__init__(**kwargs)
        self.train_sampler = None
        self.loss = GaussianSplattingLoss(loss_config=self.LOSS)

    @preTrainingCallback(priority=50)
    @torch.no_grad()
    def createSampler(self, _, dataset: 'BaseDataset') -> None:
        """Creates the sampler."""
        self.train_sampler = DatasetSampler(dataset=dataset.train(), random=True)

    @preTrainingCallback(priority=40)
    @torch.no_grad()
    def setupGaussians(self, _, dataset: 'BaseDataset') -> None:
        """Sets up the model."""
        camera_centers = torch.stack([camera_properties.T for camera_properties in dataset.train()])
        radius = (1.1 * torch.max(torch.linalg.norm(camera_centers - torch.mean(camera_centers, dim=0), dim=1))).item()
        Logger.logInfo(f'Training cameras extent: {radius:.2f}')

        if dataset.point_cloud is not None:
            point_cloud = dataset.point_cloud
        else:
            n_random_points = 100_000
            min_bounds, max_bounds = dataset.getBoundingBox()
            extent = max_bounds - min_bounds
            point_cloud = BasicPointCloud(torch.rand(n_random_points, 3, dtype=torch.float32, device=min_bounds.device) * extent + min_bounds)
        self.model.gaussians.initialize_from_point_cloud(point_cloud, radius)
        self.model.gaussians.training_setup(self)

    @trainingCallback(priority=110, start_iteration=1000, iteration_stride=1000)
    @torch.no_grad()
    def increaseSHDegree(self, *_) -> None:
        """Increase the levels of SH up to a maximum degree."""
        self.model.gaussians.increase_used_sh_degree()

    @trainingCallback(priority=100)
    def trainingIteration(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Performs a training step without actually doing the optimizer step."""
        # init modes
        self.model.train()
        dataset.train()
        self.loss.train()
        # update learning rate
        self.model.gaussians.update_learning_rate(iteration + 1)
        # get random sample from dataset
        camera_properties = self.train_sampler.get(dataset=dataset)['camera_properties']
        dataset.camera.setProperties(camera_properties)
        # render sample
        outputs = self.renderer.renderImage(camera=dataset.camera, to_chw=True)
        # calculate loss
        loss = self.loss(outputs, camera_properties)
        loss.backward()
        # track values for pruning and densification
        if iteration < self.DENSIFY_END_ITERATION:
            self.model.gaussians.add_densification_stats(outputs['viewspace_points'], outputs['visibility_mask'])

    @trainingCallback(priority=90, start_iteration='DENSIFY_START_ITERATION', end_iteration='DENSIFY_END_ITERATION', iteration_stride='DENSIFICATION_INTERVAL')
    @torch.no_grad()
    def densify(self, iteration: int, _) -> None:
        """Apply densification."""
        if iteration == self.DENSIFY_START_ITERATION or iteration == self.DENSIFY_END_ITERATION:  # matches behavior of official 3dgs codebase
            return
        self.model.gaussians.densify_and_prune(self.DENSIFY_GRAD_THRESHOLD, 0.005, iteration > self.OPACITY_RESET_INTERVAL)

    @trainingCallback(priority=80, start_iteration='OPACITY_RESET_INTERVAL', end_iteration='DENSIFY_END_ITERATION', iteration_stride='OPACITY_RESET_INTERVAL')
    @torch.no_grad()
    def resetOpacities(self, iteration: int, _) -> None:
        """Reset opacities."""
        if iteration == self.DENSIFY_END_ITERATION:  # matches behavior of official 3dgs codebase
            return
        self.model.gaussians.reset_opacities()

    @trainingCallback(priority=80, start_iteration='DENSIFY_START_ITERATION', iteration_stride='NUM_ITERATIONS')
    @torch.no_grad()
    def resetOpacitiesWhiteBackground(self, _, dataset: 'BaseDataset') -> None:
        """Reset opacities one additional time when using a white background."""
        # original implementation only supports black or white background, this is an attempt to make it work with any color
        if (dataset.camera.background_color > 0.0).any():
            self.model.gaussians.reset_opacities()

    @trainingCallback(priority=70)
    @torch.no_grad()
    def performOptimizerStep(self, *_) -> None:
        """Update parameters."""
        self.model.gaussians.optimizer.step()
        self.model.gaussians.optimizer.zero_grad()

    @postTrainingCallback(priority=1000)
    @torch.no_grad()
    def bakeActivations(self, *_) -> None:
        """Bake relevant activation functions after training."""
        self.model.gaussians.bake_activations()
        # delete optimizer to save memory
        self.model.gaussians.optimizer = None
