# -- coding: utf-8 --

"""Base/Trainer.py: Implementation of the basic trainer for models."""

import math
from pathlib import Path
from random import randrange
from statistics import mean

from operator import attrgetter
from typing import IO, Callable, Generator
import datetime
import shutil
import inspect

import torch
import torchmetrics
import torchvision

import Framework
from Logging import Logger
from Datasets.Base import BaseDataset
from Methods.Base.Model import BaseModel
from Methods.Base.Renderer import BaseRenderer
from Methods.Base.utils import CallbackTimer, postTrainingCallback, \
                               trainingCallback, getGitCommit
from Optim.Losses.Base import BaseLoss


@Framework.Configurable.configure(
    LOAD_CHECKPOINT=None,
    MODEL_NAME='Default',
    NUM_ITERATIONS=1,
    ACTIVATE_TIMING=False,
    RUN_VALIDATION=False,
    BACKUP=Framework.ConfigParameterList(
        FINAL_CHECKPOINT=True,
        RENDER_TESTSET=True,
        RENDER_TRAINSET=False,
        RENDER_VALSET=False,
        VISUALIZE_ERRORS=False,
        INTERVAL=-1,
        TRAINING_STATE=False
    ),
    WANDB=Framework.ConfigParameterList(
        ACTIVATE=False,
        ENTITY=None,
        PROJECT='nerficg',
        LOG_IMAGES=True,
        INDEX_VALIDATION=-1,
        INDEX_TRAINING=-1,
        INTERVAL=1000,
        SWEEP_MODE=Framework.ConfigParameterList(
            ACTIVE=False,
            START_ITERATION=999,  # should always be set to ITERATION_STRIDE - 1 so last training iteration is logged
            ITERATION_STRIDE=1000,
        ),
    )
)
class BaseTrainer(Framework.Configurable, torch.nn.Module):
    """Defines the basic trainer used to train a model."""

    def __init__(self, model: BaseModel, renderer: BaseRenderer) -> None:
        Framework.Configurable.__init__(self, 'TRAINING')
        torch.nn.Module.__init__(self)
        self.model: BaseModel = model
        self.renderer: BaseRenderer = renderer
        # check if training code was committed
        self.model.git_commit = getGitCommit()
        # setup training logging
        if self.WANDB.ACTIVATE:
            self.WANDB.ACTIVATE = Framework.setupWandb(project=self.WANDB.PROJECT, entity=self.WANDB.ENTITY, name=self.model.model_name)
        # create output and checkpoint directory
        self.output_directory = model.output_directory
        self.checkpoint_directory = self.output_directory / 'checkpoints'
        self.loss: BaseLoss = BaseLoss()
        Logger.logInfo(f'creating output directory: {self.output_directory}')
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Framework.config._path, str(self.output_directory / 'training_config.yaml'))

    @classmethod
    def load(cls, checkpoint_name: str) -> 'BaseTrainer':
        """Loads a saved training checkpoint from a '.train' file."""
        if checkpoint_name is None or checkpoint_name.split('.')[-1] != 'train':
            raise Framework.CheckpointError(f'Invalid checkpoint name "{checkpoint_name}"')
        try:
            checkpoint_path = Path(__file__).resolve().parents[3]
            training_instance = torch.load(checkpoint_path / checkpoint_name)
        except IOError as e:
            raise Framework.CheckpointError(f'Failed to load checkpoint "{e}"')
        return training_instance

    def save(self, path: Path) -> None:
        """Saves the current model in a '.train' file at the given path."""
        try:
            torch.save(self, path)
        except IOError as e:
            raise Framework.CheckpointError(f'Failed to save checkpoint "{e}"')

    def renderDataset(self, dataset: BaseDataset, verbose: bool = True):
        self.model.eval()
        if self.BACKUP.RENDER_TESTSET:
            self.renderer.renderSubset(self.output_directory, dataset.test(), calculate_metrics=True, visualize_errors=self.BACKUP.VISUALIZE_ERRORS, verbose=verbose)
        if self.BACKUP.RENDER_TRAINSET:
            self.renderer.renderSubset(self.output_directory, dataset.train(), calculate_metrics=False, visualize_errors=False, verbose=verbose)
        if self.BACKUP.RENDER_VALSET:
            self.renderer.renderSubset(self.output_directory, dataset.eval(), calculate_metrics=False, visualize_errors=False, verbose=verbose)

    @trainingCallback(priority=1, start_iteration='BACKUP.INTERVAL', iteration_stride='BACKUP.INTERVAL')
    def saveIntermediateCheckpoint(self, iteration: int, dataset: BaseDataset) -> None:
        """Creates an intermediate checkpoint at the current iteration."""
        self.model.save(self.checkpoint_directory / f'{iteration:07d}.pt')
        if self.BACKUP.TRAINING_STATE:
            self.save(self.checkpoint_directory / f'{iteration:07d}.train')
        self.renderDataset(dataset=dataset, verbose=False)

    @postTrainingCallback(active='BACKUP.FINAL_CHECKPOINT', priority=1)
    def saveFinalCheckpoints(self, _, dataset: BaseDataset) -> None:
        """Creates a final checkpoint before exiting the training loop."""
        Logger.logInfo('creating final model and training checkpoints')
        self.model.save(self.checkpoint_directory / 'final.pt')
        if self.BACKUP.TRAINING_STATE:
            self.save(self.checkpoint_directory / 'final.train')
        self.renderDataset(dataset=dataset)

    def logTiming(self, dataset: BaseDataset) -> None:
        """Writes runtimes to file if activated."""
        if self.ACTIVATE_TIMING:
            Logger.logInfo('writing timings')
            training_time = 0

            def addItem(f: IO[str], name: str, values: tuple[float], training_time: float) -> float:
                f.write(f'{name}:\n\t'
                        f'Total execution time: {datetime.timedelta(seconds=round(values[0]))}\n\t'
                        f'Time per iteration [ms]: {values[1] * 1000:.2f}\n\t'
                        f'Number of iterations: {values[2]}\n\n'
                        )
                return training_time + datetime.timedelta(seconds=values[0]).total_seconds()

            with open(str(self.output_directory / 'timings.txt'), 'w') as f:
                if dataset.load_timer is not None:
                    training_time = addItem(f, 'Dataset loading', dataset.load_timer.getValues(), training_time)
                for callback in self.listCallbacks():
                    training_time = addItem(f, callback.__name__, callback.timer.getValues(), training_time)
                f.write(f'Time:{training_time}')

    def run(self, dataset: 'BaseDataset') -> None:
        """Trains the model for the specified amount of iterations executing all callbacks along the way."""
        Logger.log(f'starting training for model: {self.model.model_name}')
        # main training loop
        starting_iteration = iteration = self.model.num_iterations_trained
        if starting_iteration <= 0:
            for callback in self.gatherCallbacks(-1):
                with callback.timer:
                    callback(self, starting_iteration, dataset)
        try:
            training_callbacks = self.gatherCallbacks(0)
            for iteration in Logger.logProgressBar(range(starting_iteration, self.NUM_ITERATIONS), desc='training', miniters=10):
                for callback in training_callbacks:
                    if (
                        (callback.start_iteration is not None and iteration < callback.start_iteration) or
                        (callback.end_iteration is not None and iteration > callback.end_iteration) or
                        (callback.iteration_stride is not None and (iteration - (callback.start_iteration or 0)) % callback.iteration_stride != 0)
                    ):
                        continue
                    with callback.timer:
                        callback(self, iteration, dataset)
                self.model.num_iterations_trained += 1
        except KeyboardInterrupt:
            Logger.logWarning('training manually interrupted')
        for callback in self.gatherCallbacks(1):
            with callback.timer:
                callback(self, iteration + 1, dataset)
        self.logTiming(dataset=dataset)
        Logger.log('training finished successfully')

    def gatherCallbacks(self, callback_type: int) -> list[Callable]:
        """Returns a list of all training callback functions of the requested type, and replaces config strings with values"""
        all_callbacks = self.listCallbacks()
        requested_callbacks = []
        for callback in all_callbacks:
            if not callback.callback_type == callback_type:
                continue
            for attr in ['active', 'start_iteration', 'end_iteration', 'iteration_stride']:
                try:
                    attr_value = getattr(callback, attr)
                    if isinstance(attr_value, str):
                        setattr(callback, attr, attrgetter(attr_value)(self))
                except AttributeError:
                    raise Framework.TrainingError(f'invalid config parameter for callback function "{callback.__name__}" field "{attr}":  \
                                        Class {self.__class__.__name__} has no config parameter {getattr(callback, attr)}')
            if self.ACTIVATE_TIMING and not isinstance(callback, CallbackTimer):
                callback.timer = CallbackTimer()
            if callback.iteration_stride is not None and callback.iteration_stride <= 0:
                callback.active = False
            if callback.active:
                requested_callbacks.append(callback)
        requested_callbacks.sort(key=lambda c: c.priority, reverse=True)
        return requested_callbacks

    def listCallbacks(self) -> Generator[Callable, None, None]:
        """Returns all registered training callback functions"""
        for member in inspect.getmembers(self.__class__, predicate=inspect.isfunction):
            if hasattr(member[1], 'callback_type'):
                yield member[1]

    @trainingCallback(active='WANDB.ACTIVATE', priority=500, iteration_stride='WANDB.INTERVAL')
    @torch.no_grad()
    def logWandB(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Logs all losses and visualizes training and validation samples using Weights & Biases."""
        # log losses
        self.loss.log(iteration, self.RUN_VALIDATION)
        # reset loss accumulation
        self.loss.reset()
        # visualize samples
        self.model.eval()
        if self.WANDB.LOG_IMAGES:
            for mode, index, name in zip([dataset.train, dataset.eval],
                                         [self.WANDB.INDEX_TRAINING, self.WANDB.INDEX_VALIDATION],
                                         ['training', 'validation']):
                mode()
                if len(dataset) > 0:
                    if index < 0:
                        index: int = randrange(len(dataset))
                    sample = dataset[index]
                    dataset.camera.setProperties(sample)
                    outputs = self.renderer.renderImage(
                        camera=dataset.camera,
                        to_chw=True,
                    )
                    outputs_color = self.renderer.pseudoColorOutputs(outputs, dataset.camera, dataset, index)
                    outputs_color.update(self.renderer.pseudoColorGT(dataset.camera, dataset, index))
                    outputs_color = dict(sorted(outputs_color.items(), reverse=True))
                    image: torch.Tensor = torchvision.utils.make_grid(
                        tensor=list(outputs_color.values()),
                        nrow=len(outputs_color),
                        padding=2,  # default
                        normalize=False,  # default
                        value_range=None,  # default
                        scale_each=False,  # default
                        pad_value=0.0,  # default
                    )
                    Framework.wandb.log(
                        data={name: Framework.wandb.Image(image, caption=' | '.join(outputs_color.keys()))},
                        step=iteration
                    )

    @trainingCallback(active='WANDB.ACTIVATE', priority=1, iteration_stride='WANDB.INTERVAL')
    def commitWandB(self, *_) -> None:
        """Commits current data log dict to Weights & Biases server."""
        Framework.wandb.log(data={}, commit=True)

    @trainingCallback(active='WANDB.SWEEP_MODE.ACTIVE', priority=2, start_iteration='WANDB.SWEEP_MODE.START_ITERATION', iteration_stride='WANDB.SWEEP_MODE.ITERATION_STRIDE')
    @torch.no_grad()
    def logTestMetricsWandB(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Computes and logs image quality metrics on the test set (used for hyperparameter sweeps)."""
        # init modes
        self.model.eval()
        dataset.test()
        self.loss.eval()
        # clear grads to free some memory in case we're running low
        self.optimizer.zero_grad()
        # render all test images
        psnrs = []
        ssims = []
        lpips = []
        for camera_properties in dataset:
            dataset.camera.setProperties(camera_properties)
            result = self.renderer.renderImage(camera=dataset.camera, to_chw=True)['rgb'][None].clamp_(0.0, 1.0)
            target = camera_properties.rgb[None]
            psnrs.append(torchmetrics.functional.image.peak_signal_noise_ratio(result, target, data_range=1.0).item())
            ssims.append(torchmetrics.functional.image.structural_similarity_index_measure(result, target, data_range=1.0).item())
            lpips.append(torchmetrics.functional.image.learned_perceptual_image_patch_similarity(result, target, normalize=True, net_type='vgg').item())
        # calculate mean metrics as well as their geometric mean
        psnr = mean(psnrs)
        ssim = mean(ssims)
        lpips = mean(lpips)
        # geometric mean of psnr, ssim, and lpips (as proposed in mipnerf)
        combined_metrics = math.exp(mean([-0.1 * math.log(10.0) * psnr, math.log(math.sqrt(1.0 - ssim)), math.log(lpips)]))
        # log average metrics
        Framework.wandb.log({
            'test_psnr': psnr,
            'test_ssim': ssim,
            'test_lpips': lpips,
            'combined_metrics': combined_metrics
        }, step=iteration, commit=True)
