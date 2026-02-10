"""Base/Trainer.py: Implementation of the basic trainer for models."""

import os
import math
from pathlib import Path
from random import randrange
from statistics import mean
import random

from operator import attrgetter
from typing import IO, Callable, Generator
import datetime
import shutil
import inspect

import torch
import torch.multiprocessing as mp
import torchmetrics
import torchvision

import Framework
from Datasets.utils import apply_background_color, ImageData
from Logging import Logger
from Datasets.Base import BaseDataset
from Methods.Base.Model import BaseModel
from Methods.Base.Renderer import BaseRenderer
from Methods.Base.utils import CallbackTimer, pre_training_callback, training_callback, post_training_callback
from Optim.Losses.Base import BaseLoss


@Framework.Configurable.configure(
    LOAD_CHECKPOINT=None,
    MODEL_NAME='Default',
    NUM_ITERATIONS=1,
    RUN_VALIDATION=False,
    DATA=Framework.ConfigParameterList(
        PRELOADING_LEVEL=1,  # 0: disk, 1: RAM, 2: VRAM
        FIELDS=[],  # ImageData fields to preload from dataset if available (default: all)
        PRECOMPUTE_RAYS=False,
        RAYS_TO_DEVICE=True,
    ),
    BACKUP=Framework.ConfigParameterList(
        FINAL_CHECKPOINT=True,
        RENDER_TESTSET=True,
        RENDER_TRAINSET=False,
        RENDER_VALSET=False,
        INTERMEDIATE_RENDERINGS=True,
        VISUALIZE_ERRORS=False,
        INTERVAL=-1,
        TRAINING_STATE=False
    ),
    TIMING=Framework.ConfigParameterList(
        ACTIVATE=False,
        INCLUDE_DATALOADING_IN_TOTAL=True,
        INCLUDE_PRETRAINING_IN_TOTAL=True,
        INCLUDE_POSTTRAINING_IN_TOTAL=False,
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
            NUM_IMAGES=-1,  # number of randomly selected test set images for metric computation (default: all)
        ),
    ),
    WRITE_VRAM_STATS=False,
)
class BaseTrainer(Framework.Configurable, torch.nn.Module):
    """Defines the basic trainer used to train a model."""

    def __init__(self, model: BaseModel, renderer: BaseRenderer) -> None:
        Framework.Configurable.__init__(self, 'TRAINING')
        torch.nn.Module.__init__(self)
        self.model = model
        self.renderer = renderer
        # setup training logging
        if self.WANDB.ACTIVATE:
            self.WANDB.ACTIVATE = Framework.setup_wandb(project=self.WANDB.PROJECT, entity=self.WANDB.ENTITY, name=self.model.model_name)
        # create output and checkpoint directory
        self.output_directory = model.output_directory
        self.checkpoint_directory = self.output_directory / 'checkpoints'
        self.loss = BaseLoss()
        Logger.log_info(f'creating output directory: {self.output_directory}')
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Framework.config.path, self.output_directory / 'training_config.yaml')

    @classmethod
    def load(cls, checkpoint_name: str) -> 'BaseTrainer':
        """Loads a saved training checkpoint from a '.train' file."""
        if checkpoint_name is None or checkpoint_name.split('.')[-1] != 'train':
            raise Framework.CheckpointError(f'Invalid checkpoint name "{checkpoint_name}"')
        try:
            checkpoint_path = Framework.Directories.NERFICG_ROOT
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

    def _render_dataset(self, dataset: BaseDataset, verbose: bool = True):
        self.model.eval()
        if self.BACKUP.RENDER_TESTSET:
            self.renderer.render_subset(self.output_directory, dataset.test(), calculate_metrics=True, visualize_errors=self.BACKUP.VISUALIZE_ERRORS, verbose=verbose)
        if self.BACKUP.RENDER_TRAINSET:
            self.renderer.render_subset(self.output_directory, dataset.train(), calculate_metrics=False, visualize_errors=False, verbose=verbose)
        if self.BACKUP.RENDER_VALSET:
            self.renderer.render_subset(self.output_directory, dataset.eval(), calculate_metrics=False, visualize_errors=False, verbose=verbose)

    @pre_training_callback(priority=5000)
    def _prepare_dataset(self, _, dataset: BaseDataset) -> None:
        """Prepares the dataset for training."""
        if self.DATA.PRELOADING_LEVEL not in [0, 1, 2]:
            Logger.log_warning(
                f'preloading level {self.DATA.PRELOADING_LEVEL} invalid, defaulting to 1\n'
                f'\tavailable options are 0 for no preloading, 1 for CPU memory (RAM), or 2 for GPU memory (VRAM)'
            )
            self.DATA.PRELOADING_LEVEL = 1
        if self.DATA.PRELOADING_LEVEL > 0:
            load_everything = not self.DATA.FIELDS or len(self.DATA.FIELDS) == 0
            to_default_device = self.DATA.PRELOADING_LEVEL == 2
            Logger.log_info(f'preloading training data into {"V" if to_default_device else ""}RAM')
            dataset.train()
            # gather all load tasks and callbacks
            tasks, callbacks = [], []
            for view in dataset:
                for image_data in view.available_image_data:
                    if load_everything or image_data in self.DATA.FIELDS:
                        task, callback = view.get_parallel_load_helpers(image_data)
                        tasks.append(task)
                        callbacks.append(callback)
            if len(tasks) > 0:
                # load in parallel using half of available CPU cores (at least 1)
                n_processes = max(1, (os.cpu_count() or 1) // 2)
                with mp.Pool(min(n_processes, len(tasks))) as pool:
                    iterator = pool.imap(ImageData.load_from_worker, tasks, chunksize=1)
                    results = [
                        result.clone()  # detach from shared memory
                        for result in Logger.log_progress(iterator, total=len(tasks), desc='fetch data', leave=False)
                    ]
                # store loaded data via callbacks
                for result, callback in Logger.log_progress(zip(results, callbacks, strict=True), total=len(results), desc='store data', leave=False):
                    callback(result, to_default_device)
        if self.DATA.PRECOMPUTE_RAYS:
            Logger.log_info('precomputing rays for training views')
            dataset.precompute_rays('train', not self.DATA.RAYS_TO_DEVICE)
            if self.RUN_VALIDATION and len(dataset.data['val']) > 0:
                Logger.log_info('precomputing rays for validation views')
                dataset.precompute_rays('val', not self.DATA.RAYS_TO_DEVICE)

    @training_callback(priority=1, start_iteration='BACKUP.INTERVAL', iteration_stride='BACKUP.INTERVAL')
    def _save_intermediate_checkpoint(self, iteration: int, dataset: BaseDataset) -> None:
        """Creates an intermediate checkpoint at the current iteration."""
        Logger.log_info(f'creating intermediate model and training checkpoint at iteration {iteration}')
        self.model.save(self.checkpoint_directory / f'{iteration:07d}.pt')
        if self.BACKUP.TRAINING_STATE:
            self.save(self.checkpoint_directory / f'{iteration:07d}.train')
        if self.BACKUP.INTERMEDIATE_RENDERINGS:
            self._render_dataset(dataset=dataset, verbose=True)

    @post_training_callback(active='BACKUP.FINAL_CHECKPOINT', priority=1)
    def _save_final_checkpoint(self, _, dataset: BaseDataset) -> None:
        """Creates a final checkpoint before exiting the training loop."""
        Logger.log_info('creating final model and training checkpoint')
        self.model.save(self.checkpoint_directory / 'final.pt')
        if self.BACKUP.TRAINING_STATE:
            self.save(self.checkpoint_directory / 'final.train')
        self._render_dataset(dataset=dataset)

    def _write_timings(self, dataset: BaseDataset) -> None:
        """Writes runtimes to file if activated."""
        Logger.log_info('writing timings')
        training_time = 0.0

        def add_item(output_file: IO[str], name: str, values: tuple[float, float, int]) -> float:
            output_file.write(
                f'{name}:\n'
                f'\tTotal execution time: {datetime.timedelta(seconds=round(values[0]))}\n'
                f'\tTime per iteration [ms]: {values[1] * 1000:.2f}\n'
                f'\tNumber of iterations: {values[2]}\n\n'
            )
            return datetime.timedelta(seconds=values[0]).total_seconds()

        with open(str(self.output_directory / 'timings.txt'), 'w') as timings_file:
            dataloading_time = add_item(timings_file, 'Dataset loading', dataset.load_timer.get_values())
            if self.TIMING.INCLUDE_DATALOADING_IN_TOTAL:
                training_time += dataloading_time
            for callback in self._list_callbacks():
                callback_time = add_item(timings_file, callback.__name__, callback.timer.get_values())
                if not self.TIMING.INCLUDE_PRETRAINING_IN_TOTAL and callback.callback_type == -1:
                    continue
                if not self.TIMING.INCLUDE_POSTTRAINING_IN_TOTAL and callback.callback_type == 1:
                    continue
                training_time += callback_time
            timings_file.write(f'Time:{training_time}')

    def _log_vram_stats(self) -> None:
        """Logs VRAM statistics and writes them to file if activated."""
        peak_allocated_bytes = torch.cuda.max_memory_allocated()
        peak_reserved_bytes = torch.cuda.max_memory_reserved()
        peak_allocated = peak_allocated_bytes / 1024 ** 3
        peak_reserved = peak_reserved_bytes / 1024 ** 3
        Logger.log_info(f'peak VRAM usage during training: {peak_allocated:.2f} GiB allocated ({peak_reserved:.2f} GiB reserved)')
        if self.WRITE_VRAM_STATS:
            with open(str(self.output_directory / 'vram_stats.txt'), 'w') as vram_file:
                vram_file.write(
                    f'Peak VRAM usage:\n'
                    f'\tallocated: {peak_allocated:.2f} GiB\n'
                    f'\treserved: {peak_reserved:.2f} GiB\n\n'
                )
                vram_file.write(f'VRAM_allocated:{peak_allocated_bytes} VRAM_reserved:{peak_reserved_bytes}')

    def run(self, dataset: 'BaseDataset') -> None:
        """Trains the model for the specified amount of iterations executing all callbacks along the way."""
        Logger.log(f'starting training for model: {self.model.model_name}')
        # run pre-training callbacks when starting from scratch
        starting_iteration = iteration = self.model.num_iterations_trained
        if starting_iteration <= 0:
            for callback in self._gather_callbacks(-1):
                with callback.timer:
                    callback(self, starting_iteration, dataset)
        # main training loop
        try:
            training_callbacks = self._gather_callbacks(0)
            for iteration in Logger.log_progress(range(starting_iteration, self.NUM_ITERATIONS), desc='training', miniters=10):
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
            Logger.log_warning('training manually interrupted')
        # log VRAM stats
        self._log_vram_stats()
        # run post-training callbacks
        for callback in self._gather_callbacks(1):
            with callback.timer:
                callback(self, iteration + 1, dataset)
        # write timings
        if self.TIMING.ACTIVATE:
            self._write_timings(dataset=dataset)
        Logger.log('training finished successfully')

    def _gather_callbacks(self, callback_type: int) -> list[Callable]:
        """Returns a list of all training callback functions of the requested type, and replaces config strings with values"""
        all_callbacks = self._list_callbacks()
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
                    raise Framework.TrainingError(
                        f'invalid config parameter for callback function "{callback.__name__}" field "{attr}":'
                        f'Class {self.__class__.__name__} has no config parameter {getattr(callback, attr)}'
                    )
            if self.TIMING.ACTIVATE and not isinstance(callback, CallbackTimer):
                callback.timer = CallbackTimer()
            if callback.iteration_stride is not None and callback.iteration_stride <= 0:
                callback.active = False
            if callback.active:
                requested_callbacks.append(callback)
        requested_callbacks.sort(key=lambda c: c.priority, reverse=True)
        return requested_callbacks

    def _list_callbacks(self) -> Generator[Callable, None, None]:
        """Returns all registered training callback functions"""
        for member in inspect.getmembers(self.__class__, predicate=inspect.isfunction):
            if hasattr(member[1], 'callback_type'):
                yield member[1]

    def _update_callback(self, name: str, **kwargs) -> None:
        """Updates a callback function during training."""
        for member in inspect.getmembers(self.__class__, predicate=inspect.isfunction):
            if not hasattr(member[1], 'callback_type'):
                continue
            if member[0] != name:
                continue
            callback = member[1]
            for key, value in kwargs.items():
                if hasattr(callback, key):
                    setattr(callback, key, value)
                else:
                    raise Framework.TrainingError(f'Callback function "{name}" has no attribute "{key}"')
            return

    @training_callback(active='WANDB.ACTIVATE', priority=500, iteration_stride='WANDB.INTERVAL')
    @torch.no_grad()
    def log_wandb(self, iteration: int, dataset: 'BaseDataset') -> None:
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
                    view = dataset[index]
                    outputs = self.renderer.render_image(
                        view=view,
                        to_chw=True,
                    )
                    outputs_color = self.renderer.postprocess_outputs(outputs, view, dataset, index)
                    outputs_color.update(self.renderer.postprocess_reference_data(view, dataset, index))
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

    @training_callback(active='WANDB.ACTIVATE', priority=1, iteration_stride='WANDB.INTERVAL')
    def _commit_wandb(self, *_) -> None:
        """Commits current data log dict to Weights & Biases server."""
        Framework.wandb.log(data={}, commit=True)

    @training_callback(active='WANDB.SWEEP_MODE.ACTIVE', priority=2, start_iteration='WANDB.SWEEP_MODE.START_ITERATION', iteration_stride='WANDB.SWEEP_MODE.ITERATION_STRIDE')
    @torch.no_grad()
    def _log_test_metrics_wandb(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Computes and logs image quality metrics on the test set (used for hyperparameter sweeps)."""
        if not self.WANDB.ACTIVATE:
            Logger.log_warning('Sweep mode requires wandb logging to be enabled. Skipping test metrics logging.')
            return
        # init modes
        self.model.eval()
        dataset.test()
        num_images = len(dataset)
        subset_iterable = range(num_images)
        # optionally select a random subset
        if self.WANDB.SWEEP_MODE.NUM_IMAGES > 0 and self.WANDB.SWEEP_MODE.NUM_IMAGES < num_images:
            subset_iterable = random.sample(subset_iterable, k=self.WANDB.SWEEP_MODE.NUM_IMAGES)
        # render images
        psnrs = []
        ssims = []
        lpips = []
        for i in Logger.log_progress(subset_iterable, desc='computing test metrics', leave=False):
            view = dataset[i]
            result = self.renderer.render_image(view=view, to_chw=True)['rgb'][None]
            # compose gt with background color if needed  # FIXME: integrate into data model
            color_gt = view.rgb
            if (alpha_gt := view.alpha) is not None:
                color_gt = apply_background_color(color_gt, alpha_gt, view.camera.background_color)
            target = color_gt[None]
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
