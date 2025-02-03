# -- coding: utf-8 --

"""Base/Renderer.py: Implementation of the basic renderer which processes the results of the models."""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from statistics import mean, median

import torch
import torchmetrics

import Framework
from Cameras.Base import BaseCamera
from Datasets.Base import BaseDataset
from Datasets.utils import saveImage, loadImagesParallel, list_sorted_files
from Logging import Logger
from Methods.Base.Model import BaseModel
from Visual.ColorMap import ColorMap
from Visual.utils import pseudoColorDepth, VideoWriter


class BaseRenderingComponent(ABC, torch.nn.Module):
    """Basic subcomponent of renderers used to parallelize the rendering procedure of sub-models."""

    def __init__(self) -> None:
        super().__init__()
        super(ABC, self).__init__()

    @classmethod
    def get(cls, *args) -> 'BaseRenderingComponent':
        """Returns an instance of the rendering component that includes support for multi-gpu execution if requested."""
        instance = cls(*args)
        # wrap in DataParallel if multiple GPUs are being used
        if Framework.config.GLOBAL.GPU_INDICES is not None and len(Framework.config.GLOBAL.GPU_INDICES) > 1:
            instance = torch.nn.DataParallel(
                module=instance, device_ids=Framework.config.GLOBAL.GPU_INDICES,
                output_device=Framework.config.GLOBAL.GPU_INDICES[0], dim=0
            )
        return instance

    @abstractmethod
    def forward(self, *args) -> None:
        """Implementations define forward passes."""
        pass


class BaseRenderer(Framework.Configurable, ABC):
    """Defines the basic renderer. Subclasses provide all functionality for their individual implementations."""

    def __init__(self, model: BaseModel, valid_model_types: list[type] = None) -> None:
        Framework.Configurable.__init__(self, 'RENDERER')
        ABC.__init__(self)
        # check if provided model is supported by this renderer
        if valid_model_types is not None and type(model) not in valid_model_types:
            Logger.logError(
                f'provided invalid model for renderer of type: "{type(self)}"'
                f'\n provided model type: "{type(model)}", valid options are: {valid_model_types}'
            )
            raise Framework.RendererError(f'provided invalid model for renderer of type: "{type(self)}"')
        # assign model
        self.model = model

    @abstractmethod
    def renderImage(self, camera: 'BaseCamera', to_chw: bool = False, benchmark: bool = False) -> dict[str, torch.Tensor | None]:
        """Renders model outputs for a given camera.

        Args:
            camera (BaseCamera): Camera object for rendering.
            to_chw (bool, optional): If set, returns outputs in shape chw instead of hwc. Defaults to False.
            benchmark (bool, optional): Indicates that renderer is called for benchmarking purposes. Defaults to False.

        Returns:
            dict[str, torch.Tensor | None]: Dictionary containing the rendered outputs.
            All tensors are expected to be of shape HxWxC or CxHxW, where c is either 1 or 3.
            All tensors are expected to be in the range [0, 1].
        """
        pass

    def pseudoColorOutputs(self, outputs: dict[str, torch.Tensor | None], camera: 'BaseCamera', dataset: BaseDataset, index: int) -> dict[str, torch.Tensor]:
        """Pseudo-colors the model outputs, returning tensors of shape 3xHxW."""
        outputs_color = {
            'rgb': outputs['rgb'].clamp_(0.0, 1.0),
            'alpha': outputs['alpha'].expand(outputs['rgb'].shape) if 'alpha' in outputs else torch.zeros_like(outputs['rgb']),
            'depth': pseudoColorDepth(
                    color_map='SPECTRAL',
                    depth=outputs['depth'],
                    near_far=None,
                    alpha=outputs['alpha'] if 'alpha' in outputs else None
                ) if 'depth' in outputs else torch.zeros_like(outputs['rgb']),
        }
        return outputs_color

    def pseudoColorGT(self, camera: 'BaseCamera', dataset: BaseDataset, index: int) -> dict[str, torch.Tensor]:
        """Pseudo-colors the gt labels relevant for this method, returning tensors of shape 3xHxW."""
        rgb_gt = camera.properties.rgb.clamp_(0.0, 1.0) if camera.properties.rgb is not None else \
            dataset.camera.background_color[:, None, None].expand(-1, camera.properties.height, camera.properties.width)
        alpha_gt = camera.properties.alpha.expand(rgb_gt.shape) if camera.properties.alpha is not None else torch.ones_like(rgb_gt)
        labels_color = {
            'rgb_gt': rgb_gt,
            'alpha_gt': alpha_gt,
        }
        return labels_color

    @torch.no_grad()
    def calculateImageQualityMetrics(self, results_path: Path, target_path: Path, output_path: Path, file_extension: str = 'png') -> None:
        """Calculate quality metrics (PSNR, SSIM, LPIPS)."""
        Logger.logInfo('calculating image quality metrics')
        try:
            targets = loadImagesParallel([
                str(target_path / name) for name in list_sorted_files(target_path)
                if file_extension in name
            ], scale_factor=None, num_threads=4, desc='loading gt')[0]
        except Exception:
            Logger.logWarning('failed to calculate quality metrics: no GT data available.')
            return
        results = loadImagesParallel([
            str(results_path / name) for name in list_sorted_files(results_path)
            if file_extension in name
        ], scale_factor=None, num_threads=4, desc='loading result')[0]
        for i in range(len(targets)):
            results[i] = results[i].float().to(Framework.config.GLOBAL.DEFAULT_DEVICE)
            targets[i] = targets[i].float().to(Framework.config.GLOBAL.DEFAULT_DEVICE)
        metrics = {
            'PSNR': {'metric': torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(Framework.config.GLOBAL.DEFAULT_DEVICE), 'values': [], 'num_decimals': 2},
            'SSIM': {'metric': torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(Framework.config.GLOBAL.DEFAULT_DEVICE), 'values': [], 'num_decimals': 3},
            'LPIPS': {'metric': torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(Framework.config.GLOBAL.DEFAULT_DEVICE),
                      'values': [], 'num_decimals': 3},
        }
        for result, target in Logger.logProgressBar(zip(results, targets), total=len(results), desc='calculate metrics', leave=False):
            for metric_data in metrics.values():
                metric_data['values'].append(metric_data['metric'](result[None], target[None]).item())
        for metric_data in metrics.values():
            metric_data['all'] = metric_data['metric'].compute()
            metric_data['mean'] = mean(metric_data['values'])
            metric_data['median'] = median(metric_data['values'])
        Logger.logInfo('\n'.join(['results:'] + [f'{metric_name}\t{metric_data["mean"]:.{metric_data["num_decimals"]}f}' for metric_name, metric_data in metrics.items()]))
        with open(output_path / 'metrics_8bit.txt', 'w') as f:
            # write summary (raw metrics in first three rows to facilitate parsing)
            f.write(
                '\n'.join(
                    [f'{self.model.model_name}', 'Metric\tMean\tMedian\tPixelMean'] +
                    [f'{metric_name}\t{metric_data["mean"]:.{metric_data["num_decimals"]}f}\t{metric_data["median"]:.{metric_data["num_decimals"]}f}\t{metric_data["all"]:.{metric_data["num_decimals"]}f}'
                        for metric_name, metric_data in metrics.items()] +
                    ['', '\t'.join(['Image'] + list(metrics.keys()))] +
                    [f'{i}\t' + '\t'.join([f'{metric_data["values"][i]:.{metric_data["num_decimals"]}f}' for metric_data in metrics.values()]) for i in range(len(results))] +
                    [' '.join([f'{metric_name}:{metric_data["mean"]}' for metric_name, metric_data in metrics.items()]) + '\n']
                )
            )

    @torch.no_grad()
    def visualizeError(self, results_path: Path, target_path: Path, output_path: Path, file_extension: str = 'png', video_fps: int = 30, video_bitrate: int = 12000) -> None:
        """Visualize differences between result and reference images."""
        Logger.logInfo('visualizing errors')
        # TODO make this work when image sizes are not all the same across the dataset
        try:
            target = torch.stack(
                loadImagesParallel([
                    str(target_path / name) for name in list_sorted_files(target_path)
                    if file_extension in name
                ], scale_factor=None, num_threads=4, desc='loading gt')[0]
            ).float().to(Framework.config.GLOBAL.DEFAULT_DEVICE)
        except Exception as e:
            Logger.logWarning(f'failed to visualize errors: {e}.')
            return
        result = torch.stack(
            loadImagesParallel([
                str(results_path / name) for name in list_sorted_files(results_path)
                if file_extension in name
            ], scale_factor=None, num_threads=4, desc='loading result')[0]
        ).float().to(Framework.config.GLOBAL.DEFAULT_DEVICE)
        # prepare error visualization
        output_directory_error = output_path / 'error'
        os.makedirs(output_directory_error, exist_ok=True)
        video_writer = VideoWriter(output_directory_error / 'error.mp4', width=int(result.shape[3] * 2), height=result.shape[2], fps=video_fps, bitrate=video_bitrate)
        l1_distances = torch.abs(result - target).clamp(0.0, 1.0)
        l2_distances = torch.sum((result - target) ** 2, dim=1, keepdim=True)
        min_l2, max_l2 = torch.min(l2_distances), torch.max(l2_distances)
        l2_distances = ((l2_distances - min_l2) / (max_l2 - min_l2) * 10).clamp(0.0, 1.0)
        l2_distances = torch.index_select(
            ColorMap.get('VIRIDIS'), dim=0, index=(l2_distances * 255).int().flatten()
        ).reshape(l2_distances.shape[0], *l2_distances.shape[2:], 3).permute(0, 3, 1, 2)

        for index, (l1_distance, l2_distance) in Logger.logProgressBar(
                enumerate(zip(l1_distances, l2_distances)), total=len(result), desc='visualizing errors', leave=False):
            error = torch.cat([l1_distance, l2_distance], dim=-1)
            saveImage(output_directory_error / f'{index:05d}.{file_extension}', error)
            video_writer.addFrame(error)
        video_writer.close()

    @torch.no_grad()
    def renderSubset(self, output_directory: Path, dataset: 'BaseDataset', calculate_metrics: bool = False,
                     visualize_errors: bool = False, verbose: bool = True, image_extension: str = 'png', save_gt: bool = False,
                     closest_train: bool = False, video_fps: int = 30, video_bitrate: int = 12000) -> None:
        """Render a data subset and save the results to disk.

        Args:
            output_directory (Path): Path to the output directory.
            dataset (BaseDataset): Dataset to render. Subset is determined by the dataset mode.
            calculate_metrics (bool, optional): Calculate and save image quality metrics, if GT rgb is available. Defaults to False.
            visualize_errors (bool, optional): Renders Error visualization to GT, if available. Defaults to False.
            verbose (bool, optional): If deactivated, surpresses all logging output. Defaults to True.
            image_extension (str, optional): Image file format used for the output images. Defaults to 'png'.
            save_gt (bool, optional): Save colored ground truth data alongside model outputs. Defaults to False.
            closest_train (bool, optional): Save closest training image for every view. Defaults to False.
            video_fps (int, optional): Frames per second for the output video. Defaults to 30.
            video_bitrate (int, optional): Bitrate for the output video. Defaults to 12000.
        """
        if not verbose:
            Logger.setMode(Logger.MODE_SILENT)
        self.model.eval()
        if len(dataset) > 0:
            Logger.logInfo(f'rendering {dataset.mode} set images')
            # loop over subset
            for index, sample in Logger.logProgressBar(enumerate(dataset), total=len(dataset), desc="image", leave=False):
                # render image
                dataset.camera.setProperties(sample)
                outputs = self.renderImage(dataset.camera, to_chw=True, benchmark=False)
                outputs = self.pseudoColorOutputs(outputs, dataset.camera, dataset, index)
                # append colored ground truth data
                if save_gt:
                    outputs.update(self.pseudoColorGT(dataset.camera, dataset, index))
                elif calculate_metrics and dataset.camera.properties.rgb is not None:
                    # append only ground truth rgb images for metric calculation
                    outputs['rgb_gt'] = dataset.camera.properties.rgb
                # append closest gt image
                if closest_train and dataset.data['train'] and dataset.mode != 'test':
                    outputs['closest_train'] = min(dataset.data['train'], key=lambda camera: torch.linalg.norm(sample.c2w[:3, 3] - camera.c2w[:3, 3].to(Framework.config.GLOBAL.DEFAULT_DEVICE))).rgb
                if index == 0:
                    # create output directories
                    output_directory_main = output_directory / f'{dataset.mode}_{self.model.num_iterations_trained}'
                    output_directories = {key: output_directory_main / key for key in outputs.keys()}
                    for output_directory in output_directories.values():
                        output_directory.mkdir(parents=True, exist_ok=True)
                    # initialize video writer
                    video_writer = VideoWriter([value / f'{key}.mp4' for key, value in output_directories.items()],
                                               width=dataset[0].width, height=dataset[0].height, fps=video_fps, bitrate=video_bitrate)
                # save images
                for key, output_directory in output_directories.items():
                    if outputs[key] is not None:
                        saveImage(output_directory / f'{index:05d}.{image_extension}', outputs[key])

                video_writer.addFrame(list(outputs.values()))
            video_writer.close()

            # calculate quality metrics (PSNR, SSIM, LPIPS), reload saved 8bit images for comparability
            if calculate_metrics:
                self.calculateImageQualityMetrics(output_directories['rgb'], output_directories['rgb_gt'], output_directory_main, image_extension)

            # visualize differences between result and reference images
            if visualize_errors:
                self.visualizeError(output_directories['rgb'], output_directories['rgb_gt'], output_directory_main, image_extension)

        Logger.setMode(Framework.config.GLOBAL.LOG_LEVEL)
