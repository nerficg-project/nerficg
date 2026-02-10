"""Base/Renderer.py: Implementation of the basic renderer which processes the results of the models."""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from statistics import mean, median

import torch
import torchmetrics

import Framework
from Datasets.Base import BaseDataset
from Datasets.utils import save_image, load_images, list_sorted_files, apply_background_color, View
from Logging import Logger
from Methods.Base.Model import BaseModel
from Visual.ColorMap import ColorMap
from Visual.utils import apply_color_map


class BaseRenderingComponent(ABC, torch.nn.Module):
    """Basic subcomponent of renderers used to parallelize the rendering procedure of sub-models."""

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
            Logger.log_error(
                f'provided invalid model for renderer of type: "{type(self)}"'
                f'\n provided model type: "{type(model)}", valid options are: {valid_model_types}'
            )
            raise Framework.RendererError(f'provided invalid model for renderer of type: "{type(self)}"')
        # assign model
        self.model = model

    @abstractmethod
    def render_image(self, view: View, to_chw: bool = False, benchmark: bool = False) -> dict[str, torch.Tensor | None]:
        """Renders model outputs for a given camera.

        Args:
            view (View): View object for rendering.
            to_chw (bool, optional): If set, returns outputs in shape chw instead of hwc. Defaults to False.
            benchmark (bool, optional): Indicates that renderer is called for benchmarking purposes. Defaults to False.

        Returns:
            dict[str, torch.Tensor | None]: Dictionary containing the rendered outputs.
            All tensors are expected to be of shape HxWxC or CxHxW, where c is either 1 or 3.
            All tensors are expected to be in the range [0, 1].
        """
        pass

    def postprocess_outputs(self, outputs: dict[str, torch.Tensor | None], view: View, dataset: BaseDataset, index: int) -> dict[str, torch.Tensor]:
        """Postprocesses the model outputs, returning tensors of shape 3xHxW."""
        outputs_color = {
            'rgb': outputs['rgb'].clamp_(0.0, 1.0),  # FIXME: this clamp should be method specific
            'alpha': outputs['alpha'].expand_as(outputs['rgb']) if 'alpha' in outputs else torch.zeros_like(outputs['rgb']),
            'depth': apply_color_map(
                    color_map='SPECTRAL',
                    image=outputs['depth'],
                    min_max=None,
                    mask=outputs['alpha'] if 'alpha' in outputs else None
                ) if 'depth' in outputs else torch.zeros_like(outputs['rgb']),
        }
        return outputs_color

    def postprocess_reference_data(self, view: View, dataset: BaseDataset, index: int) -> dict[str, torch.Tensor]:
        """Postprocesses the reference data relevant for this method, returning tensors of shape 3xHxW."""
        # rgb
        if (rgb_gt := view.rgb) is None:
            rgb_gt = view.camera.background_color[:, None, None].expand(3, view.camera.height, view.camera.width)
        # alpha
        if (alpha_gt := view.alpha) is None:
            alpha_gt = torch.ones_like(rgb_gt)
        else:
            alpha_gt = alpha_gt.expand_as(rgb_gt)
            rgb_gt = apply_background_color(rgb_gt, alpha_gt, view.camera.background_color)  # FIXME: integrate into data model
        return {
            'rgb_gt': rgb_gt,
            'alpha_gt': alpha_gt,
        }

    @torch.no_grad()
    def compute_image_metrics(
        self,
        results_path: Path,
        target_path: Path,
        output_path: Path,
        file_extension: str = 'png'
    ) -> None:
        """Calculate quality metrics (PSNR, SSIM, LPIPS)."""
        Logger.log_info('calculating image quality metrics')
        try:
            targets = load_images([
                str(target_path / name) for name in list_sorted_files(target_path)
                if file_extension in name
            ], scale_factor=None, num_threads=4, desc='loading gt')[0]
        except Exception:
            Logger.log_warning('failed to calculate quality metrics: no GT data available.')
            return
        results = load_images([
            str(results_path / name) for name in list_sorted_files(results_path)
            if file_extension in name
        ], scale_factor=None, num_threads=4, desc='loading result')[0]
        torch.hub.set_dir(Framework.Directories.CACHE_DIR)
        metrics = {
            'PSNR': {
                'metric': torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(Framework.config.GLOBAL.DEFAULT_DEVICE),
                'values': [], 'num_decimals': 2
            },
            'SSIM': {
                'metric': torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(Framework.config.GLOBAL.DEFAULT_DEVICE),
                'values': [], 'num_decimals': 3
            },
            'LPIPS': {
                'metric': torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(Framework.config.GLOBAL.DEFAULT_DEVICE),
                'values': [], 'num_decimals': 3
            },
        }
        for result, target in Logger.log_progress(zip(results, targets), total=len(results), desc='calculate metrics', leave=False):
            result = result.float().to(Framework.config.GLOBAL.DEFAULT_DEVICE)[None]
            target = target.float().to(Framework.config.GLOBAL.DEFAULT_DEVICE)[None]
            for metric_data in metrics.values():
                metric_data['values'].append(metric_data['metric'](result, target).item())
        for metric_data in metrics.values():
            metric_data['all'] = metric_data['metric'].compute()
            metric_data['mean'] = mean(metric_data['values'])
            metric_data['median'] = median(metric_data['values'])
        Logger.log_info('\n'.join(['results:'] + [f'{metric_name}\t{metric_data["mean"]:.{metric_data["num_decimals"]}f}' for metric_name, metric_data in metrics.items()]))
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
    def visualize_error(
        self,
        results_path: Path,
        target_path: Path,
        output_path: Path,
        file_extension: str = 'png',
    ) -> None:
        """Visualize differences between result and reference images."""
        Logger.log_info('visualizing errors')
        # TODO make this work when image sizes are not all the same across the dataset
        try:
            target = torch.stack(
                load_images([
                    str(target_path / name) for name in list_sorted_files(target_path)
                    if file_extension in name
                ], scale_factor=None, num_threads=4, desc='loading gt')[0]
            ).float().to(Framework.config.GLOBAL.DEFAULT_DEVICE)
        except Exception as e:
            Logger.log_warning(f'failed to visualize errors: {e}.')
            return
        result = torch.stack(
            load_images([
                str(results_path / name) for name in list_sorted_files(results_path)
                if file_extension in name
            ], scale_factor=None, num_threads=4, desc='loading result')[0]
        ).float().to(Framework.config.GLOBAL.DEFAULT_DEVICE)
        # prepare error visualization
        output_directory_error = output_path / 'error'
        os.makedirs(output_directory_error, exist_ok=True)
        l1_distances = torch.abs(result - target).clamp(0.0, 1.0)
        l2_distances = torch.sum((result - target) ** 2, dim=1, keepdim=True)
        min_l2, max_l2 = torch.min(l2_distances), torch.max(l2_distances)
        l2_distances = ((l2_distances - min_l2) / (max_l2 - min_l2) * 10).clamp(0.0, 1.0)
        l2_distances = torch.index_select(
            ColorMap.get('VIRIDIS'), dim=0, index=(l2_distances * 255).int().flatten()
        ).reshape(l2_distances.shape[0], *l2_distances.shape[2:], 3).permute(0, 3, 1, 2)

        for index, (l1_distance, l2_distance) in Logger.log_progress(
                enumerate(zip(l1_distances, l2_distances)), total=len(result), desc='visualizing errors', leave=False):
            error = torch.cat([l1_distance, l2_distance], dim=-1)
            save_image(output_directory_error / f'{index:05d}.{file_extension}', error)

    @torch.no_grad()
    def render_subset(
        self,
        output_directory: Path,
        dataset: 'BaseDataset',
        calculate_metrics: bool = False,
        visualize_errors: bool = False,
        verbose: bool = True,
        image_extension: str = 'png',
        save_gt: bool = False,
        closest_train: bool = False,
    ) -> None:
        """Render a data subset and save the results to disk.

        Args:
            output_directory (Path): Path to the output directory.
            dataset (BaseDataset): Dataset to render. Subset is determined by the dataset mode.
            calculate_metrics (bool, optional): Calculate and save image quality metrics, if GT rgb is available. Defaults to False.
            visualize_errors (bool, optional): Renders Error visualization to GT, if available. Defaults to False.
            verbose (bool, optional): If deactivated, suppresses all logging output. Defaults to True.
            image_extension (str, optional): Image file format used for the output images. Defaults to 'png'.
            save_gt (bool, optional): Save colored ground truth data alongside model outputs. Defaults to False.
            closest_train (bool, optional): Save the closest training image for every view. Defaults to False.
        """
        if not verbose:
            Logger.set_mode(Logger.MODE_SILENT)
        self.model.eval()
        if len(dataset) > 0:
            Logger.log_info(f'rendering {dataset.mode} set images')
            # loop over subset
            for index, view in Logger.log_progress(enumerate(dataset), total=len(dataset), desc="image", leave=False):
                # render image
                outputs = self.render_image(view, to_chw=True, benchmark=False)
                outputs = self.postprocess_outputs(outputs, view, dataset, index)
                # append colored ground truth data
                if save_gt:
                    outputs.update(self.postprocess_reference_data(view, dataset, index))
                elif (calculate_metrics or visualize_errors) and (color_gt := view.rgb) is not None:  # TODO: the None check should not load the image
                    # append only ground truth rgb images for metric calculation
                    # compose gt with background color if needed  # FIXME: integrate into data model
                    if (alpha_gt := view.alpha) is not None:
                        color_gt = apply_background_color(color_gt, alpha_gt, view.camera.background_color)
                    outputs['rgb_gt'] = color_gt
                # append closest gt image
                if closest_train and dataset.data['train'] and dataset.mode != 'test':
                    outputs['closest_train'] = min(dataset.data['train'], key=lambda other_view: torch.linalg.norm(view.position - other_view.position)).rgb  # TODO: gpu upload
                if index == 0:
                    # create output directories
                    output_directory_main = output_directory / f'{dataset.mode}_{self.model.num_iterations_trained}'
                    output_directories = {key: output_directory_main / key for key in outputs.keys()}
                    for output_directory in output_directories.values():
                        output_directory.mkdir(parents=True, exist_ok=True)
                # save images
                for key, output_directory in output_directories.items():
                    if outputs[key] is not None:
                        save_image(output_directory / f'{index:05d}.{image_extension}', outputs[key])

            # calculate quality metrics (PSNR, SSIM, LPIPS), reload saved 8bit images for comparability
            if calculate_metrics:
                self.compute_image_metrics(output_directories['rgb'], output_directories['rgb_gt'], output_directory_main, image_extension)

            # visualize differences between result and reference images
            if visualize_errors:
                self.visualize_error(output_directories['rgb'], output_directories['rgb_gt'], output_directory_main, image_extension)

        Logger.set_mode(Framework.config.GLOBAL.LOG_LEVEL)
