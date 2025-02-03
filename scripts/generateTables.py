#! /usr/bin/env python3
# -- coding: utf-8 --

"""
generateTables.py:
Recomputes image quality metrics for several scenes and methods.
Results are written to text files in the root directory where the root directory must have the format described below.
Images will be sorted by name and must have the same resolution within each scene.
Images can be in any format supported by our loadImage function (only tested with PNG).
Metrics and method sorting in the table will be identical to the config file.
To add new metrics, add a new entry to the 'known_metrics' dict below.
Masked losses will be calculated if a '_mask' directory is present in the scene directories.

root_dir
├── scene0
│   ├── gt
│   │   ├── image0.png
│   │   └── ...
│   └── method0
│   │   ├── image0.png
│   │   └── ...
│   └── _mask
│       ├── image0.png
│       └── ...
├── scene1
│   ├── gt
│   │   ├── image0.png
│   │   └── ...
│   └── method0
│   │   ├── image0.png
│   │   └── ...
│   └── _mask
│       ├── image0.png
│       └── ...
└── ...
"""

import warnings
from argparse import ArgumentParser
from typing import Callable
from pathlib import Path
import torch
from tabulate import tabulate
from statistics import mean
import torchmetrics
import yaml

import utils

with utils.discoverSourcePath():
    from Logging import Logger
    from Datasets.utils import loadImagesParallel, list_sorted_files, list_sorted_directories
    from Optim.MaskedMetrics import mPSNR, mSSIM, mLPIPS


"""
Dictionary of available metrics, consisting of string identifier, a tuple of callables for
metric calculation and output formating, as well as a boolean indicator if masks are required.
Add new metrics here to make them accessible via configuration files.
"""
known_metrics: dict[str: tuple[Callable, Callable]] = {
    'PSNR': (torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).cuda(), lambda value: f'{value:.2f}', False),
    'SSIM': (torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).cuda(), lambda value: f'{value:.3f}', False),
    'MS-SSIM': (torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).cuda(), lambda value: f'{value:.3f}', False),
    'mPSNR': (mPSNR, lambda value: f'{value:.2f}', True),
    'mSSIM': (mSSIM, lambda value: f'{value:.3f}', True),
    'mLPIPS': (mLPIPS(), lambda value: f'{value:.3f}', True),
    'LPIPS_vgg': (torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).cuda(), lambda value: f'{value:.3f}', False),
    'LPIPS_alex': (torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda(), lambda value: f'{value:.3f}', False),
    'LPIPS_squeeze': (torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='squeeze', normalize=True).cuda(), lambda value: f'{value:.3f}', False),
}


def writeEmptyConfigFile(path: Path):
    """write and a default config file to the given base directory."""
    # get scenes and methods
    scenes = list_sorted_directories(path)
    methods = list_sorted_directories(path / scenes[0])
    methods = [method for method in methods if method != 'gt' and not method.startswith('_')]
    # generate a default config dict
    example_config_dict = {
        'METRICS': ['PSNR', 'SSIM', 'LPIPS_vgg'],
        'SCENES': scenes,
        'METHODS': methods,
    }
    # write the config dict to a yaml file
    with open(str(path / 'config.yaml'), 'w') as f:
        yaml.dump(example_config_dict, f, default_flow_style=False, indent=4, canonical=False, sort_keys=False)


@torch.no_grad()
def computeMetrics(results_path: Path, targets: torch.Tensor, mask_images: torch.Tensor | None,
                   metrics: list[torchmetrics.Metric], metric_requires_mask: list[bool]) -> list[float]:
    """Calculate quality metrics."""
    method_name = results_path.name
    image_filenames = [name for name in list_sorted_files(results_path) if name.endswith('.png') or name.endswith('.jpg') or name.endswith('.jpeg')]
    results = torch.stack(
        loadImagesParallel([
            str(results_path / name) for name in image_filenames
        ], scale_factor=None, num_threads=4, desc=f'loading {method_name} images')[0]
    ).float().cuda()
    metric_values = [[] for _ in metrics]
    for result, target, mask in Logger.logProgressBar(zip(results, targets, mask_images), total=len(results), desc=f'calculate {method_name} metrics', leave=False):
        for metric, values, with_mask in zip(metrics, metric_values, metric_requires_mask):
            if with_mask:
                if mask is None:
                    values.append(0.0)
                else:
                    values.append(metric(result, target, mask))
            else:
                values.append(metric(result[None], target[None]).item())
    return [mean(values) for values in metric_values]


def main(root_dir: Path, config_only: bool):
    Logger.setMode(Logger.MODE_VERBOSE)
    config_path = root_dir / 'config.yaml'
    Logger.logInfo(f'reading config file from {config_path}')
    if not config_path.exists():
        Logger.logInfo('config file not found -> creating default config file')
        config = writeEmptyConfigFile(root_dir)
    else:
        with open(config_path, 'r') as f:
            config = yaml.unsafe_load(f)
    if config_only:
        return
    # gather methods, scenes and metrics
    Logger.logInfo(f'processing results for {root_dir.name}')
    scene_names = config['SCENES']
    gt_name = 'gt'
    method_names = config['METHODS']
    metric_names = config['METRICS']
    for metric_name in metric_names:
        if metric_name not in known_metrics:
            Logger.logError(f'requested unknown metric: {metric_name} -> exiting')
            return
    metric_functions = [known_metrics[metric_name][0] for metric_name in metric_names]
    metric_formatting = [known_metrics[metric_name][1] for metric_name in metric_names]
    metric_requires_mask = [known_metrics[metric_name][2] for metric_name in metric_names]
    results = {
        metric_name: [[] for _ in method_names]
        for metric_name in metric_names
    }
    # calculate metrics
    for scene_name in scene_names:
        Logger.logInfo(f'processing scene {scene_name}')
        scene_path = root_dir / scene_name
        gt_path = scene_path / gt_name
        if not gt_path.exists():
            Logger.logInfo(f'no ground truth images available for scene {scene_name} -> exiting')
            return
        gt_images = torch.stack(
            loadImagesParallel([
                str(gt_path / name) for name in list_sorted_files(gt_path) if name.endswith('.png') or name.endswith('.jpg') or name.endswith('.jpeg')
            ], scale_factor=None, num_threads=4, desc='loading reference images')[0]
        ).float().cuda()
        mask_path = scene_path / '_mask'
        if mask_path.exists():
            mask_images = torch.stack(
                loadImagesParallel([
                    str(mask_path / name) for name in list_sorted_files(mask_path) if name.endswith('.png') or name.endswith('.jpg') or name.endswith('.jpeg')
                ], scale_factor=None, num_threads=4, desc='loading masks')[0]
            ).float().cuda()[:, :1]
        else:
            if max(metric_requires_mask):
                Logger.logWarning(f'mask required for metric calculation but no mask images found for scene {scene_name} -> values will default to zero')
            mask_images = [None] * len(gt_images)
        for method_name in method_names:
            method_path = scene_path / method_name
            if not method_path.exists():
                Logger.logWarning(f'no results available for method {method_name} on scene {scene_name} -> filling with zeros')
                for metric_name in metric_names:
                    results[metric_name][method_names.index(method_name)].append(0.0)
            else:
                scene_metrics = computeMetrics(method_path, gt_images, mask_images, metric_functions, metric_requires_mask)
                for metric_name, metric_value in zip(metric_names, scene_metrics):
                    results[metric_name][method_names.index(method_name)].append(metric_value)
    # build tables
    headers = ['method'] + scene_names + ['mean']
    metric_tables = {
        metric_name: [
            [method_name]
            + [metric_formatting[metric_names.index(metric_name)](scene_result) for scene_result in results[metric_name][method_names.index(method_name)]]
            + [metric_formatting[metric_names.index(metric_name)](mean(results[metric_name][method_names.index(method_name)]))]
            for method_name in method_names
        ]
        for metric_name in metric_names
    }
    # write output files
    with open(root_dir / 'metrics.txt', 'w') as f:
        f.write('\n\n'.join(
            f'{metric_name}\n{tabulate(metric_table, headers, colalign=["left"] + ["center"] * (len(metric_table[0]) - 1), disable_numparse=True)}'
            for metric_name, metric_table in metric_tables.items()
        ))
    with open(root_dir / 'latex_tables.txt', 'w') as f:
        f.write('\n\n'.join(
            f'% {metric_name} (format: {table_format})\n{tabulate(metric_table, headers, colalign=["left"] + ["center"] * (len(metric_table[0]) - 1), disable_numparse=True, tablefmt=table_format)}'
            for table_format in ['latex', 'latex_raw', 'latex_booktabs', 'latex_longtable', 'plain']
            for metric_name, metric_table in metric_tables.items()
        ))


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # Parse command line arguments.
    parser = ArgumentParser(prog='Generate Statistics')
    parser.add_argument(
        '-d', '--dir', action='store', dest='results_root_dir', default=None,
        metavar='path/to/dataset/results/directory', required=True,
        help='A directory containing test set renderings of various methods for each scene of a dataset.'
    )
    parser.add_argument(
        '-c', '--config_only', action='store_true', dest='config_only',
        help='If set, only writes a default config file to the root directory (if it does not already exist), without processing any results.'
    )
    args, _ = parser.parse_known_args()
    # run main
    main(Path(args.results_root_dir), args.config_only)
