#! /usr/bin/env python3

"""monocular_depth.py: Predicts relative monocular depth on a given input image sequence."""

import subprocess
import sys
import os
from argparse import ArgumentParser
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
import cv2
import torch

import utils
with utils.DiscoverSourcePath():
    import Framework
    from Logging import Logger
    from Datasets.utils import list_sorted_files, list_sorted_directories, save_image
    from Visual.utils import apply_color_map


class MonocularDepthEstimator(ABC):
    def __init__(self):
        super().__init__()
        self.device = Framework.config.GLOBAL.DEFAULT_DEVICE
        self.load()

    @abstractmethod
    def load(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class Midas(MonocularDepthEstimator):
    def load(self):
        Logger.log_info('Loading Midas large model...')
        self.model = torch.hub.load('intel-isl/MiDaS', 'DPT_Large').to(self.device).eval()
        self.input_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms').dpt_transform

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        prediction = self.model(self.input_transforms(img).to(self.device))
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='nearest',
        )[0]
        min_pred, max_pred = prediction.min(), prediction.max()
        prediction = (prediction - min_pred) / (max_pred - min_pred)
        return prediction


class DepthAnything(MonocularDepthEstimator):
    """Depth-Anything monocular depth estimator by Yang et al. (https://github.com/LiheYoung/Depth-Anything)"""
    def load(self):
        Logger.log_info('Loading Depth-Anything model...')
        self.model_path: Path = Framework.Directories.CACHE_DIR/ 'Depth-Anything-V2'
        if not self.model_path.exists():
            # clone repository
            command: list[str] = [
                'git', 'clone',
                'https://github.com/DepthAnything/Depth-Anything-V2',
            ]
            if subprocess.run(command, cwd=Framework.Directories.CACHE_DIR, check=False).returncode != 0:
                Logger.log_error(f'Failed to clone Depth-Anything repository using: {command}.')
                sys.exit(0)
            # download checkpoint
            command: list[str] = [
                'wget',
                'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth',
            ]
            if subprocess.run(command, cwd=self.model_path, check=False).returncode != 0:
                Logger.log_error(f'Failed to download Depth-Anything checkpoint using: {command}.')
                sys.exit(0)
        sys.path.append(str(self.model_path))
        from depth_anything_v2.dpt import DepthAnythingV2
        cwd = os.getcwd()
        os.chdir(str(self.model_path))
        model_config = {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}  # vitl
        self.model = DepthAnythingV2(**model_config)
        self.model.load_state_dict(torch.load(self.model_path / 'depth_anything_v2_vitl.pth', map_location='cpu'))
        self.model.to(self.device).eval()
        os.chdir(cwd)

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        prediction = torch.from_numpy(self.model.infer_image(img)).to(self.device)[None]
        min_pred, max_pred = prediction.min(), prediction.max()
        prediction = (prediction - min_pred) / (max_pred - min_pred)
        return prediction


@torch.no_grad()
def main(*, base_path: Path, output_path: Path | None, recursive: bool, color: bool,
         method: str, model: MonocularDepthEstimator = None) -> bool:
    if model is None:
        model = getattr(sys.modules[__name__], method)()
    # run on subdirectories
    if recursive:
        subdirs = [base_path / i for i in list_sorted_directories(base_path)]
        for subdir in subdirs:
            main(
                base_path=subdir,
                output_path=output_path if output_path is None else output_path / subdir.name,
                recursive=recursive,
                color=color,
                method=method,
                model=model
            )
    # load images
    filenames = [i for i in list_sorted_files(base_path) if Path(i).suffix.lower() in ['.png', '.jpg', '.jpeg']]
    file_paths = [str(base_path / i) for i in filenames]
    if file_paths:
        Logger.log_info(f'Running depth estimator on sequence directory: "{base_path}"...')
        Logger.log_info(f'Found {len(file_paths)} images')
        # create output directory
        if output_path is None:
            output_dir = base_path / 'monoc_depth'
        else:
            output_dir = output_path
        os.makedirs(output_dir, exist_ok=True)
        # predict and save depth
        for i in Logger.log_progress(range(len(file_paths)), desc='image', leave=False):
            img = cv2.imread(file_paths[i])
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            prediction = model(img)
            np.save(output_dir / filenames[i], prediction.cpu().numpy())
            if color:
                save_image(output_dir / f'{filenames[i].split(".")[0]}_color.png',
                          apply_color_map(color_map='TURBO', image=prediction))
        Logger.log_info('done.')
    return True


if __name__ == '__main__':
    # parse command line args
    parser = ArgumentParser(
        prog='monocular_depth.py',
        description='Predicts relative monocular depth on a given input image sequence.'
    )
    parser.add_argument(
        '-i', '--input', action='store', dest='sequence_path',
        required=True, help='Path to the base directory containing the images.'
    )
    parser.add_argument(
        '-r', '--recursive', action='store_true', dest='recursive',
        help='Apply recursively on subdirectories.'
    )
    parser.add_argument(
        '-o', '--output_path', action='store', dest='output_path',
        required=False, default=None, help='If set, the predicted depth will be stored in the given directory.'
    )
    parser.add_argument(
        '-v', '--visualize', action='store_true', dest='color',
        help='Generate and store color visualizations of estimated depth.'
    )
    parser.add_argument(
        '-m', '--method',
        default='DepthAnything',
        const='DepthAnything',
        dest='method',
        nargs='?',
        choices=['Midas', 'DepthAnything'],
        help='method to use for depth estimation'
    )
    args = parser.parse_args()
    # init Framework with defaults
    Framework.setup()
    # run main
    Logger.set_mode(Logger.MODE_VERBOSE)
    main(
        base_path=Path(args.sequence_path),
        output_path=Path(args.output_path) if args.output_path is not None else None,
        recursive=args.recursive,
        color=args.color,
        method=args.method
    )
