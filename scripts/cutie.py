#! /usr/bin/env python3
# -- coding: utf-8 --

"""
cutie.py: Run manual segmentation using Cutie on a directory of video frames.
See https://github.com/hkchengrex/Cutie for more details.
"""

import os
import shutil
import subprocess
from argparse import ArgumentParser
from pathlib import Path
import torch

import utils

with utils.discoverSourcePath():
    import Framework
    from Logging import Logger
    from Datasets.utils import list_sorted_files, list_sorted_directories, \
        loadImage, saveImage


ENV_NAME = 'cutie-pytorch'
GIT_URL = 'https://github.com/hkchengrex/Cutie.git'


@torch.no_grad()
def main(*, base_path: Path, output_path: Path | None, recursive: bool, file_pattern: str = None,
         colmap_format: bool = False) -> bool:
    git_path = Path(utils.getCachePath() / 'Cutie')
    # install if necessary
    if not git_path.exists():
        Logger.logInfo('Cloning Cutie...')
        # clone repo
        command: list[str] = [
            'git', 'clone',
            GIT_URL,
            str(git_path),
        ]
        if subprocess.run(command, check=False).returncode != 0:
            Logger.logError('Failed to clone Cutie.')
            return False
        # setup conda env
        Logger.logInfo('Creating conda env...')
        command = [
            'conda', 'create',
            '--name', ENV_NAME,
            'python=3.8',
        ]
        if subprocess.run(command, check=False).returncode != 0:
            Logger.logError(f'Failed to create conda environment "{ENV_NAME}" using command: {" ".join(command)}')
            return False
        Logger.logInfo('Installing dependencies...')
        # pytorch
        command = [
            'conda', 'run',
            '--no-capture-output',
            '-n', ENV_NAME,
            'pip', 'install',
            'torch',
            'torchvision',
            '--index-url', 'https://download.pytorch.org/whl/cu118',
        ]
        if subprocess.run(command, check=False).returncode != 0:
            Logger.logError(f'Failed to install PyTorch using command: {" ".join(command)}')
            return False
        # cutie dependencies
        command = [
            'conda', 'run',
            '--no-capture-output',
            '-n', ENV_NAME,
            'pip', 'install', '-e', '.',
        ]
        if subprocess.run(command, cwd=git_path, check=False).returncode != 0:
            Logger.logError(f'Failed to install Cutie using command: {" ".join(command)}')
            return False
        command = [
            'conda', 'run',
            '--no-capture-output',
            '-n', ENV_NAME,
            'pip', 'uninstall',
            'opencv-python',
            '-y',
        ]
        if subprocess.run(command, check=False).returncode != 0:
            Logger.logError(f'Failed to uninstall OpenCV using command: {" ".join(command)}')
            return False

        command = [
            'conda', 'run',
            '--no-capture-output',
            '-n', ENV_NAME,
            'pip', 'install',
            'opencv-python-headless',
        ]
        if subprocess.run(command, check=False).returncode != 0:
            Logger.logError(f'Failed to install OpenCV Headless using command: {" ".join(command)}')
            return False
        # download checkpoint
        Logger.logInfo('Downloading checkpoint...')
        command = [
            'conda', 'run',
            '--no-capture-output',
            '-n', ENV_NAME,
            'python', 'cutie/utils/download_models.py',
        ]
        if subprocess.run(command, cwd=git_path, check=False).returncode != 0:
            Logger.logError(f'Failed to download model using command: {" ".join(command)}')
            return False
    # run on subdirectories
    if recursive:
        subdirs = [base_path / i for i in list_sorted_directories(base_path)]
        for subdir in subdirs:
            main(subdir,
                 output_path if output_path is None else output_path / subdir.name,
                 recursive,
                 file_pattern)
    # scan images
    filenames = [i for i in list_sorted_files(base_path, pattern=file_pattern)
                 if Path(i).suffix.lower() in ['.png', '.jpg']]
    # create output directory
    if output_path is None:
        output_dir = base_path / 'segmentation'
    else:
        output_dir = output_path
    os.makedirs(str(output_dir), exist_ok=True)

    tmp_img_path = git_path / 'examples' / 'tmp_images'
    workspace_path = git_path / 'workspace' / 'tmp_images'

    def cleanCutieWorkspace():
        """Clean the Cutie workspace."""
        tmp_img_path.mkdir(parents=True, exist_ok=True)
        if tmp_img_path.exists():
            for child in tmp_img_path.iterdir():
                if child.is_file():
                    child.unlink()
                else:
                    shutil.rmtree(child)
        if workspace_path.exists():
            for child in workspace_path.iterdir():
                if child.is_file():
                    child.unlink()
                else:
                    shutil.rmtree(child)

    cleanCutieWorkspace()
    # run
    if filenames:
        Logger.logInfo(f'Running Cutie on sequence directory: "{base_path}"...')
        Logger.logInfo(f'Found {len(filenames)} images')
        # copy images to workspace
        for i in Logger.logProgressBar(filenames, desc='Copying images', leave=False):
            rgb, _ = loadImage(base_path / i, scale_factor=None)
            saveImage(tmp_img_path / (i.split('.')[0] + '.jpg'), rgb)
        # run Cutie
        command: list[str] = [
            'conda', 'run',
            '--no-capture-output',
            '-n', ENV_NAME,
            'python', 'interactive_demo.py',
            '--images', str(tmp_img_path),
            '--num_objects', str(input('enter max number of objects: ')),
        ]
        if subprocess.run(command, cwd=git_path, check=False).returncode != 0:
            Logger.logError('Failed to run Cutie.')
            return False
        # copy and invert masks
        mask_path = git_path / 'workspace' / 'tmp_images' / 'binary_masks'
        for i in Logger.logProgressBar(filenames, desc='Saving masks', leave=False):
            filename = i.split('.')[0]
            rgb, _ = loadImage(mask_path / (filename + '.png'), scale_factor=None)
            outname = i
            if colmap_format:
                rgb = 1.0 - rgb
                outname = outname + '.png'
            saveImage(output_dir / outname, rgb)
        Logger.logInfo('done.')
        cleanCutieWorkspace()
    return True


if __name__ == '__main__':
    # parse command line args
    parser: ArgumentParser = ArgumentParser(
        prog='cutie.py',
        description='Run manual segmentation using Cutie on a directory of video frames.'
    )
    parser.add_argument(
        '-i', '--input_path', action='store', dest='sequence_path',
        required=True, help='Path to the base directory containing the input images.'
    )
    parser.add_argument(
        '-r', '--recusive', action='store_true', dest='recursive',
        help='Apply recursively on subdirectories.'
    )
    parser.add_argument(
        '-o', '--output_path', action='store', dest='output_path',
        required=False, default=None, help='If set, the predicted segmentations will be stored in the given directory.'
    )
    parser.add_argument(
        '-f', '--format_colmap', action='store_true', dest='colmap_format',
        help='store the output in the format required by COLMAP.'
    )
    parser.add_argument(
        '-p', '--pattern', action='store', dest='file_pattern', type=str,
        required=False, default=None, help='Optional file name pattern to filter input images.'
    )
    args = parser.parse_args()
    # init Framework with defaults
    Framework.setup()
    # run main
    Logger.setMode(Logger.MODE_VERBOSE)
    main(
        base_path=Path(args.sequence_path),
        output_path=Path(args.output_path) if args.output_path is not None else None,
        recursive=args.recursive,
        file_pattern=args.file_pattern,
        colmap_format=args.colmap_format
    )
