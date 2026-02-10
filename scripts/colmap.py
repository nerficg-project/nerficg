#! /usr/bin/env python3

"""colmap.py: Calibrates a given input image sequence using colmap."""

import shutil
import subprocess
from argparse import ArgumentParser
from pathlib import Path

import torch

from cutie import main as run_cutie
from raft import main as run_raft
from monocular_depth import main as run_monocular_depth

import utils
with utils.DiscoverSourcePath():
    import Framework
    from Logging import Logger


def download_vocab_tree() -> Path | None:
    # download vocab tree binary
    cache_path: Path = Framework.Directories.CACHE_DIR
    file_name: str = 'vocab_tree_flickr100K_words32K.bin'
    cache_path.mkdir(exist_ok=True)
    if not (cache_path / file_name).exists():
        Logger.log_info('Downloading vocab tree binary...')
        command: list[str] = [
            'wget',
            '-O', str(cache_path / file_name),
            'https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin',
        ]
        if not subprocess.run(command, check=False).returncode == 0:
            return None
    return cache_path / file_name


def main(*, base_path: Path, use_cpu: bool, camera_mode: str, camera_model: str, vocab_tree_matcher: bool, use_masks: bool,
         undistort: bool, generate_annotations: bool, align: bool) -> bool:
    Logger.log_info(f'Running colmap calibration for sequence "{base_path.name}"...')
    # check input directory
    if not base_path.exists():
        Logger.log_error(f'Input directory "{base_path}" does not exist.')
        return False
    images_path: Path = base_path / 'images'
    flow_path: Path = base_path / 'flow'
    mask_path = base_path / 'sfm_masks'
    depth_path = base_path / 'monoc_depth'
    if not images_path.exists():
        Logger.log_error('Input directory does not contain an "images" directory.')
        return False

    # check if gpu is available
    if not use_cpu and not torch.cuda.is_available():
        Logger.log_warning('No GPU detected, running on CPU.')
        use_cpu = True

    # run colmap feature extraction
    Logger.log_info('Running colmap feature extraction...')
    command: list[str] = [
        'colmap', 'feature_extractor',
        '--database_path', str(base_path / 'database.db'),
        '--image_path', str(images_path),
        '--ImageReader.single_camera', str(camera_mode == 'single'),
        '--ImageReader.single_camera_per_folder', str(camera_mode == 'single_folder'),
        '--ImageReader.single_camera_per_image', str(camera_mode == 'single_image'),
        '--ImageReader.camera_model', camera_model,
        '--FeatureExtraction.use_gpu', str(not use_cpu),
    ]
    if use_masks:
        if not mask_path.exists():
            run_cutie(base_path=images_path, output_path=mask_path, recursive=True, colmap_format=True)
        command += ['--ImageReader.mask_path', str(mask_path)]
    if subprocess.run(command, check=False).returncode != 0:
        Logger.log_error('Error while running colmap feature extraction.')
        return False

    # run colmap feature matching
    Logger.log_info('Running colmap feature matching...')
    if vocab_tree_matcher:
        Logger.log_info('Using vocab tree matcher...')
        vocab_tree_path = download_vocab_tree()
        if vocab_tree_path is None:
            Logger.log_error('Error while downloading vocab tree binary.')
            return False
    command: list[str] = [
        'colmap', 'exhaustive_matcher' if not vocab_tree_matcher else 'vocab_tree_matcher',
        '--database_path', str(base_path / 'database.db'),
        '--FeatureMatching.use_gpu', str(not use_cpu),
    ]
    if vocab_tree_matcher:
        command.append('--VocabTreeMatching.vocab_tree_path')
        command.append(str(vocab_tree_path))
    if subprocess.run(command, check=False).returncode != 0:
        Logger.log_error('Error while running colmap feature matching.')
        return False

    # run colmap bundle adjustment
    Logger.log_info('Running colmap bundle adjustment...')
    output_path: Path = base_path / 'sparse'
    output_path.mkdir(exist_ok=True)
    command: list[str] = [
        'colmap', 'mapper',
        '--database_path', str(base_path / 'database.db'),
        '--image_path', str(images_path),
        '--output_path', str(output_path),
        '--log_level', '1',
        '--Mapper.ba_global_function_tolerance', '0.000001',
        # '--Mapper.ba_use_gpu', str(not use_cpu),  # COLMAP 3.13.0 has this disabled by default
    ]
    if subprocess.run(command, check=False).returncode != 0:
        Logger.log_error('Error while running colmap bundle adjustment.')
        return False

    # run colmap orientation alignment
    if align:
        Logger.log_info('Aligning scene orientation...')
        shutil.move(base_path / 'sparse' / '0', base_path / 'sparse' / '0_unaligned')
        (base_path / 'sparse' / '0').mkdir(exist_ok=True)
        command: list[str] = [
            'colmap', 'model_orientation_aligner',
            '--image_path', str(images_path),
            '--input_path', str(base_path / 'sparse' / '0_unaligned'),
            '--output_path', str(base_path / 'sparse' / '0'),
            '--max_image_size', '4000',
            # '--method', 'IMAGE-ORIENTATION',  # default is MANHATTAN-WORLD
        ]
        if subprocess.run(command, check=False).returncode != 0:
            Logger.log_error('Error while running colmap orientation alignment.')
            return False

    # undistort images
    if undistort:
        Logger.log_info('Undistorting images...')
        command: list[str] = [
            'colmap', 'image_undistorter',
            '--image_path', str(images_path),
            '--input_path', str(base_path / 'sparse' / '0'),
            '--output_path', str(base_path / 'dense'),
            '--output_type', 'COLMAP',
        ]
        if subprocess.run(command, check=False).returncode != 0:
            Logger.log_error('Error while running colmap image undistortion.')
            return False

        shutil.move(images_path, base_path / 'images_distorted')
        shutil.move(base_path / 'sparse' / '0', base_path / 'sparse' / '0_distorted')
        shutil.copytree(base_path / 'dense' / 'sparse', base_path / 'sparse' / '0')
        shutil.copytree(base_path / 'dense' / 'images', base_path / 'images')

        for p in (mask_path, flow_path, depth_path):
            if p.exists():
                shutil.move(p, base_path / (p.name + '_distorted'))

    # Generate .ply file
    Logger.log_info('Generating PLY...')
    sparse_recon_path = base_path / 'sparse' / '0'
    command: list[str] = [
        'colmap', 'model_converter',
        '--input_path', str(sparse_recon_path),
        '--output_path', str(sparse_recon_path / 'points3D.ply'),
        '--output_type', 'PLY',
    ]
    if subprocess.run(command, check=False).returncode != 0:
        Logger.log_error('Error converting to ply.')
        return False

    # Generate additional data
    if generate_annotations:
        run_raft(base_path=images_path, output_path=flow_path, recursive=True, backward=True, color=True, model=None)
        if not mask_path.exists():
            run_cutie(base_path=images_path, output_path=mask_path, recursive=True, colmap_format=True)
        run_monocular_depth(base_path=images_path, output_path=depth_path, recursive=True, color=True,
                          method='DepthAnything', model=None)
    Logger.log_info('Done.')
    return True


if __name__ == '__main__':
    # init Framework with defaults
    args = Framework.setup()
    Logger.set_mode(Logger.MODE_VERBOSE)
    # parse command line args
    parser = ArgumentParser(
        prog='colmap.py',
        description='Calibrates a given input image sequence using COLMAP.'
    )
    parser.add_argument(
        '-i', '--input', action='store', dest='sequence_path',
        required=True,
        help='Path to the base directory of the sequence, containing the "images" dir.'
    )
    parser.add_argument(
        '--cpu', action='store_true', dest='use_cpu',
        help='Prevents the use of the GPU.'
    )
    parser.add_argument(
        '--camera_mode', action='store', dest='camera_mode', type=str, default='single',
        required=False, choices=['single', 'single_folder', 'single_image'],
        help='How many cameras models to use for calibration. One of "single", "single_folder", "single_image".'
    )
    parser.add_argument(
        '--camera_model', action='store', dest='camera_model', type=str, default='OPENCV',
        required=False, choices=['SIMPLE_PINHOLE', 'PINHOLE', 'SIMPLE_RADIAL', 'RADIAL', 'OPENCV', 'FULL_OPENCV',
                                 'SIMPLE_RADIAL_FISHEYE', 'RADIAL_FISHEYE', 'OPENCV_FISHEYE'],
        help='Specifies the intrinsic camera model. One of "SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", '
             '"OPENCV", "FULL_OPENCV", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "OPENCV_FISHEYE".'
    )
    parser.add_argument(
        '-v', '--vocab_tree_matcher', action='store_true', dest='vocab_tree_matcher',
        help='Use the vocab tree matcher instead of the exhaustive matcher (recommended for n>500 images).'
    )
    parser.add_argument(
        '-a', '--annotations', action='store_true', dest='generate_annotations',
        help='Calculates additional annotations (optical flow, monocular depth, binary segmentation) '
             'on the input sequence.'
    )
    parser.add_argument(
        '-m', '--mask', action='store_true', dest='use_masks',
        help='Use masks in the "sfm_masks" directory to exclude regions from keypoint matching.'
    )
    parser.add_argument(
        '-u', '--undistort', action='store_true', dest='undistort',
        help='undistorts the images after calibration.'
    )
    parser.add_argument(
        '-o', '--orientation', action='store_true', dest='align',
        help='aligns scene orientation.'
    )
    args = parser.parse_args(args)
    # run main
    main(
        base_path=Path(args.sequence_path),
        use_cpu=args.use_cpu,
        camera_mode=args.camera_mode,
        camera_model=args.camera_model,
        vocab_tree_matcher=args.vocab_tree_matcher,
        use_masks=args.use_masks,
        undistort=args.undistort,
        generate_annotations=args.generate_annotations,
        align=args.align
    )
