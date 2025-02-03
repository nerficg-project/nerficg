#! /usr/bin/env python3
# -- coding: utf-8 --

"""vggsfm.py: Calibrates a given input image sequence using VGGSfM (see https://github.com/facebookresearch/vggsfm)."""

import shutil
import subprocess
from argparse import ArgumentParser
from pathlib import Path

import utils

from cutie import main as runCutie
from raft import main as runRAFT
from monocularDepth import main as runMonocularDepth

with utils.discoverSourcePath():
    import Framework
    from Logging import Logger


GIT_URL = 'https://github.com/facebookresearch/vggsfm.git'
GIT_COMMIT = 'd655269'
CONDA_ENV = 'vggsfm_tmp'
PYTHON_VERSION = '3.10'
PYTORCH_VERSION = '2.1.0'
CUDA_VERSION = '12.1'


def initializeVGGSfM(cache_path: Path, vggsfm_path: Path) -> bool:
    Logger.logInfo('Initializing VGGSfM...')
    # clone repository
    Logger.logInfo('Cloning git repository...')
    command: list[str] = [
        'git', 'clone',
        GIT_URL,
    ]
    if subprocess.run(command, cwd=cache_path, check=False).returncode != 0:
        Logger.logError(f'Error while cloning VGGSfM repository from "{GIT_URL}".')
        return False
    command: list[str] = [
        'git', 'checkout',
        GIT_COMMIT,
    ]
    if subprocess.run(command, cwd=cache_path / 'vggsfm', check=False).returncode != 0:
        Logger.logError(f'Error while checking out commit "{GIT_COMMIT}" in VGGSfM repository.')
        return False
    # setup conda env
    Logger.logInfo('Creating conda env...')
    commands: list[dict[str, ...] | list[str]] = [
        [
            'conda', 'create', '--name', CONDA_ENV,
            f'python={PYTHON_VERSION}',
        ],
        [
            'conda', 'run', '--no-capture-output', '-n', CONDA_ENV,
            'conda', 'install',
            f'pytorch={PYTORCH_VERSION}', 'torchvision', f'pytorch-cuda={CUDA_VERSION}',
            '-c', 'pytorch', '-c', 'nvidia',
        ],
        [
            'conda', 'run', '--no-capture-output', '-n', CONDA_ENV,
            'conda', 'install',
            '-c', 'fvcore', '-c', 'iopath', '-c', 'conda-forge',
            'fvcore', 'iopath',
        ],
        [
            'conda', 'run', '--no-capture-output', '-n', CONDA_ENV,
            'conda', 'install',
            'pytorch3d=0.7.5',
            '-c', 'pytorch3d',
        ],
        [
            'conda', 'run', '--no-capture-output', '-n', CONDA_ENV,
            'pip', 'install',
            'hydra-core',
            '--upgrade',
        ],
        [
            'conda', 'run', '--no-capture-output', '-n', CONDA_ENV,
            'pip', 'install',
            'omegaconf', 'opencv-python', 'einops', 'visdom', 'tqdm', 'scipy', 'plotly', 'scikit-learn',
            'imageio[ffmpeg]', 'gradio', 'trimesh', 'huggingface_hub',
        ],
        {
            'cwd': vggsfm_path,
            'command': ['git', 'clone', 'https://github.com/jytime/LightGlue.git', 'dependency/LightGlue'],
        },
        {
            'cwd': vggsfm_path / 'dependency' / 'LightGlue',
            'command': [
                'conda', 'run', '--no-capture-output', '-n', CONDA_ENV,
                'python', '-m', 'pip', 'install', '-e', '.',
            ],
        },
        [
            'conda', 'run', '--no-capture-output', '-n', CONDA_ENV,
            'pip', 'install', 'numpy==1.26.3',
        ],
        [
            'conda', 'run', '--no-capture-output', '-n', CONDA_ENV,
            'pip', 'install', 'pycolmap==3.10.0', 'pyceres',
        ],
        [
            'conda', 'run', '--no-capture-output', '-n', CONDA_ENV,
            'pip', 'install', 'poselib==2.0.2',
        ],
    ]
    for command in commands:
        if isinstance(command, dict):
            if subprocess.run(command['command'], cwd=command['cwd'], check=False) != 0:
                Logger.logError(f'Error while setting up VGGSfM conda environment using command: "{command}".')
                return False
        else:
            if subprocess.run(command, check=False) != 0:
                Logger.logError(f'Error while setting up VGGSfM conda environment using command: "{command}".')
                return False
    # build dependencies
    Logger.logInfo('Installing dependencies...')
    command: list[str] = [
        'conda', 'run',
        '--no-capture-output',
        '-n', CONDA_ENV,
        'python', '-m', 'pip', 'install', '-e', '.',
    ]
    if subprocess.run(command, cwd=vggsfm_path, check=False).returncode != 0:
        Logger.logError('Error while installing VGGSfM dependencies.')
        return False
    return True


def main(*, base_path: Path, num_query_points: int, num_query_frames: int, shared_camera: bool,
         generate_annotations: bool, use_masks: bool, reinstall: bool) -> bool:
    Logger.logInfo(f'Running VGGSfM calibration for sequence "{base_path.name}"...')
    # setup directories
    base_path = base_path.resolve()
    if not base_path.exists():
        Logger.logError(f'Input directory "{base_path}" does not exist.')
        return False
    images_path: Path = base_path / 'images'
    flow_path: Path = base_path / 'flow'
    mask_path = base_path / 'masks'
    depth_path = base_path / 'monoc_depth'
    # initialize vggsfm
    cache_path: Path = utils.getCachePath()
    cache_path.mkdir(exist_ok=True)
    vggsfm_path: Path = cache_path / 'vggsfm'
    if reinstall and vggsfm_path.exists():
        shutil.rmtree(vggsfm_path)
    if not vggsfm_path.exists():
        if not initializeVGGSfM(cache_path, vggsfm_path):
            return False
    # create masks
    if use_masks and not mask_path.exists():
        runCutie(base_path=images_path, output_path=mask_path, recursive=False, colmap_format=False)
    # run VGGSfM
    command: list[str] = [
        'HYDRA_FULL_ERROR=1',
        'conda', 'run',
        '--no-capture-output',
        '-n', CONDA_ENV,
        'python', 'demo.py',
        f'SCENE_DIR={base_path}',
        f'max_query_pts={num_query_points}',
        f'shared_camera={shared_camera}',
        f'query_frame_num={num_query_frames}',
    ]
    if subprocess.run(command, cwd=vggsfm_path, check=False).returncode != 0:
        Logger.logError(f'Error while running VGGSfM using: "{command}".')
        return False
    # os.remove('demo.log')
    # generate additional data
    if generate_annotations:
        if not flow_path.exists():
            runRAFT(base_path=images_path, output_path=flow_path, recursive=False, backward=True,
                    color=True, model=None)
        if not mask_path.exists():
            runCutie(base_path=images_path, output_path=mask_path, recursive=False, colmap_format=False)
        if not depth_path.exists():
            runMonocularDepth(base_path=images_path, output_path=depth_path, recursive=False, color=True,
                              method='DepthAnything', model=None)
    return True


if __name__ == '__main__':
    # init Framework with defaults
    args = Framework.setup()
    Logger.setMode(Logger.MODE_VERBOSE)
    # parse command line args
    parser: ArgumentParser = ArgumentParser(prog='vggsfm.py',
                                            description='Calibrates a given input image sequence using VGGSfM.')
    parser.add_argument(
        '-i', '--input', action='store', dest='sequence_path',
        required=True,
        help='Path to the base directory of the sequence, containing the "images" dir.'
    )
    parser.add_argument(
        '-p', '--points_query', action='store', dest='num_query_points', default=1024, type=int,
        help='Max number of query points for the network.')
    parser.add_argument(
        '-f', '--frames_query', action='store', dest='num_query_frames', default=6, type=int,
        help='Number of query frames.')
    parser.add_argument(
        '-s', '--shared', action='store_true', dest='shared_camera',
        help='Use a shared camera model for all images.'
    )
    parser.add_argument(
        '-a', '--annotations', action='store_true', dest='generate_annotations',
        help='Calculates additional annotations (optical flow, monocular depth, binary segmentation) '
             'on the input sequence.'
    )
    parser.add_argument(
        '-m', '--mask', action='store_true', dest='use_masks',
        help='Create and use masks to exclude (dynamic) regions from keypoint matching.'
    )
    parser.add_argument(
        '-r', '--reinstall', action='store_true', dest='reinstall',
        help='Reinstall VGGSfM dependencies.'
    )
    args = parser.parse_args(args)
    # run main
    main(
        base_path=Path(args.sequence_path),
        num_query_points=args.num_query_points,
        num_query_frames=args.num_query_frames,
        shared_camera=args.shared_camera,
        generate_annotations=args.generate_annotations,
        use_masks=args.use_masks,
        reinstall=args.reinstall
    )
