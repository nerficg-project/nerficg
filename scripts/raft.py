#! /usr/bin/env python3
# -- coding: utf-8 --

"""raft.py: Predicts optical flow for the given input image sequence using RAFT."""

import os
from argparse import ArgumentParser
from pathlib import Path
import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

import utils

with utils.discoverSourcePath():
    import Framework
    from Logging import Logger
    from Datasets.utils import list_sorted_files, list_sorted_directories, \
        loadImagesParallel, saveOpticalFlowFile, flowToImage, saveImage


def predictAndSave(*, filename: str, output_dir: Path, model: torch.nn.Module, color: bool,
                   inputs1: torch.Tensor, inputs2: torch.Tensor, width: int, height: int) -> None:
    flows = model(inputs1, inputs2)[-1][:, :, :height, :width]
    saveOpticalFlowFile(output_dir / f'{filename}.flo', flows[0])
    if color:
        saveImage(output_dir / f'{filename}.png', flowToImage(flows[0]))


@torch.no_grad()
def main(*, base_path: Path, output_path: Path | None, recursive: bool, backward: bool, color: bool,
         model: torch.nn.Module = None) -> bool:
    device = Framework.config.GLOBAL.DEFAULT_DEVICE
    # load model
    if model is None:
        Logger.logInfo('Loading RAFT model...')
        model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device).eval()
    # run on subdirectories
    if recursive:
        subdirs = [base_path / i for i in list_sorted_directories(base_path)]
        for subdir in subdirs:
            main(
                base_path=subdir,
                output_path=output_path if output_path is None else output_path / subdir.name,
                recursive=recursive,
                backward=backward,
                color=color,
                model=model
            )
    # load images
    filenames = [i for i in list_sorted_files(base_path) if Path(i).suffix.lower() in ['.png', '.jpg', '.jpeg']]
    file_paths = [str(base_path / i) for i in filenames]
    if filenames:
        Logger.logInfo(f'Running RAFT on sequence directory: "{base_path}"...')
        Logger.logInfo(f'Found {len(filenames)} images')
        rgbs, _ = loadImagesParallel(file_paths, None, 4, 'loading image sequence')
        rgbs = (torch.stack(rgbs, dim=0).to(device) * 2.0) - 1.0
        # create output directory
        if output_path is None:
            output_dir = base_path / 'flow'
        else:
            output_dir = output_path
        os.makedirs(str(output_dir), exist_ok=True)
        # pad inputs to multiple of 8
        *_, h, w = rgbs.shape
        delta_h = (8 - (h % 8)) % 8
        delta_w = (8 - (w % 8)) % 8
        if delta_h != 0 or delta_w != 0:
            rgbs = torch.nn.functional.pad(rgbs, (0, delta_w, 0, delta_h, 0, 0, 0, 0), 'constant', 0)
        # predict and save flows
        for i in Logger.logProgressBar(range(len(rgbs) - 1), desc='image', leave=False):
            inputs1 = rgbs[i:i+1]
            inputs2 = rgbs[i+1:i+2]
            predictAndSave(filename=f'{filenames[i].split(".")[0]}_forward', output_dir=output_dir, model=model,
                           color=color, inputs1=inputs1, inputs2=inputs2, width=w, height=h)
            if backward:
                predictAndSave(filename=f'{filenames[i+1].split(".")[0]}_backward', output_dir=output_dir, model=model,
                               color=color, inputs1=inputs2, inputs2=inputs1, width=w, height=h)
        Logger.logInfo('done.')
    return True


if __name__ == '__main__':
    # parse command line args
    parser: ArgumentParser = ArgumentParser(
        prog='raft.py',
        description='Predicts optical flow for the given input image sequence using RAFT.'
    )
    parser.add_argument(
        '-i', '--input', action='store', dest='sequence_path',
        required=True, help='Path to the base directory containing the images.'
    )
    parser.add_argument(
        '-r', '--recusive', action='store_true', dest='recursive',
        help='Scan for subdirectories and estimate flow.'
    )
    parser.add_argument(
        '-b', '--backward', action='store_true', dest='backward',
        help='Generate flow predictions in backward direction.'
    )
    parser.add_argument(
        '-o', '--output_path', action='store', dest='output_path',
        required=False, default=None,
        help='If set, the flow predictions will be stored in the given directory.'
    )
    parser.add_argument(
        '-v', '--visualize', action='store_true', dest='color',
        help='Generate and store color visualizations of estimated flow.'
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
        backward=args.backward,
        color=args.color
    )
