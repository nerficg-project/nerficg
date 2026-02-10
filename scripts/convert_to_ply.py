#! /usr/bin/env python3

"""convert_to_ply.py: Extracts a trained model into a .ply file using a method-specific format."""

from argparse import ArgumentParser
from pathlib import Path

from plyfile import PlyData, PlyElement
import numpy as np

import utils
with utils.DiscoverSourcePath():
    import Framework
    from Logging import Logger
    from Implementations import Methods as MI


def save_as_ply(output_path: Path, ply_data_dict: dict[str, np.ndarray | list[str]], use_ascii: bool) -> None:
    elements = [PlyElement.describe(data, key) for key, data in ply_data_dict.items() if key != 'comments']  # noqa
    comments = ply_data_dict.get('comments', [])
    PlyData(elements=elements, text=use_ascii, comments=comments).write(output_path)


def main(*, base_dir: Path, use_ascii: bool) -> None:
    # setup framework
    Framework.setup(config_path=str(base_dir / 'training_config.yaml'), require_custom_config=True)
    # load model
    model = MI.get_model(
        method=Framework.config.GLOBAL.METHOD_TYPE,
        checkpoint=str(base_dir / 'checkpoints' / 'final.pt'),
    ).eval()

    Logger.log_info('starting ply export')

    # extract data from model
    ply_data_dict = model.get_ply_dict()

    if ply_data_dict:
        # create and write .ply file
        save_as_ply(base_dir / 'final.ply', ply_data_dict, use_ascii)
        Logger.log_info('done')
    else:
        Logger.log_warning('model does not support ply export')


if __name__ == '__main__':
    parser = ArgumentParser(
        prog='convert_to_ply.py',
        description='Extracts a trained model into a .ply file using a method-specific format.'
    )
    parser.add_argument(
        '-d', '--dir', action='store', dest='base_dir', default=None,
        metavar='path/to/output/directory', required=True,
        help='A directory containing the outputs of a completed training.'
    )
    parser.add_argument(
        '-t', '--text', action='store_true', dest='use_ascii', default=False,
        help='Whether to use ascii instead of binary format.'
    )
    args, _ = parser.parse_known_args()
    Logger.set_mode(Logger.MODE_VERBOSE)
    main(base_dir=Path(args.base_dir), use_ascii=args.use_ascii)
