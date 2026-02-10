#! /usr/bin/env python3

"""create_config.py: Creates a new config file with default values for a given method and dataset."""

import os
from argparse import ArgumentParser
import warnings
import yaml
from pathlib import Path

import utils
with utils.DiscoverSourcePath():
    import Framework
    from Logging import Logger
    from Implementations import Methods as MI
    from Implementations import Datasets as DI


def main(*, method_name: str, dataset_name: str, all_sequences: bool, output_filename: str) -> None:
    Logger.set_mode(Logger.MODE_VERBOSE)
    # create config with global defaults
    Framework.config = Framework.ConfigWrapper(GLOBAL=Framework.get_default_global_config())
    Framework.config.GLOBAL.METHOD_TYPE = method_name
    Framework.config.GLOBAL.DATASET_TYPE = dataset_name
    # add renderer, model and training parameters
    method = MI.import_method(method_name)
    Framework.config.MODEL = method.MODEL.get_default_parameters()
    Framework.config.RENDERER = method.RENDERER.get_default_parameters()
    Framework.config.TRAINING = method.TRAINING_INSTANCE.get_default_parameters()
    # add dataset parameters
    dataset_class = DI.get_dataset_class(dataset_name)
    Framework.config.DATASET = dataset_class.get_default_parameters()
    # dump config into file
    output_path = Framework.Directories.CONFIG_DIR
    dataset_path = None
    if all_sequences:
        output_path = output_path / output_filename
        dataset_path = Path(Framework.config.DATASET.PATH).parents[0]
        os.makedirs(output_path, exist_ok=True)
        if not dataset_path.is_dir():
            Logger.log_error(f'failed to gather sequences from "{dataset_path}": directory not found')
            return
        config_file_names = [i.name for i in dataset_path.iterdir() if i.is_dir()]
    else:
        config_file_names = [output_filename]
    for config_file_name in config_file_names:
        config_file_path = output_path / (config_file_name if config_file_name.endswith('.yaml') else f'{config_file_name}.yaml')
        try:
            Framework.config.TRAINING.MODEL_NAME = config_file_name
            if dataset_path is not None:
                Framework.config.DATASET.PATH = str(dataset_path / config_file_name)
            with open(config_file_path, 'w') as f:
                yaml.dump(Framework.ConfigParameterList.toDict(Framework.config), f,
                          default_flow_style=False, indent=4, canonical=False, sort_keys=False)
                Logger.log_info(f'configuration file successfully created: {config_file_path}')
        except IOError as e:
            Logger.log_error(f'failed to create configuration file: "{e}"')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # parse command line args
    parser = ArgumentParser(
        prog='create_config.py',
        description='Creates a new config file with default values for a given method and dataset.'
    )
    parser.add_argument(
        '-m', '--method', action='store', dest='method_name',
        metavar='method directory name', required=True,
        help='Name of the method you want to train. Name should match the directory in lib/methods.'
    )
    parser.add_argument(
        '-d', '--dataset', action='store', dest='dataset_name',
        metavar='dataset name', required=True,
        help='Name of the dataset you want to train on. Name should match the python file in src/Datasets.'
    )
    parser.add_argument(
        '-a', '--all', action='store_true', dest='all_sequences',
        help='If set, creates a directory containing a config file for each sequence in the dataset.'
    )
    parser.add_argument(
        '-o', '--output', action='store', dest='output_filename',
        metavar='output config filename', required=True,
        help='Name of the generated config file.'
    )
    args = parser.parse_args()
    if args.output_filename.endswith('.yaml'):
        args.output_filename = args.output_filename[:-5]

    # run main
    main(
        method_name=args.method_name,
        dataset_name=args.dataset_name,
        all_sequences=args.all_sequences,
        output_filename=args.output_filename
    )
