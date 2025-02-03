#! /usr/bin/env python3
# -- coding: utf-8 --

"""sequentialTrain.py: Sequentially runs model trainings for a list or directory of config files."""

import os
import shutil
from argparse import ArgumentParser
from pathlib import Path
import datetime
from statistics import mean
from tabulate import tabulate
from multiprocessing import Process, Queue

import utils
with utils.discoverSourcePath():
    import Framework
    import train
    from Logging import Logger
    from Datasets.utils import list_sorted_files


def gatherConfigs() -> tuple[list[str], Path]:
    Logger.log('collecting config files')
    # parse arguments to retrieve config file locations
    parser: ArgumentParser = ArgumentParser(prog='SequentialTrain')
    parser.add_argument(
        '-c', '--configs', action='store', dest='config_paths', default=[],
        metavar='paths/to/config_files', required=False, nargs='+',
        help='Multiple whitespace separated training config files.'
    )
    parser.add_argument(
        '-d', '--dir', action='store', dest='config_dir', default=None,
        metavar='path/to/configdir/', required=False,
        help='A directory containing training configuration files.'
    )
    args, _ = parser.parse_known_args()
    # add all configs from -c flag
    config_paths: list[str] = args.config_paths
    # set output common directory
    output_directory = f'sequential_train_{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}'
    # add all configs from config dir
    if args.config_dir is not None:
        config_dir_path = Path(args.config_dir)
        if not config_dir_path.is_dir():
            raise Framework.TrainingError(f'config dir is not a valid directory: {config_dir_path}')
        directory_configs = [str(config_dir_path / i) for i in list_sorted_files(config_dir_path) if '.yaml' in i]
        config_paths = config_paths + directory_configs
        output_directory = f'{config_dir_path.name}_{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}'
    # output directory and all configs for execution
    return output_directory, config_paths


def writeOutputMetrics(metric_files: dict[str, Path], output_directory: Path) -> None:
    if metric_files:
        output_file = output_directory / 'summary.txt'
        Logger.log(f'Gathering final quality metrics for {len(metric_files)} runs in: {output_file}')
        parsed_values = []
        for metric_file in metric_files.values():
            with open(metric_file) as f:
                for line in f:
                    pass
            parsed_values.append({i[0]: float(i[1]) for i in [j.split(':') for j in line.split(' ')]})
        headers = ['Metric'] + list(metric_files.keys()) + ['Mean']
        tab = [[metric_name] + [(run[metric_name]) for run in parsed_values] for metric_name in parsed_values[0].keys()]
        for row in tab:
            row.append(mean(row[1:]))
        with open(output_file, 'w') as f:
            f.write(tabulate(tabular_data=tab, headers=headers, floatfmt=".3f"))


def training_process_helper(output, config):
    training_instance = train.main(config_path=config)
    output.put((training_instance.output_directory, training_instance.model.model_name, training_instance.NUM_ITERATIONS))


def main():
    Logger.setMode(Logger.MODE_VERBOSE)
    # get list of training configs
    output_directory, config_paths = gatherConfigs()
    if not config_paths:
        raise Framework.TrainingError('no valid config file found')
    # create output directory
    output_directory = Path(__file__).resolve().parents[1] / 'output' / output_directory
    os.makedirs(str(output_directory), exist_ok=False)
    # run trainings
    Logger.log(f'running sequential training for {len(config_paths)} configurations')
    Logger.log(f'outputs will be available at: {output_directory}')
    success, failed = 0, 0
    metric_files = {}
    for config in config_paths:
        try:
            # run training for single config
            output = Queue()
            training_process = Process(target=training_process_helper, args=(output, config))
            training_process.start()
            training_process.join()
            output_directory_run, model_name, num_iterations = output.get(block=False)
            shutil.move(output_directory_run, output_directory)
            metric_file: Path = output_directory / Path(output_directory_run).name / \
                f'test_{num_iterations}' / 'metrics_8bit.txt'
            # detect output metric file
            if metric_file.is_file():
                metric_files[model_name] = metric_file
            success += 1
        except Exception as e:
            Logger.logError(f'training failed for config file: "{config}" with error:\n{e}')
            failed += 1
    Logger.log(f'\nfinished sequential training ({success} successful, {failed} failed)')
    # combine training results in single table
    writeOutputMetrics(metric_files, output_directory)


if __name__ == '__main__':
    main()
