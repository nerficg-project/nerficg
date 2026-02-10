#! /usr/bin/env python3

"""benchmark.py: Benchmarks the specified method on the specified dataset based on an exemplary config file."""

import os
import shutil
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean

import torch
from tabulate import tabulate
import yaml

import utils
with utils.DiscoverSourcePath():
    import Framework
    from Implementations import Methods as MI
    from Implementations import Datasets as DI
    from Datasets.utils import list_sorted_directories
    try:
        import Thirdparty.TinyCudaNN as tcnn
    except ImportError:
        tcnn = None


output_formatting = {
    'PSNR': lambda value: f'{value:.2f}',
    'SSIM': lambda value: f'{value:.3f}',
    'LPIPS': lambda value: f'{value:.3f}',
    'VRAM_allocated': lambda value: f'{(value / 1024 ** 3):.2f}',
    'VRAM_reserved': lambda value: f'{(value / 1024 ** 3):.2f}',
    'default': str,
}

custom_configurations = {
    'dataset': {
        'mipnerf360': {
            'IMAGE_SCALE_FACTOR': lambda scene: 0.5 if scene in ['bonsai', 'counter', 'kitchen', 'room'] else 0.25,
        },
        'OmniBlender': {
            'NEAR_PLANE': lambda scene: 0.05 if scene in ['fisher-hut'] else (0.01 if scene in ['archiviz-flat', 'barbershop', 'classroom', 'restroom', 'LOU'] else 0.1),
        }
    },
    'global': {},
    'training': {},
    'model': {},
    'renderer': {},
    'method': {},
}

# if enabled and config has TRAINING.USE_MCMC=True, TRAINING.MAX_PRIMITIVES will be updated based on scene name
ENABLE_CUSTOM_MCMC_COUNTS = True
mcmc_counts = {
    'bicycle': 6_131_954,
    'bonsai': 1_244_819,
    'counter': 1_222_956,
    'flowers': 3_636_448,
    'garden': 5_834_784,
    'kitchen': 1_852_335,
    'room': 1_593_376,
    'stump': 4_961_797,
    'treehill': 3_783_761,
}


def main():
    # setup
    Framework.setup(require_custom_config=True)
    warnings.filterwarnings('ignore')
    original_config_name = Framework.config.path.stem

    # determine scenes to run
    dataset_root_path = Path(Framework.config.DATASET.PATH).resolve().parents[0]
    scene_names = list_sorted_directories(dataset_root_path)
    print(f'found scenes {scene_names}')

    # prepare customizations
    dataset_customizations = custom_configurations['dataset'].get(os.path.basename(dataset_root_path), {})

    output_directories = []
    for scene_name in scene_names:
        print(f'\n=== running scene "{scene_name}" ===')

        # make sure to reset everything between runs
        torch.cuda.reset_peak_memory_stats()
        Framework.set_random_seed()

        # modify config for current scene
        for key, value in dataset_customizations.items():
            Framework.config.DATASET[key] = value(scene_name)
            print(f'overriding DATASET.{key} = {Framework.config.DATASET[key]}')
        Framework.config.TRAINING.MODEL_NAME = f'{scene_name}_{original_config_name}'
        Framework.config.DATASET.PATH = str(dataset_root_path / scene_name)
        if ENABLE_CUSTOM_MCMC_COUNTS and Framework.config.TRAINING.get('USE_MCMC', False):
            if scene_name in mcmc_counts:
                Framework.config.TRAINING.MAX_PRIMITIVES = mcmc_counts[scene_name]
                print(f'overriding TRAINING.MAX_PRIMITIVES = {Framework.config.TRAINING.MAX_PRIMITIVES:,}')

        # set up training instance and dataset
        training_instance = MI.get_training_instance(
            method=Framework.config.GLOBAL.METHOD_TYPE,
            checkpoint=None
        )
        dataset = DI.get_dataset(
            dataset_type=Framework.config.GLOBAL.DATASET_TYPE,
            path=Framework.config.DATASET.PATH
        )

        # append output directory
        output_directory = training_instance.output_directory
        output_directories.append(output_directory)

        # update config with scene-specific overrides
        scene_config_path = output_directory / 'training_config.yaml'
        with open(scene_config_path, 'r') as f:
            scene_config = yaml.safe_load(f)
        for key, value in dataset_customizations.items():
            scene_config['DATASET'][key] = Framework.config.DATASET[key]
        scene_config['TRAINING']['MODEL_NAME'] = Framework.config.TRAINING.MODEL_NAME
        scene_config['DATASET']['PATH'] = Framework.config.DATASET.PATH
        if ENABLE_CUSTOM_MCMC_COUNTS and Framework.config.TRAINING.get('USE_MCMC', False):
            scene_config['TRAINING']['MAX_PRIMITIVES'] = Framework.config.TRAINING.MAX_PRIMITIVES
        with open(scene_config_path, 'w') as f:
            yaml.safe_dump(scene_config, f, sort_keys=False)

        # run training
        training_instance.run(dataset)

        # clean up
        del training_instance
        del dataset
        if tcnn is not None:
            tcnn.free_temporary_memory()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    parsed_values = []
    parsed_timings = []
    parsed_vram_stats = []
    for scene_name, output_directory in zip(scene_names, output_directories):
        metrics_file_path = output_directory / f'test_{Framework.config.TRAINING.NUM_ITERATIONS}'
        try:
            with open(metrics_file_path / 'metrics_8bit.txt') as f:
                for line in f:
                    pass
            parsed_values.append({i[0]: float(i[1]) for i in [j.split(':') for j in line.split(' ')]})
        except Exception:
            parsed_values.append({
                'PSNR': float(0.0),
                'SSIM': float(0.0),
                'LPIPS': float(0.0),
            })
        if Framework.config.TRAINING.TIMING.ACTIVATE:
            try:
                with open(output_directory / 'timings.txt') as f:
                    for line in f:
                        pass
                parsed_timings.append({i[0]: float(i[1]) for i in [j.split(':') for j in line.split(' ')]})
            except Exception as e:
                raise Exception(f'timings file is missing or invalid: {e}') from e
        if Framework.config.TRAINING.WRITE_VRAM_STATS:
            try:
                with open(output_directory / 'vram_stats.txt') as f:
                    for line in f:
                        pass
                parsed_vram_stats.append({i[0]: float(i[1]) for i in [j.split(':') for j in line.split(' ')]})
            except Exception as e:
                raise Exception(f'vram stats file is missing or invalid: {e}') from e
    headers = ['Metric'] + scene_names + ['Mean']
    table = [[metric_name] + [
        output_formatting.get(metric_name, output_formatting['default'])(run[metric_name]) for run in parsed_values
    ] for metric_name in parsed_values[0].keys()]
    for row in table:
        row.append(output_formatting.get(row[0], output_formatting['default'])(
            mean(run[row[0]] for run in parsed_values)
        ))
    if parsed_timings:
        timing_table = [[timing_name] + [
            output_formatting.get(timing_name, output_formatting['default'])(
                timedelta(seconds=round(run[timing_name]))
            ) for run in parsed_timings
        ] for timing_name in parsed_timings[0].keys()]
        for row in timing_table:
            row.append(output_formatting.get(row[0], output_formatting['default'])(
                timedelta(seconds=round(mean(run[row[0]] for run in parsed_timings)))
            ))
        table += timing_table
    if parsed_vram_stats:
        vram_stats_table = [[vram_stat_name] + [
            output_formatting.get(vram_stat_name, output_formatting['default'])(
                run[vram_stat_name]
            ) for run in parsed_vram_stats
        ] for vram_stat_name in parsed_vram_stats[0].keys()]
        for row in vram_stats_table:
            row.append(output_formatting.get(row[0], output_formatting['default'])(
                mean(run[row[0]] for run in parsed_vram_stats)
            ))
        table += vram_stats_table
    table_string = tabulate(table, headers, colalign=['left'] + ['center'] * (len(table[0]) - 1), disable_numparse=True)
    print(f'\n=== summary ===')
    print(table_string)
    # create output directory, move all results there, and create summary file
    output_directory_name = f'{original_config_name}_{datetime.now():%Y-%m-%d-%H-%M-%S}'
    output_directory = Framework.Directories.OUTPUT_DIR / output_directory_name
    os.makedirs(output_directory, exist_ok=False)
    for result_directory in output_directories:
        shutil.move(result_directory, output_directory)
    with open(output_directory / 'summary.txt', 'w') as f:
        f.write(table_string)
    with open(output_directory / 'latex_tables.txt', 'w') as f:
        f.write('\n\n'.join(
            tabulate(
                table,
                headers,
                colalign=['left'] + ['center'] * (len(table[0]) - 1),
                disable_numparse=True,
                tablefmt=table_format)
            for table_format in ['plain', 'latex', 'latex_raw', 'latex_booktabs', 'latex_longtable']
        ))


if __name__ == '__main__':
    main()
