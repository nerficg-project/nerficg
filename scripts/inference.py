#! /usr/bin/env python3

"""inference.py: Renders outputs from a pretrained model."""

from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import torch

import utils
with utils.DiscoverSourcePath():
    import Framework
    from Logging import Logger
    from Implementations import Methods as MI
    from Implementations import Datasets as DI
    from Visual.Trajectories import CameraTrajectory


def main(*, base_dir: Path, checkpoint_name: str, subsets: list[str] | None, calculate_metrics: bool,
         closest_train: bool, visualize_errors: bool, benchmark: bool) -> None:
    # setup framework
    Framework.setup(config_path=str(base_dir / 'training_config.yaml'), require_custom_config=True)
    # load dataset, model, and renderer
    dataset = DI.get_dataset(
        dataset_type=Framework.config.GLOBAL.DATASET_TYPE,
        path=Framework.config.DATASET.PATH
    )
    model = MI.get_model(
        method=Framework.config.GLOBAL.METHOD_TYPE,
        checkpoint=str(base_dir / 'checkpoints' / checkpoint_name),
    ).eval()
    renderer = MI.get_renderer(
        method=Framework.config.GLOBAL.METHOD_TYPE,
        model=model
    )
    # render subsets
    if subsets:
        if 'all' in subsets:
            subsets = dataset.subsets + CameraTrajectory.list_options()
        subsets = list(set(subsets))
        subsets.sort()
        for subset in subsets:
            if subset not in dataset.subsets:
                if subset not in CameraTrajectory.list_options():
                    Logger.log_warning(f'Skipping unknown subset or camera trajectory: {subset}.')
                    continue
                trajectory = CameraTrajectory.get(subset)()
                trajectory.add_to_dataset(dataset, reference_set='train')
            dataset.set_mode(subset)
            renderer.render_subset(
                output_directory=base_dir / 'inference',
                dataset=dataset,
                calculate_metrics=calculate_metrics,
                visualize_errors=visualize_errors,
                verbose=True,
                image_extension='png',
                save_gt=False,
                closest_train=closest_train,
            )
    # performance benchmark
    if benchmark:
        NUM_ITERATIONS = 100  # number of times the test set is rendered to calculate the online FPS
        Logger.log_info('Benchmarking online FPS (this might take a while)...')
        # if no test images are available, use the training set
        if len(dataset.test()) == 0:
            if len(dataset.train()) == 0:
                raise Framework.InferenceError('No images found for benchmarking.')
        # warmup
        for view in Logger.log_progress(dataset, leave=False, desc='Warming Up'):
            renderer.render_image(view, benchmark=True)
        # render
        num_test_images = len(dataset)
        torch.cuda.synchronize()
        start_time = perf_counter()
        for _ in Logger.log_progress(range(NUM_ITERATIONS), leave=False, desc='Benchmarking Performance'):
            for view in dataset:
                renderer.render_image(view, benchmark=True)
        torch.cuda.synchronize()
        end_time = perf_counter()
        # write output
        benchmark_output_path = base_dir / f'performance_{model.num_iterations_trained}.txt'
        total_time = end_time - start_time
        total_time_ms = total_time * 1000
        total_num_images = NUM_ITERATIONS * num_test_images
        avg_fps = total_num_images / total_time
        avg_ms_per_image = total_time_ms / total_num_images
        with open(str(benchmark_output_path), 'w') as f:
            f.write(f'Number of test set renders: {NUM_ITERATIONS}\n')
            f.write(f'Number of test set images: {num_test_images}\n')
            f.write(f'Test set image size: {view.camera.width}x{view.camera.height}\n')
            f.write(f'Total rendering time: {total_time_ms:.2f} ms\n')
            f.write(f'Average rendering time per image: {avg_ms_per_image:.2f} ms\n')
            f.write(f'Average FPS: {avg_fps:.2f}\n')
        Logger.log_info(f'Average FPS: {avg_fps:.2f} ({avg_ms_per_image:.2f} ms)')
        Logger.log_info(f'Performance benchmark results written to {benchmark_output_path}.')
    Logger.log_info('Done.')


if __name__ == '__main__':
    parser = ArgumentParser(
        prog='inference.py',
        description='Renders outputs from a pretrained model.'
    )
    parser.add_argument(
        '-d', '--dir', action='store', dest='base_dir', default=None,
        metavar='path/to/output/directory', required=True,
        help='A directory containing the outputs of a completed training.'
    )
    parser.add_argument(
        '-s', '--subsets', action='store', dest='subsets', default=None,
        metavar='subsets', required=False, nargs='*', type=str,
        help='Dataset subsets to render. If "all", all available subsets are rendered.'
    )
    parser.add_argument(
        '-m', '--metrics', action='store_true', dest='calculate_metrics',
        help='Calculates standard metrics for the rendered images, if ground truth is available.'
    )
    parser.add_argument(
        '-b', '--benchmark', action='store_true', dest='benchmark',
        help='Calculates the online FPS by repeatedly rendering the test set without output visualization/saving.'
    )
    parser.add_argument(
        '--closest_train', action='store_true', dest='closest_train',
        help='Renders the closest ground truth training image to the requested camera pose.'
    )
    parser.add_argument(
        '--visualize_errors', action='store_true', dest='visualize_errors',
        help='Visualizes the errors between the rendered and ground truth images, if available.'
    )
    parser.add_argument(
        '--checkpoint', action='store', dest='checkpoint_name', default='final.pt',
        metavar='checkpoint_name', required=False,
        help='The name of the checkpoint file to use for inference.'
    )
    args, _ = parser.parse_known_args()
    Logger.set_mode(Logger.MODE_VERBOSE)
    main(
        base_dir=Path(args.base_dir),
        checkpoint_name=args.checkpoint_name,
        subsets=args.subsets,
        calculate_metrics=args.calculate_metrics,
        closest_train=args.closest_train,
        visualize_errors=args.visualize_errors,
        benchmark=args.benchmark,
    )
