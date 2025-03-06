# -- coding: utf-8 --
"""Framework.py: Contains all functions that are part of the framework's default setup process."""

__author__ = 'Moritz Kappel and Florian Hahlbohm'
__credits__ = ['Moritz Kappel', 'Florian Hahlbohm', 'Timon Scholz']
__license__ = 'MIT'
__maintainer__ = 'Moritz Kappel'
__email__ = 'kappel@cg.cs.tu-bs.de'
__status__ = 'Development'

development_versions: dict[str, str] = {
    'python': '3.11.8',
    'pytorch': '2.5.1',
    'cuda': '11.8',
    'cudnn': '90100'
}

import ast
import os
import warnings
import platform
import random
from argparse import ArgumentParser
from multiprocessing import set_start_method
from pathlib import Path
import yaml
from munch import Munch
from typing import Any, Type, TypeVar

import numpy as np
import torch

from Logging import Logger


T = TypeVar("T")


class ConfigParameterList(Munch):

    def recursiveUpdate(self, other: 'ConfigParameterList'):
        # check type
        if not isinstance(other, ConfigParameterList):
            raise TypeError()
        # copy list
        other = other.copy()
        # recursively update sublists
        for key, value in [(i, other[i]) for i in other]:
            if isinstance(value, ConfigParameterList) and hasattr(self, key) and isinstance(self[key], ConfigParameterList):
                self[key].recursiveUpdate(value)
                del other[key]
        # update remaining contents
        self.update(other)


class ConfigWrapper(ConfigParameterList):
    _warned = set()

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            default_global = getDefaultGlobalConfig()
            if item in default_global.__dict__:
                if item not in self._warned:
                    Logger.logWarning(f'Parameter missing in loaded config: GLOBAL.{item}')
                    self._warned.add(item)
                return default_global.__getattr__(item)

            raise AttributeError(f'Parameter missing in loaded config: {item}') from None


class Configurable:

    _configuration: ConfigParameterList = ConfigParameterList()

    def __init__(self, config_file_data_field: str) -> None:
        # gather default class configuration
        self.config_file_data_field: str = config_file_data_field
        instance_params = self.__class__._configuration.copy()
        # overwrite from config file
        if not hasattr(config, self.config_file_data_field):
            Logger.logWarning(f'data field {self.config_file_data_field} requested by class {self.__class__.__name__} is not available in config file. \
                              Using default config.')
        else:
            instance_params.recursiveUpdate(config[self.config_file_data_field])
        # assign to instance
        for param in instance_params:
            self.__dict__[param] = instance_params[param]

    @classmethod
    def getDefaultParameters(cls):
        return cls._configuration

    @staticmethod
    def configure(**params):
        newParams = ConfigParameterList(params)

        def configDecorator(cls: Type[T]) -> Type[T]:
            if not issubclass(cls, Configurable):
                raise FrameworkError(f'configure decorator must be applied to subclass of Configurable, but got {cls.__class__}')
            cls._configuration = cls._configuration.copy()
            cls._configuration.recursiveUpdate(newParams)
            return cls
        return configDecorator


def setup(require_custom_config: bool = False, config_path: str | None = None) -> list[str] | None:
    """Performs a complete training setup based on the config file provided via comment line (or default)"""
    # set multiprocessing start method
    try:
        set_start_method('spawn')
    except RuntimeError:
        Logger.logWarning('multiprocessing start method already set')
    # parse arguments and load config
    config_args: dict[str, str] = {}
    unknown_args = None
    if config_path is None:
        # parse arguments to retrieve config file location
        parser: ArgumentParser = ArgumentParser(prog='Framework', add_help=False)
        parser.add_argument(
            '-c', '--config', action='store', dest='config_path', default=None,
            metavar='path/to/config_file.yaml', required=False, nargs='*',
            help='The .yaml file containing the project configuration.'
        )
        args, unknown_args = parser.parse_known_args()
        if args.config_path is not None:
            # parse extra args overwriting values in config (for wandb sweeps)
            config_path: str = args.config_path[0]
            config_args: dict[str, str] = {}
            for config_arg in args.config_path[1:]:
                try:
                    key, value = config_arg.split('=')
                    config_args[key] = value
                except ValueError:
                    raise FrameworkError(f'invalid config overwrite argument syntax: "{config_arg}" (expected syntax: config_key=config_value).')

    # initialize config and
    loadConfig(config_path, require_custom_config, config_args)
    # filter warnings
    if config.GLOBAL.FILTER_WARNINGS:
        warnings.filterwarnings('ignore')
    # call init methods
    checkLibraryVersions()
    setupTorch()
    setRandomSeed()
    # return unused arguments to application
    return unknown_args


def loadConfig(config_path, require_custom_config, config_args) -> None:
    """Loads project configuration from .yaml file."""
    global config
    Logger.setMode(Logger.MODE_VERBOSE)
    if config_path is not None:
        try:
            yaml_dict: dict[str, Any] = yaml.unsafe_load(open(config_path))
            config = ConfigWrapper.fromDict(yaml_dict)
            Logger.setMode(config.GLOBAL.LOG_LEVEL)
            Logger.log(f'configuration file loaded: {config_path}')
            Logger.logDebug(config)
            config._path = os.path.abspath(config_path)
        except IOError:
            raise FrameworkError(f'invalid config file path: "{config_path}"')
    else:
        if require_custom_config:
            raise FrameworkError('missing config file, please provide a config file path using the "-c / --config" argument.')
        config = ConfigWrapper(GLOBAL=getDefaultGlobalConfig())
        Logger.setMode(config.GLOBAL.LOG_LEVEL)
        Logger.log('using default configuration')

    # override single config elements from command line arguments
    for config_arg, value in config_args.items():
        try:
            value = ast.literal_eval(value)
        except ValueError:
            pass  # keep value as string
        elements = config_arg.split('.')
        param_name = elements[-1]
        param_path = elements[:-1]
        target_munch = config
        try:
            for key in param_path:
                target_munch = getattr(target_munch, key)
        except AttributeError:
            raise FrameworkError(f'invalid config file key "{key}" in config overwrite argument "{config_arg}={value}"')
        setattr(target_munch, param_name, value)


def getDefaultGlobalConfig() -> ConfigParameterList:
    """Returns the default values of all global configuration parameters."""
    return ConfigParameterList(
        LOG_LEVEL=Logger.MODE_VERBOSE,
        GPU_INDICES=[0],
        RANDOM_SEED=1618033989,
        ANOMALY_DETECTION=False,
        FILTER_WARNINGS=True,
        METHOD_TYPE=None,
        DATASET_TYPE=None
    )


def checkLibraryVersions() -> None:
    """Print icurrent library versions and add them to global config."""

    def printVersion(lib: str, version: str) -> None:
        tested_version = development_versions[lib.lower()] if lib.lower() in development_versions else None
        if version != tested_version:
            Logger.logWarning(f'current {lib} version: {version} (tested with {tested_version})')
        else:
            Logger.logInfo(f'current {lib} version: {version}')

    # python
    config.GLOBAL.PYTHON_VERSION = platform.python_version()
    printVersion('Python', config.GLOBAL.PYTHON_VERSION)

    # pytorch
    config.GLOBAL.TORCH_VERSION = torch.__version__.split('+')[0]
    printVersion('Pytorch', config.GLOBAL.TORCH_VERSION)

    # cuda
    config.GLOBAL.CUDA_VERSION = torch.version.cuda
    printVersion('CUDA', config.GLOBAL.CUDA_VERSION)

    # cudnn
    config.GLOBAL.CUDNN_VERSION = str(torch.backends.cudnn.version())
    printVersion('CUDNN', config.GLOBAL.CUDNN_VERSION)


def setRandomSeed() -> None:
    """Gets or sets the random seed for reproducibility (check https://pytorch.org/docs/stable/notes/randomness.html)"""
    # use a random seed if provided by config file
    if config.GLOBAL.RANDOM_SEED is None:
        config.GLOBAL.RANDOM_SEED = np.random.randint(0, 4294967295)
    Logger.logInfo(f'deterministic mode enabled using random seed: {config.GLOBAL.RANDOM_SEED}')
    torch.manual_seed(config.GLOBAL.RANDOM_SEED)
    random.seed(config.GLOBAL.RANDOM_SEED)
    np.random.seed(config.GLOBAL.RANDOM_SEED)  # legacy
    # torch.use_deterministic_algorithms(True) # decreases performance
    # torch.backends.cudnn.benchmark = False  # decreases performance


def setupTorch() -> None:
    """Initializes PyTorch by setting the default tensor type, GPU device and misc flags."""
    # cache directory for torch checkpoints etc.
    cache_path = Path(__file__).parents[1] / '.cache'
    # set torch hub cache directory
    torch.hub.set_dir(str(cache_path))
    # set default torch extension path
    os.environ['TORCH_EXTENSIONS_DIR'] = str(cache_path / 'Extensions')
    Logger.logInfo('initializing pytorch runtime')

    if config.GLOBAL.GPU_INDICES is None:
        Logger.logInfo('No GPU indices specified: entering CPU mode')
        config.GLOBAL.DEFAULT_DEVICE = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            Logger.logInfo('entering GPU mode')
            valid_indices: list[int] = [i for i in config.GLOBAL.GPU_INDICES if i in range(torch.cuda.device_count())]
            for item in [i for i in config.GLOBAL.GPU_INDICES if i not in valid_indices]:
                Logger.logWarning(f'requested GPU index {item} not available on this machine')
            config.GLOBAL.GPU_INDICES = valid_indices
            Logger.logInfo(f'using GPU(s): {str(config.GLOBAL.GPU_INDICES).replace(",", " (main),", 1)}')
            config.GLOBAL.DEFAULT_DEVICE = torch.device(f'cuda:{config.GLOBAL.GPU_INDICES[0]}')
            torch.cuda.set_device(config.GLOBAL.DEFAULT_DEVICE)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.fastest = True
        else:
            Logger.logWarning(
                f'GPU indices {config.GLOBAL.GPU_INDICES } requested but not supported by the machine: switching to CPU mode'
            )
            config.GLOBAL.DEFAULT_DEVICE = torch.device('cpu')
            config.GLOBAL.GPU_INDICES = []

    torch.autograd.set_detect_anomaly(config.GLOBAL.ANOMALY_DETECTION)
    # torch.set_default_device(config.GLOBAL.DEFAULT_DEVICE)  # TODO performance issues, replace default_tensor_type at some point
    torch.set_default_tensor_type(torch.cuda.FloatTensor if config.GLOBAL.GPU_INDICES else torch.FloatTensor)
    return


def setupWandb(project: str, entity: str, name: str) -> bool:
    """Sets up wandb for training visualization."""
    try:
        global wandb
        import wandb
        Logger.logInfo('setting up wandb')
        # os.environ["WANDB_CONSOLE"] = "off"
        os.environ["WANDB_SILENT"] = "true"
        log_path = Path(__file__).resolve().parents[1] / 'output'
        log_path.mkdir(parents=True, exist_ok=True)
        wandb.init(project=project, entity=entity, name=name, config=config.toDict(),
                   dir=str(log_path))
        Logger.logInfo(f'wandb logs will be available at: {wandb.run.url}')
    except ModuleNotFoundError:
        Logger.logWarning('wandb module not found: cannot log training details')
        config.TRAINING.WANDB.ACTIVATE = False
        return False
    return True


def teardown():
    global config, wandb
    if 'wandb' in globals():
        wandb.finish()
        del wandb
    if 'config' in globals():
        del config
    # torch.cuda.synchronize()
    # torch.cuda.empty_cache()
    Logger.logInfo('framework teardown complete')


# custom exceptions
class FrameworkError(Exception):
    """A generic exception class for errors that occur within this framework."""
    def __init__(self, msg):
        super().__init__(msg)
        Logger.logError(f'({self.__class__.__name__}) {msg}')


class MethodError(FrameworkError):
    """An error regarding general method implementation."""


class CheckpointError(FrameworkError):
    """Error loading or saving a model checkpoint."""


class RendererError(FrameworkError):
    """Error occurring during rendering."""


class ModelError(FrameworkError):
    """Errors regarding the model implementation."""


class TrainingError(FrameworkError):
    """A general error during training."""


class InferenceError(FrameworkError):
    """Raise in case of an exception inference configuration."""


class CameraError(FrameworkError):
    """Error within a camera model."""


class DatasetError(FrameworkError):
    """Error loadding or processing a dataset."""


class LossError(FrameworkError):
    """An error occuring during loss definition or calculation."""


class SamplerError(FrameworkError):
    """Sampler failed to sample values."""


class VisualizationError(FrameworkError):
    """Error visualization data."""


class GUIError(FrameworkError):
    """Error during GUI rendering or synchronization."""


class ExtensionError(FrameworkError):
    """Failed to load a third party or custom CUDA extension."""

    def __init__(self, name: str, install_command: str | list[str]) -> None:
        global config
        self.__extension_name__ = name
        self.__install_command__ = install_command
        msg = (f'Dependency "{name}" not found.\n'
               f'\tInstall the extension using "{install_command if isinstance(install_command, str) else " ".join(install_command)}",\n'
               f'\tor use "./scripts/install.py -m {config.GLOBAL.METHOD_TYPE}" to automatically install all dependencies of method "{config.GLOBAL.METHOD_TYPE}".'
               if 'config' in globals() and config.GLOBAL.METHOD_TYPE is not None else '')
        Exception.__init__(self, msg)
