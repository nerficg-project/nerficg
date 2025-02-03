#! /usr/bin/env python3
# -- coding: utf-8 --

"""install.py: installs specific extensions or all extensions required by a given method."""

import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path
import importlib
import warnings
from types import ModuleType

import utils

with utils.discoverSourcePath():
    import Framework
    from Implementations import Methods as MI
    from Logging import Logger


def installExtension(install_name: str, install_command: list[str]) -> bool:
    """Installs a single extension."""
    Logger.logInfo(f'Installing extension {install_name}...')
    result = subprocess.run(install_command, check=False)
    if result.returncode != 0:
        Logger.logError(f'Failed to install extension "{install_name}" with command: "{install_command if isinstance(install_command, str) else " ".join(install_command)}"')
    return result.returncode == 0


def importExtension(extension_path: str) -> ModuleType:
    """Imports an extension module."""
    extension_spec = Path(extension_path).resolve()
    if extension_spec.is_dir():
        extension_spec = extension_spec / '__init__.py'
    extension_spec = importlib.util.spec_from_file_location(str(extension_spec.stem), str(extension_spec))
    extension_module = importlib.util.module_from_spec(extension_spec)
    extension_spec.loader.exec_module(extension_module)
    return extension_module


def main(extension_path: str, method_name: str) -> None:
    """Installs extensions required by a given method or a specific extension."""
    Framework.setup()
    essential_modules = set(sys.modules.keys())
    if extension_path is not None:
        try:
            extension_module = importExtension(extension_path)
            install_command = extension_module.__install_command__
            install_name = extension_module.__extension_name__
        except Framework.ExtensionError as e:
            install_name = e.__extension_name__
            install_command = e.__install_command__
        except FileNotFoundError:
            Logger.logError(f'Invalid extension path "{extension_path}": Module not found.')
            return
        except AttributeError as e:
            Logger.logError(f'Invalid extension module "{extension_path}": {e}')
            return
        if not installExtension(install_name, install_command):
            return
    if method_name is not None:
        if method_name not in MI.options:
            Logger.logError(f'Invalid method name "{method_name}".\nAvailable methods are: {MI.options}')
            return
        Logger.logInfo(f'Installing extensions for method "{method_name}"...')
        last_installed = None
        with utils.discoverSourcePath():
            while True:
                try:
                    all_modules = set(sys.modules.keys())
                    for module in all_modules.symmetric_difference(essential_modules):
                        del sys.modules[module]
                    MI._import(method_name)
                    break
                except Framework.ExtensionError as e:
                    if last_installed == e.__extension_name__:
                        Logger.logError(f'Failed to install extension "{e.__extension_name__}" with command: "{e.__install_command__ if isinstance(e.__install_command__, str) else " ".join(e.__install_command__)}"')
                        return
                    last_installed = e.__extension_name__
                    if not installExtension(e.__extension_name__, e.__install_command__):
                        return
                except Exception as e:
                    Logger.logError(f'Unexpected error during method import: {e}')
                    return
    Logger.logInfo('done')


if __name__ == '__main__':
    # parse arguments
    parser: ArgumentParser = ArgumentParser(prog='Install')
    parser.add_argument(
        '-m', '--method', action='store', dest='method_name', default=None,
        metavar='method_name', required=False,
        help='Name of the method to install extensions for.'
    )
    parser.add_argument(
        '-e', '--extension', action='store', dest='extension_path', default=None,
        metavar='extension_path', required=False,
        help='Path to extension to be installed.'
    )
    args = parser.parse_args()
    # run
    Logger.setMode(Logger.MODE_VERBOSE)
    warnings.filterwarnings('ignore')
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    main(args.extension_path, args.method_name)
