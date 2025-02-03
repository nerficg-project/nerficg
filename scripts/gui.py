#! /usr/bin/env python3
# -- coding: utf-8 --

"""gui.py: opens graphical user interface."""

import sys

import torch.multiprocessing as mp

import utils
with utils.discoverSourcePath():
    from Logging import Logger

if __name__ == '__main__':
    Logger.setMode(Logger.MODE_VERBOSE)
    with utils.discoverSourcePath():
        try:
            from ICGui.launchViewer import main
        except ImportError as e:
            Logger.logError(f'Failed to open GUI: {e}\n'
                            f'Make sure the icgui submodule is initialized and '
                            f'updated before running this script. ')
            sys.exit(0)
        mp.set_start_method('spawn')
        main()
