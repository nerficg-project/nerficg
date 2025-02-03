# -- coding: utf-8 --

"""Base/utils.py: Contains utility functions used for the implementation of the available NeRF methods."""

import time
import git
from contextlib import AbstractContextManager, nullcontext
from typing import Callable
from pathlib import Path

import torch

import Framework
from Logging import Logger


class CallbackTimer(AbstractContextManager):
    """Measures system-wide time elapsed during function call."""

    def __init__(self) -> None:
        self.duration: float = 0
        self.num_calls: int = 0

    def getValues(self) -> tuple[float, float, int]:
        """Returns absolute time, average time per call, and number of total calls of the callback."""
        return self.duration, self.duration / self.num_calls if self.num_calls > 0 else self.duration, self.num_calls

    def __enter__(self) -> None:
        """Starts the timer."""
        self.start = time.perf_counter()

    def __exit__(self, *_) -> None:
        """Stops the timer and adds the elapsed time to the total execution duration."""
        if Framework.config.GLOBAL.GPU_INDICES is not None:
            torch.cuda.synchronize(device=None)
        self.end = time.perf_counter()
        self.duration += (self.end - self.start)
        self.num_calls += 1


def callbackDecoratorFactory(callback_type: int = 0, active: bool = True, priority: int = 50,
                             start_iteration: (int | str | None) = None, end_iteration: (int | str | None) = None,
                             iteration_stride: (int | str | None) = None) -> Callable:
    """
    Decorator registering class members as training Callbacks. If argument is of type string, the value will be copied from the corresponding config variable.
    Arguments:
        callback_type       Indicates if callback is executed before, during or after training.
        active              Used to deactivate callbacks
        priority            Determines order of callback execution (higher priority first).
        start_iteration     Index of first iteration where this callback is called.
        end_iteration       Last iteration where this callback is called (exclusive).
        iteration_stride    Number of iterations between callback calls.
    """
    def decorator(function: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)
        wrapper.callback_type = callback_type
        wrapper.active = active
        wrapper.priority = priority
        wrapper.start_iteration = start_iteration
        wrapper.end_iteration = end_iteration
        wrapper.iteration_stride = iteration_stride
        wrapper.timer = nullcontext()
        wrapper.__name__ = function.__name__
        return wrapper
    return decorator


def trainingCallback(active: bool | str = True, priority: int = 50, start_iteration: (int | str | None) = None,
                     end_iteration: (int | str | None) = None, iteration_stride: (int | str | None) = None) -> Callable:
    """Training callback decorator."""
    return callbackDecoratorFactory(0, active, priority, start_iteration, end_iteration, iteration_stride)


def preTrainingCallback(active: bool | str = True, priority: int = 50) -> Callable:
    """Pre-training callback decorator."""
    return callbackDecoratorFactory(-1, active, priority)


def postTrainingCallback(active: bool | str = True, priority: int = 50) -> Callable:
    """Post-training callback decorator."""
    return callbackDecoratorFactory(1, active, priority)


def getGitCommit() -> str | None:
    """Writes current git commit to model"""
    Logger.logInfo('Checking git status')
    parent_path = Path(__file__).resolve().parents[3]
    try:
        repo = git.Repo(parent_path)
        if repo.is_dirty(untracked_files=True):
            Logger.logWarning('Detected uncommitted changes in your git repository. Using the latest commit as reference.')
        return f'{repo.active_branch}:{repo.head.commit.hexsha}'
    except git.InvalidGitRepositoryError:
        Logger.logInfo(f'Could not find git repository at "{parent_path}"')
    return None
