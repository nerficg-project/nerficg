# -- coding: utf-8 --

"""Base/Model.py: Abstract base class for scene models."""

from abc import ABC, abstractmethod
import datetime
from pathlib import Path
from typing import Callable
import torch

import Framework
from Methods.Base.utils import getGitCommit
from Logging import Logger


class BaseModel(Framework.Configurable, ABC, torch.nn.Module):
    """Defines the basic PyTorch neural model."""

    def __init__(self, name: str = None) -> None:
        Framework.Configurable.__init__(self, 'MODEL')
        ABC.__init__(self)
        torch.nn.Module.__init__(self)
        self.model_name: str = name if name is not None else 'Default'
        self.creation_date: str = f'{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}'
        self.num_iterations_trained: int = 0
        self.git_commit: str = None
        self.output_directory: Path = Path(__file__).resolve().parents[3] / 'output' / str(Framework.config.GLOBAL.METHOD_TYPE) / f'{self.model_name}_{self.creation_date}'

    @abstractmethod
    def build(self) -> 'BaseModel':
        """
        Automatically called after model constructor during model initialization / checkpoint loading.
        This function should create / register all submodules, model parameters and buffers with correct shape based on the current configuration.
        If a parameter is of dynamic shape, i.e. the shape depends on the training data and might differ between checkpoints,
        this parameter should be registered as None type using: self.register_buffer('param_name', None).
        """
        return self

    def forward(self) -> None:
        """Invalidates forward passes of model as all models are executed exclusively through renderers."""
        Logger.logError('Model cannot be executed directly. Use a Renderer instead.')

    def __repr__(self) -> str:
        """Returns string representation of the model's metadata."""
        params_string = ''
        additional_parameters = type(self).getDefaultParameters().keys()
        if additional_parameters:
            params_string += '\t Additional parameters:'
            for param in additional_parameters:
                params_string += f'\n\t\t{param}: {self.__dict__[param]}'
        return f'<instance of class: {self.__class__.__name__}\n' \
               f'\t model name: {self.model_name}\n' \
               f'\t created on: {self.creation_date}\n' \
               f'\t output directory: {self.output_directory}\n' \
               f'\t git commit: {self.git_commit}\n' \
               f'\t trained for: {self.num_iterations_trained} iterations\n' \
               f'{params_string}>'

    @classmethod
    def load(cls, checkpoint_name: str | None,
             map_location: Callable = lambda storage, location: storage) -> 'BaseModel':
        """Loads a saved model from '.pt' file."""
        if checkpoint_name is None or checkpoint_name.split('.')[-1] != 'pt':
            raise Framework.ModelError(f'Invalid model checkpoint: "{checkpoint_name}"')
        try:
            # load checkpoint
            checkpoint_path = Path(__file__).resolve().parents[3]
            checkpoint = torch.load(checkpoint_path / checkpoint_name, map_location=map_location)
            # create new model
            model = cls()
            # load model configuration
            for param in ['model_name', 'creation_date', 'num_iterations_trained', 'git_commit', 'output_directory'] + list(cls.getDefaultParameters().keys()):
                try:
                    model.__dict__[param] = checkpoint[param]
                except KeyError:
                    Logger.logWarning(f'failed to load model parameter "{param}" -> using default value: "{model.__dict__[param]}"')
            # build the model
            model.build()
            # load model parameters
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            # print warnings for missing keys
            for key in missing_keys:
                Logger.logWarning(f'missing key in model checkpoint: "{key}"')
            # add parameters of dynamic size
            for key in unexpected_keys:
                target = model
                attr_name = key
                while '.' in attr_name:
                    sub_target, attr_name = key.split('.', 1)
                    target = getattr(model, sub_target)
                if attr_name in target._parameters:
                    setattr(target, attr_name, torch.nn.Parameter(checkpoint['model_state_dict'][key]))
                else:
                    if hasattr(target, attr_name):
                        delattr(target, attr_name)
                    target.register_buffer(attr_name, checkpoint['model_state_dict'][key])
            model.to(Framework.config.GLOBAL.DEFAULT_DEVICE)
        except IOError as e:
            raise Framework.ModelError(f'failed to load model from file: "{e}"')
        # check git commit
        if model.git_commit is not None:
            git_commit_id = getGitCommit()
            if git_commit_id != model.git_commit:
                Logger.logWarning(f'Git status mismatch (Model "{model.git_commit}", Current "{git_commit_id}").\n'
                                  '\tCheck out the correct branch/commit for reproducibility.')
        return model

    def save(self, path: Path) -> None:
        """Saves the current model as '.pt' file."""
        try:
            checkpoint = {'model_state_dict': self.state_dict()}
            for param in ['model_name', 'creation_date', 'num_iterations_trained', 'git_commit', 'output_directory'] + list(type(self).getDefaultParameters().keys()):
                checkpoint[param] = self.__dict__[param]
            torch.save(checkpoint, path)
        except IOError as e:
            Logger.logWarning(f'failed to save model: "{e}"')

    def exportTorchScript(self, path: Path) -> None:
        """Exports model as torch script module (e.g. for execution in c++)"""
        try:
            script_module = torch.jit.script(self)
            script_module.save(str(path))
        except IOError as e:
            Logger.logWarning(f'failed to generate script module: "{e}"')

    def numModuleParameters(self, trainable_only=False) -> int:
        """Returns the model's number of parameters."""
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))
