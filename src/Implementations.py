# -- coding: utf-8 --

"""Implementations.py: Dynamically provides access to the implemented datasets and methods."""

import sys
import importlib
from types import ModuleType
from typing import Type
from pathlib import Path

import Framework
from Logging import Logger

from Methods.Base.Model import BaseModel
from Methods.Base.Renderer import BaseRenderer
from Methods.Base.Trainer import BaseTrainer
from Datasets.Base import BaseDataset


class Methods:
    """A class containing all implemented methods"""
    path: Path = Path(__file__).resolve().parents[0] / 'Methods'
    options: tuple[str] = tuple([i.name for i in path.iterdir() if i.is_dir() and i.name not in ['Base', '__pycache__']])
    modules: dict[str, ModuleType] = {}

    @staticmethod
    def _import(method: str) -> ModuleType:
        with setImportPaths():
            m = importlib.import_module(f'Methods.{method}')
        return m

    @staticmethod
    def importMethod(method: str) -> ModuleType:
        """imports the requested method module"""
        if method not in Methods.options:
            raise Framework.MethodError(f'requested invalid method type: {method}\navailable methods are: {Methods.options}')
        if method not in Methods.modules:
            try:
                Methods.modules[method] = Methods._import(method)
            except Exception as e:
                raise Framework.MethodError(f'failed to import method {method}:\n{e}')
        return Methods.modules[method]

    @staticmethod
    def getModel(method: str, checkpoint: str = None, name: str = 'Default') -> BaseModel:
        """returns a model of the given type loaded with the provided checkpoint"""
        Logger.logInfo('creating model')
        model_class: Type[BaseModel] = Methods.importMethod(method).MODEL
        return model_class.load(checkpoint) if checkpoint is not None else model_class(name).build()

    @staticmethod
    def getRenderer(method: str, model: BaseModel) -> BaseRenderer:
        """returns a renderer for the specified method initialized with the given model instance"""
        Logger.logInfo('creating renderer')
        model_class: Type[BaseRenderer] = Methods.importMethod(method).RENDERER
        return model_class(model)

    @staticmethod
    def getTrainingInstance(method: str, checkpoint: str | None = None) -> BaseTrainer:
        """returns a trainer of the given type loaded with the provided checkpoint"""
        Logger.logInfo('creating training instance')
        model_class: Type[BaseTrainer] = Methods.importMethod(method).TRAINING_INSTANCE
        if checkpoint is not None:
            return model_class.load(checkpoint)
        model = Methods.getModel(method=method, name=Framework.config.TRAINING.MODEL_NAME)
        renderer = Methods.getRenderer(method=method, model=model)
        return model_class(model=model, renderer=renderer)


class Datasets:
    """Dynamically loads and provides access to the implemented datasets."""
    path: Path = Path(__file__).resolve().parents[0] / 'Datasets'
    options: tuple[str] = tuple([i.name.split('.')[0] for i in path.iterdir() if i.is_file() and i.name not in ['Base.py', 'utils.py', 'datasets.md']])
    loaded: dict[str, Type[BaseDataset]] = {}

    @staticmethod
    def importDataset(dataset_type: str) -> None:
        if dataset_type in Datasets.options:
            try:
                with setImportPaths():
                    m = importlib.import_module(f'Datasets.{dataset_type}')
                Datasets.loaded[dataset_type] = m.CustomDataset
            except Exception:
                raise Framework.DatasetError(f'failed to import dataset: {dataset_type}')
        else:
            raise Framework.DatasetError(f'requested invalid dataset type: {dataset_type}\navailable datasets are: {Datasets.options}')

    @staticmethod
    def getDatasetClass(dataset_type: str) -> Type[BaseDataset]:
        if dataset_type not in Datasets.loaded:
            Datasets.importDataset(dataset_type)
        return Datasets.loaded[dataset_type]

    @staticmethod
    def getDataset(dataset_type: str, path: str) -> BaseDataset:
        """Returns a dataset instance of the given type loaded from the given path."""
        dataset_class: Type[BaseDataset] = Datasets.getDatasetClass(dataset_type)
        return dataset_class(path)


class setImportPaths:
    """helper context adding source code directory to pythonpath during dynamic imports"""

    def __init__(self, sub_path: Path = Path('')):
        self.sub_path = sub_path

    def __enter__(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[0] / self.sub_path))

    def __exit__(self, *_):
        sys.path.pop(0)
