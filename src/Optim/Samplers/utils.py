"""Samplers/utils.py: Utilities for image and ray sampling routines."""

import torch

import Framework


class SequentialSampler:
    def __init__(self, num_elements: int) -> None:
        self.num_elements = num_elements
        self.indices = torch.arange(self.num_elements)
        self.reset()

    def shuffle(self) -> None:
        pass

    def reset(self) -> None:
        self.current_id = 0
        self.shuffle()

    def get(self, num_samples: int) -> torch.Tensor:
        if num_samples > self.num_elements:
            raise Framework.SamplerError(f"cannot draw {num_samples} samples from {self.num_elements} elements")
        if self.current_id + num_samples > self.num_elements:
            self.reset()
        samples = self.indices[self.current_id:self.current_id + num_samples]
        self.current_id += num_samples
        return samples


class RandomSequentialSampler(SequentialSampler):

    def shuffle(self) -> None:
        self.indices = self.indices[torch.randperm(self.num_elements)]


class IncrementalSequentialSampler:

    def __init__(self, num_elements: int) -> None:
        self.num_elements = num_elements
        self.current_size = 0
        self.indices = torch.arange(self.num_elements)
        self.reset()

    def reset(self) -> None:
        self.current_size = min(self.current_size + 1, self.num_elements)
        self.current_indices = self.indices[:self.current_size]
        self.current_id = 0

    def get(self, num_samples: int) -> torch.Tensor:
        if num_samples > self.current_size:
            raise Framework.SamplerError(f'cannot draw {num_samples} samples from {self.current_size} elements')
        if self.current_id + num_samples > self.current_size:
            self.reset()
        samples = self.current_indices[self.current_id:self.current_id + num_samples]
        self.current_id += num_samples
        return samples
