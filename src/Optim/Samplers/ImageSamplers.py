"""Samplers/ImageSamplers.py: Samplers selecting a subset of rays from a given ray batch."""

from abc import ABC, abstractmethod

import torch

from Optim.Samplers.utils import RandomSequentialSampler, SequentialSampler


class ImageSampler(ABC):
    """Abstract base class for image samplers."""

    def __init__(self, num_elements: int) -> None:
        super().__init__()
        self.num_elements = num_elements

    @abstractmethod
    def get(self, ray_batch_size: int) -> torch.Tensor:
        pass

    def update(self, **_) -> None:
        pass


class SequentialImageSampler(ImageSampler):

    def __init__(self, num_elements: int) -> None:
        super().__init__(num_elements)
        self.sampler = SequentialSampler(num_elements=self.num_elements)

    def get(self, ray_batch_size: int) -> torch.Tensor:
        return self.sampler.get(num_samples=ray_batch_size)


class SequentialRandomImageSampler(SequentialImageSampler):

    def __init__(self, num_elements: int) -> None:
        super().__init__(num_elements)
        self.sampler = RandomSequentialSampler(num_elements=self.num_elements)


class RandomImageSampler(ImageSampler):

    def get(self, ray_batch_size: int) -> torch.Tensor:
        return torch.randint(low=0, high=self.num_elements, size=(ray_batch_size,))


class MultinomialImageSampler(ImageSampler):

    def __init__(self, num_elements: int) -> None:
        super().__init__(num_elements)
        self.pdf = torch.ones(size=(self.num_elements,))

    def get(self, ray_batch_size: int) -> torch.Tensor:
        return torch.multinomial(input=self.pdf, num_samples=ray_batch_size)

    @torch.no_grad()
    def update(self, ray_ids: torch.Tensor, weights: torch.Tensor, constant_addend: float = False) -> None:
        if constant_addend is not None:
            self.pdf += constant_addend
        self.pdf[ray_ids] = weights
