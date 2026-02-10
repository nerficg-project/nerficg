"""Samplers/DatasetSamplers.py: Samplers returning a batch of rays from a dataset."""

import Framework
from Datasets.Base import BaseDataset
from Datasets.utils import View, RayBatch
from Optim.Samplers.ImageSamplers import ImageSampler
from Optim.Samplers.utils import IncrementalSequentialSampler, RandomSequentialSampler, SequentialSampler


class DatasetSampler:

    def __init__(
        self,
        dataset: BaseDataset,
        random: bool = True,
        img_sampler_cls: type[ImageSampler] | None = None
    ) -> None:
        self.mode = dataset.mode
        self.id_sampler = RandomSequentialSampler(num_elements=len(dataset)) if random else SequentialSampler(num_elements=len(dataset))
        self.img_samplers = [img_sampler_cls(num_elements=(view.camera.width * view.camera.height)) for view in dataset] if img_sampler_cls else None

    def get(self, dataset: BaseDataset, ray_batch_size: int | None = None) -> dict[str, int | View | RayBatch | None]:
        if dataset.mode != self.mode:
            raise Framework.SamplerError(f'DatasetSampler initialized for mode "{self.mode}" got dataset with active mode "{dataset.mode}"')
        sample_id = self.id_sampler.get(num_samples=1).item()
        view = dataset[sample_id]
        image_sampler = ray_ids = ray_batch = None
        if self.img_samplers and ray_batch_size is not None:
            image_sampler = self.img_samplers[sample_id]
            ray_ids = image_sampler.get(ray_batch_size).to(Framework.config.GLOBAL.DEFAULT_DEVICE)
            if dataset.ray_collection[self.mode] is not None:
                ray_batch = dataset.ray_collection[self.mode][sample_id][ray_ids]
            else:
                ray_batch = view.get_rays()[ray_ids]
        return {
            'sample_id': sample_id,
            'view': view,
            'image_sampler': image_sampler,
            'ray_ids': ray_ids,
            'ray_batch': ray_batch
        }


class RayPoolSampler:
    def __init__(
        self,
        dataset: BaseDataset,
        img_sampler_cls: type[ImageSampler]
    ) -> None:
        self.mode = dataset.mode
        self.image_sampler = img_sampler_cls(num_elements=dataset.get_total_ray_count())

    def get(self, dataset: BaseDataset, ray_batch_size: int) -> dict[str, None | ImageSampler | RayBatch]:
        if dataset.mode != self.mode:
            raise Framework.SamplerError(f'RayPoolSampler initialized for mode "{self.mode}" got dataset with active mode "{dataset.mode}"')
        sample_id = view = None
        rays_all = dataset.get_all_rays()
        ray_ids = self.image_sampler.get(ray_batch_size).to(rays_all.device)
        ray_batch = rays_all[ray_ids].to(device=Framework.config.GLOBAL.DEFAULT_DEVICE)
        return {
            'sample_id': sample_id,
            'view': view,
            'image_sampler': self.image_sampler,
            'ray_ids': ray_ids,
            'ray_batch': ray_batch
        }


class IncrementalDatasetSampler(DatasetSampler):

    def __init__(
        self,
        dataset: BaseDataset,
        img_sampler_cls: type[ImageSampler] | None = None
    ) -> None:
        super().__init__(dataset, False, img_sampler_cls)
        self.id_sampler = IncrementalSequentialSampler(num_elements=len(dataset))
