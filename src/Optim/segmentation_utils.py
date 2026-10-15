"""
Optim/segmentation_utils.py: Binary segmentation masks for dataset views via selectable segmentation models:
SegFormer (https://github.com/NVlabs/SegFormer/) and SAM 3 (https://github.com/facebookresearch/sam3/).
"""

from abc import ABC, abstractmethod
from pathlib import Path

import torch

import Framework
from Datasets.Base import BaseDataset
from Datasets.utils import ImageData, apply_image_scale_factor, save_image
from Logging import Logger
from Thirdparty.Transformers import Sam3Model, Sam3Processor, SegformerImageProcessor, SegformerForSemanticSegmentation

_ALLOW_NVIDIA_LICENSED_CODE = True
_ALLOW_SAM_LICENSED_CODE = True

SEGFORMER_CHECKPOINT = 'nvidia/segformer-b0-finetuned-ade-512-512'
SAM3_CHECKPOINT = 'facebook/sam3'


class Segmenter(ABC):
    """Base class for segmentation models computing binary masks of a target given by a text label."""

    @abstractmethod
    def segment(self, rgb: torch.Tensor, label: str) -> torch.Tensor:
        """Returns a binary float mask (1, H, W) of the label for the image rgb (3, H, W) in [0, 1]."""
        pass


class SegFormerSegmenter(Segmenter):
    """Semantic segmentation with SegFormer, labels are ADE20K class names (e.g. 'sky')."""

    def __init__(self) -> None:
        if not _ALLOW_NVIDIA_LICENSED_CODE:
            raise Framework.FrameworkError(
                'SegFormer is subject to the `NVIDIA Source Code License for SegFormer` (https://github.com/NVlabs/SegFormer/blob/master/LICENSE).'
                'Set _ALLOW_NVIDIA_LICENSED_CODE = True to enable.'
            )
        Logger.log_warning(
            'SegFormer is subject to the `NVIDIA Source Code License for SegFormer` (https://github.com/NVlabs/SegFormer/blob/master/LICENSE), '
            'a non-commercial license. For commercial use, disable SegFormer by setting _ALLOW_NVIDIA_LICENSED_CODE = False.'
        )
        self.processor = SegformerImageProcessor.from_pretrained(SEGFORMER_CHECKPOINT)
        self.model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_CHECKPOINT).cuda().eval()

    @torch.no_grad()
    def segment(self, rgb: torch.Tensor, label: str) -> torch.Tensor:
        x = self.processor(images=rgb, do_resize=False, do_rescale=False, return_tensors='pt')['pixel_values'].cuda()
        return (self.model(x).logits.argmax(dim=1) == self.model.config.label2id[label]).float()


class SAM3Segmenter(Segmenter):
    """Promptable concept segmentation with SAM 3, labels are free-form noun phrases. The mask is the union of all instances."""

    def __init__(self, threshold: float = 0.5, mask_threshold: float = 0.5, half_precision: bool = True) -> None:
        if not _ALLOW_SAM_LICENSED_CODE:
            raise Framework.FrameworkError(
                'SAM 3 is subject to the `SAM License` (https://github.com/facebookresearch/sam3/blob/main/LICENSE).'
                'Set _ALLOW_SAM_LICENSED_CODE = True to enable.'
            )
        try:
            self.processor = Sam3Processor.from_pretrained(SAM3_CHECKPOINT)
            self.model = Sam3Model.from_pretrained(SAM3_CHECKPOINT).cuda().eval()
            self.threshold = threshold
            self.mask_threshold = mask_threshold
            self.half_precision = half_precision
        except OSError as e:
            raise Framework.CheckpointError(
                f'Failed to load {SAM3_CHECKPOINT}: {e}\n'
                f'The SAM 3 weights are gated: accept the license at https://huggingface.co/{SAM3_CHECKPOINT}, '
                'then log in with `hf auth login` (or set HF_TOKEN).'
            )

    def _predict(self, rgb: torch.Tensor, label: str):
        inputs = self.processor(images=rgb, text=label, do_rescale=False, return_tensors='pt').to('cuda')
        # Runs SAM 3 under bf16 autocast, which is about 2.6x faster than fp32 at nearly identical masks.
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.half_precision):
            return self.model(**inputs)

    @torch.no_grad()
    def segment(self, rgb: torch.Tensor, label: str, ) -> torch.Tensor:
        outputs = self._predict(rgb, label)
        masks = self.processor.post_process_instance_segmentation(
            outputs=outputs,
            threshold=self.threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=[rgb.shape[1:]],
        )[0]['masks']
        return masks.any(dim=0, keepdim=True).float() if len(masks) > 0 else rgb.new_zeros((1, *rgb.shape[1:]))

    @torch.no_grad()
    def segment_instances(self, rgb: torch.Tensor, label: str) -> dict[str, torch.Tensor]:
        """
        Returns all instances above the score threshold as 'scores' (K), 'boxes' (K, 4) in xyxy pixel coordinates, and
        'masks' (K, H, W) holding the unthresholded mask probabilities at the input resolution.
        """
        outputs = self._predict(rgb, label)
        scores = outputs.pred_logits[0].float().sigmoid()
        if outputs.presence_logits is not None:
            scores = scores * outputs.presence_logits[0].float().sigmoid()
        keep = scores > self.threshold
        height, width = rgb.shape[1:]
        masks = scores.new_zeros((0, height, width))
        if keep.any():  # interpolation does not support empty inputs
            masks = torch.nn.functional.interpolate(
                outputs.pred_masks[:, keep].float().sigmoid(), size=(height, width), mode='bilinear', align_corners=False,
            )[0]
        boxes = outputs.pred_boxes[0, keep].float() * torch.tensor([width, height, width, height], device=scores.device)
        return {'scores': scores[keep], 'boxes': boxes, 'masks': masks}


SEGMENTATION_MODELS: dict[str, type[Segmenter]] = {
    'SegFormer': SegFormerSegmenter,
    'SAM3': SAM3Segmenter,
}


@torch.no_grad()
def compute_segmentation_masks(
    dataset: BaseDataset,
    output_directory: Path,
    label: str = 'sky',
    scale_factor: float = 1.0,
    model: str = 'SegFormer',
) -> None:
    """
    Segments the given label in all views of the dataset's current subset with the selected segmentation model (see
    SEGMENTATION_MODELS), assigning them to the views and writing the masks to disk if an output directory is provided.
    Views are scaled by scale_factor before segmentation, the resulting masks are always stored at full view resolution.
    """
    segmenter = SEGMENTATION_MODELS[model]()
    output_directory.mkdir(parents=True, exist_ok=True)
    for view in Logger.log_progress(dataset, leave=False, desc=f'Computing {label} masks ({model})', total=len(dataset)):
        rgb = view.rgb
        full_size = rgb.shape[1:]
        if scale_factor != 1.0:
            rgb = apply_image_scale_factor(rgb, scale_factor)
        mask = segmenter.segment(rgb, label)
        if mask.shape[1:] != full_size:
            mask = torch.nn.functional.interpolate(mask[None], size=full_size, mode='nearest-exact')[0]
        mask_path = output_directory / f'{view._rgb.path.name}.png'
        save_image(mask_path, mask)
        view.segmentation = ImageData(path=mask_path, n_channels=1)
