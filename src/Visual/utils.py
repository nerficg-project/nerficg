# -- coding: utf-8 --

"""Visual/utils.py: Utilities for visualization tasks."""

from pathlib import Path
import av
from typing import Any

import torch
from torch import Tensor

from Logging import Logger
from Visual.ColorMap import ColorMap


def pseudoColorDepth(color_map: str, depth: Tensor, near_far: tuple[float, float] | None = None, alpha: Tensor | None = None, interpolate: bool = False) -> Tensor:
    """Produces a pseudo-colorized depth image for the given depth tensor."""
    # correct depth based on alpha mask
    if alpha is not None:
        depth = depth.clone()
        depth[alpha > 1e-5] = depth[alpha > 1e-5] / alpha[alpha > 1e-5]
    # normalize depth to [0, 1]
    if near_far is None:
        # calculate near and far planes from depth map
        masked_depth = depth[alpha > 0.99] if alpha is not None else depth
        near_plane = masked_depth.min().item() if masked_depth.numel() > 0 else 0.0
        far_plane = masked_depth.max().item() if masked_depth.numel() > 0 else 1.0
    else:
        near_plane, far_plane = near_far
    depth: Tensor = torch.clamp((depth - near_plane) / (far_plane - near_plane), min=0.0, max=1.0)
    # apply color map
    depth = ColorMap.apply(depth, color_map, interpolate)
    # mask color with alpha
    if alpha is not None:
        depth *= alpha
    return depth


class VideoWriter:
    """Wrapper class to facilitate creation of videos using PyAV."""

    def __init__(
            self,
            video_paths: Path | list[Path],
            width: int,
            height: int,
            fps: int,
            bitrate: int,
            video_codec: str = 'libx264rgb',
            options: dict[str, Any] | None = None
    ) -> None:
        self.containers = []
        self.streams = []
        padding_right = width % 2
        padding_bottom = height % 2
        self.pad = torch.nn.ReplicationPad2d((0, padding_right, 0, padding_bottom))
        if not isinstance(video_paths, list):
            video_paths = [video_paths]
        for video_path in video_paths:
            container = av.open(str(video_path), mode='w')
            stream = container.add_stream(video_codec, rate=fps)
            stream.codec_context.bit_rate = bitrate * 1e3  # kbps
            stream.width = width + padding_right
            stream.height = height + padding_bottom
            stream.pix_fmt = 'yuv420p' if video_codec != 'libx264rgb' else 'rgb24'
            stream.options = options or {}
            self.containers.append(container)
            self.streams.append(stream)

    def addFrame(self, frames: Tensor | list[Tensor]) -> None:
        """Adds given frames to the internal video streams."""
        if not isinstance(frames, list):
            frames = [frames]
        if len(self.streams) != len(frames):
            Logger.logWarning(
                f'number of frames does not match with number of video streams ({len(frames)} vs. {len(self.streams)})'
            )
        for frame, stream, container in zip(frames, self.streams, self.containers):
            frame = self.pad((frame * 255)).byte().permute(1, 2, 0).cpu().numpy()
            frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            frame.pict_type = 0
            for packet in stream.encode(frame):
                container.mux(packet)

    def close(self) -> None:
        """Flushes all streams and closes all containers."""
        for stream, container in zip(self.streams, self.containers):
            for packet in stream.encode():
                container.mux(packet)
            container.close()
