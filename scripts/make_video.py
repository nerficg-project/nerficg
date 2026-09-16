#! /usr/bin/env python3

"""make_video.py: Turns a directory of rendered frames into a video and an animated gif."""

from argparse import ArgumentParser
from pathlib import Path

import cv2
from natsort import natsorted
from PIL import Image


def read_frame(path: Path) -> 'cv2.typing.MatLike':
    """Loads an image, failing loudly rather than yielding the None cv2.imread returns."""
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f'Failed to read image "{path}".')
    return image


def main(*, frame_directory: Path, output_stem: Path, fps: int, gif_scale: float) -> None:
    """Encodes all images in a directory, in natural filename order, into an .mp4 and a .gif."""
    if fps < 1:
        raise SystemExit(f'fps must be at least 1, got {fps}.')
    if not 0.0 < gif_scale <= 1.0:
        raise SystemExit(f'gif_scale must be in (0.0, 1.0], got {gif_scale}.')
    extensions = ('.png', '.jpg', '.jpeg')
    frames = natsorted(p for p in frame_directory.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    if not frames:
        raise SystemExit(f'No images found in "{frame_directory}".')

    height, width = read_frame(frames[0]).shape[:2]
    # OpenCV's VideoWriter silently discards frames whose size differs from the first one,
    # which would desync the video from the gif, so mismatches are rejected up front.
    for frame in frames:
        frame_height, frame_width = read_frame(frame).shape[:2]
        if (frame_width, frame_height) != (width, height):
            raise SystemExit(
                f'All frames must share one size, but "{frame}" is {frame_width}x{frame_height} '
                f'while "{frames[0]}" is {width}x{height}.'
            )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    video_path = output_stem.with_name(f'{output_stem.name}.mp4')
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f'Could not open "{video_path}" for writing; the mp4v encoder may be unavailable.')
    try:
        for frame in frames:
            writer.write(read_frame(frame))
    finally:
        writer.release()

    gif_size = (max(1, round(width * gif_scale)), max(1, round(height * gif_scale)))
    gif_path = output_stem.with_name(f'{output_stem.name}.gif')
    gif_frames = (Image.open(frame).convert('RGB').resize(gif_size, Image.LANCZOS) for frame in frames)
    first_frame = next(gif_frames)
    # Gifs store frame delays in centiseconds, so the playback rate is quantized and only
    # matches the video exactly when 100 is divisible by fps.
    first_frame.save(
        gif_path, save_all=True, append_images=gif_frames,
        duration=round(1000 / fps), loop=0, optimize=True,
    )
    print(f'Encoded {len(frames)} frames ({width}x{height}) to {video_path} and {gif_path}.')


if __name__ == '__main__':
    parser = ArgumentParser(
        prog='make_video.py',
        description='Turns a directory of rendered frames into a video and an animated gif.'
    )
    parser.add_argument(
        '-d', '--dir', action='store', dest='frame_directory', required=True,
        metavar='path/to/frame/directory',
        help='A directory of images, encoded in natural filename order. All must share one size.'
    )
    parser.add_argument(
        '-o', '--output', action='store', dest='output_stem', required=True,
        metavar='path/to/output',
        help='Output path. ".mp4" and ".gif" are appended to it, so any suffix is kept as part of the name.'
    )
    parser.add_argument(
        '--fps', action='store', dest='fps', default=30, type=int,
        metavar='fps', help='Frames per second of the encoded outputs.'
    )
    parser.add_argument(
        '--gif_scale', action='store', dest='gif_scale', default=0.5, type=float,
        metavar='factor', help='Resolution of the gif relative to the input frames, in (0.0, 1.0].'
    )
    args = parser.parse_args()
    main(
        frame_directory=Path(args.frame_directory),
        output_stem=Path(args.output_stem),
        fps=args.fps,
        gif_scale=args.gif_scale,
    )
