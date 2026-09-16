#! /usr/bin/env python3

"""make_video.py: Turns a directory of rendered frames into a video and an animated gif."""

from argparse import ArgumentParser
from pathlib import Path

import cv2
from PIL import Image


def main(*, frame_directory: Path, output_stem: Path, fps: int, gif_scale: float) -> None:
    """Encodes all images in a directory, in filename order, into an .mp4 and a .gif."""
    frames = sorted(p for p in frame_directory.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
    if not frames:
        raise SystemExit(f'No images found in "{frame_directory}".')
    height, width = cv2.imread(str(frames[0])).shape[:2]

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    video_path = output_stem.with_suffix('.mp4')
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        writer.write(cv2.imread(str(frame)))
    writer.release()

    gif_size = (max(1, int(width * gif_scale)), max(1, int(height * gif_scale)))
    gif_frames = [Image.open(frame).convert('RGB').resize(gif_size, Image.LANCZOS) for frame in frames]
    gif_path = output_stem.with_suffix('.gif')
    gif_frames[0].save(
        gif_path, save_all=True, append_images=gif_frames[1:],
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
        help='A directory of images, encoded in filename order.'
    )
    parser.add_argument(
        '-o', '--output', action='store', dest='output_stem', required=True,
        metavar='path/to/output', help='Output path without extension; .mp4 and .gif are appended.'
    )
    parser.add_argument(
        '--fps', action='store', dest='fps', default=30, type=int,
        metavar='fps', help='Frames per second of the encoded outputs.'
    )
    parser.add_argument(
        '--gif_scale', action='store', dest='gif_scale', default=0.5, type=float,
        metavar='factor', help='Resolution of the gif relative to the input frames.'
    )
    args = parser.parse_args()
    main(
        frame_directory=Path(args.frame_directory),
        output_stem=Path(args.output_stem),
        fps=args.fps,
        gif_scale=args.gif_scale,
    )
