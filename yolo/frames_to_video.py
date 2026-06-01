import argparse
import re
from pathlib import Path

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def natural_key(path: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def collect_images(input_dir: Path):
    images = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    images = sorted(images, key=natural_key)

    return images


def main():
    parser = argparse.ArgumentParser(description="Convert image frames to video.")

    parser.add_argument("--input-dir", required=True, help="图片帧目录")
    parser.add_argument("--output", required=True, help="输出视频路径，例如 output/video.mp4")
    parser.add_argument("--fps", type=float, default=25, help="视频帧率")
    parser.add_argument("--resize-width", type=int, default=None, help="可选，统一缩放宽度")
    parser.add_argument("--resize-height", type=int, default=None, help="可选，统一缩放高度")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    images = collect_images(input_dir)

    if not images:
        raise RuntimeError(f"没有找到图片帧: {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_frame = cv2.imread(str(images[0]))

    if first_frame is None:
        raise RuntimeError(f"无法读取第一张图片: {images[0]}")

    if args.resize_width is not None and args.resize_height is not None:
        width = args.resize_width
        height = args.resize_height
    else:
        height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # type: ignore
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (width, height))

    for idx, img_path in enumerate(images, start=1):
        frame = cv2.imread(str(img_path))

        if frame is None:
            print(f"跳过无法读取的图片: {img_path}")
            continue

        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))

        writer.write(frame)

        if idx % 100 == 0:
            print(f"已写入 {idx}/{len(images)} 帧")

    writer.release()

    print("=" * 60)
    print("视频生成完成")
    print("=" * 60)
    print(f"输入帧数: {len(images)}")
    print(f"输出视频: {output_path}")
    print(f"FPS: {args.fps}")
    print("=" * 60)


if __name__ == "__main__":
    main()