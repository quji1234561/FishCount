import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def natural_key(path: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def parse_gt_line(line: str):
    """
    支持 MOT 风格 gt.txt：
    frame_id, object_id, x, y, w, h, ...

    例如：
    1,1,100,120,30,20,1,1,1
    或：
    1 1 100 120 30 20 1 1 1
    """
    line = line.strip()

    if not line or line.startswith("#"):
        return None

    parts = [p for p in re.split(r"[,\s]+", line) if p]

    if len(parts) < 6:
        return None

    try:
        frame_id = int(float(parts[0]))
        object_id = int(float(parts[1]))
        x = float(parts[2])
        y = float(parts[3])
        w = float(parts[4])
        h = float(parts[5])

        conf = 1.0
        if len(parts) >= 7:
            conf = float(parts[6])

        return frame_id, object_id, x, y, w, h, conf

    except ValueError:
        return None


def read_gt(gt_path: Path, filter_conf: bool = False):
    annotations = defaultdict(list)

    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = parse_gt_line(line)

            if item is None:
                continue

            frame_id, object_id, x, y, w, h, conf = item

            if filter_conf and conf <= 0:
                continue

            annotations[frame_id].append(
                {
                    "object_id": object_id,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "conf": conf,
                }
            )

    return annotations


def build_frame_image_map(img_dir: Path):
    images = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS],
        key=natural_key,
    )

    if not images:
        raise FileNotFoundError(f"图片目录中没有找到图片: {img_dir}")

    frame_to_img = {}

    # 优先使用纯数字文件名映射帧号，例如 000001.jpg -> frame_id=1
    for img_path in images:
        try:
            frame_id = int(img_path.stem)
            frame_to_img[frame_id] = img_path
        except ValueError:
            pass

    # 如果图片名不是纯数字，则按自然排序顺序映射为 1,2,3...
    if not frame_to_img:
        for idx, img_path in enumerate(images, start=1):
            frame_to_img[idx] = img_path

    return frame_to_img


def convert_box_to_yolo(x, y, w, h, img_w, img_h):
    """
    MOT bbox:
    x, y, w, h 是左上角坐标和宽高

    YOLO bbox:
    class_id center_x center_y width height
    坐标需要归一化到 0~1
    """
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(img_w), x + w)
    y2 = min(float(img_h), y + h)

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 1 or box_h <= 1:
        return None

    center_x = x1 + box_w / 2.0
    center_y = y1 + box_h / 2.0

    center_x /= img_w
    center_y /= img_h
    box_w /= img_w
    box_h /= img_h

    center_x = min(max(center_x, 0.0), 1.0)
    center_y = min(max(center_y, 0.0), 1.0)
    box_w = min(max(box_w, 0.0), 1.0)
    box_h = min(max(box_h, 0.0), 1.0)

    return center_x, center_y, box_w, box_h


def prepare_dirs(output_dir: Path):
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_data_yaml(output_dir: Path, class_name: str):
    yaml_path = output_dir / "data.yaml"

    content = f"""path: {output_dir.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: {class_name}
"""

    yaml_path.write_text(content, encoding="utf-8")

    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Convert MOT gt.txt to YOLO dataset format.")

    parser.add_argument("--gt", required=True, help="gt.txt 路径")
    parser.add_argument("--img-dir", required=True, help="图片帧目录，例如 data/BT-001/img1")
    parser.add_argument("--output", default="output/yolo_dataset_sampled", help="输出 YOLO 数据集目录")

    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--class-name", default="fish", help="类别名称")

    parser.add_argument("--filter-conf", action="store_true", help="过滤 conf<=0 的标注")
    parser.add_argument("--split-mode", choices=["random", "sequential"], default="random", help="划分方式")

    parser.add_argument("--frame-step", type=int, default=1, help="抽帧间隔，例如 10 表示每 10 帧取 1 帧")
    parser.add_argument("--max-frames", type=int, default=None, help="最多使用多少帧，默认不限制")

    parser.add_argument("--clean", action="store_true", help="转换前清空输出目录，避免旧图片残留")

    args = parser.parse_args()

    gt_path = Path(args.gt)
    img_dir = Path(args.img_dir)
    output_dir = Path(args.output)

    if not gt_path.exists():
        raise FileNotFoundError(f"gt.txt 不存在: {gt_path}")

    if not img_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {img_dir}")

    if args.frame_step < 1:
        raise ValueError("--frame-step 必须 >= 1")

    if args.clean and output_dir.exists():
        print(f"清空旧输出目录: {output_dir}")
        shutil.rmtree(output_dir)

    prepare_dirs(output_dir)

    annotations = read_gt(gt_path, filter_conf=args.filter_conf)
    frame_to_img = build_frame_image_map(img_dir)

    valid_frames = sorted([fid for fid in annotations.keys() if fid in frame_to_img])

    if not valid_frames:
        raise RuntimeError("没有找到能和 gt.txt 匹配的图片帧，请检查 frame_id 和图片文件名。")

    original_frame_count = len(valid_frames)

    # 抽帧
    if args.frame_step > 1:
        valid_frames = valid_frames[::args.frame_step]

    # 限制最大帧数
    if args.max_frames is not None and args.max_frames > 0:
        valid_frames = valid_frames[:args.max_frames]

    if not valid_frames:
        raise RuntimeError("抽帧后没有剩余帧，请减小 --frame-step 或取消 --max-frames。")

    # 划分 train / val
    if args.split_mode == "random":
        random.seed(args.seed)
        random.shuffle(valid_frames)

    val_count = int(len(valid_frames) * args.val_ratio)
    val_count = max(1, val_count) if len(valid_frames) > 1 else 0

    if args.split_mode == "sequential":
        valid_frames = sorted(valid_frames)
        split_index = int(len(valid_frames) * (1 - args.val_ratio))
        train_frames = set(valid_frames[:split_index])
        val_frames = set(valid_frames[split_index:])
    else:
        val_frames = set(valid_frames[:val_count])
        train_frames = set(valid_frames[val_count:])

    stats = {
        "train_images": 0,
        "val_images": 0,
        "boxes": 0,
        "skipped_boxes": 0,
    }

    for frame_id in sorted(valid_frames):
        split = "val" if frame_id in val_frames else "train"

        img_path = frame_to_img[frame_id]

        dst_img_path = output_dir / "images" / split / img_path.name
        dst_label_path = output_dir / "labels" / split / f"{img_path.stem}.txt"

        shutil.copy2(img_path, dst_img_path)

        with Image.open(img_path) as img:
            img_w, img_h = img.size

        label_lines = []

        for ann in annotations[frame_id]:
            box = convert_box_to_yolo(
                ann["x"],
                ann["y"],
                ann["w"],
                ann["h"],
                img_w,
                img_h,
            )

            if box is None:
                stats["skipped_boxes"] += 1
                continue

            center_x, center_y, box_w, box_h = box

            label_lines.append(
                f"0 {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}"
            )

            stats["boxes"] += 1

        dst_label_path.write_text("\n".join(label_lines), encoding="utf-8")

        if split == "train":
            stats["train_images"] += 1
        else:
            stats["val_images"] += 1

    yaml_path = write_data_yaml(output_dir, args.class_name)

    print("=" * 60)
    print("YOLO 数据集转换完成")
    print("=" * 60)
    print(f"原始可用帧数: {original_frame_count}")
    print(f"抽帧后帧数: {len(valid_frames)}")
    print(f"训练图片数: {stats['train_images']}")
    print(f"验证图片数: {stats['val_images']}")
    print(f"标注框数量: {stats['boxes']}")
    print(f"跳过无效框数量: {stats['skipped_boxes']}")
    print(f"输出目录: {output_dir.resolve()}")
    print(f"data.yaml: {yaml_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()