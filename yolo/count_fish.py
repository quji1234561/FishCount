import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def is_image(path: Path):
    return path.suffix.lower() in IMAGE_EXTS


def is_video(path: Path):
    return path.suffix.lower() in VIDEO_EXTS


def collect_images(source: Path):
    if source.is_file() and is_image(source):
        return [source]

    if source.is_dir():
        images = [p for p in source.iterdir() if is_image(p)]
        return sorted(images, key=lambda x: x.name)

    return []


def predict_boxes(model, frame, imgsz, conf, iou, device, class_id):
    predict_kwargs = {
        "source": frame,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "verbose": False,
    }

    if device is not None:
        predict_kwargs["device"] = device

    results = model.predict(**predict_kwargs)
    result = results[0]

    boxes = []

    if result.boxes is None:
        return boxes

    for box in result.boxes:
        cls = int(box.cls.item()) if box.cls is not None else -1

        if class_id is not None and cls != class_id:
            continue

        confidence = float(box.conf.item()) if box.conf is not None else 0.0
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        boxes.append(
            {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "conf": confidence,
                "cls": cls,
            }
        )

    return boxes


def get_class_name(names, cls_id):
    if isinstance(names, dict):
        return names.get(cls_id, "fish")

    if isinstance(names, list) and 0 <= cls_id < len(names):
        return names[cls_id]

    return "fish"


def draw_result(frame, boxes, names=None):
    annotated = frame.copy()

    for box in boxes:
        x1 = box["x1"]
        y1 = box["y1"]
        x2 = box["x2"]
        y2 = box["y2"]
        conf = box["conf"]
        cls = box["cls"]

        label_name = get_class_name(names, cls)
        label = f"{label_name} {conf:.2f}"

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        text_y = max(20, y1 - 6)

        cv2.putText(
            annotated,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    count_text = f"Fish Count: {len(boxes)}"

    cv2.rectangle(
        annotated,
        (10, 10),
        (280, 58),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        annotated,
        count_text,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated


def process_images(model, source: Path, output_dir: Path, args):
    image_paths = collect_images(source)

    if not image_paths:
        raise RuntimeError(f"没有找到图片: {source}")

    image_output_dir = output_dir / "images"
    image_output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for idx, img_path in enumerate(image_paths, start=1):
        frame = cv2.imread(str(img_path))

        if frame is None:
            print(f"跳过无法读取的图片: {img_path}")
            continue

        boxes = predict_boxes(
            model=model,
            frame=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            class_id=args.class_id,
        )

        annotated = draw_result(frame, boxes, names=model.names)

        out_path = image_output_dir / f"{img_path.stem}_count{img_path.suffix}"
        cv2.imwrite(str(out_path), annotated)

        avg_conf = sum(b["conf"] for b in boxes) / len(boxes) if boxes else 0.0

        rows.append(
            {
                "frame_id": idx,
                "file_name": img_path.name,
                "fish_count": len(boxes),
                "avg_conf": f"{avg_conf:.4f}",
            }
        )

        print(f"{img_path.name}: fish_count={len(boxes)}")

    return rows


def process_video(model, source: Path, output_dir: Path, args):
    cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_path = output_dir / f"{source.stem}_count.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # type: ignore
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    rows = []
    frame_id = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_id += 1

        boxes = predict_boxes(
            model=model,
            frame=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            class_id=args.class_id,
        )

        annotated = draw_result(frame, boxes, names=model.names)
        writer.write(annotated)

        avg_conf = sum(b["conf"] for b in boxes) / len(boxes) if boxes else 0.0

        rows.append(
            {
                "frame_id": frame_id,
                "file_name": source.name,
                "fish_count": len(boxes),
                "avg_conf": f"{avg_conf:.4f}",
            }
        )

        if frame_id % 50 == 0:
            print(f"已处理 {frame_id} 帧")

    cap.release()
    writer.release()

    print(f"输出视频: {output_video_path}")

    return rows


def save_csv(rows, output_dir: Path):
    csv_path = output_dir / "fish_count.csv"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["frame_id", "file_name", "fish_count", "avg_conf"]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"统计结果已保存: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Count fish in images or videos using YOLOv8.")

    parser.add_argument("--weights", required=True, help="模型权重路径")
    parser.add_argument("--source", required=True, help="图片、图片文件夹或视频路径")
    parser.add_argument("--output", default="output/count_results", help="输出目录")

    parser.add_argument("--imgsz", type=int, default=416, help="预测输入尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IOU 阈值")
    parser.add_argument("--device", default=None, help="预测设备，例如 0 / cpu，不填自动选择")

    parser.add_argument("--class-id", type=int, default=0, help="统计类别 ID，默认 0。设置为 -1 表示统计所有类别")

    args = parser.parse_args()

    if args.class_id == -1:
        args.class_id = None

    weights_path = Path(args.weights).resolve()
    source_path = Path(args.source).resolve()
    output_dir = Path(args.output).resolve()

    if not weights_path.exists():
        raise FileNotFoundError(f"模型不存在: {weights_path}")

    if not source_path.exists():
        raise FileNotFoundError(f"输入源不存在: {source_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))

    if source_path.is_file() and is_video(source_path):
        rows = process_video(model, source_path, output_dir, args)
    else:
        rows = process_images(model, source_path, output_dir, args)

    save_csv(rows, output_dir)

    print("=" * 60)
    print("鱼类数量统计完成")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()