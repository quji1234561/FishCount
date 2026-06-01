import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def natural_key(path: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def parse_gt_line(line: str):
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    parts = [p for p in re.split(r"[,\s]+", line) if p]
    if len(parts) < 6:
        return None

    try:
        frame_id = int(float(parts[0]))
        obj_id = int(float(parts[1]))
        x = float(parts[2])
        y = float(parts[3])
        w = float(parts[4])
        h = float(parts[5])

        conf = 1.0
        if len(parts) >= 7:
            conf = float(parts[6])

        return frame_id, obj_id, x, y, w, h, conf
    except ValueError:
        return None


def read_gt(gt_path: Path, filter_conf: bool = False):
    annotations = defaultdict(list)

    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = parse_gt_line(line)
            if item is None:
                continue

            frame_id, obj_id, x, y, w, h, conf = item

            if filter_conf and conf <= 0:
                continue

            annotations[frame_id].append([x, y, x + w, y + h])

    return annotations


def build_frame_image_map(img_dir: Path):
    images = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS],
        key=natural_key,
    )

    if not images:
        raise FileNotFoundError(f"图片目录中没有找到图片: {img_dir}")

    frame_to_img = {}

    for img_path in images:
        try:
            frame_id = int(img_path.stem)
            frame_to_img[frame_id] = img_path
        except ValueError:
            pass

    if not frame_to_img:
        for idx, img_path in enumerate(images, start=1):
            frame_to_img[idx] = img_path

    return frame_to_img


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def match_boxes(pred_boxes, gt_boxes, iou_threshold):
    """
    贪心匹配：
    每个预测框最多匹配一个真实框；
    每个真实框最多被匹配一次。
    """
    pairs = []

    for pred_idx, pred in enumerate(pred_boxes):
        for gt_idx, gt in enumerate(gt_boxes):
            iou = box_iou(pred, gt)
            if iou >= iou_threshold:
                pairs.append((iou, pred_idx, gt_idx))

    pairs.sort(reverse=True, key=lambda x: x[0])

    matched_preds = set()
    matched_gts = set()
    matched_ious = []

    for iou, pred_idx, gt_idx in pairs:
        if pred_idx in matched_preds or gt_idx in matched_gts:
            continue

        matched_preds.add(pred_idx)
        matched_gts.add(gt_idx)
        matched_ious.append(iou)

    tp = len(matched_ious)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    return tp, fp, fn, matched_ious


def predict_boxes(model, img_path: Path, imgsz, conf, iou, device, class_id):
    kwargs = {
        "source": str(img_path),
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "verbose": False,
    }

    if device is not None:
        kwargs["device"] = device

    results = model.predict(**kwargs)
    result = results[0]

    pred_boxes = []

    if result.boxes is None:
        return pred_boxes

    for box in result.boxes:
        cls = int(box.cls.item()) if box.cls is not None else -1

        if class_id is not None and cls != class_id:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        pred_boxes.append([x1, y1, x2, y2])

    return pred_boxes


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def save_csv(rows, csv_path: Path):
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "frame_id",
            "file_name",
            "gt_count",
            "pred_count",
            "count_error",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "mean_iou",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_count_curve(rows, output_dir: Path):
    frame_ids = [r["frame_id"] for r in rows]
    gt_counts = [r["gt_count"] for r in rows]
    pred_counts = [r["pred_count"] for r in rows]

    plt.figure(figsize=(10, 5))
    plt.plot(frame_ids, gt_counts, label="GT Count")
    plt.plot(frame_ids, pred_counts, label="Pred Count")
    plt.xlabel("Frame ID")
    plt.ylabel("Fish Count")
    plt.title("GT Count vs Predicted Count")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "count_curve.png", dpi=200)
    plt.close()


def plot_count_error(rows, output_dir: Path):
    frame_ids = [r["frame_id"] for r in rows]
    errors = [r["count_error"] for r in rows]

    plt.figure(figsize=(10, 5))
    plt.bar(frame_ids, errors)
    plt.xlabel("Frame ID")
    plt.ylabel("Pred Count - GT Count")
    plt.title("Count Error per Frame")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "count_error.png", dpi=200)
    plt.close()


def plot_metrics_bar(metrics, output_dir: Path):
    names = ["precision", "recall", "f1", "count_accuracy"]
    values = [metrics[name] for name in names]

    plt.figure(figsize=(8, 5))
    plt.bar(names, values)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Evaluation Metrics")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_bar.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO fish detection with gt.txt.")

    parser.add_argument("--weights", required=True, help="模型权重路径")
    parser.add_argument("--gt", required=True, help="gt.txt 路径")
    parser.add_argument("--img-dir", required=True, help="图片帧目录")
    parser.add_argument("--output", default="output/eval_results", help="评估结果输出目录")

    parser.add_argument("--imgsz", type=int, default=416, help="预测输入尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="预测置信度阈值")
    parser.add_argument("--nms-iou", type=float, default=0.45, help="YOLO NMS IOU 阈值")
    parser.add_argument("--eval-iou", type=float, default=0.5, help="评估匹配 IOU 阈值")

    parser.add_argument("--device", default=None, help="设备，例如 0 / cpu")
    parser.add_argument("--class-id", type=int, default=0, help="类别 ID，默认 0；-1 表示所有类别")

    parser.add_argument("--frame-step", type=int, default=1, help="评估抽帧间隔")
    parser.add_argument("--max-frames", type=int, default=None, help="最多评估多少帧")
    parser.add_argument("--filter-conf", action="store_true", help="过滤 gt.txt 中 conf<=0 的标注")

    args = parser.parse_args()

    if args.class_id == -1:
        args.class_id = None

    weights_path = Path(args.weights)
    gt_path = Path(args.gt)
    img_dir = Path(args.img_dir)
    output_dir = Path(args.output)

    if not weights_path.exists():
        raise FileNotFoundError(f"模型不存在: {weights_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"gt.txt 不存在: {gt_path}")
    if not img_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {img_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    gt_annotations = read_gt(gt_path, filter_conf=args.filter_conf)
    frame_to_img = build_frame_image_map(img_dir)

    frame_ids = sorted([fid for fid in gt_annotations.keys() if fid in frame_to_img])

    if args.frame_step > 1:
        frame_ids = frame_ids[::args.frame_step]

    if args.max_frames is not None and args.max_frames > 0:
        frame_ids = frame_ids[:args.max_frames]

    if not frame_ids:
        raise RuntimeError("没有可评估的帧，请检查 gt.txt 和图片路径。")

    model = YOLO(str(weights_path))

    rows = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_ious = []
    count_errors = []
    gt_counts = []
    pred_counts = []

    for idx, frame_id in enumerate(frame_ids, start=1):
        img_path = frame_to_img[frame_id]
        gt_boxes = gt_annotations[frame_id]

        pred_boxes = predict_boxes(
            model=model,
            img_path=img_path,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.nms_iou,
            device=args.device,
            class_id=args.class_id,
        )

        tp, fp, fn, matched_ious = match_boxes(
            pred_boxes=pred_boxes,
            gt_boxes=gt_boxes,
            iou_threshold=args.eval_iou,
        )

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

        gt_count = len(gt_boxes)
        pred_count = len(pred_boxes)
        count_error = pred_count - gt_count

        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_ious.extend(matched_ious)
        count_errors.append(count_error)
        gt_counts.append(gt_count)
        pred_counts.append(pred_count)

        rows.append(
            {
                "frame_id": frame_id,
                "file_name": img_path.name,
                "gt_count": gt_count,
                "pred_count": pred_count,
                "count_error": count_error,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "mean_iou": round(mean_iou, 4),
            }
        )

        if idx % 20 == 0:
            print(f"已评估 {idx}/{len(frame_ids)} 帧")

    precision = safe_div(total_tp, total_tp + total_fp)
    recall = safe_div(total_tp, total_tp + total_fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    mae = float(np.mean(np.abs(count_errors)))
    rmse = float(math.sqrt(np.mean(np.square(count_errors))))
    mean_gt = float(np.mean(gt_counts)) if gt_counts else 0.0

    count_accuracy = 1.0 - safe_div(mae, mean_gt)
    count_accuracy = max(0.0, min(1.0, count_accuracy))

    metrics = {
        "frames": len(frame_ids),
        "total_gt_boxes": int(sum(gt_counts)),
        "total_pred_boxes": int(sum(pred_counts)),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou": round(float(np.mean(all_ious)) if all_ious else 0.0, 4),
        "count_mae": round(mae, 4),
        "count_rmse": round(rmse, 4),
        "count_accuracy": round(count_accuracy, 4),
        "conf": args.conf,
        "eval_iou": args.eval_iou,
    }

    save_csv(rows, output_dir / "frame_metrics.csv")

    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    plot_count_curve(rows, output_dir)
    plot_count_error(rows, output_dir)
    plot_metrics_bar(metrics, output_dir)

    print("=" * 60)
    print("评估完成")
    print("=" * 60)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"输出目录: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()