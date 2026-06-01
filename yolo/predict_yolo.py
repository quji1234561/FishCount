import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Predict images or videos using trained YOLOv8 model.")

    parser.add_argument("--weights", required=True, help="模型权重路径，例如 output/yolo_runs/fish_yolov8n/weights/best.pt")
    parser.add_argument("--source", required=True, help="图片、图片文件夹或视频路径")

    parser.add_argument("--imgsz", type=int, default=416, help="预测输入尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IOU 阈值")
    parser.add_argument("--device", default=None, help="预测设备，例如 0 / cpu，不填自动选择")

    parser.add_argument("--project", default="output/yolo_runs", help="预测输出根目录")
    parser.add_argument("--name", default="predict_fish", help="预测输出名称")
    parser.add_argument("--exist-ok", action="store_true", help="允许覆盖同名输出目录")

    parser.add_argument("--save-txt", action="store_true", help="保存 YOLO txt 预测结果")
    parser.add_argument("--save-conf", action="store_true", help="txt 中保存置信度")

    args = parser.parse_args()

    weights_path = Path(args.weights).resolve()
    source_path = Path(args.source).resolve()
    project_dir = Path(args.project).resolve()

    if not weights_path.exists():
        raise FileNotFoundError(f"模型不存在: {weights_path}")

    if not source_path.exists():
        raise FileNotFoundError(f"输入源不存在: {source_path}")

    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))

    predict_kwargs = {
        "source": str(source_path),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "save": True,
        "project": str(project_dir),
        "name": args.name,
        "exist_ok": args.exist_ok,
        "save_txt": args.save_txt,
        "save_conf": args.save_conf,
    }

    if args.device is not None:
        predict_kwargs["device"] = args.device

    results = model.predict(**predict_kwargs)

    print("=" * 60)
    print("YOLOv8 预测完成")
    print("=" * 60)

    if results:
        print(f"预测输出目录: {results[0].save_dir}")

    print("=" * 60)


if __name__ == "__main__":
    main()