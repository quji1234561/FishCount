import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 fish detector.")

    parser.add_argument("--data", required=True, help="data.yaml 路径")
    parser.add_argument("--model", default="yolov8n.pt", help="预训练模型，例如 yolov8n.pt / yolov8s.pt")

    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=416, help="输入图片尺寸")
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--device", default=None, help="训练设备，例如 0 / cpu，不填则自动选择")

    parser.add_argument("--project", default="output/yolo_runs", help="训练输出根目录")
    parser.add_argument("--name", default="fish_yolov8n", help="本次训练名称")
    parser.add_argument("--patience", type=int, default=10, help="早停耐心值")
    parser.add_argument("--exist-ok", action="store_true", help="允许覆盖同名输出目录")

    parser.add_argument("--cache", action="store_true", help="缓存图片，加快训练，但会占用更多内存")
    parser.add_argument("--no-plots", action="store_true", help="不生成训练图表，稍微加快训练")

    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    project_dir = Path(args.project).resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml 不存在: {data_path}")

    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(project_dir),
        "name": args.name,
        "patience": args.patience,
        "exist_ok": args.exist_ok,
        "cache": args.cache,
        "plots": not args.no_plots,
    }

    if args.device is not None:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)

    save_dir = Path(results.save_dir) # type: ignore

    print("=" * 60)
    print("YOLOv8 训练完成")
    print("=" * 60)
    print(f"训练输出目录: {save_dir}")
    print(f"最佳模型路径: {save_dir / 'weights' / 'best.pt'}")
    print(f"最后模型路径: {save_dir / 'weights' / 'last.pt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()