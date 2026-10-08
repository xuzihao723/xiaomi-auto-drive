import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on CARLA data.")
    parser.add_argument("--data", type=Path, required=True, help="YOLO data.yaml path.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLOv8 model.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu, 0, 0,1, etc.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--name", default="train")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument(
        "--mosaic",
        type=float,
        default=1.0,
        help="Mosaic augmentation probability (0 disables it).",
    )
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mixed precision. Disabled by default for stability on this environment.",
    )
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data.resolve()),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project.resolve()),
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        resume=args.resume,
        amp=args.amp,
        cache=args.cache,
        fraction=args.fraction,
        save_period=args.save_period,
        close_mosaic=args.close_mosaic,
        mosaic=args.mosaic,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        deterministic=True,
        plots=True,
    )
    print("Training complete")
    print(results)


if __name__ == "__main__":
    main()
