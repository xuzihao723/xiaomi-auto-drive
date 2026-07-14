import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/detect/predict"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument(
        "--compact-labels",
        action="store_true",
        help="Use shorter display names in crowded demo scenes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from ultralytics import YOLO

    project = args.output.parent.resolve()
    name = args.output.name

    model = YOLO(str(args.weights))
    if args.compact_labels:
        model.model.names = {
            0: "Car",
            1: "Pedestrian",
            2: "Light",
            3: "Sign",
        }
    results = model.predict(
        source=str(args.source.resolve()),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=str(project),
        name=name,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        line_width=args.line_width,
        exist_ok=True,
    )
    print("Inference complete")
    print(f"output: {args.output}")
    print(f"batches: {len(results)}")


if __name__ == "__main__":
    main()
