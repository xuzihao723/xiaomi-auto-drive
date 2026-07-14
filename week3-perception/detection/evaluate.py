import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a YOLOv8 checkpoint.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--name", default="eval")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/evaluation_metrics.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data.resolve()),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project.resolve()),
        name=args.name,
        plots=True,
    )

    print("Evaluation complete")
    print(f"box_map: {metrics.box.map:.6f}")
    print(f"box_map50: {metrics.box.map50:.6f}")
    print(f"box_map75: {metrics.box.map75:.6f}")
    print(f"results_dir: {metrics.save_dir}")

    per_class = {}
    for index, class_name in metrics.names.items():
        per_class[class_name] = {
            "precision": float(metrics.box.p[index]),
            "recall": float(metrics.box.r[index]),
            "mAP50": float(metrics.box.ap50[index]),
            "mAP50_95": float(metrics.box.ap[index]),
        }
    try:
        results_dir = Path(metrics.save_dir).resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        results_dir = Path(metrics.save_dir)
    output = {
        "split": args.split,
        "imgsz": args.imgsz,
        "device": args.device,
        "overall": {
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "mAP75": float(metrics.box.map75),
        },
        "per_class": per_class,
        "results_dir": results_dir.as_posix(),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"metrics_json: {args.output_json}")


if __name__ == "__main__":
    main()
