import argparse
import json
import statistics
import time
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark YOLOv8 inference FPS.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=150)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/fps_benchmark.json"),
    )
    return parser.parse_args()


def collect_images(source, max_images):
    if source.is_file():
        return [source]
    images = [
        path
        for path in sorted(source.iterdir())
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return images[:max_images]


def main():
    args = parse_args()

    from ultralytics import YOLO

    images = collect_images(args.source, args.max_images)
    if not images:
        raise RuntimeError(f"No images found in {args.source}")

    model = YOLO(str(args.weights))

    for image_path in images[: args.warmup]:
        model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )

    timings = []
    for image_path in images:
        start = time.perf_counter()
        model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )
        timings.append(time.perf_counter() - start)

    mean_latency = statistics.mean(timings)
    median_latency = statistics.median(timings)
    fps = 1.0 / mean_latency if mean_latency > 0 else 0.0

    result = {
        "weights": str(args.weights),
        "source": str(args.source),
        "images": len(images),
        "imgsz": args.imgsz,
        "device": args.device,
        "mean_latency_ms": mean_latency * 1000.0,
        "median_latency_ms": median_latency * 1000.0,
        "fps": fps,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("FPS benchmark complete")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
