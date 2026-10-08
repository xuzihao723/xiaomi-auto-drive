import argparse
import json
import statistics
import time
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark YOLOv8 inference FPS.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--secondary-weights",
        type=Path,
        help="Optional second checkpoint for sequential class-fusion benchmarking.",
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=150)
    parser.add_argument("--half", action="store_true", help="Use FP16 where supported.")
    parser.add_argument(
        "--preload", action="store_true",
        help="Decode images before timing so disk I/O is excluded.",
    )
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
    secondary_model = YOLO(str(args.secondary_weights)) if args.secondary_weights else None
    sources = images
    if args.preload:
        import cv2
        sources = [cv2.imread(str(path)) for path in images]
        if any(image is None for image in sources):
            raise RuntimeError("Failed to preload one or more images")

    predict_kwargs = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "device": args.device,
        "verbose": False,
    }
    if args.half:
        predict_kwargs["half"] = True

    def synchronize():
        try:
            import torch
            if str(args.device) != "cpu" and torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    for source in sources[: args.warmup]:
        model.predict(
            source=source,
            **predict_kwargs,
        )
        if secondary_model is not None:
            secondary_model.predict(
                source=source,
                **predict_kwargs,
            )

    timings = []
    for source in sources:
        synchronize()
        start = time.perf_counter()
        model.predict(
            source=source,
            **predict_kwargs,
        )
        if secondary_model is not None:
            secondary_model.predict(
                source=source,
                **predict_kwargs,
            )
        synchronize()
        timings.append(time.perf_counter() - start)

    mean_latency = statistics.mean(timings)
    median_latency = statistics.median(timings)
    fps = 1.0 / mean_latency if mean_latency > 0 else 0.0
    sorted_timings = sorted(timings)

    def percentile(fraction):
        index = min(len(sorted_timings) - 1, round((len(sorted_timings) - 1) * fraction))
        return sorted_timings[index]

    result = {
        "weights": str(args.weights),
        "secondary_weights": str(args.secondary_weights) if args.secondary_weights else None,
        "method": "sequential_class_fusion" if args.secondary_weights else "single_model",
        "source": str(args.source),
        "images": len(images),
        "imgsz": args.imgsz,
        "device": args.device,
        "half": args.half,
        "preloaded": args.preload,
        "mean_latency_ms": mean_latency * 1000.0,
        "median_latency_ms": median_latency * 1000.0,
        "p95_latency_ms": percentile(0.95) * 1000.0,
        "min_latency_ms": min(timings) * 1000.0,
        "max_latency_ms": max(timings) * 1000.0,
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
