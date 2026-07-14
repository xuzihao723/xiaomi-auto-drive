import argparse
import json
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args():
    parser = argparse.ArgumentParser(description="Create an MP4 demo from images.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--max-frames", type=int, default=1200)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("reports/video_summary.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    import cv2

    images = [
        path
        for path in sorted(args.source.iterdir())
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ][: args.max_frames]
    if not images:
        raise RuntimeError(f"No images found in {args.source}")

    first = cv2.imread(str(images[0]))
    if first is None:
        raise RuntimeError(f"Failed to read first image: {images[0]}")

    height, width = first.shape[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )

    written = 0
    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
        written += 1

    writer.release()
    print(f"Demo video written: {args.output}")
    summary = {
        "output": str(args.output),
        "frames": written,
        "fps": args.fps,
        "duration_seconds": written / args.fps,
        "width": width,
        "height": height,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"frames: {written}, fps: {args.fps}")
    print(f"summary: {args.summary_json}")


if __name__ == "__main__":
    main()
