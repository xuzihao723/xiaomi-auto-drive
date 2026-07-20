"""Create a one-minute demo combining Week 3 detection and Week 4 segmentation."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "segmentation"))

from dataset import MEAN, STD  # noqa: E402
from inference import load_model  # noqa: E402
from postprocess import overlay_segmentation  # noqa: E402


CLASS_NAMES = {0: "Car", 1: "Pedestrian", 2: "TrafficLight", 3: "TrafficSign"}
COLORS = {0: (20, 210, 255), 1: (255, 80, 180), 2: (80, 220, 80), 3: (220, 120, 40)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--segmentation-weights", type=Path, required=True)
    parser.add_argument("--road-user-weights", type=Path, required=True)
    parser.add_argument("--traffic-control-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def preprocess(images, size, device):
    tensors = []
    for image in images:
        rgb = cv2.cvtColor(cv2.resize(image, size), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - MEAN) / STD
        tensors.append(torch.from_numpy(rgb.transpose(2, 0, 1)))
    return torch.stack(tensors).float().to(device)


def draw_detections(image, detections):
    counts = {name: 0 for name in CLASS_NAMES.values()}
    for x1, y1, x2, y2, confidence, class_id in detections:
        class_id = int(class_id)
        name = CLASS_NAMES[class_id]
        counts[name] += 1
        color = COLORS[class_id]
        pt1, pt2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(image, pt1, pt2, color, 2, cv2.LINE_AA)
        label = f"{name} {confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.47, 1)
        top = max(0, pt1[1] - text_height - 7)
        cv2.rectangle(image, (pt1[0], top), (pt1[0] + text_width + 6, pt1[1]), color, -1)
        cv2.putText(
            image,
            label,
            (pt1[0] + 3, max(text_height + 1, pt1[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (10, 10, 10),
            1,
            cv2.LINE_AA,
        )
    return counts


def add_header(image, frame_number, total_frames):
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 38), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)
    cv2.putText(
        image,
        "Week 4 | Detection + U-Net Segmentation",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"{frame_number:04d}/{total_frames:04d}",
        (image.shape[1] - 105, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()
    from ultralytics import YOLO

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    target_frames = int(round(args.fps * args.seconds))
    paths = sorted(
        path
        for path in args.source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )[:target_frames]
    if len(paths) < target_frames:
        raise RuntimeError(f"Need {target_frames} frames for {args.seconds}s, found {len(paths)}")

    segmenter, checkpoint = load_model(args.segmentation_weights, device)
    segmentation_size = tuple(checkpoint.get("image_size", [640, 360]))
    road_user = YOLO(str(args.road_user_weights))
    traffic_control = YOLO(str(args.traffic_control_weights))

    first = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read {paths[0]}")
    height, width = first.shape[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create {args.output}")

    total_counts = {name: 0 for name in CLASS_NAMES.values()}
    drivable_frames = 0
    lane_curve_count = 0
    written = 0
    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start : start + args.batch_size]
        images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in batch_paths]
        if any(image is None for image in images):
            raise RuntimeError(f"Failed to read a frame in batch starting at {start}")

        with torch.inference_mode():
            logits = segmenter(preprocess(images, segmentation_size, device))
            masks = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        common = dict(
            source=[str(path) for path in batch_paths],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=0.7,
            max_det=300,
            device=0 if device.type == "cuda" else "cpu",
            verbose=False,
        )
        road_results = road_user.predict(**common)
        traffic_results = traffic_control.predict(**common)

        for image, mask, road_result, traffic_result in zip(images, masks, road_results, traffic_results):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            frame, polygon, curves = overlay_segmentation(image, mask)
            drivable_frames += int(polygon is not None)
            lane_curve_count += len(curves)
            road = road_result.boxes.data.detach().cpu().numpy()
            traffic = traffic_result.boxes.data.detach().cpu().numpy()
            road = road[np.isin(road[:, 5], (0, 1))]
            traffic = traffic[np.isin(traffic[:, 5], (2, 3))]
            detections = np.concatenate((road, traffic), axis=0)
            if len(detections):
                detections = detections[np.argsort(-detections[:, 4])]
            counts = draw_detections(frame, detections)
            for name, count in counts.items():
                total_counts[name] += count
            add_header(frame, written + 1, target_frames)
            writer.write(frame)
            written += 1
        print(f"frames={written}/{target_frames}", flush=True)
    writer.release()

    summary = {
        "output": str(args.output),
        "frames": written,
        "fps": args.fps,
        "duration_seconds": written / args.fps,
        "resolution": [width, height],
        "segmentation_weights": str(args.segmentation_weights),
        "road_user_weights": str(args.road_user_weights),
        "traffic_control_weights": str(args.traffic_control_weights),
        "detections": total_counts,
        "frames_with_drivable_polygon": drivable_frames,
        "fitted_lane_curves": lane_curve_count,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
