"""Run the two-checkpoint class-specialized detector and save annotated frames."""

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_NAMES = {0: "Car", 1: "Pedestrian", 2: "TrafficLight", 3: "TrafficSign"}
COLORS = {
    0: (20, 210, 255),
    1: (255, 80, 180),
    2: (80, 220, 80),
    3: (220, 120, 40),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--road-user-weights", type=Path, required=True)
    parser.add_argument("--traffic-control-weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-images", type=int, default=1000)
    return parser.parse_args()


def draw_detections(image, detections):
    canvas = image.copy()
    for x1, y1, x2, y2, confidence, class_id in detections:
        class_id = int(class_id)
        color = COLORS[class_id]
        pt1, pt2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(canvas, pt1, pt2, color, 2, cv2.LINE_AA)
        label = f"{CLASS_NAMES[class_id]} {confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        top = max(0, pt1[1] - text_height - 7)
        cv2.rectangle(canvas, (pt1[0], top), (pt1[0] + text_width + 6, pt1[1]), color, -1)
        cv2.putText(
            canvas,
            label,
            (pt1[0] + 3, max(text_height + 1, pt1[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (10, 10, 10),
            1,
            cv2.LINE_AA,
        )
    return canvas


def main():
    args = parse_args()
    from ultralytics import YOLO

    images = sorted(
        path
        for path in args.source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )[: args.max_images]
    if not images:
        raise FileNotFoundError(f"No images found under {args.source}")
    args.output.mkdir(parents=True, exist_ok=True)

    road_user_model = YOLO(str(args.road_user_weights))
    traffic_control_model = YOLO(str(args.traffic_control_weights))
    output_index = 0
    for start in range(0, len(images), args.batch):
        paths = images[start : start + args.batch]
        common = dict(
            source=[str(path) for path in paths],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=0.7,
            max_det=300,
            device=args.device,
            verbose=False,
        )
        road_results = road_user_model.predict(**common)
        traffic_results = traffic_control_model.predict(**common)
        for path, road_result, traffic_result in zip(paths, road_results, traffic_results):
            road = road_result.boxes.data.detach().cpu().numpy()
            traffic = traffic_result.boxes.data.detach().cpu().numpy()
            road = road[np.isin(road[:, 5], (0, 1))]
            traffic = traffic[np.isin(traffic[:, 5], (2, 3))]
            detections = np.concatenate((road, traffic), axis=0)
            detections = detections[np.argsort(-detections[:, 4])]
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"Could not read {path}")
            annotated = draw_detections(image, detections)
            destination = args.output / f"{output_index:06d}.jpg"
            if not cv2.imwrite(str(destination), annotated):
                raise RuntimeError(f"Could not write {destination}")
            output_index += 1
        print(f"Processed {output_index}/{len(images)} frames", flush=True)

    print(f"Annotated frames: {args.output}")


if __name__ == "__main__":
    main()
