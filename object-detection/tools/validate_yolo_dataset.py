import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml


CLASS_NAMES = {
    0: "Car",
    1: "Pedestrian",
    2: "TrafficLight",
    3: "TrafficSign",
}
COLORS = {
    0: (0, 210, 255),
    1: (255, 120, 0),
    2: (0, 255, 0),
    3: (255, 0, 255),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate and preview a YOLO dataset.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path, required=True)
    parser.add_argument("--preview-count", type=int, default=6)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val", "test"),
        default=("train", "val", "test"),
        help="Dataset splits to validate (default: all).",
    )
    return parser.parse_args()


def resolve_dataset(data_yaml):
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config["path"])
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return root, config


def parse_label(label_path):
    rows = []
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 fields")
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
        if class_id not in CLASS_NAMES:
            raise ValueError(f"{label_path}:{line_number}: invalid class {class_id}")
        cx, cy, width, height = values
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            raise ValueError(f"{label_path}:{line_number}: center outside [0,1]")
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise ValueError(f"{label_path}:{line_number}: size outside (0,1]")
        if cx - width / 2.0 < -1e-5 or cx + width / 2.0 > 1.0 + 1e-5:
            raise ValueError(f"{label_path}:{line_number}: horizontal box overflow")
        if cy - height / 2.0 < -1e-5 or cy + height / 2.0 > 1.0 + 1e-5:
            raise ValueError(f"{label_path}:{line_number}: vertical box overflow")
        rows.append((class_id, cx, cy, width, height))
    return rows


def draw_sample(image_path, rows):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    for class_id, cx, cy, box_width, box_height in rows:
        left = int((cx - box_width / 2.0) * width)
        right = int((cx + box_width / 2.0) * width)
        top = int((cy - box_height / 2.0) * height)
        bottom = int((cy + box_height / 2.0) * height)
        color = COLORS[class_id]
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        cv2.putText(
            image,
            CLASS_NAMES[class_id],
            (left, max(20, top - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return image


def make_preview(samples, destination, limit):
    # Prioritize samples containing TrafficSign, then samples with more objects.
    samples = sorted(
        samples,
        key=lambda item: (
            sum(row[0] == 3 for row in item[2]),
            len(item[2]),
        ),
        reverse=True,
    )[:limit]
    tiles = []
    for image_path, _, rows in samples:
        image = draw_sample(image_path, rows)
        image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.putText(
            image,
            image_path.name,
            (10, 345),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(image)
    if not tiles:
        return
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[index : index + 2]) for index in range(0, len(tiles), 2)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), np.vstack(rows))


def main():
    args = parse_args()
    root, config = resolve_dataset(args.data.resolve())
    result = {
        "status": "passed",
        "dataset_root": str(root),
        "splits": {},
    }

    for split in args.splits:
        image_dir = root / config[split]
        label_dir = root / "labels" / split
        image_paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        label_paths = sorted(label_dir.glob("*.txt"))
        if len(image_paths) != len(label_paths):
            raise RuntimeError(
                f"{split}: image/label mismatch {len(image_paths)} != {len(label_paths)}"
            )

        samples = []
        object_counts = Counter()
        frame_counts = Counter()
        empty_labels = 0
        for image_path in image_paths:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(label_path)
            rows = parse_label(label_path)
            if not rows:
                empty_labels += 1
            object_counts.update(row[0] for row in rows)
            frame_counts.update({row[0] for row in rows})
            samples.append((image_path, label_path, rows))

        make_preview(
            samples,
            args.preview_dir / f"grouped_preview_{split}.jpg",
            args.preview_count,
        )
        result["splits"][split] = {
            "images": len(image_paths),
            "labels": len(label_paths),
            "empty_labels": empty_labels,
            "object_counts": {
                CLASS_NAMES[class_id]: object_counts[class_id]
                for class_id in CLASS_NAMES
            },
            "frame_counts": {
                CLASS_NAMES[class_id]: frame_counts[class_id]
                for class_id in CLASS_NAMES
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
