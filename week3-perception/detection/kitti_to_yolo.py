import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
CLASS_TO_ID = {"Car": 0, "Pedestrian": 1}
PSEUDO_CLASS_TO_ID = {9: 2, 11: 3}
ID_TO_NAME = {
    0: "Car",
    1: "Pedestrian",
    2: "TrafficLight",
    3: "TrafficSign",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert KITTI 2D labels to YOLO labels."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="KITTI dataset root containing image_2 and label_2.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output YOLO dataset root.",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=None,
        help="Optional data.yaml path to rewrite.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--split-mode",
        choices=["contiguous", "random"],
        default="random",
        help="Random split is recommended so every target class reaches each split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="Hardlink saves disk space when source and output are on the same drive.",
    )
    parser.add_argument(
        "--visualize",
        type=int,
        default=0,
        help="Number of converted samples to draw into output preview directory.",
    )
    parser.add_argument(
        "--pseudo-model",
        type=Path,
        default=None,
        help="Optional COCO YOLOv8 checkpoint used to pseudo-label traffic controls.",
    )
    parser.add_argument("--pseudo-imgsz", type=int, default=960)
    parser.add_argument("--pseudo-device", default="cpu")
    parser.add_argument("--traffic-light-conf", type=float, default=0.20)
    parser.add_argument("--traffic-sign-conf", type=float, default=0.06)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("reports/dataset_summary.json"),
    )
    return parser.parse_args()


def clean_output(output_root):
    for dirname in ["images", "labels", "preview"]:
        path = output_root / dirname
        if path.exists():
            shutil.rmtree(path)


def read_kitti_labels(label_path, image_width, image_height):
    yolo_rows = []
    ignored = Counter()

    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 8:
            ignored["bad_field_count"] += 1
            continue

        class_name = parts[0]
        if class_name not in CLASS_TO_ID:
            ignored[class_name] += 1
            continue

        left, top, right, bottom = [float(value) for value in parts[4:8]]
        left = max(0.0, min(float(image_width), left))
        right = max(0.0, min(float(image_width), right))
        top = max(0.0, min(float(image_height), top))
        bottom = max(0.0, min(float(image_height), bottom))

        box_width = right - left
        box_height = bottom - top
        if box_width <= 1.0 or box_height <= 1.0:
            ignored["tiny_or_invalid_bbox"] += 1
            continue

        center_x = (left + right) / 2.0 / image_width
        center_y = (top + bottom) / 2.0 / image_height
        norm_width = box_width / image_width
        norm_height = box_height / image_height

        yolo_rows.append(
            (
                CLASS_TO_ID[class_name],
                center_x,
                center_y,
                norm_width,
                norm_height,
            )
        )

    return yolo_rows, ignored


def make_splits(samples, train_ratio, val_ratio, split_mode, seed):
    samples = list(samples)
    if split_mode == "random":
        rng = random.Random(seed)
        rng.shuffle(samples)

    train_end = int(len(samples) * train_ratio)
    val_end = train_end + int(len(samples) * val_ratio)
    return {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }


def copy_image(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def write_label(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"
        for class_id, cx, cy, width, height in rows
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_data_yaml(data_yaml, output_root):
    path_text = Path(
        os.path.relpath(output_root.resolve(), data_yaml.parent.resolve())
    ).as_posix()

    text = "\n".join(
        [
            f"path: {path_text}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            "names:",
            "  0: Car",
            "  1: Pedestrian",
            "  2: TrafficLight",
            "  3: TrafficSign",
            "",
        ]
    )
    data_yaml.write_text(text, encoding="utf-8")


def draw_previews(output_root, sample_paths, limit):
    if limit <= 0:
        return

    import cv2

    preview_dir = output_root / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        0: (0, 210, 255),
        1: (255, 120, 0),
        2: (0, 255, 0),
        3: (255, 0, 255),
    }

    for image_path in sample_paths[:limit]:
        split = image_path.parent.name
        label_path = output_root / "labels" / split / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                cx, cy, width, height = [float(value) for value in parts[1:]]
                image_height, image_width = image.shape[:2]
                left = int((cx - width / 2.0) * image_width)
                right = int((cx + width / 2.0) * image_width)
                top = int((cy - height / 2.0) * image_height)
                bottom = int((cy + height / 2.0) * image_height)
                color = colors.get(class_id, (255, 255, 255))
                cv2.rectangle(image, (left, top), (right, bottom), color, 2)
                cv2.putText(
                    image,
                    ID_TO_NAME.get(class_id, str(class_id)),
                    (left, max(20, top - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        cv2.imwrite(str(preview_dir / f"{split}_{image_path.name}"), image)


def collect_pseudo_labels(args, images):
    if args.pseudo_model is None:
        return {}, Counter()

    from ultralytics import YOLO

    model = YOLO(str(args.pseudo_model))
    thresholds = {9: args.traffic_light_conf, 11: args.traffic_sign_conf}
    predictions = model.predict(
        source=[str(path) for path in images],
        classes=list(PSEUDO_CLASS_TO_ID),
        imgsz=args.pseudo_imgsz,
        conf=min(thresholds.values()),
        iou=0.60,
        device=args.pseudo_device,
        batch=16,
        verbose=False,
        stream=True,
    )

    pseudo_by_stem = {}
    counts = Counter()
    for image_path, result in zip(images, predictions):
        rows = []
        if result.boxes is not None:
            height, width = result.orig_shape
            for cls_value, confidence, xyxy in zip(
                result.boxes.cls.tolist(),
                result.boxes.conf.tolist(),
                result.boxes.xyxy.tolist(),
            ):
                coco_id = int(cls_value)
                if confidence < thresholds[coco_id]:
                    continue
                left, top, right, bottom = xyxy
                box_width = right - left
                box_height = bottom - top
                if box_width < 6 or box_height < 6:
                    continue
                if coco_id == 11 and (box_width > 180 or box_height > 180):
                    continue
                rows.append(
                    (
                        PSEUDO_CLASS_TO_ID[coco_id],
                        (left + right) / 2.0 / width,
                        (top + bottom) / 2.0 / height,
                        box_width / width,
                        box_height / height,
                    )
                )
                counts[PSEUDO_CLASS_TO_ID[coco_id]] += 1
        pseudo_by_stem[image_path.stem] = rows
    return pseudo_by_stem, counts


def main():
    args = parse_args()
    image_dir = args.dataset_root / "image_2"
    label_dir = args.dataset_root / "label_2"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")

    images = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No images found under {image_dir}")

    pseudo_by_stem, pseudo_counts = collect_pseudo_labels(args, images)

    clean_output(args.output_root)
    splits = make_splits(
        images,
        args.train_ratio,
        args.val_ratio,
        args.split_mode,
        args.seed,
    )

    class_counts = Counter()
    split_class_counts = {name: Counter() for name in splits}
    split_class_frames = {name: Counter() for name in splits}
    ignored_counts = Counter()
    empty_label_files = 0
    written_images = []

    for split, split_images in splits.items():
        for image_path in split_images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                import cv2

                image = cv2.imread(str(image_path))
                if image is None:
                    raise RuntimeError(f"Failed to read image: {image_path}")
                image_height, image_width = image.shape[:2]
                rows, ignored = read_kitti_labels(
                    label_path, image_width, image_height
                )
            else:
                rows, ignored = [], Counter({"missing_label": 1})
            rows.extend(pseudo_by_stem.get(image_path.stem, []))

            if not rows:
                empty_label_files += 1
            for row in rows:
                class_counts[row[0]] += 1
                split_class_counts[split][row[0]] += 1
            for class_id in {row[0] for row in rows}:
                split_class_frames[split][class_id] += 1
            ignored_counts.update(ignored)

            dst_image = args.output_root / "images" / split / image_path.name
            dst_label = args.output_root / "labels" / split / f"{image_path.stem}.txt"
            copy_image(image_path, dst_image, args.copy_mode)
            write_label(dst_label, rows)
            written_images.append(dst_image)

    if args.data_yaml:
        args.data_yaml.parent.mkdir(parents=True, exist_ok=True)
        write_data_yaml(args.data_yaml, args.output_root)

    draw_previews(args.output_root, written_images, args.visualize)

    print("YOLO conversion complete")
    print(f"source: {args.dataset_root}")
    print(f"output: {args.output_root}")
    for split, split_images in splits.items():
        print(f"{split}: {len(split_images)} images")
    print(
        "class counts: "
        + ", ".join(f"{ID_TO_NAME[i]}={class_counts[i]}" for i in ID_TO_NAME)
    )
    print(f"empty label files: {empty_label_files}")
    print(f"ignored labels: {dict(ignored_counts)}")
    if args.visualize:
        print(f"preview: {args.output_root / 'preview'}")

    summary = {
        "source_images": len(images),
        "split_mode": args.split_mode,
        "seed": args.seed,
        "splits": {key: len(value) for key, value in splits.items()},
        "class_counts": {ID_TO_NAME[i]: class_counts[i] for i in ID_TO_NAME},
        "per_split_object_counts": {
            split: {ID_TO_NAME[i]: counts[i] for i in ID_TO_NAME}
            for split, counts in split_class_counts.items()
        },
        "per_split_frame_counts": {
            split: {ID_TO_NAME[i]: counts[i] for i in ID_TO_NAME}
            for split, counts in split_class_frames.items()
        },
        "pseudo_label_counts": {
            ID_TO_NAME[i]: pseudo_counts[i] for i in PSEUDO_CLASS_TO_ID.values()
        },
        "empty_label_files": empty_label_files,
        "ignored_labels": dict(ignored_counts),
        "pseudo_model": str(args.pseudo_model) if args.pseudo_model else None,
        "pseudo_thresholds": {
            "TrafficLight": args.traffic_light_conf,
            "TrafficSign": args.traffic_sign_conf,
        },
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"summary: {args.summary_json}")


if __name__ == "__main__":
    main()
