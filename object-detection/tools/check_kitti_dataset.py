import argparse
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
KNOWN_CLASSES = {"Car", "Pedestrian"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check a KITTI-style 2D detection dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root containing image_2 and label_2.",
    )
    return parser.parse_args()


def read_label_file(path):
    rows = []
    errors = []

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 15:
            errors.append(f"{path.name}:{line_no} has {len(parts)} fields")
            continue

        try:
            bbox = tuple(float(value) for value in parts[4:8])
        except ValueError:
            errors.append(f"{path.name}:{line_no} has invalid bbox values")
            continue

        rows.append((parts[0], bbox))

    return rows, errors


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
    labels = sorted(label_dir.glob("*.txt"))

    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}

    class_counts = Counter()
    empty_labels = 0
    unknown_counts = Counter()
    field_errors = []
    bbox_errors = []

    for label_path in labels:
        rows, errors = read_label_file(label_path)
        field_errors.extend(errors)
        if not rows:
            empty_labels += 1

        for class_name, bbox in rows:
            if class_name in KNOWN_CLASSES:
                class_counts[class_name] += 1
            else:
                unknown_counts[class_name] += 1

            left, top, right, bottom = bbox
            if right <= left or bottom <= top:
                bbox_errors.append(f"{label_path.name}: invalid bbox {bbox}")
            if left < 0 or top < 0 or right > 1280 or bottom > 720:
                bbox_errors.append(f"{label_path.name}: bbox out of 1280x720 {bbox}")

    print("KITTI dataset check")
    print(f"dataset_root: {args.dataset_root}")
    print(f"images: {len(images)}")
    print(f"labels: {len(labels)}")
    print(f"matched pairs: {len(image_stems & label_stems)}")
    print(f"missing labels: {len(image_stems - label_stems)}")
    print(f"missing images: {len(label_stems - image_stems)}")
    print(f"empty label files: {empty_labels}")
    print(f"class counts: {dict(class_counts)}")
    print(f"unknown classes: {dict(unknown_counts)}")
    print(f"field errors: {len(field_errors)}")
    print(f"bbox errors: {len(bbox_errors)}")

    for message in field_errors[:10]:
        print(f"FIELD_ERROR {message}")
    for message in bbox_errors[:10]:
        print(f"BBOX_ERROR {message}")

    if image_stems != label_stems or field_errors or bbox_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
