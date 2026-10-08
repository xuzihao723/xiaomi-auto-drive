"""Build a reproducible, train-only oversampling view of a YOLO dataset.

Validation and test images are linked exactly once. Training images containing
selected minority classes receive extra hard-linked aliases. No sample crosses
split boundaries, so the independent evaluation sets remain untouched.
"""

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path


CLASS_NAMES = {
    0: "Car",
    1: "Pedestrian",
    2: "TrafficLight",
    3: "TrafficSign",
}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--extra",
        action="append",
        default=["1:4", "3:2"],
        help="CLASS_ID:EXTRA_COPIES; repeatable (default: Pedestrian +4, TrafficSign +2).",
    )
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def parse_extra(values):
    result = {}
    for value in values:
        class_id_text, copies_text = value.split(":", 1)
        class_id, copies = int(class_id_text), int(copies_text)
        if class_id not in CLASS_NAMES or copies < 0:
            raise ValueError(f"Invalid --extra value: {value}")
        result[class_id] = copies
    return result


def read_classes(label_path):
    classes = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.split()
        if fields:
            classes.append(int(fields[0]))
    return classes


def link_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def find_image(images_dir, stem):
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found for label stem: {stem}")


def main():
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if source == output or source in output.parents:
        raise ValueError("Output must not be the source directory or a child of it.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")

    extra_copies = parse_extra(args.extra)
    summary = {
        "source": str(source),
        "output": str(output),
        "extra_copies_per_minority_image": {
            CLASS_NAMES[class_id]: copies for class_id, copies in extra_copies.items()
        },
        "splits": {},
        "integrity": {
            "validation_or_test_oversampled": False,
            "cross_split_copying": False,
        },
    }

    for split in ("train", "val", "test"):
        source_images = source / "images" / split
        source_labels = source / "labels" / split
        output_images = output / "images" / split
        output_labels = output / "labels" / split
        labels = sorted(source_labels.glob("*.txt"))
        source_objects = Counter()
        output_objects = Counter()
        source_image_count = 0
        output_image_count = 0
        repeated_by_reason = Counter()

        for label_path in labels:
            image_path = find_image(source_images, label_path.stem)
            classes = read_classes(label_path)
            source_objects.update(classes)
            source_image_count += 1

            aliases = [(label_path.stem, "original")]
            if split == "train":
                for class_id, copies in sorted(extra_copies.items()):
                    if class_id in classes:
                        for index in range(1, copies + 1):
                            aliases.append(
                                (
                                    f"{label_path.stem}__{CLASS_NAMES[class_id].lower()}rep{index:02d}",
                                    CLASS_NAMES[class_id],
                                )
                            )

            for alias, reason in aliases:
                link_file(image_path, output_images / f"{alias}{image_path.suffix.lower()}")
                link_file(label_path, output_labels / f"{alias}.txt")
                output_objects.update(classes)
                output_image_count += 1
                if reason != "original":
                    repeated_by_reason[reason] += 1

        summary["splits"][split] = {
            "source_images": source_image_count,
            "output_images": output_image_count,
            "added_training_aliases": output_image_count - source_image_count,
            "aliases_by_reason": dict(repeated_by_reason),
            "source_objects": {
                CLASS_NAMES[class_id]: source_objects[class_id] for class_id in CLASS_NAMES
            },
            "output_objects": {
                CLASS_NAMES[class_id]: output_objects[class_id] for class_id in CLASS_NAMES
            },
        }

    summary_path = args.summary or (output / "balanced_sampling_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
