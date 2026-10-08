import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


CLASS_NAMES = {
    0: "Car",
    1: "Pedestrian",
    2: "TrafficLight",
    3: "TrafficSign",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a leakage-resistant YOLO dataset from scenario groups."
    )
    parser.add_argument("--original-yolo", type=Path, required=True)
    parser.add_argument("--scenario-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--original-block-size", type=int, default=100)
    parser.add_argument(
        "--copy-mode", choices=["copy", "hardlink"], default="hardlink"
    )
    return parser.parse_args()


def link_or_copy(source, destination, mode):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            destination.hardlink_to(source)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def read_counts(label_path):
    counts = Counter()
    frame_classes = set()
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid YOLO label row: {label_path}: {line}")
            class_id = int(parts[0])
            if class_id not in CLASS_NAMES:
                raise ValueError(f"Unknown class {class_id}: {label_path}")
            counts[class_id] += 1
            frame_classes.add(class_id)
    return counts, frame_classes


def write_yaml(path, output):
    relative = Path(os.path.relpath(output.resolve(), path.parent.resolve())).as_posix()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"path: {relative}",
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
        ),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    if args.output.exists():
        shutil.rmtree(args.output)

    groups = defaultdict(list)
    original_samples = []
    for old_split in ("train", "val", "test"):
        image_dir = args.original_yolo / "images" / old_split
        label_dir = args.original_yolo / "labels" / old_split
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            original_samples.append((image_path, label_path))

    original_samples.sort(key=lambda sample: int(sample[0].stem))
    original_block_policy = {
        0: "train",
        1: "train",
        2: "train",
        3: "train",
        4: "train",
        5: "train",
        6: "buffer",
        7: "val",
        8: "buffer",
        9: "test",
    }
    discarded_original_images = 0
    for image_path, label_path in original_samples:
        frame_index = int(image_path.stem)
        block_index = frame_index // args.original_block_size
        split = original_block_policy.get(block_index, "buffer")
        if split == "buffer":
            discarded_original_images += 1
            continue
        group = f"town10_block_{block_index:02d}"
        groups[(split, group)].append(
            (image_path, label_path, f"town10_{image_path.stem}")
        )

    manifest_path = args.scenario_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for scenario in manifest["config"]["scenarios"]:
        scenario_dir = args.scenario_root / "scenarios" / scenario["name"]
        if not (scenario_dir / "scenario.json").exists():
            raise FileNotFoundError(f"Incomplete scenario: {scenario['name']}")
        for image_path in sorted((scenario_dir / "images").iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = scenario_dir / "labels" / f"{image_path.stem}.txt"
            groups[(scenario["split"], scenario["name"])].append(
                (image_path, label_path, image_path.stem)
            )

    split_groups = defaultdict(set)
    split_images = Counter()
    split_objects = defaultdict(Counter)
    split_frames = defaultdict(Counter)
    seen_stems = set()
    seen_hashes = defaultdict(set)

    for (split, group), samples in sorted(groups.items()):
        split_groups[split].add(group)
        for image_path, label_path, output_stem in samples:
            if output_stem in seen_stems:
                raise RuntimeError(f"Duplicate output stem: {output_stem}")
            seen_stems.add(output_stem)
            if not label_path.exists():
                raise FileNotFoundError(label_path)
            destination_image = (
                args.output / "images" / split / f"{output_stem}{image_path.suffix.lower()}"
            )
            destination_label = args.output / "labels" / split / f"{output_stem}.txt"
            link_or_copy(image_path, destination_image, args.copy_mode)
            link_or_copy(label_path, destination_label, args.copy_mode)

            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            for other_split, hashes in seen_hashes.items():
                if other_split != split and digest in hashes:
                    raise RuntimeError(
                        f"Exact image leakage between {other_split} and {split}: {image_path}"
                    )
            seen_hashes[split].add(digest)

            counts, frame_classes = read_counts(label_path)
            split_images[split] += 1
            split_objects[split].update(counts)
            for class_id in frame_classes:
                split_frames[split][class_id] += 1

    group_sets = list(split_groups.items())
    for index, (split_a, groups_a) in enumerate(group_sets):
        for split_b, groups_b in group_sets[index + 1 :]:
            overlap = groups_a & groups_b
            if overlap:
                raise RuntimeError(
                    f"Scenario leakage between {split_a} and {split_b}: {sorted(overlap)}"
                )

    write_yaml(args.data_yaml, args.output)
    summary = {
        "split_strategy": "scenario_grouped",
        "original_sequence_policy": (
            "100-frame contiguous groups; blocks 0-5 train, block 7 val, "
            "block 9 test; blocks 6 and 8 are leakage buffers"
        ),
        "discarded_original_buffer_images": discarded_original_images,
        "splits": dict(split_images),
        "groups": {split: sorted(values) for split, values in split_groups.items()},
        "group_overlap": False,
        "exact_image_hash_overlap": False,
        "per_split_object_counts": {
            split: {
                CLASS_NAMES[class_id]: split_objects[split][class_id]
                for class_id in CLASS_NAMES
            }
            for split in ("train", "val", "test")
        },
        "per_split_frame_counts": {
            split: {
                CLASS_NAMES[class_id]: split_frames[split][class_id]
                for class_id in CLASS_NAMES
            }
            for split in ("train", "val", "test")
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
