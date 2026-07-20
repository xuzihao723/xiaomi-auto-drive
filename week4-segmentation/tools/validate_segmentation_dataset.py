"""Validate image/mask pairing, class range and split separation."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = {"splits": {}, "errors": []}
    hashes = {}
    for split in ("train", "val", "test", "audit"):
        images = sorted((args.data / "images" / split).glob("*.png"))
        masks = {path.name: path for path in (args.data / "masks" / split).glob("*.png")}
        counts = Counter()
        sizes = Counter()
        for image_path in images:
            mask_path = masks.get(image_path.name)
            if mask_path is None:
                summary["errors"].append(f"missing mask: {split}/{image_path.name}")
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                summary["errors"].append(f"unreadable pair: {split}/{image_path.name}")
                continue
            if image.shape[:2] != mask.shape:
                summary["errors"].append(f"shape mismatch: {split}/{image_path.name}")
            values, pixel_counts = np.unique(mask, return_counts=True)
            if any(int(value) not in (0, 1, 2) for value in values):
                summary["errors"].append(f"invalid class: {split}/{image_path.name}")
            for value, count in zip(values, pixel_counts):
                counts[int(value)] += int(count)
            sizes[f"{image.shape[1]}x{image.shape[0]}"] += 1
            image_hash = digest(image_path)
            if image_hash in hashes:
                summary["errors"].append(
                    f"cross-split duplicate: {hashes[image_hash]} and {split}/{image_path.name}"
                )
            hashes[image_hash] = f"{split}/{image_path.name}"
        extra_masks = sorted(set(masks) - {path.name for path in images})
        for name in extra_masks:
            summary["errors"].append(f"mask without image: {split}/{name}")
        summary["splits"][split] = {
            "images": len(images),
            "masks": len(masks),
            "pixel_counts": {str(key): counts[key] for key in range(3)},
            "image_sizes": dict(sizes),
        }
    summary["passed"] = not summary["errors"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
