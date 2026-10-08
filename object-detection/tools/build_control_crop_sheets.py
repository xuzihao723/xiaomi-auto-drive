"""Build numbered context crops for exhaustive traffic-control label review."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


NAMES = {2: "TrafficLight", 3: "TrafficSign"}
COLORS = {2: (60, 210, 60), 3: (220, 80, 220)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--rows", type=int, default=4)
    return parser.parse_args()


def find_image(directory, stem):
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(stem)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    crop_width, crop_height = 480, 300
    records, crops = [], []
    review_id = 0
    for label_path in sorted(args.labels.glob("*.txt")):
        image = cv2.imread(str(find_image(args.images, label_path.stem)))
        height, width = image.shape[:2]
        for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            fields = raw.split()
            if not fields or int(fields[0]) not in NAMES:
                continue
            class_id = int(fields[0])
            x, y, w, h = map(float, fields[1:5])
            x1, y1 = int((x - w / 2) * width), int((y - h / 2) * height)
            x2, y2 = int((x + w / 2) * width), int((y + h / 2) * height)
            box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
            context_w = max(180, box_w * 7)
            context_h = max(120, box_h * 6)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            left = max(0, min(width - context_w, cx - context_w // 2))
            top = max(0, min(height - context_h, cy - context_h // 2))
            right, bottom = min(width, left + context_w), min(height, top + context_h)
            crop = image[top:bottom, left:right].copy()
            scale_x, scale_y = crop_width / crop.shape[1], (crop_height - 34) / crop.shape[0]
            crop = cv2.resize(crop, (crop_width, crop_height - 34), interpolation=cv2.INTER_CUBIC)
            p1 = (int((x1 - left) * scale_x), int((y1 - top) * scale_y))
            p2 = (int((x2 - left) * scale_x), int((y2 - top) * scale_y))
            cv2.rectangle(crop, p1, p2, COLORS[class_id], 3, cv2.LINE_AA)
            header = np.full((34, crop_width, 3), 20, dtype=np.uint8)
            review_id += 1
            title = f"ID={review_id:03d} {label_path.stem} line={line_number} {NAMES[class_id]}"
            cv2.putText(header, title, (7, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 1, cv2.LINE_AA)
            crops.append(np.vstack([header, crop]))
            records.append({
                "review_id": review_id,
                "stem": label_path.stem,
                "label_line": line_number,
                "original_class_id": class_id,
                "original_class_name": NAMES[class_id],
                "decision": "pending",
                "review_note": "",
            })

    per_sheet = args.columns * args.rows
    for sheet_number, start in enumerate(range(0, len(crops), per_sheet), 1):
        sheet = np.full((args.rows * crop_height, args.columns * crop_width, 3), 235, dtype=np.uint8)
        for offset, crop in enumerate(crops[start:start + per_sheet]):
            row, column = divmod(offset, args.columns)
            y, x = row * crop_height, column * crop_width
            sheet[y:y + crop_height, x:x + crop_width] = crop
        cv2.imwrite(str(args.output / f"control_crops_{sheet_number:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])

    with (args.output / "control_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"controls={len(records)} sheets={(len(crops) + per_sheet - 1) // per_sheet}")


if __name__ == "__main__":
    main()
