"""Render unmatched model proposals as numbered crops for human漏标 review."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from build_traffic_review_sheets import COLORS, find_image, iou, load_controls


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--match-iou", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records, crops = [], []
    crop_width, crop_height = 480, 300
    for label_path in sorted(args.labels.glob("*.txt")):
        image = cv2.imread(str(find_image(args.images, label_path.stem)))
        height, width = image.shape[:2]
        existing = load_controls(label_path, width, height)
        proposal_path = args.proposals / label_path.name
        if not proposal_path.exists():
            continue
        proposals = load_controls(proposal_path, width, height, include_confidence=True)
        for proposal in proposals:
            if proposal["confidence"] < args.confidence:
                continue
            if any(proposal["class_id"] == item["class_id"] and iou(proposal, item) >= args.match_iou for item in existing):
                continue
            x1, y1, x2, y2 = (proposal[key] for key in ("x1", "y1", "x2", "y2"))
            box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
            context_w, context_h = max(180, box_w * 7), max(120, box_h * 6)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            left = max(0, min(width - context_w, cx - context_w // 2))
            top = max(0, min(height - context_h, cy - context_h // 2))
            right, bottom = min(width, left + context_w), min(height, top + context_h)
            crop = image[top:bottom, left:right].copy()
            sx, sy = crop_width / crop.shape[1], (crop_height - 34) / crop.shape[0]
            crop = cv2.resize(crop, (crop_width, crop_height - 34), interpolation=cv2.INTER_CUBIC)
            p1, p2 = (int((x1-left)*sx), int((y1-top)*sy)), (int((x2-left)*sx), int((y2-top)*sy))
            cv2.rectangle(crop, p1, p2, COLORS[proposal["class_id"]], 3, cv2.LINE_AA)
            header = np.full((34, crop_width, 3), 20, dtype=np.uint8)
            review_id = len(records) + 1
            title = f"P={review_id:03d} {label_path.stem} {proposal['class_name']} conf={proposal['confidence']:.2f}"
            cv2.putText(header, title, (7, 23), cv2.FONT_HERSHEY_SIMPLEX, .51, (255,255,255), 1, cv2.LINE_AA)
            crops.append(np.vstack([header, crop]))
            records.append({
                "proposal_id": review_id, "stem": label_path.stem,
                "class_id": proposal["class_id"], "class_name": proposal["class_name"],
                "confidence": proposal["confidence"], "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "decision": "pending", "review_note": "",
            })
    columns, rows, per_sheet = 4, 4, 16
    for sheet_number, start in enumerate(range(0, len(crops), per_sheet), 1):
        sheet = np.full((rows*crop_height, columns*crop_width, 3), 235, dtype=np.uint8)
        for offset, crop in enumerate(crops[start:start+per_sheet]):
            row, column = divmod(offset, columns)
            sheet[row*crop_height:(row+1)*crop_height, column*crop_width:(column+1)*crop_width] = crop
        cv2.imwrite(str(args.output / f"proposal_crops_{sheet_number:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
    with (args.output / "proposal_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    print(f"unmatched_proposals={len(records)} sheets={(len(crops)+15)//16}")


if __name__ == "__main__":
    main()
