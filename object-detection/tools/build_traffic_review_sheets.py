"""Render traffic-control label review sheets and a machine-readable manifest."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


COLORS = {2: (60, 210, 60), 3: (220, 80, 220)}
NAMES = {2: "TrafficLight", 3: "TrafficSign"}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proposals", type=Path)
    parser.add_argument("--proposal-conf", type=float, default=0.15)
    parser.add_argument("--sheet-columns", type=int, default=2)
    parser.add_argument("--sheet-rows", type=int, default=3)
    return parser.parse_args()


def find_image(images_dir, stem):
    for extension in IMAGE_EXTENSIONS:
        path = images_dir / f"{stem}{extension}"
        if path.exists():
            return path
    raise FileNotFoundError(stem)


def load_controls(label_path, width, height, include_confidence=False):
    controls = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = raw_line.split()
        if not fields:
            continue
        class_id = int(fields[0])
        if class_id not in NAMES:
            continue
        x, y, w, h = map(float, fields[1:5])
        x1 = max(0, int(round((x - w / 2) * width)))
        y1 = max(0, int(round((y - h / 2) * height)))
        x2 = min(width - 1, int(round((x + w / 2) * width)))
        y2 = min(height - 1, int(round((y + h / 2) * height)))
        controls.append(
            {
                "line": line_number,
                "class_id": class_id,
                "class_name": NAMES[class_id],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": x2 - x1,
                "height": y2 - y1,
                "confidence": float(fields[5]) if include_confidence and len(fields) > 5 else None,
            }
        )
    return controls


def iou(first, second):
    x1, y1 = max(first["x1"], second["x1"]), max(first["y1"], second["y1"])
    x2, y2 = min(first["x2"], second["x2"]), min(first["y2"], second["y2"])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = first["width"] * first["height"] + second["width"] * second["height"] - intersection
    return intersection / union if union else 0.0


def annotate(image, stem, controls, proposals):
    canvas = image.copy()
    counts = {2: 0, 3: 0}
    for index, control in enumerate(controls, start=1):
        class_id = control["class_id"]
        counts[class_id] += 1
        color = COLORS[class_id]
        p1 = (control["x1"], control["y1"])
        p2 = (control["x2"], control["y2"])
        cv2.rectangle(canvas, p1, p2, color, 3, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{index}:{'L' if class_id == 2 else 'S'}",
            (p1[0], max(18, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    unmatched = []
    for proposal in proposals:
        if any(
            proposal["class_id"] == control["class_id"] and iou(proposal, control) >= 0.5
            for control in controls
        ):
            continue
        unmatched.append(proposal)
        color = (255, 220, 40) if proposal["class_id"] == 2 else (40, 170, 255)
        p1 = (proposal["x1"], proposal["y1"])
        p2 = (proposal["x2"], proposal["y2"])
        cv2.rectangle(canvas, p1, p2, color, 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"P:{'L' if proposal['class_id'] == 2 else 'S'} {proposal['confidence']:.2f}",
            (p1[0], max(18, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), (20, 20, 20), -1)
    cv2.putText(
        canvas,
        f"{stem} existing: L={counts[2]} S={counts[3]} unmatched proposals={len(unmatched)}",
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def make_sheets(frames, output_dir, columns, rows):
    per_sheet = columns * rows
    thumb_width, thumb_height = 960, 540
    for sheet_index, start in enumerate(range(0, len(frames), per_sheet), start=1):
        chunk = frames[start : start + per_sheet]
        sheet = np.full(
            (rows * thumb_height, columns * thumb_width, 3), 245, dtype=np.uint8
        )
        for local_index, frame in enumerate(chunk):
            thumb = cv2.resize(frame, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
            row, col = divmod(local_index, columns)
            y, x = row * thumb_height, col * thumb_width
            sheet[y : y + thumb_height, x : x + thumb_width] = thumb
        cv2.imwrite(str(output_dir / f"review_sheet_{sheet_index:02d}.jpg"), sheet)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames, manifest = [], []
    labels = sorted(args.labels.glob("*.txt"))
    for image_index, label_path in enumerate(labels, start=1):
        image_path = find_image(args.images, label_path.stem)
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read {image_path}")
        height, width = image.shape[:2]
        controls = load_controls(label_path, width, height)
        proposals = []
        if args.proposals:
            proposal_path = args.proposals / f"{label_path.stem}.txt"
            if proposal_path.exists():
                proposals = [
                    item
                    for item in load_controls(
                        proposal_path, width, height, include_confidence=True
                    )
                    if item["confidence"] is not None
                    and item["confidence"] >= args.proposal_conf
                ]
        frames.append(annotate(image, label_path.stem, controls, proposals))
        if not controls:
            manifest.append(
                {
                    "image_index": image_index,
                    "stem": label_path.stem,
                    "label_line": "",
                    "class_id": "",
                    "class_name": "",
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": "",
                    "width": "",
                    "height": "",
                    "review_status": "pending_empty_frame_check",
                    "review_note": "",
                }
            )
        for control in controls:
            manifest.append(
                {
                    "image_index": image_index,
                    "stem": label_path.stem,
                    "label_line": control["line"],
                    "class_id": control["class_id"],
                    "class_name": control["class_name"],
                    "x1": control["x1"],
                    "y1": control["y1"],
                    "x2": control["x2"],
                    "y2": control["y2"],
                    "width": control["width"],
                    "height": control["height"],
                    "review_status": "pending",
                    "review_note": "",
                }
            )
    make_sheets(frames, args.output, args.sheet_columns, args.sheet_rows)
    manifest_path = args.output / "review_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"images={len(frames)} manifest_rows={len(manifest)}")
    print(manifest_path)


if __name__ == "__main__":
    main()
