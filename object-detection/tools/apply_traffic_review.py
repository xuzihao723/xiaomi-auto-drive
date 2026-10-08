"""Apply an auditable human-review decision set without overwriting pseudo labels."""

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--control-review", type=Path, required=True)
    parser.add_argument("--proposal-review", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-view", type=Path)
    return parser.parse_args()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def count_controls(directory):
    counts = Counter()
    for path in directory.glob("*.txt"):
        for raw in path.read_text(encoding="utf-8").splitlines():
            fields = raw.split()
            if fields and int(fields[0]) in (2, 3):
                counts[int(fields[0])] += 1
    return counts


def main():
    args = parse_args()
    config = json.loads(args.corrections.read_text(encoding="utf-8"))
    controls = {int(row["review_id"]): row for row in read_csv(args.control_review)}
    proposals = {int(row["proposal_id"]): row for row in read_csv(args.proposal_review)}
    source_labels = args.source / "labels" / "test"
    source_images = args.source / "images" / "test"
    reviewed_labels = args.output / "reviewed_labels"
    reviewed_labels.mkdir(parents=True, exist_ok=True)
    for path in source_labels.glob("*.txt"):
        shutil.copy2(path, reviewed_labels / path.name)

    drop_ids = set(config["drop_review_ids"])
    reclassify = {int(key): int(value) for key, value in config["reclassify_review_ids"].items()}
    operations = []
    by_file = {}
    for review_id in sorted(drop_ids | set(reclassify)):
        row = controls[review_id]
        by_file.setdefault(row["stem"], {})[int(row["label_line"])] = (
            "drop" if review_id in drop_ids else "reclassify",
            reclassify.get(review_id), review_id,
        )
    for stem, changes in by_file.items():
        label_path = reviewed_labels / f"{stem}.txt"
        lines = label_path.read_text(encoding="utf-8").splitlines()
        revised = []
        for line_number, raw in enumerate(lines, 1):
            if line_number not in changes:
                revised.append(raw)
                continue
            action, new_class, review_id = changes[line_number]
            fields = raw.split()
            if action == "drop":
                operations.append({"action": "drop", "review_id": review_id, "stem": stem, "line": line_number, "old_class": fields[0], "new_class": ""})
                continue
            old_class = fields[0]
            fields[0] = str(new_class)
            revised.append(" ".join(fields))
            operations.append({"action": "reclassify", "review_id": review_id, "stem": stem, "line": line_number, "old_class": old_class, "new_class": new_class})
        label_path.write_text("\n".join(revised) + ("\n" if revised else ""), encoding="utf-8")

    for proposal_id in config["add_proposal_ids"]:
        row = proposals[int(proposal_id)]
        stem = row["stem"]
        image = cv2.imread(str(next(source_images.glob(f"{stem}.*"))))
        height, width = image.shape[:2]
        x1, y1, x2, y2 = (float(row[key]) for key in ("x1", "y1", "x2", "y2"))
        x, y = ((x1+x2)/2/width, (y1+y2)/2/height)
        w, h = ((x2-x1)/width, (y2-y1)/height)
        label_path = reviewed_labels / f"{stem}.txt"
        with label_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{int(row['class_id'])} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        operations.append({"action": "add", "review_id": proposal_id, "stem": stem, "line": "", "old_class": "", "new_class": int(row["class_id"])})

    before, after = count_controls(source_labels), count_controls(reviewed_labels)
    summary = {
        "images_reviewed": 150,
        "before": {"TrafficLight": before[2], "TrafficSign": before[3], "total": before[2] + before[3]},
        "after": {"TrafficLight": after[2], "TrafficSign": after[3], "total": after[2] + after[3]},
        "operations": dict(Counter(item["action"] for item in operations)),
        "source_labels_preserved": True,
    }
    (args.output / "traffic_review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "traffic_review_operations.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(operations[0]))
        writer.writeheader(); writer.writerows(operations)

    if args.dataset_view:
        image_out = args.dataset_view / "images" / "test"
        label_out = args.dataset_view / "labels" / "test"
        image_out.mkdir(parents=True, exist_ok=True); label_out.mkdir(parents=True, exist_ok=True)
        for source in source_images.iterdir():
            target = image_out / source.name
            if not target.exists():
                try: os.link(source, target)
                except OSError: shutil.copy2(source, target)
        for source in reviewed_labels.glob("*.txt"):
            shutil.copy2(source, label_out / source.name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
