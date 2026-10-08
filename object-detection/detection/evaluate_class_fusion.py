"""Evaluate a class-specialized fusion of two YOLO detectors.

The road-user model contributes Car/Pedestrian detections (classes 0/1), while
the traffic-control model contributes TrafficLight/TrafficSign detections
(classes 2/3). Metrics are calculated once on the unchanged YOLO test split.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--road-user-weights", type=Path, required=True)
    parser.add_argument("--traffic-control-weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def resolve_split(data_path, split):
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = Path(config.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    split_path = Path(config[split])
    if not split_path.is_absolute():
        split_path = root / split_path
    names_raw = config["names"]
    if isinstance(names_raw, list):
        names = {index: name for index, name in enumerate(names_raw)}
    else:
        names = {int(index): name for index, name in names_raw.items()}
    return split_path, names


def label_path_for(image_path):
    parts = list(image_path.parts)
    try:
        image_index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as error:
        raise ValueError(f"Image path has no 'images' directory: {image_path}") from error
    parts[image_index] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_targets(label_path, width, height):
    classes, boxes = [], []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.split()
        if not fields:
            continue
        class_id, x, y, w, h = map(float, fields[:5])
        x1 = (x - w / 2) * width
        y1 = (y - h / 2) * height
        x2 = (x + w / 2) * width
        y2 = (y + h / 2) * height
        classes.append(int(class_id))
        boxes.append((x1, y1, x2, y2))
    return (
        torch.tensor(classes, dtype=torch.float32),
        torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
    )


def match_predictions(pred_classes, true_classes, iou, iouv):
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0]), dtype=bool)
    correct_class = true_classes[:, None] == pred_classes
    class_iou = (iou * correct_class).cpu().numpy()
    for index, threshold in enumerate(iouv.cpu().tolist()):
        matches = np.array(np.nonzero(class_iou >= threshold)).T
        if not matches.shape[0]:
            continue
        if matches.shape[0] > 1:
            matches = matches[class_iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), index] = True
    return torch.tensor(correct, dtype=torch.bool)


def main():
    args = parse_args()
    from ultralytics import YOLO
    from ultralytics.utils.metrics import ConfusionMatrix, ap_per_class, box_iou

    split_path, names = resolve_split(args.data.resolve(), args.split)
    images = sorted(
        path for path in split_path.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No images found in {split_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    road_user_model = YOLO(str(args.road_user_weights))
    traffic_control_model = YOLO(str(args.traffic_control_weights))
    iouv = torch.linspace(0.5, 0.95, 10)
    confusion = ConfusionMatrix(names=names, task="detect")
    stats = {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}

    for start in range(0, len(images), args.batch):
        batch_paths = images[start : start + args.batch]
        common = dict(
            source=[str(path) for path in batch_paths],
            imgsz=args.imgsz,
            conf=0.001,
            iou=0.7,
            max_det=300,
            device=args.device,
            verbose=False,
        )
        road_results = road_user_model.predict(**common)
        traffic_results = traffic_control_model.predict(**common)

        for image_path, road_result, traffic_result in zip(
            batch_paths, road_results, traffic_results
        ):
            road = road_result.boxes.data.detach().cpu()
            traffic = traffic_result.boxes.data.detach().cpu()
            road = road[(road[:, 5] == 0) | (road[:, 5] == 1)]
            traffic = traffic[(traffic[:, 5] == 2) | (traffic[:, 5] == 3)]
            detections = torch.cat((road, traffic), dim=0)
            if detections.numel():
                detections = detections[torch.argsort(detections[:, 4], descending=True)]

            height, width = road_result.orig_shape
            true_classes, true_boxes = load_targets(
                label_path_for(image_path), width, height
            )
            if detections.shape[0]:
                correct = match_predictions(
                    detections[:, 5],
                    true_classes,
                    box_iou(true_boxes, detections[:, :4]),
                    iouv,
                )
                stats["tp"].append(correct.numpy())
                stats["conf"].append(detections[:, 4].numpy())
                stats["pred_cls"].append(detections[:, 5].numpy())
            else:
                stats["tp"].append(np.zeros((0, len(iouv)), dtype=bool))
                stats["conf"].append(np.zeros(0))
                stats["pred_cls"].append(np.zeros(0))
            stats["target_cls"].append(true_classes.numpy())

            confusion.process_batch(
                {
                    "bboxes": detections[:, :4],
                    "conf": detections[:, 4],
                    "cls": detections[:, 5],
                },
                {"bboxes": true_boxes, "cls": true_classes},
                conf=0.25,
                iou_thres=0.45,
            )

    combined = {key: np.concatenate(value, axis=0) for key, value in stats.items()}
    tp, fp, precision, recall, f1, ap, classes, *_ = ap_per_class(
        combined["tp"],
        combined["conf"],
        combined["pred_cls"],
        combined["target_cls"],
        plot=True,
        save_dir=args.output_dir,
        names=names,
        prefix="Box",
    )
    confusion.plot(save_dir=args.output_dir, normalize=False)
    confusion.plot(save_dir=args.output_dir, normalize=True)

    per_class = {}
    for row, class_id in enumerate(classes):
        per_class[names[int(class_id)]] = {
            "precision": float(precision[row]),
            "recall": float(recall[row]),
            "mAP50": float(ap[row, 0]),
            "mAP50_95": float(ap[row].mean()),
            "targets": int((combined["target_cls"] == class_id).sum()),
            "true_positives_at_best_f1": int(tp[row]),
            "false_positives_at_best_f1": int(fp[row]),
        }

    summary = {
        "method": "class_specialized_prediction_fusion",
        "road_user_weights": str(args.road_user_weights),
        "road_user_classes": ["Car", "Pedestrian"],
        "traffic_control_weights": str(args.traffic_control_weights),
        "traffic_control_classes": ["TrafficLight", "TrafficSign"],
        "split": args.split,
        "images": len(images),
        "targets": int(combined["target_cls"].shape[0]),
        "overall": {
            "precision": float(precision.mean()),
            "recall": float(recall.mean()),
            "mAP50": float(ap[:, 0].mean()),
            "mAP50_95": float(ap.mean()),
        },
        "per_class": per_class,
        "output_dir": str(args.output_dir),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
