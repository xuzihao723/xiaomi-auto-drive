"""Run deterministic unit checks for model, metrics and geometric post-processing."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "segmentation"))

from metrics import SegmentationMetrics  # noqa: E402
from model import build_model  # noqa: E402
from postprocess import fit_lane_curves, largest_drivable_polygon  # noqa: E402


def main():
    checks = {}

    model = build_model(base_channels=8).eval()
    with torch.inference_mode():
        output = model(torch.zeros(2, 3, 64, 128))
    checks["model_output_shape"] = list(output.shape) == [2, 3, 64, 128]

    target = torch.tensor([[[0, 1], [2, 1]]])
    metric = SegmentationMetrics(3)
    metric.update(target, target)
    perfect = metric.compute()
    checks["perfect_metric"] = (
        perfect["pixel_accuracy"] == 1.0
        and perfect["mean_iou_road_lane"] == 1.0
        and perfect["per_class_dice"] == [1.0, 1.0, 1.0]
    )

    mask = np.zeros((180, 320), dtype=np.uint8)
    road = np.array([[70, 179], [135, 70], [185, 70], [255, 179]], dtype=np.int32)
    cv2.fillPoly(mask, [road], 1)
    y = np.arange(65, 180)
    left_x = (145 - 0.42 * (y - 65)).astype(np.int32)
    right_x = (175 + 0.42 * (y - 65)).astype(np.int32)
    for xs in (left_x, right_x):
        for x_value, y_value in zip(xs, y):
            cv2.circle(mask, (int(x_value), int(y_value)), 2, 2, -1)
    polygon = largest_drivable_polygon(mask, min_area=100)
    curves = fit_lane_curves(mask, min_points=20)
    checks["drivable_polygon"] = polygon is not None and len(polygon) >= 4
    checks["two_lane_curves"] = len(curves) == 2 and all(len(item["points"]) >= 2 for item in curves)

    passed = all(checks.values())
    result = {"passed": passed, "checks": checks}
    output = ROOT / "reports" / "self_checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
