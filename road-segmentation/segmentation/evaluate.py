"""Evaluate the best U-Net checkpoint on the untouched test split."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CarlaSegmentationDataset, MEAN, STD
from inference import load_model
from metrics import SegmentationMetrics
from postprocess import overlay_segmentation


CLASS_NAMES = ["Background", "DrivableArea", "LaneMarking"]


def denormalize(image_tensor):
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = np.clip((image * STD + MEAN) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def scenario_from_name(name):
    return Path(name).stem.rsplit("_", 1)[0]


def add_title(image, title):
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 32), (20, 20, 20), -1)
    cv2.putText(canvas, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--previews", type=int, default=12)
    parser.add_argument("--split", default="test", help="Dataset split to evaluate")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.weights, device)
    image_size = tuple(checkpoint.get("image_size", [640, 360]))
    dataset = CarlaSegmentationDataset(args.data, args.split, image_size, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    overall = SegmentationMetrics(3)
    scenario_metrics = defaultdict(lambda: SegmentationMetrics(3))
    args.output.mkdir(parents=True, exist_ok=True)
    preview_dir = args.output / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_indices = set(np.linspace(0, len(dataset) - 1, min(args.previews, len(dataset)), dtype=int).tolist())
    sample_index = 0

    with torch.inference_mode():
        for images, masks, names in loader:
            logits = model(images.to(device, non_blocking=True))
            predictions = logits.argmax(dim=1).cpu()
            overall.update(predictions, masks)
            for local_index, name in enumerate(names):
                scenario_metrics[scenario_from_name(name)].update(
                    predictions[local_index : local_index + 1], masks[local_index : local_index + 1]
                )
                if sample_index in preview_indices:
                    image = denormalize(images[local_index])
                    pred_mask = predictions[local_index].numpy().astype(np.uint8)
                    gt_mask = masks[local_index].numpy().astype(np.uint8)
                    ground_truth, _, _ = overlay_segmentation(image, gt_mask)
                    prediction, _, _ = overlay_segmentation(image, pred_mask)
                    comparison = np.concatenate(
                        [add_title(image, "RGB"), add_title(ground_truth, "Ground truth"), add_title(prediction, "U-Net prediction")],
                        axis=1,
                    )
                    cv2.imwrite(str(preview_dir / f"{sample_index:03d}_{Path(name).stem}.jpg"), comparison)
                sample_index += 1

    result = {
        "split": args.split,
        "samples": len(dataset),
        "weights": str(args.weights),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "device": str(device),
        "image_size": list(image_size),
        "classes": CLASS_NAMES,
        "overall": overall.compute(),
        "by_scenario": {name: metric.compute() for name, metric in sorted(scenario_metrics.items())},
        "preview_count": len(preview_indices),
    }
    output_path = args.output / f"{args.split}_metrics.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    print(f"metrics={output_path}")


if __name__ == "__main__":
    main()
