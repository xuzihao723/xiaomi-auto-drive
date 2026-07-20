"""Run U-Net inference and produce an overlay with lane fitting."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset import MEAN, STD
from model import build_model
from postprocess import overlay_segmentation


def load_model(weights, device):
    checkpoint = torch.load(weights, map_location=device, weights_only=False)
    config = checkpoint.get("model_config", {})
    model = build_model(
        num_classes=int(config.get("num_classes", 3)),
        base_channels=int(config.get("base_channels", 32)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def predict_mask(model, image_bgr, size, device):
    width, height = size
    rgb = cv2.cvtColor(cv2.resize(image_bgr, size), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    with torch.inference_mode():
        mask = model(tensor).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.weights, device)
    size = tuple(checkpoint.get("image_size", [640, 360]))
    image = cv2.imread(str(args.source), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.source)
    mask = predict_mask(model, image, size, device)
    overlay, polygon, curves = overlay_segmentation(image, mask)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), overlay)
    result = {
        "source": str(args.source),
        "weights": str(args.weights),
        "drivable_polygon": None if polygon is None else polygon.reshape(-1, 2).tolist(),
        "lane_curves": [
            {"side": item["side"], "coefficients": item["coefficients"], "points": item["points"].tolist()}
            for item in curves
        ],
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "lane_curves": len(curves), "drivable": polygon is not None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
