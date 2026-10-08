"""Export a compact inference-only U-Net checkpoint from a training checkpoint."""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    compact = {
        key: checkpoint[key]
        for key in ("model", "epoch", "metrics", "model_config", "image_size", "classes")
        if key in checkpoint
    }
    if "model" not in compact:
        raise KeyError(f"No model state in {args.checkpoint}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(compact, args.output)
    print(
        json.dumps(
            {
                "source": str(args.checkpoint),
                "output": str(args.output),
                "epoch": compact.get("epoch"),
                "bytes": args.output.stat().st_size,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
