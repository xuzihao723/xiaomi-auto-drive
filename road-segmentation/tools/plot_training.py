"""Plot loss and segmentation IoU curves from the training CSV."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No training rows in {args.csv}")

    epochs = [int(row["epoch"]) for row in rows]
    values = lambda key: [float(row[key]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    axes[0].plot(epochs, values("train_loss"), label="Train loss")
    axes[0].plot(epochs, values("val_loss"), label="Validation loss")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="U-Net loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, values("train_miou"), label="Train road/lane mIoU")
    axes[1].plot(epochs, values("val_miou"), label="Validation road/lane mIoU")
    axes[1].plot(epochs, values("val_road_iou"), label="Validation road IoU", linestyle="--")
    axes[1].plot(epochs, values("val_lane_iou"), label="Validation lane IoU", linestyle="--")
    axes[1].set(xlabel="Epoch", ylabel="IoU", title="Segmentation accuracy", ylim=(0.0, 1.0))
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
