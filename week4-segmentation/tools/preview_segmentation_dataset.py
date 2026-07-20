"""Create a contact-sheet preview of RGB images and semantic masks."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def colorize(mask):
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    color[mask == 1] = (0, 180, 0)
    color[mask == 2] = (0, 220, 255)
    return color


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    images = sorted((args.data / "images" / args.split).glob("*.png"))[
        args.start : args.start + args.limit
    ]
    panels = []
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(args.data / "masks" / args.split / image_path.name), cv2.IMREAD_GRAYSCALE)
        overlay = cv2.addWeighted(image, 1.0, colorize(mask), 0.4, 0.0)
        cv2.putText(overlay, image_path.stem, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(overlay)
    if not panels:
        raise RuntimeError("No images found")
    columns = 2
    rows = (len(panels) + columns - 1) // columns
    height, width = panels[0].shape[:2]
    sheet = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        sheet[row * height : (row + 1) * height, column * width : (column + 1) * width] = panel
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet):
        raise RuntimeError(f"Failed to write {args.output}")
    print(args.output)


if __name__ == "__main__":
    main()
