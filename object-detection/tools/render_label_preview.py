import argparse
import importlib.util
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Render one YOLO label preview.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--label", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_validator():
    path = Path(__file__).with_name("validate_yolo_dataset.py")
    spec = importlib.util.spec_from_file_location("dataset_validator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    args = parse_args()
    validator = load_validator()
    rows = validator.parse_label(args.label)
    preview = validator.draw_sample(args.image, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), preview):
        raise RuntimeError(f"Could not write preview: {args.output}")
    print(args.output)


if __name__ == "__main__":
    main()
