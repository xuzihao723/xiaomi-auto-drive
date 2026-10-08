"""Export one YOLO checkpoint to ONNX FP32 and TensorRT FP16."""

import argparse
import json
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    exports = {}
    onnx_path = Path(model.export(format="onnx", imgsz=args.imgsz, batch=1, dynamic=False, simplify=True, opset=17))
    onnx_target = args.output / f"{args.weights.stem}_fp32.onnx"
    shutil.move(str(onnx_path), onnx_target)
    exports["onnx_fp32"] = str(onnx_target)

    engine_path = Path(model.export(format="engine", imgsz=args.imgsz, batch=1, dynamic=False, half=True, device=0, workspace=4))
    engine_target = args.output / f"{args.weights.stem}_fp16.engine"
    shutil.move(str(engine_path), engine_target)
    exports["tensorrt_fp16"] = str(engine_target)
    summary = {"source": str(args.weights), "imgsz": args.imgsz, "batch": 1, "exports": exports}
    (args.output / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
