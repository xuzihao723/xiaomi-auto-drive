"""Report local inference backend availability for reproducible benchmarks."""

import importlib.util
import json
import platform
import argparse
from pathlib import Path

import torch
import ultralytics


def module_status(name):
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"available": False, "version": None}
    module = __import__(name)
    return {"available": True, "version": getattr(module, "__version__", "unknown")}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path)
args = parser.parse_args()

result = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "ultralytics": ultralytics.__version__,
    "onnx": module_status("onnx"),
    "onnxruntime": module_status("onnxruntime"),
    "tensorrt": module_status("tensorrt"),
}
if result["onnxruntime"]["available"]:
    import onnxruntime

    result["onnxruntime"]["providers"] = onnxruntime.get_available_providers()
payload = json.dumps(result, indent=2)
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
print(payload)
