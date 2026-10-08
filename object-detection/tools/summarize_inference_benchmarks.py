"""Aggregate verified inference benchmarks and draw a comparison chart."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "reports" / "inference_benchmarks"

CASES = [
    ("pytorch_cpu", "PyTorch CPU", "pytorch_cpu.json"),
    ("onnx_cpu", "ONNX CPU", "onnx_cpu.json"),
    ("pytorch_gpu", "PyTorch GPU", "pytorch_gpu.json"),
    ("onnx_gpu", "ONNX GPU", "onnx_gpu.json"),
    ("tensorrt_gpu_fp16", "TensorRT GPU FP16", "tensorrt_gpu_fp16.json"),
    ("fusion_pytorch_cpu", "双模型融合 CPU", "fusion_pytorch_cpu.json"),
    ("fusion_pytorch_gpu", "双模型融合 GPU", "fusion_pytorch_gpu.json"),
]


def load(name):
    return json.loads((BENCHMARK_DIR / name).read_text(encoding="utf-8"))


def main():
    rows = []
    for key, label, filename in CASES:
        item = load(filename)
        rows.append(
            {
                "key": key,
                "label": label,
                "backend": Path(item["weights"]).suffix.lower().lstrip(".") or "pytorch",
                "device": item["device"],
                "images": item["images"],
                "imgsz": item["imgsz"],
                "preloaded": item["preloaded"],
                "mean_latency_ms": item["mean_latency_ms"],
                "median_latency_ms": item["median_latency_ms"],
                "p95_latency_ms": item["p95_latency_ms"],
                "fps": item["fps"],
            }
        )

    by_key = {row["key"]: row for row in rows}
    pt_cpu = by_key["pytorch_cpu"]["mean_latency_ms"]
    pt_gpu = by_key["pytorch_gpu"]["mean_latency_ms"]
    for row in rows:
        baseline = pt_cpu if "cpu" in row["key"] else pt_gpu
        row["speedup_vs_same_device_pytorch"] = baseline / row["mean_latency_ms"]

    summary = {
        "method": {
            "source": "data/yolo_carla/images/test",
            "images": 150,
            "imgsz": 640,
            "batch": 1,
            "warmup": 20,
            "preloaded": True,
            "timed_scope": "model preprocessing + inference + NMS/postprocess; excludes disk image decoding",
            "hardware": "NVIDIA GeForce RTX 4070 Laptop GPU 8 GB / laptop CPU",
            "limitation": "software inference benchmark only; not real-vehicle camera-to-control end-to-end latency",
        },
        "results": rows,
        "key_findings": {
            "fastest_single_model": "tensorrt_gpu_fp16",
            "tensorrt_speedup_vs_pytorch_gpu": by_key["tensorrt_gpu_fp16"]["speedup_vs_same_device_pytorch"],
            "onnx_gpu_speedup_vs_pytorch_gpu": by_key["onnx_gpu"]["speedup_vs_same_device_pytorch"],
            "onnx_cpu_speedup_vs_pytorch_cpu": by_key["onnx_cpu"]["speedup_vs_same_device_pytorch"],
            "fusion_gpu_speedup_vs_fusion_cpu": by_key["fusion_pytorch_cpu"]["mean_latency_ms"] / by_key["fusion_pytorch_gpu"]["mean_latency_ms"],
        },
    }

    output_json = ROOT / "reports" / "inference_benchmark_summary.json"
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    chart_rows = rows[:5]
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#ECA82C", "#54A24B"]
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    bars = ax.bar([row["label"] for row in chart_rows], [row["fps"] for row in chart_rows], color=colors)
    ax.set_ylabel("FPS (batch=1, 640 px)")
    ax.set_title("Laptop inference backend comparison (150 preloaded test images)")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, labels=[f"{row['fps']:.1f}" for row in chart_rows], padding=3)
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    output_png = ROOT / "reports" / "inference_benchmark_comparison.png"
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output_json)
    print(output_png)


if __name__ == "__main__":
    main()
