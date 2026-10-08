import argparse
import json
import platform
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run repeatable CUDA forward/backward stress checks."
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--matrix-size", type=int, default=2048)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/cuda_stability.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment")

    device = torch.device("cuda:0")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    a = torch.randn(
        args.matrix_size,
        args.matrix_size,
        device=device,
        requires_grad=True,
    )
    b = torch.randn(
        args.matrix_size,
        args.matrix_size,
        device=device,
        requires_grad=True,
    )

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    final_loss = None

    for iteration in range(1, args.iterations + 1):
        product = torch.mm(a, b)
        loss = product.square().mean()
        loss.backward()
        with torch.no_grad():
            a -= 1e-7 * a.grad
            b -= 1e-7 * b.grad
            a.grad.zero_()
            b.grad.zero_()
        if iteration % 20 == 0:
            torch.cuda.synchronize(device)
            print(
                f"iteration={iteration}/{args.iterations} "
                f"loss={loss.item():.6f}"
            )
        final_loss = float(loss.item())

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    result = {
        "status": "passed",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "iterations": args.iterations,
        "matrix_size": args.matrix_size,
        "elapsed_seconds": elapsed,
        "iterations_per_second": args.iterations / elapsed,
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "final_loss": final_loss,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
