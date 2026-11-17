"""Inference-cost benchmark: parameters, FLOPs, latency, memory, throughput.

Measures an existing checkpoint on GPU and CPU with batch size 1 (the
deployment-relevant case) and batch size 16 (throughput).  Results are written
to JSON and a CSV row so they can be merged into the final comparison table.

Usage::

    PYTHONPATH=src .venv/bin/python -m nriqa.inference \
        --checkpoint runs/.../best.pt --out results/inference/siglip.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from typing import Any, Dict, Optional

import numpy as np
import torch

from .evaluate import load_model_from_checkpoint
from .models import count_parameters


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def measure_latency(
    model: torch.nn.Module,
    device: torch.device,
    size: int = 224,
    batch_size: int = 1,
    warmup: int = 10,
    iters: int = 50,
) -> Dict[str, float]:
    model.eval().to(device)
    x = torch.rand(batch_size, 3, size, size, device=device)
    for _ in range(warmup):
        model(x)
    _sync(device)
    times = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        model(x)
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    times_np = np.asarray(times)
    return {
        "latency_ms_mean": float(times_np.mean()),
        "latency_ms_median": float(np.median(times_np)),
        "latency_ms_p95": float(np.quantile(times_np, 0.95)),
        "throughput_img_s": float(batch_size / (times_np.mean() / 1000.0)),
    }


@torch.no_grad()
def measure_flops(model: torch.nn.Module, device: torch.device, size: int = 224) -> Optional[float]:
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        return None
    model.eval().to(device)
    x = torch.rand(1, 3, size, size, device=device)
    counter = FlopCounterMode(display=False)
    try:
        with counter:
            model(x)
        return float(counter.get_total_flops())
    except Exception:
        return None


def measure_peak_memory(model: torch.nn.Module, device: torch.device, size: int = 224,
                        batch_size: int = 1) -> Optional[float]:
    if device.type != "cuda":
        return None
    model.eval().to(device)
    torch.cuda.reset_peak_memory_stats(device)
    x = torch.rand(batch_size, 3, size, size, device=device)
    with torch.no_grad():
        model(x)
    _sync(device)
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))


def benchmark_checkpoint(checkpoint: str, device: str = "cuda:0", size: int = 224) -> Dict[str, Any]:
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model, ckpt = load_model_from_checkpoint(checkpoint, dev)
    cfg = ckpt.get("cfg", {})
    params = count_parameters(model)
    result: Dict[str, Any] = {
        "checkpoint": os.path.abspath(checkpoint),
        "model": cfg.get("model", {}).get("name", "unknown"),
        "config": cfg.get("_config_path") or cfg.get("experiment", {}).get("name"),
        "params": int(params),
        "input_size": int(size),
        "device": str(dev),
        "gpu_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
        "cpu_name": platform.processor(),
        "gpu_batch1": measure_latency(model, dev, size=size, batch_size=1),
        "gpu_batch16": measure_latency(model, dev, size=size, batch_size=16, iters=30),
        "flops": measure_flops(model, dev, size=size),
        "peak_memory_mb_batch1": measure_peak_memory(model, dev, size=size, batch_size=1),
        "peak_memory_mb_batch16": measure_peak_memory(model, dev, size=size, batch_size=16),
    }
    if torch.cuda.is_available():
        result["cpu_batch1"] = measure_latency(model.cpu(), torch.device("cpu"), size=size,
                                               batch_size=1, warmup=3, iters=15)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="results/inference.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args(argv)
    result = benchmark_checkpoint(args.checkpoint, args.device, args.size)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
