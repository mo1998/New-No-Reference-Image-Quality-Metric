"""Benchmark suite for pretrained/classical NR-IQA methods.

Runs PyIQA metrics on the *same* content split and preprocessing as our models,
so published numbers can be compared with a like-for-like protocol.  Results
are written per metric (per-image predictions) plus one summary CSV.

Usage (inside the project venv that has pyiqa installed)::

    PYTHONPATH=src .venv/bin/python -m nriqa.benchmark --metrics brisque niqe musiq
    PYTHONPATH=src .venv/bin/python -m nriqa.benchmark --all --device cuda:1
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from .config import load_config, resolve_device
from .data import build_transform, load_dataframe, make_loader, split_dataframe
from .metrics import compute_metrics

# Metrics that are exposed by PyIQA 0.1.13 and relevant to NR-IQA
DEFAULT_METRICS = [
    "brisque",
    "niqe",
    "musiq",
    "clipiqa",
    "clipiqa+",
    "maniqa",
    "topiq_nr",
    "hyperiqa",
    "dbcnn",
    "tres",
    "arniqa",
    "liqe",
]

PUBLISHED = {
    # Reference values from the papers, KADID-10k SROCC / params.
    # These are context, not the numbers measured by this script.
    "brisque": {"kadid_srocc": 0.528, "params": 0},
    "niqe": {"kadid_srocc": 0.374, "params": 0},
    "musiq": {"kadid_srocc": 0.875, "params": 27_000_000},
    "clipiqa": {"kadid_srocc": 0.916, "params": 150_000_000},
    "maniqa": {"kadid_srocc": 0.946, "params": 24_000_000},
    "liqe": {"kadid_srocc": 0.930, "params": 151_000_000},
    "arniqa": {"kadid_srocc": 0.908, "params": 24_000_000},
    "hyperiqa": {"kadid_srocc": 0.852, "params": 27_000_000},
    "dbcnn": {"kadid_srocc": 0.851, "params": 24_000_000},
    "tres": {"kadid_srocc": 0.859, "params": 152_000_000},
    "topiq_nr": {"kadid_srocc": None, "params": 27_000_000},
}


def _clear():
    torch.cuda.empty_cache()


@torch.no_grad()
def measure_latency(metric: torch.nn.Module, device: torch.device, size: int = 224,
                    iters: int = 30) -> float:
    """Median single-image latency in milliseconds (GPU, batch=1)."""
    if device.type != "cuda":
        return float("nan")
    x = torch.rand(1, 3, size, size, device=device)
    for _ in range(5):
        metric(x)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        metric(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


@torch.no_grad()
def score_dataframe(
    metric: torch.nn.Module,
    df: pd.DataFrame,
    transform: Any,
    device: torch.device,
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    loader = make_loader(
        df, transform, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, drop_last=False, seed=seed,
    )
    rows: List[Dict[str, Any]] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        scores = metric(images)
        scores = scores.detach().float().cpu().view(-1).numpy()
        for i in range(len(scores)):
            rows.append(
                {
                    "name": batch["name"][i],
                    "ref": batch["ref"][i],
                    "dataset": batch["dataset"][i],
                    "mos": float(batch["mos"][i]),
                    "score": float(scores[i]),
                    "dist_type": int(batch["dist_type"][i]),
                    "level": int(batch["level"][i]),
                }
            )
    return pd.DataFrame(rows)


def evaluate_metric(
    name: str,
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    device: torch.device,
    split_name: str,
    out_dir: str,
) -> Optional[Dict[str, Any]]:
    import pyiqa

    print(f"\n=== {name} on {split_name} ({len(df)} images) ===")
    try:
        metric = pyiqa.create_metric(name, device=device)
        n_params = sum(p.numel() for p in metric.parameters())
        lower_better = bool(getattr(metric, "lower_better", False))
        size = int(cfg["data"]["size"])
        transform = build_transform(size, train=False, norm="none", eval_mode=cfg["data"]["eval_mode"])
        t0 = time.time()
        preds = score_dataframe(
            metric, df, transform, device,
            batch_size=int(cfg["eval"].get("bench_batch_size", 8)),
            num_workers=int(cfg["data"].get("num_workers", 4)),
            seed=int(cfg["experiment"]["seed"]),
        )
        elapsed = time.time() - t0
        # Orient so that higher is better before correlations
        oriented = -preds["score"].to_numpy() if lower_better else preds["score"].to_numpy()
        metrics = compute_metrics(oriented, preds["mos"].to_numpy())
        latency = measure_latency(metric, device, size=size)
        preds["oriented"] = oriented
        preds["metric"] = name
        preds["split"] = split_name
        preds.to_csv(os.path.join(out_dir, f"{name}_{split_name}_predictions.csv"), index=False)
        row = {
            "metric": name,
            "split": split_name,
            "n": int(len(preds)),
            "params": int(n_params),
            "latency_ms": latency,
            "srocc": metrics["srocc"],
            "plcc": metrics["plcc"],
            "rmse": metrics["rmse"],
            "krcc": metrics["krcc"],
            "lower_better": lower_better,
            "eval_seconds": round(elapsed, 2),
        }
        print(f"  SROCC={metrics['srocc']:.4f} PLCC={metrics['plcc']:.4f} "
              f"params={n_params:,} latency={latency:.2f}ms")
        del metric
        _clear()
        return row
    except Exception as exc:
        print(f"  FAILED: {exc!r}")
        return {"metric": name, "split": split_name, "error": repr(exc)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--metrics", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out-dir", default="results/benchmarks")
    ap.add_argument("--datasets", nargs="*", default=["kadid_test", "tid2013"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-mode", default=None, choices=[None, "center", "resize", "full"])
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg["eval"]["bench_batch_size"] = args.batch_size
    if args.eval_mode is not None:
        cfg["data"]["eval_mode"] = args.eval_mode
    device = resolve_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    metrics = args.metrics or (DEFAULT_METRICS if args.all else DEFAULT_METRICS)

    # Build the canonical KADID content split
    kadid = load_dataframe("kadid10k", cfg["data"]["root"])
    _, _, test_df = split_dataframe(
        kadid, mode=cfg["data"]["split"], test_frac=cfg["data"]["test_frac"],
        val_frac=cfg["data"]["val_frac"], seed=cfg["experiment"]["seed"],
        fold=cfg["data"]["fold"], kfold=cfg["data"]["kfold"],
    )
    frames: Dict[str, pd.DataFrame] = {}
    if "kadid_test" in args.datasets:
        frames["kadid_test"] = test_df
    if "tid2013" in args.datasets:
        frames["tid2013"] = load_dataframe("tid2013", cfg["data"]["tid_root"])

    rows: List[Dict[str, Any]] = []
    for name in metrics:
        for split_name, df in frames.items():
            row = evaluate_metric(name, df, cfg, device, split_name, args.out_dir)
            if row is not None:
                rows.append(row)
                pd.DataFrame(rows).to_csv(
                    os.path.join(args.out_dir, "pyiqa_summary.csv"), index=False
                )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        print("\n" + summary.to_string(index=False))
    with open(os.path.join(args.out_dir, "pyiqa_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
