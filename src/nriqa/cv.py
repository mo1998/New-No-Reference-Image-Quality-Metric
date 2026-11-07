"""Run k-fold content cross-validation and aggregate the results.

Each fold is an independent training run with a disjoint 20% test fold and a
content-held-out validation set.  The script launches runs sequentially (so
GPU memory is not oversubscribed), then aggregates test/cross-dataset metrics
with mean, standard deviation and per-fold values.

Usage::

    PYTHONPATH=src .venv/bin/python -m nriqa.cv \
        --config configs/siglip_b16_kadid.yaml --k 5 --name siglip_cv
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def run_fold(
    cfg_path: str,
    fold: int,
    k: int,
    name: str,
    output_root: str,
    device: str,
    epochs: int | None,
    python: str,
) -> Dict[str, Any]:
    run_name = f"{name}_f{fold}"
    cmd = [
        python, "-m", "nriqa.run",
        "--config", cfg_path,
        "--name", run_name,
        "--output-dir", output_root,
        "--fold", str(fold),
        "--device", device,
        "--set", "data.split=kfold",
        "--set", f"data.kfold={k}",
        "--set", "eval.bootstrap=0",
    ]
    if epochs is not None:
        cmd += ["--set", f"train.epochs={epochs}"]
    env = dict(os.environ)
    env["PYTHONPATH"] = "src" + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    print("running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, env=env, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"fold {fold} failed with exit code {proc.returncode}")
    # Find the run directory created for this fold
    candidates = []
    for d in os.listdir(output_root):
        p = os.path.join(output_root, d, "summary.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as fh:
                s = json.load(fh)
            if s.get("experiment") == run_name:
                candidates.append((os.path.getmtime(p), p, s))
    if not candidates:
        raise RuntimeError(f"no summary found for fold {fold}")
    _, path, summary = sorted(candidates)[-1]
    summary["summary_path"] = path
    return summary


def aggregate(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    def collect(key: str) -> List[float]:
        vals = []
        for s in summaries:
            if key == "test":
                v = s.get("test", {}).get("srocc")
            elif key == "test_plcc":
                v = s.get("test", {}).get("plcc")
            elif key == "tid":
                v = s.get("cross_tid2013", {}).get("srocc")
            elif key == "tid_plcc":
                v = s.get("cross_tid2013", {}).get("plcc")
            else:
                v = None
            if v is not None and np.isfinite(v):
                vals.append(float(v))
        return vals

    out: Dict[str, Any] = {"n_folds": len(summaries), "folds": []}
    for s in summaries:
        out["folds"].append(
            {
                "best_epoch": s.get("best_epoch"),
                "test_srocc": s.get("test", {}).get("srocc"),
                "test_plcc": s.get("test", {}).get("plcc"),
                "tid_srocc": s.get("cross_tid2013", {}).get("srocc"),
                "params": s.get("params"),
                "summary": s.get("summary_path"),
            }
        )
    for key in ("test", "test_plcc", "tid", "tid_plcc"):
        vals = collect(key)
        if vals:
            out[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "values": [round(v, 6) for v in vals],
            }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--name", default=None)
    ap.add_argument("--output-root", default="runs_cv")
    ap.add_argument("--results-dir", default="results/cv")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    args = ap.parse_args(argv)

    name = args.name or os.path.splitext(os.path.basename(args.config))[0]
    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    folds = args.folds if args.folds is not None else list(range(args.k))
    summaries = []
    for fold in folds:
        summaries.append(
            run_fold(args.config, fold, args.k, name, args.output_root, args.device, args.epochs, args.python)
        )
    agg = aggregate(summaries)
    agg["name"] = name
    agg["config"] = args.config
    out_path = os.path.join(args.results_dir, f"{name}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(agg, fh, indent=2)
    rows = []
    for f in agg["folds"]:
        rows.append(f)
    pd.DataFrame(rows).to_csv(out_path.replace(".json", ".csv"), index=False)
    print(json.dumps({k: v for k, v in agg.items() if k != "folds"}, indent=2))
    print("saved", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
