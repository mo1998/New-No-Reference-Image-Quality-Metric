"""Collect every completed run into one comparison table.

The output is intentionally simple: one row per run with the metrics needed for
the paper/report tables.  It reads only ``summary.json`` files, so runs that
crashed or are still training are ignored.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List

import pandas as pd


def _get(d: Dict[str, Any], *path, default=None):
    for key in path:
        if not isinstance(d, dict):
            return default
        d = d.get(key)
    return d if d is not None else default


def collect(patterns: List[str]) -> pd.DataFrame:
    rows = []
    seen = set()
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
            except Exception:
                continue
            cfg_model = _get(d, "model", default=None)
            test = d.get("test", {}) or {}
            ci = d.get("test_ci", {}) or {}
            tid = d.get("cross_tid2013", {}) or {}
            row = {
                "experiment": d.get("experiment"),
                "model": _get(d, "model_name", default=None)
                or (cfg_model.get("name") if isinstance(cfg_model, dict) else None),
                "params": d.get("params"),
                "best_epoch": d.get("best_epoch"),
                "test_srocc": test.get("srocc"),
                "test_plcc": test.get("plcc"),
                "test_rmse": test.get("rmse"),
                "test_ci_low": _get(ci, "srocc", "low"),
                "test_ci_high": _get(ci, "srocc", "high"),
                "tid_srocc": tid.get("srocc"),
                "tid_plcc": tid.get("plcc"),
                "train_images": _get(d, "split", "train_images"),
                "train_refs": _get(d, "split", "train_refs"),
                "run_dir": os.path.dirname(path),
                "summary_path": path,
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["experiment"])
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patterns", nargs="+", default=[
        "runs/*/summary.json",
        "runs_cv65/*/summary.json",
        "runs_cv65_384/*/summary.json",
        "runs_ablate/*/summary.json",
        "runs_cv/*/summary.json",
    ])
    ap.add_argument("--out", default="results/all_runs.csv")
    args = ap.parse_args(argv)
    df = collect(args.patterns)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.drop(columns=["summary_path"], errors="ignore").to_string(index=False))
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
