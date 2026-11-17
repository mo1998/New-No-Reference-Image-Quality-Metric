"""Lightweight random hyperparameter search.

No external HPO service is required: the search only needs a JSON space and a
base config.  Model selection uses validation SROCC (never test).  The test
metric is stored for diagnostics but is not used to rank trials.

Space format::

    {
      "train.lr":            {"type": "loguniform", "low": 1e-5, "high": 1e-3},
      "train.weight_decay":  {"type": "loguniform", "low": 1e-4, "high": 1e-1},
      "model.kwargs.freeze_blocks": {"type": "choice", "values": [4, 6, 8, 10]},
      "train.loss.rank":     {"type": "uniform", "low": 0.0, "high": 0.5}
    }

Usage::

    PYTHONPATH=src .venv/bin/python -m nriqa.hpo \
        --config configs/siglip_b16_aux_kadid.yaml \
        --space configs/hpo/siglip_space.json --trials 12 --epochs 12
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .tracking import aggregate_summaries


def sample_value(spec: Dict[str, Any], rng: random.Random) -> Any:
    kind = spec.get("type", "choice")
    if kind == "loguniform":
        lo, hi = float(spec["low"]), float(spec["high"])
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    if kind == "uniform":
        return float(rng.uniform(float(spec["low"]), float(spec["high"])))
    if kind == "int":
        return int(rng.randint(int(spec["low"]), int(spec["high"])))
    if kind == "choice":
        return rng.choice(spec["values"])
    if kind == "bool":
        return bool(rng.choice([True, False]))
    raise ValueError(f"Unknown space type: {kind}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--space", required=True)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--output-dir", default="runs_hpo")
    ap.add_argument("--out", default="results/hpo/trials.csv")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name-prefix", default="hpo")
    args = ap.parse_args(argv)

    with open(args.space, "r", encoding="utf-8") as fh:
        space = json.load(fh)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rng = random.Random(args.seed)
    rows: List[Dict[str, Any]] = []

    for trial in range(args.trials):
        params = {key: sample_value(spec, rng) for key, spec in space.items()}
        name = f"{args.name_prefix}_t{trial:03d}"
        # Do not test on the same val split as the final model: this is a
        # pilot HPO search, so use the standard split and report both.
        overrides = [f"{k}={v}" for k, v in params.items()]
        overrides += [f"train.epochs={args.epochs}", "eval.bootstrap=0"]
        cmd = [args.python, "-m", "nriqa.run", "--config", args.config,
               "--name", name, "--output-dir", args.output_dir, "--device", args.device]
        for item in overrides:
            cmd += ["--set", item]
        print("\n=== trial", trial, params, "===")
        env = dict(os.environ)
        env["PYTHONPATH"] = "src" + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        proc = subprocess.run(cmd, env=env, text=True)
        row = {"trial": trial, "name": name, **params}
        if proc.returncode != 0:
            row["error"] = f"exit {proc.returncode}"
        else:
            summaries = [s for s in aggregate_summaries([os.path.join(args.output_dir, "*")])
                         if s.get("experiment") == name]
            if summaries:
                s = summaries[-1]
                row.update(
                    {
                        "val_srocc": s.get("val", {}).get("srocc"),
                        "test_srocc": s.get("test", {}).get("srocc"),
                        "test_plcc": s.get("test", {}).get("plcc"),
                        "tid_srocc": s.get("cross_tid2013", {}).get("srocc"),
                        "best_epoch": s.get("best_epoch"),
                        "params": s.get("params"),
                        "run_dir": s.get("run_dir"),
                    }
                )
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(json.dumps(row, indent=2))

    df = pd.DataFrame(rows)
    if "val_srocc" in df:
        df = df.sort_values("val_srocc", ascending=False)
    df.to_csv(args.out, index=False)
    print("\n", df.to_string(index=False))
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
