"""Run a named ablation grid and aggregate results.

A variant is a dict of ``run.py --set`` overrides.  Each variant is launched as
an independent process so a failure cannot poison the rest of the grid.

Example::

    PYTHONPATH=src .venv/bin/python -m nriqa.ablate \
        --config configs/siglip_b16_aux_kadid.yaml \
        --variants configs/ablations/siglip.json \
        --out results/ablations/siglip.csv
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List

import pandas as pd

from .config import load_config
from .tracking import aggregate_summaries


def build_cmd(cfg_path: str, name: str, output_dir: str, overrides: List[str],
              python: str, device: str) -> List[str]:
    cmd = [python, "-m", "nriqa.run", "--config", cfg_path, "--name", name,
           "--output-dir", output_dir, "--device", device]
    for item in overrides:
        cmd += ["--set", item]
    return cmd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--variants", required=True, help="JSON list of {name, overrides}")
    ap.add_argument("--out", default="results/ablations/summary.csv")
    ap.add_argument("--output-dir", default="runs_ablate")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    with open(args.variants, "r", encoding="utf-8") as fh:
        variants = json.load(fh)
    if isinstance(variants, dict):
        variants = variants.get("variants", [])
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for variant in variants:
        name = variant["name"]
        overrides = [f"{k}={v}" for k, v in (variant.get("overrides") or {}).items()]
        cmd = build_cmd(args.config, name, args.output_dir, overrides, args.python, args.device)
        print("\n=== variant:", name, "===")
        print(" ".join(cmd))
        if args.dry_run:
            continue
        env = dict(os.environ)
        env["PYTHONPATH"] = "src" + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        proc = subprocess.run(cmd, env=env, text=True)
        if proc.returncode != 0:
            rows.append({"variant": name, "error": f"exit {proc.returncode}"})
            continue
        # Locate the most recent summary for this variant
        summaries = aggregate_summaries([os.path.join(args.output_dir, "*")])
        matches = [s for s in summaries if s.get("experiment") == name]
        if not matches:
            rows.append({"variant": name, "error": "no summary"})
            continue
        s = sorted(matches, key=lambda x: x.get("wall_time_s", 0))[-1]
        rows.append(
            {
                "variant": name,
                "params": s.get("params"),
                "best_epoch": s.get("best_epoch"),
                "test_srocc": s.get("test", {}).get("srocc"),
                "test_plcc": s.get("test", {}).get("plcc"),
                "tid_srocc": s.get("cross_tid2013", {}).get("srocc"),
                "tid_plcc": s.get("cross_tid2013", {}).get("plcc"),
                "run_dir": s.get("run_dir"),
            }
        )
        pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    print("\n", df.to_string(index=False))
    df.to_csv(args.out, index=False)
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
