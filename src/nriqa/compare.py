"""Compare two or more prediction files with content-level statistics.

Prediction CSVs are produced by ``nriqa.run`` (``predictions_test.csv``) and
contain at least ``name``, ``ref``, ``mos`` and ``pred``.  The script aligns
rows by image name, reports point metrics, and runs paired content-level
bootstrap tests against a chosen reference method.

Usage::

    PYTHONPATH=src .venv/bin/python -m nriqa.compare \
        --pred runs/A/predictions_test.csv runs/B/predictions_test.csv \
        --labels baseline proposed --out results/compare.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from .metrics import bootstrap_ci, compute_metrics, paired_bootstrap_test


def load_prediction(path: str, label: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"name", "ref", "mos", "pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df.copy()
    df["method"] = label or os.path.basename(os.path.dirname(path))
    return df[["name", "ref", "mos", "pred", "method"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--reference", default=None, help="Label of the reference method")
    ap.add_argument("--out", default="results/compare.json")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    labels = args.labels or [None] * len(args.pred)
    if len(labels) != len(args.pred):
        raise SystemExit("--labels must match --pred in length")
    frames = [load_prediction(p, l) for p, l in zip(args.pred, labels)]
    common = set(frames[0]["name"])
    for f in frames[1:]:
        common &= set(f["name"])
    common = sorted(common)
    if not common:
        raise SystemExit("No overlapping image names across prediction files")
    aligned = [f.set_index("name").loc[common].reset_index() for f in frames]

    out: Dict[str, object] = {"n_common": len(common), "methods": {}}
    for f in aligned:
        label = str(f["method"].iloc[0])
        pred = f["pred"].to_numpy(dtype=float)
        mos = f["mos"].to_numpy(dtype=float)
        refs = f["ref"].astype(str).to_numpy()
        metrics = compute_metrics(pred, mos)
        ci = bootstrap_ci(pred, mos, refs, n_boot=args.bootstrap, seed=args.seed)
        out["methods"][label] = {"metrics": metrics, "ci": ci}

    ref_label = args.reference or str(aligned[0]["method"].iloc[0])
    ref_idx = [str(f["method"].iloc[0]) for f in aligned].index(ref_label)
    ref = aligned[ref_idx]
    out["reference"] = ref_label
    out["tests"] = {}
    for i, f in enumerate(aligned):
        label = str(f["method"].iloc[0])
        if i == ref_idx:
            continue
        # Align to the reference frame by name
        merged = ref.merge(f, on="name", suffixes=("_ref", "_cmp"))
        tests = {}
        for metric in ("srocc", "plcc"):
            tests[metric] = paired_bootstrap_test(
                merged["pred_cmp"].to_numpy(),
                merged["pred_ref"].to_numpy(),
                merged["mos_cmp"].to_numpy(),
                merged["ref_ref"].astype(str).to_numpy(),
                metric=metric,
                n_boot=args.bootstrap,
                seed=args.seed,
            )
        out["tests"][label] = tests

    print(json.dumps(out, indent=2))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
