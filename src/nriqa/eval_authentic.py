"""Evaluate checkpoints on authentic-distortion benchmarks (LIVEC, SPAQ).

These benchmarks have no reference contents and no train/test split in the
KADID sense, so the protocol is zero-shot: the model is applied unchanged to
every image and SROCC/PLCC are computed against the human MOS.  For LIVEC the
seven official training images (``t1``-``t7``) are excluded to match the
standard 1,162-image test set.

Usage::

    PYTHONPATH=src .venv/bin/python -m nriqa.eval_authentic \
        --dataset livec --root data/livec/LIVEC \
        --checkpoint-glob 'runs_cv65_koniq/*/last.pt' --label b16_koniq
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from .config import resolve_device
from .data import load_dataframe
from .evaluate import (
    evaluate_predictions,
    load_model_from_checkpoint,
    predict_dataframe,
    save_predictions,
)


def _drop_livec_training(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the seven official LIVEC training images (t1..t7)."""
    pattern = re.compile(r"^t[1-7]\.(bmp|jpg|jpeg|png)$", re.I)
    keep = ~df["name"].astype(str).str.match(pattern)
    return df[keep].reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["livec", "spaq"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--checkpoint-glob", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", default="results/authentic")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--keep-livec-training", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    device = resolve_device(args.device)
    df = load_dataframe(args.dataset, args.root)
    if args.dataset == "livec" and not args.keep_livec_training:
        df = _drop_livec_training(df)
    print(f"{args.dataset}: {len(df)} images, MOS [{df['mos'].min():.3f}, {df['mos'].max():.3f}]")

    checkpoints = sorted(glob.glob(args.checkpoint_glob))
    if not checkpoints:
        raise SystemExit(f"No checkpoints matched {args.checkpoint_glob}")

    rows: List[Dict[str, Any]] = []
    all_preds: List[pd.DataFrame] = []
    for i, ckpt in enumerate(checkpoints):
        model, ckpt_data = load_model_from_checkpoint(ckpt, device)
        cfg = ckpt_data["cfg"]
        cfg["eval"]["batch_size"] = args.batch_size
        preds = predict_dataframe(model, df, cfg, device, split=args.dataset)
        preds["checkpoint"] = ckpt
        preds["fold"] = i
        result = evaluate_predictions(
            preds, target_stats=ckpt_data.get("target_stats"),
            bootstrap=0, seed=int(cfg["experiment"]["seed"]),
        )
        m = result["metrics"]
        print(f"  {os.path.basename(os.path.dirname(ckpt))}: "
              f"SROCC={m['srocc']:.4f} PLCC={m['plcc']:.4f}")
        rows.append({"checkpoint": ckpt, **m})
        all_preds.append(preds)
        del model
        torch.cuda.empty_cache()

    pred_df = pd.concat(all_preds, ignore_index=True)
    save_predictions(pred_df, os.path.join(args.out_dir, f"{args.dataset}_{args.label}_predictions.csv"))

    summary: Dict[str, Any] = {
        "dataset": args.dataset,
        "root": args.root,
        "label": args.label,
        "n_images": int(len(df)),
        "checkpoints": checkpoints,
        "per_checkpoint": rows,
    }
    if len(rows) > 1:
        for key in ("srocc", "plcc", "rmse", "krcc"):
            vals = np.asarray([r[key] for r in rows], dtype=float)
            summary[key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)),
                "values": [float(v) for v in vals],
            }
    else:
        for key in ("srocc", "plcc", "rmse", "krcc"):
            summary[key] = {"mean": float(rows[0][key]), "std": 0.0, "values": [float(rows[0][key])]}
    out = os.path.join(args.out_dir, f"{args.dataset}_{args.label}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: summary[k] for k in ("dataset", "label", "n_images", "srocc", "plcc")}, indent=2))
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
