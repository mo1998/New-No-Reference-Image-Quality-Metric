"""Generate per-fold soft targets from a cross-validated teacher.

The teacher checkpoints produced by ``nriqa.cv`` are already trained on the
same content folds as the student, so their predictions on the training split
are legitimate soft targets (no test content is used).  The output is one CSV
per fold with ``name,soft`` columns, consumed by the student config through
``data.soft_csv: teacher_preds/fold{fold}.csv``.

Usage::

    PYTHONPATH=src .venv/bin/python -m nriqa.distill \
        --teacher-glob 'runs_cv65_l16_384/*/last.pt' \
        --out teacher_preds --device cuda:1
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Any, Dict

import pandas as pd
import torch

from .config import resolve_device
from .data import load_dataframe, split_dataframe
from .evaluate import denormalize, load_model_from_checkpoint, predict_dataframe


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher-glob", required=True)
    ap.add_argument("--out", default="teacher_preds")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    device = resolve_device(args.device)
    checkpoints = sorted(glob.glob(args.teacher_glob))
    if not checkpoints:
        raise SystemExit(f"No checkpoints matched {args.teacher_glob}")

    for ckpt_path in checkpoints:
        model, ckpt = load_model_from_checkpoint(ckpt_path, device)
        cfg: Dict[str, Any] = ckpt["cfg"]
        fold = int(cfg["data"].get("fold", 0))
        df = load_dataframe(cfg["data"]["dataset"], cfg["data"]["root"])
        train_df, _, _ = split_dataframe(
            df,
            mode=cfg["data"]["split"],
            test_frac=cfg["data"]["test_frac"],
            val_frac=cfg["data"]["val_frac"],
            seed=cfg["experiment"]["seed"],
            fold=fold,
            kfold=cfg["data"]["kfold"],
        )
        cfg["eval"]["tta_mode"] = "none"
        cfg["eval"]["bootstrap"] = 0
        cfg["eval"]["batch_size"] = args.batch_size
        print(f"fold {fold}: teacher={ckpt_path} train={len(train_df)}")
        preds = predict_dataframe(model, train_df, cfg, device, split="train")
        soft = denormalize(preds["pred"].to_numpy(), ckpt.get("target_stats"))
        out = pd.DataFrame({"name": preds["name"], "soft": soft, "teacher_mos": preds["mos"]})
        path = os.path.join(args.out, f"fold{fold}.csv")
        out.to_csv(path, index=False)
        print(f"  saved {path}")
        del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
