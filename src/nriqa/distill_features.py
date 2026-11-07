"""Extract per-fold pooled teacher features for feature distillation.

The student sees the teacher's pooled representation for each training image
and is trained with a cosine feature-matching loss in addition to score
distillation.  Features are stored once per fold, so student training incurs
no teacher forward pass.

Usage::

    PYTHONPATH=src .venv/bin/python -m nriqa.distill_features \
        --teacher-glob 'runs_cv65_l16_384/*/last.pt' \
        --out teacher_feats --size 384 --device cuda:1
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

from .config import resolve_device
from .data import build_transform, load_dataframe, make_loader, split_dataframe
from .evaluate import load_model_from_checkpoint


@torch.no_grad()
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher-glob", required=True)
    ap.add_argument("--out", default="teacher_feats")
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    device = resolve_device(args.device)
    checkpoints = sorted(glob.glob(args.teacher_glob))
    if not checkpoints:
        raise SystemExit(f"No checkpoints matched {args.teacher_glob}")

    for ckpt_path in checkpoints:
        model, ckpt = load_model_from_checkpoint(ckpt_path, device)
        cfg = ckpt["cfg"]
        fold = int(cfg["data"].get("fold", 0))
        size = args.size or int(cfg["data"]["size"])
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
        transform = build_transform(
            size, train=False, norm=cfg["data"]["norm"], eval_mode="center"
        )
        loader = make_loader(
            train_df, transform, batch_size=args.batch_size,
            num_workers=int(cfg["data"].get("num_workers", 8)),
            shuffle=False, seed=int(cfg["experiment"]["seed"]),
        )
        feats, names = [], []
        print(f"fold {fold}: {len(train_df)} images, teacher={ckpt_path}")
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            f = model.forward_features(x)
            feats.append(f.float().cpu().numpy().astype(np.float16))
            names.extend(list(batch["name"]))
        features = np.concatenate(feats, axis=0)
        path = os.path.join(args.out, f"fold{fold}.npz")
        np.savez_compressed(path, names=np.asarray(names), features=features)
        print(f"  saved {path} shape={features.shape} dtype={features.dtype}")
        del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
