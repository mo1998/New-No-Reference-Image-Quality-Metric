"""Frozen foundation-backbone feature extraction.

Compact pretrained encoders (CLIP ViT-B/16, SigLIP, DINOv2, ...) provide a
strong semantic prior that is complementary to handcrafted NSS statistics.
This module extracts and caches their penultimate features once, which makes
linear/MLP probing and feature-level distillation cheap and reproducible.

Usage::

    PYTHONPATH=src .venv/bin/python -m nriqa.foundation \
        --model vit_base_patch16_clip_224.openai --out features/clip_b16

The cached ``.npz`` contains features plus all evaluation metadata.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data import IQADataset, load_dataframe, split_dataframe


def build_feature_extractor(model_name: str, pretrained: bool = True,
                            device: torch.device = torch.device("cpu")):
    import timm

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    model.eval().to(device)
    data_cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)
    return model, transform, data_cfg


@torch.no_grad()
def extract_split(
    model: torch.nn.Module,
    df: pd.DataFrame,
    transform: Any,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 8,
) -> Dict[str, np.ndarray]:
    ds = IQADataset(df, transform=transform)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda", persistent_workers=num_workers > 0,
    )
    feats, mos, refs, names, dtypes, levels = [], [], [], [], [], []
    t0 = time.time()
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        f = model(x)
        if isinstance(f, (tuple, list)):
            f = f[0]
        feats.append(f.detach().float().cpu().numpy())
        mos.extend(batch["mos"].tolist())
        refs.extend(list(batch["ref"]))
        names.extend(list(batch["name"]))
        dtypes.extend(batch["dist_type"].tolist())
        levels.extend(batch["level"].tolist())
    print(f"    {len(df)} images in {time.time() - t0:.1f}s")
    return {
        "features": np.concatenate(feats, axis=0).astype(np.float32),
        "mos": np.asarray(mos, dtype=np.float32),
        "ref": np.asarray(refs),
        "name": np.asarray(names),
        "dist_type": np.asarray(dtypes, dtype=np.int16),
        "level": np.asarray(levels, dtype=np.int16),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="vit_base_patch16_clip_224.openai")
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out_dir = args.out or os.path.join("features", args.model.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"loading {args.model} on {device}")
    model, transform, data_cfg = build_feature_extractor(args.model, True, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,} data_config={data_cfg}")

    kadid = load_dataframe("kadid10k", cfg["data"]["root"])
    train_df, val_df, test_df = split_dataframe(
        kadid, mode=cfg["data"]["split"], test_frac=cfg["data"]["test_frac"],
        val_frac=cfg["data"]["val_frac"], seed=cfg["experiment"]["seed"],
        fold=cfg["data"]["fold"], kfold=cfg["data"]["kfold"],
    )
    for split_name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        print(f"  extracting {split_name} ({len(df)})")
        payload = extract_split(model, df, transform, device, args.batch_size,
                                int(cfg["data"]["num_workers"]))
        np.savez_compressed(os.path.join(out_dir, f"kadid_{split_name}.npz"), **payload)
    print("  extracting tid2013 (3000)")
    tid = load_dataframe("tid2013", cfg["data"]["tid_root"])
    payload = extract_split(model, tid, transform, device, args.batch_size,
                            int(cfg["data"]["num_workers"]))
    np.savez_compressed(os.path.join(out_dir, "tid2013.npz"), **payload)

    with open(os.path.join(out_dir, "info.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"model={args.model}\nparams={n_params}\ndata_config={data_cfg}\n")
    print("saved features to", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
