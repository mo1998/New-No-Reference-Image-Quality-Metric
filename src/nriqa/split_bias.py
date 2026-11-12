"""Zero-shot control for the random-vs-content split experiment."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from .config import load_config, resolve_device
from .data import build_transform, load_dataframe, make_loader, split_dataframe
from .metrics import compute_metrics


def get_splits(root: str = "data/kadid10k", seed: int = 42):
    df = load_dataframe("kadid10k", root)
    _, _, content_test = split_dataframe(
        df, mode="holdout", test_frac=0.2, val_frac=0.1, seed=seed
    )
    _, _, random_test = split_dataframe(
        df, mode="random", test_frac=2000 / 10125, val_frac=1000 / 10125, seed=seed
    )
    return content_test, random_test


@torch.no_grad()
def score_metric(name: str, df: pd.DataFrame, device: torch.device, batch_size: int = 8) -> pd.DataFrame:
    import pyiqa

    metric = pyiqa.create_metric(name, device=device)
    transform = build_transform(224, train=False, norm="none", eval_mode="center")
    loader = make_loader(df, transform, batch_size=batch_size, num_workers=8, shuffle=False, seed=42)
    rows = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        s = metric(x).detach().float().cpu().view(-1).numpy()
        for i in range(len(s)):
            rows.append(
                {
                    "name": batch["name"][i],
                    "mos": float(batch["mos"][i]),
                    "score": float(s[i]),
                }
            )
    del metric
    torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def main() -> int:
    device = resolve_device("cuda:0")
    content, random = get_splits()
    print(f"content test={len(content)} random test={len(random)}")
    os.makedirs("results/split_bias", exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for name in ["maniqa", "topiq_nr", "clipiqa", "arniqa"]:
        for split_name, df in (("content", content), ("random", random)):
            preds = score_metric(name, df, device)
            m = compute_metrics(preds["score"], preds["mos"])
            preds.to_csv(f"results/split_bias/zero_{name}_{split_name}.csv", index=False)
            print(f"{name:10s} {split_name:7s} SROCC={m['srocc']:.4f} PLCC={m['plcc']:.4f}")
            rows.append({"model": name, "split": split_name, **m, "n": len(preds)})
            pd.DataFrame(rows).to_csv("results/split_bias/zero_shot.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
