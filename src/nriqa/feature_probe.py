"""Train a small head on frozen MANIQA / LIQE features under both splits.

This is the stable form of the split audit for architectures whose PyIQA
implementations are not safely end-to-end trainable.  The same features are
used for every split/seed; only the head is trained.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import resolve_device
from .data import load_dataframe, split_dataframe
from .metrics import compute_metrics


class Head(nn.Module):
    def __init__(self, dim: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _fit_head(Xtr, ytr, Xva, yva, Xte, yte, device, seed: int = 42) -> Dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    Xte = (Xte - mu) / sd
    ym, ys = ytr.mean(), ytr.std() + 1e-6
    tr = torch.utils.data.TensorDataset(
        torch.tensor(Xtr, dtype=torch.float32), torch.tensor((ytr - ym) / ys, dtype=torch.float32)
    )
    dl = torch.utils.data.DataLoader(tr, batch_size=128, shuffle=True, num_workers=0)
    head = Head(Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    best, best_state, patience = -1.0, None, 0
    for epoch in range(200):
        head.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(head(xb), yb)
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            pv = head(Xva_t).cpu().numpy() * ys + ym
        s = compute_metrics(pv, yva)["srocc"]
        if s > best + 1e-5:
            best, patience = s, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            patience += 1
            if patience >= 20:
                break
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        pt = head(Xte_t).cpu().numpy() * ys + ym
    m = compute_metrics(pt, yte)
    m["best_val_srocc"] = float(best)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="results/split_bias")
    ap.add_argument("--dataset", default="kadid10k")
    ap.add_argument("--root", default="data/kadid10k")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()

    device = resolve_device("cuda:0")
    d = np.load(args.features)
    X = d["features"].astype(np.float32)
    names = [str(n) for n in d["names"]]
    idx = {n: i for i, n in enumerate(names)}
    df = load_dataframe(args.dataset, args.root)
    df = df[df["name"].isin(idx)].reset_index(drop=True)
    print(f"{args.model}: {len(df)} rows, features {X.shape}")

    rows: List[Dict] = []
    for seed in args.seeds:
        for split_name in ("content", "random"):
            if split_name == "content":
                tr, va, te = split_dataframe(df, mode="holdout", test_frac=args.test_frac, val_frac=args.val_frac, seed=seed)
            else:
                tr, va, te = split_dataframe(df, mode="random", test_frac=args.test_frac, val_frac=args.val_frac, seed=seed)
            def mat(part):
                return np.stack([X[idx[n]] for n in part["name"]])
            m = _fit_head(
                mat(tr), tr["mos"].to_numpy(np.float32),
                mat(va), va["mos"].to_numpy(np.float32),
                mat(te), te["mos"].to_numpy(np.float32),
                device, seed,
            )
            row = {"model": args.model, "split": split_name, "seed": seed, **m}
            rows.append(row)
            print(f"{args.model} {split_name} seed={seed} SROCC={m['srocc']:.4f} PLCC={m['plcc']:.4f}")
            pd.DataFrame(rows).to_csv(os.path.join(args.out, f"probe_{args.model}_{args.dataset}.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
