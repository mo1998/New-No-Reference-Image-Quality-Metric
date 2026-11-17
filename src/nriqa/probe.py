"""Linear / MLP probing of frozen features for NR-IQA.

This is both a diagnostic (how much quality information is linearly available
in a backbone?) and a strong, extremely cheap baseline.  It uses the exact
content split from the main protocol and never fits on the test split.

Usage::

    PYTHONPATH=src .venv/bin/python -m nriqa.probe --features features/clip_b16
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge

from .metrics import bootstrap_ci, compute_metrics


def load_features(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def standardize(train: np.ndarray, *others: np.ndarray) -> Tuple[np.ndarray, ...]:
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    out = [(train - mu) / sd]
    out.extend([(x - mu) / sd for x in others])
    return tuple(out)


def orient(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-9 and np.std(b) > 1e-9 else 0.0


def ridge_probe(xtr, ytr, xva, yva, xte, yte, xtid, ytid):
    best = None
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        model = Ridge(alpha=alpha)
        model.fit(xtr, ytr)
        pv = model.predict(xva)
        s = compute_metrics(pv, yva)["srocc"]
        if best is None or (np.isfinite(s) and s > best[0]):
            best = (s, alpha, model)
    _, alpha, model = best
    return {
        "alpha": alpha,
        "test": compute_metrics(model.predict(xte), yte),
        "tid": compute_metrics(model.predict(xtid), ytid),
        "val": compute_metrics(model.predict(xva), yva),
    }


class MLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def mlp_probe(xtr, ytr, xva, yva, xte, yte, xtid, ytid, device="cuda:1",
              epochs: int = 120, seed: int = 42):
    torch.manual_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    xtr_t = torch.tensor(xtr, dtype=torch.float32, device=dev)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=dev)
    xva_t = torch.tensor(xva, dtype=torch.float32, device=dev)
    model = MLP(xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = (-1.0, None, -1)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr_t), device=dev)
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            pred = model(xtr_t[idx])
            loss = F.mse_loss(pred, ytr_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(xva_t).cpu().numpy()
        s = compute_metrics(pv, yva)["srocc"]
        if s > best[0]:
            best = (s, {k: v.clone() for k, v in model.state_dict().items()}, ep)
    model.load_state_dict(best[1])
    model.eval()
    with torch.no_grad():
        pte = model(torch.tensor(xte, dtype=torch.float32, device=dev)).cpu().numpy()
        ptid = model(torch.tensor(xtid, dtype=torch.float32, device=dev)).cpu().numpy()
    with torch.no_grad():
        pva = model(torch.tensor(xva, dtype=torch.float32, device=dev)).cpu().numpy()
    return {
        "best_epoch": best[2],
        "test": compute_metrics(pte, yte),
        "tid": compute_metrics(ptid, ytid),
        "val": compute_metrics(pva, yva),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--bootstrap", type=int, default=500)
    args = ap.parse_args(argv)

    train = load_features(os.path.join(args.features, "kadid_train.npz"))
    val = load_features(os.path.join(args.features, "kadid_val.npz"))
    test = load_features(os.path.join(args.features, "kadid_test.npz"))
    tid = load_features(os.path.join(args.features, "tid2013.npz"))

    xtr, xva, xte, xtid = standardize(
        train["features"], val["features"], test["features"], tid["features"]
    )
    ytr, yva, yte, ytid = train["mos"], val["mos"], test["mos"], tid["mos"]

    results: Dict[str, Any] = {"features": args.features}
    results["ridge"] = ridge_probe(xtr, ytr, xva, yva, xte, yte, xtid, ytid)
    results["mlp"] = mlp_probe(xtr, ytr, xva, yva, xte, yte, xtid, ytid, device=args.device)

    # Content-level CI for the best probe (MLP)
    probe = results["mlp"]
    preds = []
    # Recompute MLP predictions for the CI rather than storing them twice
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # The MLP is small; refit quickly with the same protocol is avoided by
    # trusting the point estimate here.  CI is reported by the main trainer.

    print(json.dumps({k: v for k, v in results.items() if k != "features"}, indent=2))
    out = args.out or os.path.join(args.features, "probe_results.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
