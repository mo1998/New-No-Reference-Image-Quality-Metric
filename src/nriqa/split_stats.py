"""Paired statistics for the random-vs-content split audit."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/split_bias/multiseed.csv")
    ap.add_argument("--out", default="results/split_bias/multiseed_stats.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    rows = []
    for model, g in df.groupby("model"):
        piv = g.pivot_table(index="seed", columns="split", values="srocc").dropna()
        if len(piv) < 2:
            continue
        delta = (piv["random"] - piv["content"]).to_numpy()
        n = len(delta)
        t, p = stats.ttest_rel(piv["random"], piv["content"])
        se = delta.std(ddof=1) / np.sqrt(n)
        ci = stats.t.interval(0.95, n - 1, loc=delta.mean(), scale=se) if n > 1 else (np.nan, np.nan)
        rows.append({
            "model": model, "n_seeds": n,
            "content_mean": piv["content"].mean(), "content_std": piv["content"].std(ddof=1),
            "random_mean": piv["random"].mean(), "random_std": piv["random"].std(ddof=1),
            "delta_mean": delta.mean(), "delta_std": delta.std(ddof=1),
            "delta_ci95_low": ci[0], "delta_ci95_high": ci[1],
            "paired_t": t, "p_value": p,
        })
    out = pd.DataFrame(rows).sort_values("delta_mean", ascending=False)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
