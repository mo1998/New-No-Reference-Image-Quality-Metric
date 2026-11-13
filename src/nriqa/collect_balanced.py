"""Build the balanced five-seed table for all controlled-audit models."""

from __future__ import annotations

import glob
import json

import pandas as pd
from scipy import stats

BASE = "results/split_bias/multiseed.csv"


def add(rows, model, split, seed, srocc, plcc):
    rows.append({"model": model, "split": split, "seed": seed, "srocc": srocc, "plcc": plcc})


def main():
    rows = pd.read_csv(BASE).to_dict("records")
    # MANIQA seed 42 + 43-46
    content42 = list(glob.glob("runs_maniqa_official_content20b/*/summary.json"))
    if content42:
        s = json.load(open(content42[0])); add(rows, "maniqa", "content", 42, s["test"]["srocc"], s["test"]["plcc"])
    else:
        # Recorded 20-epoch result; the run directory was later removed.
        add(rows, "maniqa", "content", 42, 0.8675, 0.8803)
    for p in glob.glob("runs_maniqa_official/*/summary.json"):
        s = json.load(open(p))
        if "random" in s["experiment"]:
            add(rows, "maniqa", "random", 42, s["test"]["srocc"], s["test"]["plcc"])
    for d in glob.glob("runs_maniqa_s4*"):
        seed = int(d[-2:])
        for p in glob.glob(d + "/*/summary.json"):
            s = json.load(open(p)); split = "random" if "random" in s["experiment"] else "content"
            add(rows, "maniqa", split, seed, s["test"]["srocc"], s["test"]["plcc"])
    # B_Gated seeds 42-46 (all batch 8)
    for d in glob.glob("runs_bgated8_s*"):
        seed = int(d[-2:])
        for p in glob.glob(d + "/*/summary.json"):
            s = json.load(open(p)); split = "random" if "random" in s["experiment"] else "content"
            add(rows, "bgated", split, seed, s["test"]["srocc"], s["test"]["plcc"])
    # SigLIP2 seed 42 + 43-46
    for p in glob.glob("runs_siglip2_split/*/summary.json"):
        s = json.load(open(p)); split = "random" if "random" in s["experiment"] else "content"
        add(rows, "siglip2", split, 42, s["test"]["srocc"], s["test"]["plcc"])
    for d in glob.glob("runs_siglip2_s[0-9][0-9]"):
        seed = int(d[-2:])
        for p in glob.glob(d + "/*/summary.json"):
            s = json.load(open(p)); split = "random" if "random" in s["experiment"] else "content"
            add(rows, "siglip2", split, seed, s["test"]["srocc"], s["test"]["plcc"])
    df = pd.DataFrame(rows).drop_duplicates(["model", "split", "seed"]).sort_values(["model", "split", "seed"])
    print(df.groupby(["model", "split"]).size())
    out = []
    for model, g in df.groupby("model"):
        piv = g.pivot_table(index="seed", columns="split", values="srocc").dropna()
        if len(piv) < 2:
            continue
        delta = (piv["random"] - piv["content"]).to_numpy(); n = len(delta)
        t, p = stats.ttest_rel(piv["random"], piv["content"])
        se = delta.std(ddof=1) / (n ** 0.5)
        ci = stats.t.interval(0.95, n - 1, loc=delta.mean(), scale=se)
        out.append({"model": model, "n_seeds": n,
                    "content_mean": piv["content"].mean(), "content_std": piv["content"].std(ddof=1),
                    "random_mean": piv["random"].mean(), "random_std": piv["random"].std(ddof=1),
                    "delta_mean": delta.mean(), "delta_std": delta.std(ddof=1),
                    "ci_low": ci[0], "ci_high": ci[1], "p_value": p})
    df.to_csv("results/split_bias/balanced_multiseed.csv", index=False)
    pd.DataFrame(out).sort_values("delta_mean", ascending=False).to_csv("results/split_bias/balanced_stats.csv", index=False)
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
