"""Build the controlled KADID comparison table from the five-seed runs.

Inputs are produced by ``collect_balanced.py``:

* ``results/split_bias/balanced_stats.csv``     - per-model mean/std/delta/p
* ``results/split_bias/balanced_multiseed.csv`` - per-seed SROCC/PLCC

The parameter counts are recorded from each run's training log (``params`` and
``params_trainable`` fields) because several baselines are loaded through
third-party packages that do not expose a stable parameter-count helper.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats

STATS = "results/split_bias/balanced_stats.csv"
SEEDS = "results/split_bias/balanced_multiseed.csv"
OUT = "results/split_bias/controlled_comparison.csv"

# params_m: total backbone+head parameters in millions (from the run log).
# trainable_m: parameters updated during training (LoRA adapters count as
# trainable, the frozen backbone does not).
REGISTRY = {
    "ours": {
        "label": "ours SigLIP-B/16 + MSOS/WaveQ fusion",
        "params_m": 93.4,
        "trainable_m": 93.4,
        "note": "5 seeds; 5-fold content CV 0.9481",
    },
    "siglip2": {
        "label": "SigLIP2-L/16-384 + linear head",
        "params_m": 316.6,
        "trainable_m": 316.6,
        "note": "5 seeds; same splits as ours",
    },
    "bgated_lora": {
        "label": "B_Gated + LoRA r16 (paper recipe)",
        "params_m": 319.7,
        "trainable_m": 3.41,
        "note": "5 seeds; LoRA on q/k/v/out, gated head",
    },
    "bgated": {
        "label": "B_Gated + gated head (no LoRA)",
        "params_m": 316.5,
        "trainable_m": 316.5,
        "note": "5 seeds; earlier approximation without adapters",
    },
    "maniqa_official": {
        "label": "MANIQA (official recipe)",
        "params_m": 135.7,
        "trainable_m": 135.7,
        "note": "5 seeds; input scaled to [-1, 1], raw-MOS MSE",
    },
    "topiq": {
        "label": "TOPIQ-NR",
        "params_m": 45.2,
        "trainable_m": 45.2,
        "note": "5 seeds; frozen backbone probe",
    },
    "liqe": {
        "label": "LIQE",
        "params_m": 151.3,
        "trainable_m": 151.3,
        "note": "5 seeds; native cross-entropy head",
    },
    "arniqa": {
        "label": "ARNIQA",
        "params_m": 23.5,
        "trainable_m": 23.5,
        "note": "5 seeds",
    },
    "clipiqa": {
        "label": "CLIP-IQA",
        "params_m": 102.3,
        "trainable_m": 102.3,
        "note": "5 seeds; probe",
    },
}

ALIASES = {"maniqa": "maniqa_official"}


def main():
    stats_df = pd.read_csv(STATS)
    seeds = pd.read_csv(SEEDS)
    ours = seeds[(seeds.model == "ours") & (seeds.split == "content")].set_index("seed")["srocc"]

    rows = []
    for _, r in stats_df.iterrows():
        key = ALIASES.get(r.model, r.model)
        meta = REGISTRY.get(key, {"label": r.model, "params_m": float("nan"),
                                  "trainable_m": float("nan"), "note": ""})
        content = seeds[(seeds.model == r.model) & (seeds.split == "content")].set_index("seed")["srocc"]
        paired = ours.index.intersection(content.index)
        p_vs_ours = float("nan")
        if len(paired) >= 2 and r.model != "ours":
            p_vs_ours = stats.ttest_rel(ours.loc[paired], content.loc[paired]).pvalue
        rows.append({
            "model": key,
            "label": meta["label"],
            "params_m": meta["params_m"],
            "trainable_m": meta["trainable_m"],
            "content_srocc": r.content_mean,
            "content_std": r.content_std,
            "random_srocc": r.random_mean,
            "random_std": r.random_std,
            "delta": r.delta_mean,
            "p_content_vs_random": r.p_value,
            "p_content_vs_ours": p_vs_ours,
            "note": meta["note"],
        })
    df = pd.DataFrame(rows).sort_values("content_srocc", ascending=False)
    df.to_csv(OUT, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
