"""Build the controlled KADID comparison table from the five-seed runs.

Inputs are produced by ``collect_balanced.py``:

* ``results/split_bias/balanced_stats.csv``     - per-model mean/std/delta/p
* ``results/split_bias/balanced_multiseed.csv`` - per-seed SROCC/PLCC

The parameter counts are recorded from each run's training log (``params`` and
``params_trainable`` fields) because several baselines are loaded through
third-party packages that do not expose a stable parameter-count helper.
"""

from __future__ import annotations

import json
import os

import pandas as pd
from scipy import stats

STATS = "results/split_bias/balanced_stats.csv"
SEEDS = "results/split_bias/balanced_multiseed.csv"
OUT = "results/split_bias/controlled_comparison.csv"

# Five-fold content CV artefacts (written by ``nriqa.cv``).  Models without an
# entry were not run five-fold; the column is left empty rather than filled
# with the five-seed estimate, which is a different estimator.
CV_FILES = {
    "ours": "results/cv/siglip_b16_384_koniq_cv65.json",
    "siglip2": "results/cv/siglip2_l16_384_cv65.json",
    "bgated_lora": "results/cv/bgated_lora_cv65.json",
    "bgated": "results/cv/bgated_cv65.json",
    "maniqa_official": "results/cv/maniqa_cv65.json",
    "topiq": "results/cv/topiq_cv65.json",
    "liqe": "results/cv/liqe_cv65.json",
    "arniqa": "results/cv/arniqa_cv65.json",
    "clipiqa": "results/cv/clipiqa_cv65.json",
}

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


def load_cv(model):
    path = CV_FILES.get(model)
    if not path or not os.path.exists(path):
        return None, None, None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    test = data.get("test") or {}
    return test.get("mean"), test.get("std"), test.get("values")


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
        cv_mean, cv_std, cv_folds = load_cv(key)
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
            "content_cv_srocc": cv_mean,
            "content_cv_std": cv_std,
            "content_cv_folds": cv_folds,
            "note": meta["note"],
        })
    df = pd.DataFrame(rows).sort_values("content_srocc", ascending=False)
    df.to_csv(OUT, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
