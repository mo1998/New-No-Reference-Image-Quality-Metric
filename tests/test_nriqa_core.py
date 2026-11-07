"""Core tests for the NR-IQA framework (no long training required).

Run with:  PYTHONPATH=src .venv/bin/python -m pytest tests/test_nriqa_core.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nriqa.config import DEFAULT_CONFIG, deep_update, load_config  # noqa: E402
from nriqa.losses import compute_loss, pairwise_rank_loss, pearson_loss  # noqa: E402
from nriqa.metrics import bootstrap_ci, compute_metrics, fit_logistic  # noqa: E402


def test_config_inheritance():
    cfg = load_config("configs/siglip_b16_kadid.yaml")
    assert cfg["model"]["name"] == "timm_regressor"
    assert cfg["data"]["norm"] == "none"
    assert cfg["train"]["epochs"] == 20
    # base defaults survive the include chain
    assert cfg["data"]["split"] == "holdout"
    assert cfg["eval"]["report_ci"] is True


def test_deep_update_does_not_mutate_base():
    base = {"a": {"b": 1}}
    out = deep_update(base, {"a": {"c": 2}})
    assert out["a"] == {"b": 1, "c": 2}
    assert base == {"a": {"b": 1}}


def test_metrics_monotone_and_bootstrap():
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    y = x + 0.2 * rng.normal(size=300)
    m = compute_metrics(x, y)
    assert m["srocc"] > 0.9 and m["plcc"] > 0.9
    assert fit_logistic(x, y) is not None
    refs = np.repeat(np.arange(12), 25)
    ci = bootstrap_ci(x, y, refs, n_boot=100)
    assert ci["srocc"]["low"] <= ci["srocc"]["mean"] <= ci["srocc"]["high"]


def test_metrics_constant_input_is_nan_not_crash():
    m = compute_metrics(np.zeros(10), np.arange(10))
    assert np.isnan(m["srocc"])


def test_rank_and_pearson_losses():
    target = torch.tensor([0.0, 1.0, 2.0, 3.0])
    good = torch.tensor([0.0, 1.0, 2.0, 3.0])
    bad = torch.tensor([3.0, 2.0, 1.0, 0.0])
    assert pearson_loss(good, target) < pearson_loss(bad, target)
    assert pairwise_rank_loss(good, target) < pairwise_rank_loss(bad, target)


def test_aux_loss_label_shift():
    """KADID ids are 1..25; the aux head has 25 classes (0..24)."""
    pred = torch.zeros(4)
    target = torch.zeros(4)
    aux = {"dist_logits": torch.randn(4, 25, requires_grad=True),
           "level_logits": torch.randn(4, 5, requires_grad=True)}
    dist = torch.tensor([1, 25, 1, 24])
    level = torch.tensor([1, 5, 1, 4])
    loss, parts = compute_loss(
        pred, target,
        {"mse": 1.0, "aux_distortion": 0.1, "aux_level": 0.1},
        aux=aux, dist_type=dist, level=level,
    )
    assert torch.isfinite(loss)
    assert "aux_dist" in parts and "aux_level" in parts


def test_split_no_leak_and_sizes():
    from nriqa.data import split_dataframe
    import pandas as pd

    rng = np.random.default_rng(1)
    refs = [f"I{i:02d}.png" for i in range(1, 82)]
    rows = []
    for r in refs:
        for j in range(3):
            rows.append({"dataset": "kadid10k", "name": f"{r[:-4]}_{j + 1:02d}_01.png",
                         "path": "", "ref": r, "mos": rng.uniform(1, 5),
                         "mos_std": 0.1, "dist_type": 1, "level": j + 1})
    df = pd.DataFrame(rows)
    tr, va, te = split_dataframe(df, mode="holdout", test_frac=0.2, val_frac=0.1, seed=42)
    assert len(tr) + len(va) + len(te) == len(df)
    assert not set(tr.ref) & set(va.ref)
    assert not set(tr.ref) & set(te.ref)
    assert not set(va.ref) & set(te.ref)
    assert te.ref.nunique() == 16


def test_msosnet_and_waveqnet_forward():
    from nriqa.models import MSOSNet, WaveQNet

    x = torch.rand(2, 3, 96, 96)
    for model in (MSOSNet(widths=(8, 12), scales=2), WaveQNet(levels=2, embed_dim=32, depth=1, nhead=4)):
        model.eval()
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2,)
        assert torch.isfinite(y).all()


def test_aux_forward_shapes():
    from nriqa.models import MSOSNet

    model = MSOSNet(widths=(8, 12), scales=2, aux_heads=True).eval()
    with torch.no_grad():
        score, aux = model.forward_aux(torch.rand(2, 3, 64, 64))
    assert score.shape == (2,)
    assert aux["dist_logits"].shape == (2, 25)
    assert aux["level_logits"].shape == (2, 5)
