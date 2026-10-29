"""Sanity tests for the evaluation protocol. Run with:  pytest tests/ -q

These tests do not need the dataset - they check the mathematical machinery
(content split, correlation metrics, logistic mapping) on synthetic data.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from eval_protocol import (  # noqa: E402
    correlation_report,
    orient_higher_is_better,
    split_by_reference,
)


def test_split_by_reference_no_leak():
    refs = [f'I{i:02d}.png' for i in range(1, 82)]
    train, test = split_by_reference(refs, 0.2, seed=42)
    assert len(train) == 65 and len(test) == 16
    assert not set(train) & set(test)
    assert sorted(train + test) == refs


def test_monotone_recovery():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = 2.0 * x + 0.5 * rng.normal(size=500)
    rep = correlation_report(x, y)
    assert rep['srocc'] > 0.9 and rep['plcc'] > 0.9
    rep_neg = correlation_report(orient_higher_is_better(x), y)
    assert rep_neg['srocc'] < -0.9


def test_psnr_scale_finite():
    # The logistic fit must stay numerically stable on raw scales such as PSNR (dB).
    rng = np.random.default_rng(1)
    x = 20.0 + 12.0 * rng.random(500)
    y = 1.5 + 2.5 * np.tanh((x - 30.0) / 8.0) + 0.2 * rng.normal(size=500)
    rep = correlation_report(x, y)
    assert np.isfinite(rep['plcc']) and np.isfinite(rep['rmse'])
    assert rep['srocc'] > 0.8


def test_nonfinite_scores_sanitized():
    # A +inf score (e.g. uncapped PSNR) must never produce a nan PLCC.
    rng = np.random.default_rng(2)
    x = rng.normal(size=200)
    y = x + 0.3 * rng.normal(size=200)
    x[::50] = np.inf  # sprinkle a few infinities
    rep = correlation_report(x, y)
    assert np.isfinite(rep['plcc']) and np.isfinite(rep['rmse'])
