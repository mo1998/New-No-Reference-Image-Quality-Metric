"""Evaluation protocol for the IQA metrics.

This follows the conventions used across the IQA literature: SROCC is the
Spearman rank correlation between predicted scores and DMOS; PLCC and RMSE are
computed after a 4-parameter logistic mapping, because metric outputs and DMOS
live on different scales (the VQEG convention). Every metric is oriented so
that higher means better quality before correlations are computed; use
orient_higher_is_better() for distance metrics such as LPIPS or DISTS.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

SEED = 42


def split_by_reference(ref_names, test_frac=0.2, seed=SEED):
    """Split reference (content) names into train and test sets.

    KADID-10k has 81 reference images with 125 distortions each, so the split
    has to happen at the reference level: otherwise the pristine content leaks
    into the test set and inflates every correlation. Returns (train_refs,
    test_refs) as sorted lists.
    """
    # Each reference image (e.g. I01) is one "content"; distortions are I01_01_01
    refs = sorted(set(ref_names))
    # Fixed seed so the split is identical on every run (course rule)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(refs))
    n_test = int(round(len(refs) * test_frac))
    # Keep both lists sorted for stable, diffable output
    test = sorted(refs[i] for i in perm[:n_test])
    train = sorted(refs[i] for i in perm[n_test:])
    return train, test


def _logistic_4p(x, b1, b2, b3, b4):
    """Standard 4-parameter logistic mapping used in IQA (VQEG convention).

    b1 and b2 are the upper and lower asymptotes, b3 is the horizontal shift
    and b4 is the scale.
    """
    return (b1 - b2) / (1.0 + np.exp(-(x - b3) / np.abs(b4))) + b2


def fit_logistic(x, y):
    """Fit a logistic mapping from metric scores x to opinion scores y.

    x is standardised first (zero mean, unit variance) so the optimiser stays
    stable on raw scales such as PSNR in dB. Returns a dict with the fitted
    parameters and the normalisation, or None if the fit fails - in that case
    callers fall back to an identity mapping.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Standardise inputs: PSNR is in dB (tens of units) and would otherwise
    # make the optimiser unstable
    mu = float(x.mean())
    sd = float(x.std()) or 1.0
    xs = (x - mu) / sd

    # Initial guess: asymptotes at the observed min/max of y
    p0 = [float(np.max(y)), float(np.min(y)), 0.0, 1.0]
    bounds = (
        [min(float(y.min()), 0.0) - np.ptp(y), -np.inf, -10.0, 0.01],
        [np.inf, max(float(y.max()), 0.0) + np.ptp(y), 10.0, 10.0],
    )
    # Suppress overflow warnings from the exponential; we check finiteness below
    with np.errstate(over='ignore', invalid='ignore'):
        try:
            params, _ = curve_fit(_logistic_4p, xs, y, p0=p0, bounds=bounds,
                                  maxfev=50000, method='trf')
        except Exception as exc:  # pragma: no cover - safety net
            print(f'WARNING: logistic fit failed ({exc}); using identity mapping')
            return None

    mapped = _logistic_4p(xs, *params)
    if not (np.isfinite(params).all() and np.isfinite(mapped).all()):
        print('WARNING: logistic fit produced non-finite values; using identity mapping')
        return None
    # Keep the normalisation so apply_logistic() can map new scores the same way
    return {'params': params, 'mu': mu, 'sd': sd}


def apply_logistic(fit, x):
    """Map raw metric scores through a fitted logistic, or return them unchanged if the fit is None."""
    x = np.asarray(x, dtype=float)
    if fit is None:
        return x
    xs = (x - fit['mu']) / fit['sd']
    return _logistic_4p(xs, *fit['params'])


def correlation_report(scores, mos):
    """SROCC, PLCC and RMSE of `scores` against `mos`.

    `scores` must already be oriented so that higher means better quality.
    SROCC needs no fit; PLCC and RMSE use the logistic-mapped scores.
    """
    scores = np.asarray(scores, dtype=float)
    mos = np.asarray(mos, dtype=float)
    # Defensive: a stray inf/nan (e.g. from an identical pair) must not
    # silently kill the whole correlation; clamp into the observed finite range
    finite = np.isfinite(scores)
    if not finite.all():
        n_bad = int((~finite).sum())
        vals = scores[finite]
        lo = float(vals.min()) if vals.size else 0.0
        hi = float(vals.max()) if vals.size else 0.0
        print(f'WARNING: {n_bad} non-finite score(s) clamped into the finite range')
        scores = np.nan_to_num(scores, nan=hi, posinf=hi, neginf=lo)
    # SROCC is rank-based, so it needs no scale alignment
    srocc = stats.spearmanr(scores, mos).statistic
    # PLCC/RMSE use the logistic-mapped scores (VQEG convention)
    fit = fit_logistic(scores, mos)
    mapped = apply_logistic(fit, scores)
    plcc = stats.pearsonr(mapped, mos).statistic
    rmse = float(np.sqrt(np.mean((mapped - mos) ** 2)))
    return {'srocc': float(srocc), 'plcc': float(plcc), 'rmse': rmse}


def orient_higher_is_better(scores):
    """Flip distance metrics (LPIPS, DISTS) so higher is better before correlating."""
    # Negating flips "lower is better" into "higher is better" for SROCC/PLCC
    return -np.asarray(scores, dtype=float)
