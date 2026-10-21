"""Evaluation protocol for Project 28 (no-reference IQA on KADID-10k).

Implements the metric conventions used across the IQA literature:

- SROCC: Spearman rank-order correlation between predicted scores and DMOS.
- PLCC / RMSE: computed after fitting a 4-parameter logistic mapping, because
  metric outputs and DMOS live on different scales (VQEG convention).
- Sign convention: every metric is oriented so that *higher = better quality*
  before correlations are computed (use orient_higher_is_better() for
  distance metrics such as LPIPS/DISTS).
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

SEED = 42


def split_by_reference(ref_names, test_frac=0.2, seed=SEED):
    """Split reference (content) names into train/test sets.

    KADID-10k has 81 reference images x 125 distortions each. The split must
    happen at the *reference* level so that no content appears in both splits:
    a random split over distorted images would leak the pristine content into
    the test set and inflate every correlation.

    Returns (train_refs, test_refs) as sorted lists.
    """
    refs = sorted(set(ref_names))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(refs))
    n_test = int(round(len(refs) * test_frac))
    test = sorted(refs[i] for i in perm[:n_test])
    train = sorted(refs[i] for i in perm[n_test:])
    return train, test


def _logistic_4p(x, b1, b2, b3, b4):
    """Standard 4-parameter logistic mapping used in IQA (VQEG convention).

    b1/b2 are the upper/lower asymptotes, b3 the horizontal shift, b4 the scale.
    """
    return (b1 - b2) / (1.0 + np.exp(-(x - b3) / np.abs(b4))) + b2


def fit_logistic(x, y):
    """Fit the logistic mapping from metric scores x to opinion scores y.

    x is standardised first (zero mean, unit variance) which keeps the
    optimisation numerically stable on raw metric scales such as PSNR (tens of
    dB). Returns a dict with the fitted parameters and the normalisation, or
    None if the fit fails - callers then fall back to an identity mapping.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mu = float(x.mean())
    sd = float(x.std()) or 1.0
    xs = (x - mu) / sd

    p0 = [float(np.max(y)), float(np.min(y)), 0.0, 1.0]
    bounds = (
        [min(float(y.min()), 0.0) - np.ptp(y), -np.inf, -10.0, 0.01],
        [np.inf, max(float(y.max()), 0.0) + np.ptp(y), 10.0, 10.0],
    )
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
    return {'params': params, 'mu': mu, 'sd': sd}


def apply_logistic(fit, x):
    """Map raw metric scores through a fitted logistic (identity if fit is None)."""
    x = np.asarray(x, dtype=float)
    if fit is None:
        return x
    xs = (x - fit['mu']) / fit['sd']
    return _logistic_4p(xs, *fit['params'])


def correlation_report(scores, mos):
    """SROCC, PLCC and RMSE of `scores` against `mos`.

    `scores` must already be oriented so that higher = better quality.
    PLCC/RMSE are computed after the logistic mapping (SROCC needs no fit).
    """
    scores = np.asarray(scores, dtype=float)
    mos = np.asarray(mos, dtype=float)
    finite = np.isfinite(scores)
    if not finite.all():
        n_bad = int((~finite).sum())
        vals = scores[finite]
        lo = float(vals.min()) if vals.size else 0.0
        hi = float(vals.max()) if vals.size else 0.0
        print(f'WARNING: {n_bad} non-finite score(s) clamped into the finite range')
        scores = np.nan_to_num(scores, nan=hi, posinf=hi, neginf=lo)
    srocc = stats.spearmanr(scores, mos).statistic
    fit = fit_logistic(scores, mos)
    mapped = apply_logistic(fit, scores)
    plcc = stats.pearsonr(mapped, mos).statistic
    rmse = float(np.sqrt(np.mean((mapped - mos) ** 2)))
    return {'srocc': float(srocc), 'plcc': float(plcc), 'rmse': rmse}


def orient_higher_is_better(scores):
    """Distance metrics (LPIPS, DISTS) are lower-better: negate for correlation."""
    return -np.asarray(scores, dtype=float)
