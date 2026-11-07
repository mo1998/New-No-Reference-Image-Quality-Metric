"""IQA evaluation metrics and content-level uncertainty estimates.

Conventions (matching the VQEG/IQA literature):

* SROCC is rank correlation and needs no fitting.
* PLCC and RMSE are reported after a monotonic 4-parameter logistic mapping
  from predicted scores to MOS, fitted on the evaluated split.
* Uncertainty is estimated by bootstrapping *reference contents*, because
  distorted images of the same content are not independent observations.

All functions accept arbitrary numeric arrays and return plain Python floats
so results are JSON-serialisable.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

EPS = 1e-12


# ---------------------------------------------------------------------------
# Logistic mapping
# ---------------------------------------------------------------------------
def _logistic4(x: np.ndarray, b1: float, b2: float, b3: float, b4: float) -> np.ndarray:
    denom = 1.0 + np.exp(-(x - b3) / max(abs(b4), EPS))
    return (b1 - b2) / denom + b2


def fit_logistic(x: Sequence[float], y: Sequence[float]) -> Optional[Dict[str, Any]]:
    """Fit y = b2 + (b1-b2)/(1+exp(-(x-b3)/|b4|)).

    Inputs are standardised before fitting for numerical stability.  Returns
    ``None`` if the optimiser fails; callers then use predictions unchanged.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 4 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    if np.ptp(x) < EPS or np.ptp(y) < EPS:
        return None
    mu, sd = float(x.mean()), float(x.std())
    if sd < EPS:
        return None
    xs = (x - mu) / sd
    p0 = [float(np.max(y)), float(np.min(y)), 0.0, 1.0]
    span = float(np.ptp(y)) or 1.0
    bounds = (
        [float(np.min(y)) - 5.0 * span, -np.inf, -10.0, 0.01],
        [float(np.max(y)) + 5.0 * span, np.inf, 10.0, 10.0],
    )
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            params, _ = curve_fit(
                _logistic4, xs, y, p0=p0, bounds=bounds, maxfev=50000, method="trf"
            )
        mapped = _logistic4(xs, *params)
        if not (np.isfinite(params).all() and np.isfinite(mapped).all()):
            return None
    except Exception:
        return None
    return {"params": params, "mu": mu, "sd": sd}


def apply_logistic(fit: Optional[Dict[str, Any]], x: Sequence[float]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if fit is None:
        return x
    xs = (x - fit["mu"]) / fit["sd"]
    return _logistic4(xs, *fit["params"])


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------
def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.ptp(x) < EPS or np.ptp(y) < EPS:
        return float("nan")
    return float(stats.spearmanr(x, y)[0])


def _safe_kendall(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.ptp(x) < EPS or np.ptp(y) < EPS:
        return float("nan")
    return float(stats.kendalltau(x, y)[0])


def compute_metrics(
    pred: Sequence[float],
    mos: Sequence[float],
    fit_logistic_mapping: bool = True,
) -> Dict[str, float]:
    """SROCC, PLCC, RMSE and KRCC for predictions vs MOS."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    mos = np.asarray(mos, dtype=np.float64).ravel()
    if pred.size != mos.size:
        raise ValueError(f"pred/mos size mismatch: {pred.size} vs {mos.size}")
    mask = np.isfinite(pred) & np.isfinite(mos)
    pred, mos = pred[mask], mos[mask]
    if pred.size < 3:
        return {"srocc": float("nan"), "plcc": float("nan"), "rmse": float("nan"),
                "krcc": float("nan"), "n": int(pred.size)}

    srocc = _safe_spearman(pred, mos)
    krcc = _safe_kendall(pred, mos)
    fit = fit_logistic(pred, mos) if fit_logistic_mapping else None
    mapped = apply_logistic(fit, pred)
    if np.ptp(mapped) < EPS:
        plcc = float("nan")
        rmse = float(np.sqrt(np.mean((mapped - mos) ** 2)))
    else:
        plcc = float(stats.pearsonr(mapped, mos)[0])
        rmse = float(np.sqrt(np.mean((mapped - mos) ** 2)))
    return {
        "srocc": srocc,
        "plcc": plcc,
        "rmse": rmse,
        "krcc": krcc,
        "n": int(pred.size),
    }


# ---------------------------------------------------------------------------
# Content-level bootstrap
# ---------------------------------------------------------------------------
def _group_indices(refs: np.ndarray) -> Dict[str, np.ndarray]:
    groups: Dict[str, List[int]] = {}
    for i, r in enumerate(refs):
        groups.setdefault(str(r), []).append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}


def bootstrap_ci(
    pred: Sequence[float],
    mos: Sequence[float],
    refs: Sequence[str],
    n_boot: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """Bootstrap confidence intervals by resampling reference contents.

    Returns ``{metric: {mean, low, high, std}}``.  The point estimate is
    computed on the full sample; the interval is the percentile interval of
    the bootstrap replicates.
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    mos = np.asarray(mos, dtype=np.float64).ravel()
    refs = np.asarray([str(r) for r in refs])
    if not (pred.size == mos.size == refs.size):
        raise ValueError("pred, mos and refs must have equal length")
    groups = _group_indices(refs)
    keys = sorted(groups)
    if len(keys) < 2 or n_boot <= 0:
        point = compute_metrics(pred, mos)
        return {k: {"mean": v, "low": float("nan"), "high": float("nan"), "std": float("nan")}
                for k, v in point.items() if k in ("srocc", "plcc", "rmse", "krcc")}

    rng = np.random.default_rng(seed)
    reps: Dict[str, List[float]] = {"srocc": [], "plcc": [], "rmse": [], "krcc": []}
    for _ in range(n_boot):
        chosen = rng.choice(len(keys), size=len(keys), replace=True)
        idx = np.concatenate([groups[keys[i]] for i in chosen])
        rep = compute_metrics(pred[idx], mos[idx])
        for k in reps:
            if np.isfinite(rep[k]):
                reps[k].append(rep[k])

    point = compute_metrics(pred, mos)
    alpha = (1.0 - confidence) / 2.0
    out: Dict[str, Dict[str, float]] = {}
    for k, vals in reps.items():
        if not vals:
            out[k] = {"mean": point[k], "low": float("nan"), "high": float("nan"),
                      "std": float("nan")}
            continue
        arr = np.asarray(vals)
        out[k] = {
            "mean": float(point[k]),
            "low": float(np.quantile(arr, alpha)),
            "high": float(np.quantile(arr, 1.0 - alpha)),
            "std": float(arr.std(ddof=1)),
        }
    return out


def paired_bootstrap_test(
    pred_a: Sequence[float],
    pred_b: Sequence[float],
    mos: Sequence[float],
    refs: Sequence[str],
    metric: str = "srocc",
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Two-sided paired bootstrap test for metric(A) - metric(B).

    Contents are resampled jointly for both methods, preserving the pairing.
    Returns the observed difference, its bootstrap CI and a two-sided p-value.
    """
    pred_a = np.asarray(pred_a, dtype=np.float64).ravel()
    pred_b = np.asarray(pred_b, dtype=np.float64).ravel()
    mos = np.asarray(mos, dtype=np.float64).ravel()
    refs = np.asarray([str(r) for r in refs])
    groups = _group_indices(refs)
    keys = sorted(groups)
    if len(keys) < 2:
        return {"diff": float("nan"), "p_value": float("nan"), "low": float("nan"), "high": float("nan")}
    fn: Callable[[np.ndarray, np.ndarray], Dict[str, float]] = compute_metrics

    obs_a = fn(pred_a, mos)[metric]
    obs_b = fn(pred_b, mos)[metric]
    obs = float(obs_a - obs_b)
    rng = np.random.default_rng(seed)
    diffs: List[float] = []
    for _ in range(n_boot):
        chosen = rng.choice(len(keys), size=len(keys), replace=True)
        idx = np.concatenate([groups[keys[i]] for i in chosen])
        a = fn(pred_a[idx], mos[idx])[metric]
        b = fn(pred_b[idx], mos[idx])[metric]
        if np.isfinite(a) and np.isfinite(b):
            diffs.append(a - b)
    if not diffs:
        return {"diff": obs, "p_value": float("nan"), "low": float("nan"), "high": float("nan")}
    arr = np.asarray(diffs)
    # Two-sided bootstrap p-value: twice the smaller tail around zero
    p = 2.0 * min(float(np.mean(arr <= 0.0)), float(np.mean(arr >= 0.0)))
    p = min(1.0, max(p, 1.0 / max(len(arr), 1)))
    return {
        "diff": obs,
        "p_value": p,
        "low": float(np.quantile(arr, 0.025)),
        "high": float(np.quantile(arr, 0.975)),
        "n_boot": int(len(arr)),
    }


def summarize(metrics: Dict[str, float]) -> str:
    """Compact human-readable one-liner."""
    return (
        f"SROCC={metrics.get('srocc', float('nan')):.4f} "
        f"PLCC={metrics.get('plcc', float('nan')):.4f} "
        f"RMSE={metrics.get('rmse', float('nan')):.4f} "
        f"(n={int(metrics.get('n', 0))})"
    )
