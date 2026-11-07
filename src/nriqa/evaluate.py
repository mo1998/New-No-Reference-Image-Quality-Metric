"""Inference and evaluation utilities.

Predictions are stored per image so that every reported correlation can be
recomputed, sliced by distortion type, or bootstrapped without rerunning the
model.  All evaluation is reference-free: the reference filename is carried
only for grouping/statistics, never as a model input.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import (
    IQADataset,
    build_transform,
    load_dataframe,
    make_loader,
)
from .metrics import bootstrap_ci, compute_metrics


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool = False,
    tta: bool = False,
) -> pd.DataFrame:
    """Run inference and return a per-image dataframe of predictions."""
    model.eval()
    rows: List[Dict[str, Any]] = []
    use_amp = amp and device.type == "cuda"
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(images)
            if tta:
                # Horizontal flip is label-preserving for quality (not for
                # spatially asymmetric artifacts such as directional blur, so
                # this is opt-in and reported separately).
                pred = 0.5 * (pred + model(torch.flip(images, dims=[3])))
        pred = pred.float().cpu().numpy()
        n = len(pred)
        for i in range(n):
            rows.append(
                {
                    "name": batch["name"][i],
                    "ref": batch["ref"][i],
                    "dataset": batch["dataset"][i],
                    "mos": float(batch["mos"][i]),
                    "pred": float(pred[i]),
                    "dist_type": int(batch["dist_type"][i]),
                    "level": int(batch["level"][i]),
                }
            )
    return pd.DataFrame(rows)


def predict_dataframe(
    model: torch.nn.Module,
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    device: torch.device,
    split: str = "test",
) -> pd.DataFrame:
    """Build the clean-protocol loader for ``df`` and run inference."""
    data_cfg = cfg["data"]
    transform = build_transform(
        int(data_cfg["size"]),
        train=False,
        norm=data_cfg["norm"],
        eval_mode=data_cfg["eval_mode"],
    )
    loader = make_loader(
        df,
        transform,
        batch_size=int(cfg["eval"]["batch_size"]),
        num_workers=int(data_cfg["num_workers"]),
        shuffle=False,
        seed=int(cfg["experiment"]["seed"]),
    )
    preds = predict_loader(
        model, loader, device,
        amp=bool(cfg["train"].get("amp", False)),
        tta=bool(cfg["eval"].get("tta", False)),
    )
    preds["split"] = split
    return preds


def denormalize(pred: np.ndarray, target_stats: Optional[Dict[str, Any]]) -> np.ndarray:
    """Invert the training-time target scaling."""
    if not target_stats:
        return pred
    mode = target_stats.get("mode", "zscore")
    if mode == "zscore":
        return pred * float(target_stats["std"]) + float(target_stats["mean"])
    if mode == "minmax":
        return pred * (float(target_stats["max"]) - float(target_stats["min"])) + float(target_stats["min"])
    return pred


def evaluate_predictions(
    preds: pd.DataFrame,
    target_stats: Optional[Dict[str, float]] = None,
    bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute point metrics, CIs and per-distortion breakdown from a frame."""
    pred = preds["pred"].to_numpy(dtype=float)
    if target_stats:
        pred = denormalize(pred, target_stats)
    mos = preds["mos"].to_numpy(dtype=float)
    refs = preds["ref"].astype(str).to_numpy()
    metrics = compute_metrics(pred, mos)
    out: Dict[str, Any] = {"metrics": metrics, "n": int(len(preds))}
    if bootstrap and bootstrap > 0 and len(set(refs)) > 1:
        ci = bootstrap_ci(pred, mos, refs, n_boot=int(bootstrap), seed=seed)
        out["ci"] = ci
    # Per-distortion-type SROCC: diagnostic for which distortions are learned
    per_type = []
    for dtype, group in preds.groupby("dist_type"):
        if group["mos"].nunique() > 1 and len(group) >= 5:
            g_pred = group["pred"].to_numpy(dtype=float)
            if target_stats:
                g_pred = denormalize(g_pred, target_stats)
            m = compute_metrics(g_pred, group["mos"].to_numpy(dtype=float))
            per_type.append(
                {"dist_type": int(dtype), "n": int(len(group)),
                 "srocc": m["srocc"], "plcc": m["plcc"]}
            )
    out["per_distortion"] = per_type
    return out


def save_predictions(preds: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    preds.to_csv(path, index=False)


def load_model_from_checkpoint(path: str, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Rebuild a model from a training checkpoint."""
    from .models import build_model

    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["cfg"]
    model = build_model(cfg)
    state = ckpt.get("ema_state") or ckpt["model_state"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model, ckpt


def evaluate_cross_dataset(
    model: torch.nn.Module,
    cfg: Dict[str, Any],
    device: torch.device,
    target_stats: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Zero-shot evaluation on TID2013 (or another configured dataset)."""
    data_cfg = cfg["data"]
    root = data_cfg.get("tid_root", "data/tid2013")
    df = load_dataframe("tid2013", root)
    preds = predict_dataframe(model, df, cfg, device, split="tid2013")
    preds["pred_raw"] = denormalize(preds["pred"].to_numpy(), target_stats)
    result = evaluate_predictions(
        preds, target_stats=target_stats,
        bootstrap=int(cfg["eval"].get("bootstrap", 0)),
        seed=int(cfg["experiment"]["seed"]),
    )
    result["predictions"] = preds
    return result
