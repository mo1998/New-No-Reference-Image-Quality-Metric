"""Loss functions for NR-IQA.

The default objective is a weighted combination of

* normalised MSE            -- absolute calibration,
* 1 - Pearson correlation   -- linear agreement with MOS (PLCC proxy),
* pairwise ranking hinge    -- order agreement (SROCC proxy),
* optional distortion/level classification -- representation shaping.

No loss peeks at a reference image.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """1 - Pearson correlation, safe for constant batches."""
    if pred.numel() < 2:
        return pred.new_tensor(0.0)
    x = pred - pred.mean()
    y = target - target.mean()
    denom = torch.sqrt((x * x).sum() * (y * y).sum()).clamp_min(eps)
    return 1.0 - (x * y).sum() / denom


def pairwise_rank_loss(
    pred: torch.Tensor, target: torch.Tensor, margin: float = 0.05, max_pairs: int = 4096
) -> torch.Tensor:
    """Hinge loss enforcing the ordering of pairs whose MOS differs by > margin."""
    if pred.numel() < 2:
        return pred.new_tensor(0.0)
    diff_t = target[:, None] - target[None, :]
    mask = diff_t > margin
    if not bool(mask.any()):
        return pred.new_tensor(0.0)
    diff_p = pred[:, None] - pred[None, :]
    losses = F.relu(margin - diff_p[mask])
    if losses.numel() > max_pairs:
        idx = torch.randperm(losses.numel(), device=losses.device)[:max_pairs]
        losses = losses[idx]
    return losses.mean()


def content_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    refs: Sequence[str],
    margin: float = 0.05,
    max_pairs: int = 4096,
) -> torch.Tensor:
    """Pairwise ranking loss restricted to distorted versions of the same content.

    Global ranking losses spend most pairs on content differences; this loss
    spends them on the within-content ordering that SROCC ultimately measures.
    """
    if pred.numel() < 2:
        return pred.new_tensor(0.0)
    import numpy as _np
    ref_arr = _np.asarray([str(r) for r in refs])
    same = torch.tensor(ref_arr[:, None] == ref_arr[None, :], device=pred.device)
    diff_t = target[:, None] - target[None, :]
    mask = same & (diff_t > margin)
    if not bool(mask.any()):
        return pred.new_tensor(0.0)
    diff_p = pred[:, None] - pred[None, :]
    losses = F.relu(margin - diff_p[mask])
    if losses.numel() > max_pairs:
        idx = torch.randperm(losses.numel(), device=losses.device)[:max_pairs]
        losses = losses[idx]
    return losses.mean()


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_cfg: Dict[str, Any],
    aux: Optional[Dict[str, torch.Tensor]] = None,
    dist_type: Optional[torch.Tensor] = None,
    level: Optional[torch.Tensor] = None,
    refs: Optional[Sequence[str]] = None,
    soft: Optional[torch.Tensor] = None,
    teacher_feat: Optional[torch.Tensor] = None,
    student_proj: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Weighted NR-IQA objective; returns (total, detached component dict)."""
    mse = F.mse_loss(pred, target)
    plcc = pearson_loss(pred, target)
    rank = pairwise_rank_loss(
        pred, target, margin=float(loss_cfg.get("margin", 0.05))
    )
    content_rank = (
        content_rank_loss(pred, target, refs, margin=float(loss_cfg.get("margin", 0.05)))
        if (refs is not None and float(loss_cfg.get("content_rank", 0.0)) > 0)
        else pred.new_tensor(0.0)
    )
    distill = pred.new_tensor(0.0)
    if soft is not None and float(loss_cfg.get("distill", 0.0)) > 0:
        distill = F.mse_loss(pred, soft)
    feature_distill = pred.new_tensor(0.0)
    if (teacher_feat is not None and student_proj is not None
            and float(loss_cfg.get("feature_distill", 0.0)) > 0):
        cos = F.cosine_similarity(student_proj, teacher_feat, dim=1, eps=1e-6).mean()
        feature_distill = 1.0 - cos
    total = (
        float(loss_cfg.get("mse", 1.0)) * mse
        + float(loss_cfg.get("plcc", 0.0)) * plcc
        + float(loss_cfg.get("rank", 0.0)) * rank
        + float(loss_cfg.get("content_rank", 0.0)) * content_rank
        + float(loss_cfg.get("distill", 0.0)) * distill
        + float(loss_cfg.get("feature_distill", 0.0)) * feature_distill
    )
    parts = {"mse": float(mse.detach()), "plcc_loss": float(plcc.detach()),
             "rank_loss": float(rank.detach()),
             "content_rank_loss": float(content_rank.detach()),
             "distill_loss": float(distill.detach()),
             "feature_distill_loss": float(feature_distill.detach())}

    if aux is not None:
        w_dist = float(loss_cfg.get("aux_distortion", 0.0))
        w_level = float(loss_cfg.get("aux_level", 0.0))
        if w_dist > 0 and "dist_logits" in aux and dist_type is not None:
            # Dataset distortion ids are 1-based (KADID 1-25, TID2013 1-24);
            # shift to 0-based class indices.
            loss_d = F.cross_entropy(aux["dist_logits"], (dist_type.long() - 1).clamp_min(0))
            total = total + w_dist * loss_d
            parts["aux_dist"] = float(loss_d.detach())
        if w_level > 0 and "level_logits" in aux and level is not None:
            loss_l = F.cross_entropy(aux["level_logits"], (level.long() - 1).clamp_min(0))
            total = total + w_level * loss_l
            parts["aux_level"] = float(loss_l.detach())
    return total, parts
