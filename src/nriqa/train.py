"""Reproducible training loop for NR-IQA models.

Design choices that matter for this task:

* model selection uses a *content-held-out validation split*, never the test
  split, so reported test correlations are unbiased;
* targets are normalised with statistics computed on the training contents
  only, and the inverse mapping is stored in the checkpoint;
* AMP, gradient clipping, cosine schedule with warm-up, optional EMA and early
  stopping are all config-driven and logged;
* every epoch writes a JSONL record and TensorBoard scalars.

The trainer is deliberately small enough to audit end-to-end.
"""

from __future__ import annotations

import copy
import math
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import environment_info, resolve_device, run_dir, set_seed
from .data import make_splits_and_loaders
from .evaluate import (
    denormalize,
    evaluate_cross_dataset,
    evaluate_predictions,
    predict_dataframe,
    predict_loader,
    save_predictions,
)
from .losses import compute_loss
from .metrics import compute_metrics, summarize
from .models import build_model, count_parameters
from .tracking import RunLogger


# ---------------------------------------------------------------------------
# Target scaling
# ---------------------------------------------------------------------------
def target_statistics(mos: np.ndarray, mode: str) -> Dict[str, Any]:
    """Statistics used to map MOS into the regression target space."""
    mos = np.asarray(mos, dtype=np.float64)
    if mode == "zscore":
        mean, std = float(mos.mean()), float(mos.std())
        std = std if std > 1e-8 else 1.0
        return {"mode": "zscore", "mean": mean, "std": std}
    if mode == "minmax":
        lo, hi = float(mos.min()), float(mos.max())
        return {"mode": "minmax", "min": lo, "max": hi if hi > lo else lo + 1.0}
    if mode == "none":
        return {"mode": "none"}
    raise ValueError(f"Unknown label_norm: {mode}")


def normalize_targets(mos: torch.Tensor, stats: Dict[str, Any]) -> torch.Tensor:
    if stats["mode"] == "zscore":
        return (mos - stats["mean"]) / stats["std"]
    if stats["mode"] == "minmax":
        return (mos - stats["min"]) / (stats["max"] - stats["min"])
    return mos


def inverse_targets(value: np.ndarray, stats: Optional[Dict[str, Any]]) -> np.ndarray:
    if not stats:
        return value
    if stats["mode"] == "zscore":
        return value * stats["std"] + stats["mean"]
    if stats["mode"] == "minmax":
        return value * (stats["max"] - stats["min"]) + stats["min"]
    return value


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for ema_b, b in zip(self.model.buffers(), model.buffers()):
            if ema_b.dtype.is_floating_point:
                ema_b.mul_(d).add_(b.detach(), alpha=1.0 - d)
            else:
                ema_b.copy_(b)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------
def _lr_lambda(epoch: int, warmup: int, total: int) -> float:
    if warmup > 0 and epoch < warmup:
        return float(epoch + 1) / float(warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    def __init__(self, cfg: Dict[str, Any], device: Optional[str] = None):
        self.cfg = cfg
        self.device = resolve_device(device or "auto")
        set_seed(int(cfg["experiment"]["seed"]))
        torch.backends.cudnn.benchmark = True

        self.out_dir = run_dir(cfg)
        self.logger = RunLogger(
            self.out_dir, enable_tensorboard=bool(cfg["logging"].get("tensorboard", True))
        )
        self.logger.log_config(cfg, environment_info())

        packed = make_splits_and_loaders(cfg)
        self.packed = packed
        self.train_df, self.val_df, self.test_df = packed["train_df"], packed["val_df"], packed["test_df"]
        self.loaders = packed["loaders"]
        self.summary_data = packed["summary"]

        self.model = build_model(cfg).to(self.device)
        self.n_params = count_parameters(self.model)
        self.n_params_trainable = count_parameters(self.model, trainable_only=True)

        tcfg = cfg["train"]
        self.target_stats = target_statistics(
            self.train_df["mos"].to_numpy(), tcfg.get("label_norm", "zscore")
        )
        self.optimizer = self._make_optimizer(tcfg)
        self.warmup = int(tcfg.get("warmup_epochs", 0))
        self.epochs = int(tcfg["epochs"])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda e: _lr_lambda(e, self.warmup, self.epochs)
        )
        self.use_amp = bool(tcfg.get("amp", True)) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.ema = (
            ModelEMA(self.model, float(tcfg["ema"]))
            if float(tcfg.get("ema", 0.0) or 0.0) > 0
            else None
        )
        self.select_best = bool(tcfg.get("select_best", True))
        self.best_metric = -float("inf")
        self.best_epoch = -1
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_ema_state: Optional[Dict[str, torch.Tensor]] = None
        self.patience = 0

    def _make_optimizer(self, tcfg: Dict[str, Any]) -> torch.optim.Optimizer:
        name = str(tcfg.get("optimizer", "adamw")).lower()
        lr = float(tcfg["lr"])
        wd = float(tcfg.get("weight_decay", 0.0))
        if name == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        if name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        if name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(), lr=lr, weight_decay=wd, momentum=0.9, nesterov=True
            )
        raise ValueError(f"Unknown optimizer: {name}")

    # -- one epoch ---------------------------------------------------------
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        tcfg = self.cfg["train"]
        loss_cfg = tcfg["loss"]
        totals = {"loss": 0.0, "mse": 0.0, "plcc_loss": 0.0, "rank_loss": 0.0}
        n_batches = 0
        use_aux = (
            float(loss_cfg.get("aux_distortion", 0.0)) > 0
            and hasattr(self.model, "forward_aux")
        )
        use_feature = (
            float(loss_cfg.get("feature_distill", 0.0)) > 0
            and hasattr(self.model, "forward_aux")
        )
        for batch in self.loaders["train"]:
            images = batch["image"].to(self.device, non_blocking=True)
            mos = batch["mos"].to(self.device, dtype=torch.float32)
            target = normalize_targets(mos, self.target_stats)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                if use_aux or use_feature:
                    pred, aux = self.model.forward_aux(images)
                else:
                    pred, aux = self.model(images), None
                soft = None
                if "soft" in batch and float(loss_cfg.get("distill", 0.0)) > 0:
                    soft = normalize_targets(
                        batch["soft"].to(self.device, dtype=torch.float32), self.target_stats
                    )
                teacher_feat = None
                student_proj = None
                if use_feature and "tfeat" in batch:
                    teacher_feat = batch["tfeat"].to(self.device, dtype=torch.float32)
                    student_proj = aux.get("proj") if aux is not None else None
                loss, parts = compute_loss(
                    pred, target, loss_cfg, aux=aux,
                    dist_type=batch["dist_type"].to(self.device) if use_aux else None,
                    level=batch["level"].to(self.device) if use_aux else None,
                    refs=batch["ref"], soft=soft,
                    teacher_feat=teacher_feat, student_proj=student_proj,
                )
            self.scaler.scale(loss).backward()
            if float(tcfg.get("grad_clip", 0.0)) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(tcfg["grad_clip"])
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.ema is not None:
                self.ema.update(self.model)
            totals["loss"] += float(loss.detach())
            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + v
            n_batches += 1
        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    @torch.no_grad()
    def validate(self) -> Tuple[Dict[str, float], float]:
        eval_model = self.ema.model if self.ema is not None else self.model
        preds = predict_loader(
            eval_model, self.loaders["val"], self.device, amp=self.use_amp
        )
        raw = inverse_targets(preds["pred"].to_numpy(), self.target_stats)
        metrics = compute_metrics(raw, preds["mos"].to_numpy())
        norm_mos = normalize_targets(
            torch.tensor(preds["mos"].to_numpy(), dtype=torch.float32), self.target_stats
        ).numpy()
        val_loss = float(np.mean((preds["pred"].to_numpy() - norm_mos) ** 2))
        return metrics, val_loss

    # -- full run ----------------------------------------------------------
    def fit(self) -> Dict[str, Any]:
        tcfg = self.cfg["train"]
        es = tcfg.get("early_stop", {})
        patience_limit = int(es.get("patience", 0) or 0)
        min_delta = float(es.get("min_delta", 0.0) or 0.0)
        monitor = str(es.get("monitor", "val_srocc"))
        print(
            f"[{self.cfg['experiment']['name']}] device={self.device} "
            f"params={self.n_params:,} split={self.summary_data} out={self.out_dir}"
        )

        last_train: Dict[str, float] = {}
        for epoch in range(self.epochs):
            t0 = time.time()
            last_train = self.train_epoch(epoch)
            val_metrics, val_loss = self.validate()
            lr = float(self.optimizer.param_groups[0]["lr"])
            record = {
                "epoch": epoch + 1,
                "lr": lr,
                "train_loss": last_train["loss"],
                "train_mse": last_train["mse"],
                "train_plcc_loss": last_train["plcc_loss"],
                "train_rank_loss": last_train.get("rank_loss", 0.0),
                "train_content_rank_loss": last_train.get("content_rank_loss", 0.0),
                "val_loss": val_loss,
                "val_srocc": val_metrics["srocc"],
                "val_plcc": val_metrics["plcc"],
                "val_rmse": val_metrics["rmse"],
                "val_krcc": val_metrics["krcc"],
                "epoch_time_s": round(time.time() - t0, 2),
            }
            if "aux_dist" in last_train:
                record["train_aux_dist"] = last_train["aux_dist"]
            if "aux_level" in last_train:
                record["train_aux_level"] = last_train["aux_level"]
            if "distill_loss" in last_train:
                record["train_distill_loss"] = last_train["distill_loss"]
            if "feature_distill_loss" in last_train:
                record["train_feature_distill_loss"] = last_train["feature_distill_loss"]
            self.logger.log_epoch(record)
            score = val_metrics["srocc"] if monitor == "val_srocc" else -val_loss
            improved = score > self.best_metric + min_delta
            if improved:
                self.best_metric = score
                self.best_epoch = epoch + 1
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.best_ema_state = (
                    copy.deepcopy(self.ema.state_dict()) if self.ema is not None else None
                )
                self.patience = 0
                if self.select_best:
                    self._save_checkpoint("best.pt", epoch + 1, val_metrics)
            else:
                self.patience += 1
            self.scheduler.step()
            print(
                f"  epoch {epoch + 1:3d}/{self.epochs} "
                f"loss={last_train['loss']:.4f} val_loss={val_loss:.4f} "
                f"SROCC={val_metrics['srocc']:.4f} PLCC={val_metrics['plcc']:.4f} "
                f"lr={lr:.2e} ({record['epoch_time_s']}s)"
                + (" *" if improved else "")
            )
            if self.select_best and patience_limit > 0 and self.patience >= patience_limit:
                print(f"  early stop at epoch {epoch + 1} (patience={patience_limit})")
                break
        self._save_checkpoint("last.pt", epoch + 1, val_metrics)

        # Restore the selected checkpoint.  When select_best is False the last
        # epoch is used and all 65 non-test contents can be used for training.
        if self.select_best and self.best_state is not None:
            state = self.best_ema_state if self.best_ema_state is not None else self.best_state
            self.model.load_state_dict(state)
        elif not self.select_best:
            if self.ema is not None:
                # Deployment/model-selection-free runs should still evaluate
                # the EMA weights, not the last noisy training step.
                self.model.load_state_dict(self.ema.state_dict())
            self.best_epoch = epoch + 1
        self.model.eval()
        return self.finalize()

    # -- checkpoint / final -------------------------------------------------
    def _checkpoint_payload(self, epoch: int, val_metrics: Dict[str, float]) -> Dict[str, Any]:
        return {
            "cfg": {k: v for k, v in self.cfg.items() if not k.startswith("_")},
            "model_state": self.model.state_dict(),
            "ema_state": self.ema.state_dict() if self.ema is not None else None,
            "target_stats": self.target_stats,
            "epoch": epoch,
            "val_metrics": val_metrics,
            "n_params": self.n_params,
        }

    def _save_checkpoint(self, name: str, epoch: int, val_metrics: Dict[str, float]) -> None:
        if not self.cfg["logging"].get("save_checkpoint", True):
            return
        torch.save(self._checkpoint_payload(epoch, val_metrics), os.path.join(self.out_dir, name))

    def finalize(self) -> Dict[str, Any]:
        eval_bs = int(self.cfg["eval"]["batch_size"])
        bootstrap = int(self.cfg["eval"].get("bootstrap", 0))
        seed = int(self.cfg["experiment"]["seed"])

        test_preds = predict_dataframe(self.model, self.test_df, self.cfg, self.device, split="test")
        test_result = evaluate_predictions(
            test_preds,
            target_stats=self.target_stats,
            bootstrap=bootstrap if self.cfg["eval"].get("report_ci", True) else 0,
            seed=seed,
        )
        val_preds = predict_dataframe(self.model, self.val_df, self.cfg, self.device, split="val")
        val_result = evaluate_predictions(
            val_preds, target_stats=self.target_stats, bootstrap=0, seed=seed
        )

        summary: Dict[str, Any] = {
            "experiment": self.cfg["experiment"]["name"],
            "model": self.cfg["model"]["name"],
            "model_kwargs": self.cfg["model"].get("kwargs", {}),
            "run_dir": self.out_dir,
            "split": self.summary_data,
            "params": self.n_params,
            "params_trainable": self.n_params_trainable,
            "target_stats": self.target_stats,
            "best_epoch": self.best_epoch,
            "test": test_result["metrics"],
            "test_ci": test_result.get("ci", {}),
            "val": val_result["metrics"],
            "per_distortion": test_result.get("per_distortion", []),
        }
        if self.cfg["logging"].get("save_predictions", True):
            save_predictions(test_preds, os.path.join(self.out_dir, "predictions_test.csv"))
            save_predictions(val_preds, os.path.join(self.out_dir, "predictions_val.csv"))

        # Zero-shot cross-dataset evaluation (train on KADID -> test TID2013)
        if "tid2013" in [d.lower() for d in self.cfg["data"].get("eval_datasets", [])]:
            try:
                cross = evaluate_cross_dataset(
                    self.model, self.cfg, self.device, target_stats=self.target_stats
                )
                summary["cross_tid2013"] = cross["metrics"]
                summary["cross_tid2013_ci"] = cross.get("ci", {})
                if self.cfg["logging"].get("save_predictions", True):
                    save_predictions(
                        cross["predictions"], os.path.join(self.out_dir, "predictions_tid2013.csv")
                    )
                print(f"  cross TID2013: {summarize(cross['metrics'])}")
            except Exception as exc:  # keep the run usable if TID is absent
                summary["cross_tid2013_error"] = repr(exc)
                print(f"  WARNING: cross-dataset evaluation failed: {exc}")

        self.logger.finish(summary)
        print(f"  TEST: {summarize(test_result['metrics'])}")
        if "cross_tid2013" in summary:
            print(f"  CROSS: {summarize(summary['cross_tid2013'])}")
        print(f"  artifacts: {self.out_dir}")
        return summary


def train(cfg: Dict[str, Any], device: Optional[str] = None) -> Dict[str, Any]:
    return Trainer(cfg, device=device).fit()
