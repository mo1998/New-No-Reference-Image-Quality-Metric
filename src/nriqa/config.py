"""Configuration, seeding and reproducibility helpers.

A single YAML/JSON file fully determines an experiment.  The defaults below are
the canonical protocol used by every headline result in this repository; a
config only needs to override what it changes, which keeps the ablation grid
small and auditable.
"""

from __future__ import annotations

import copy
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Canonical defaults.  Changing a value here changes every experiment, so the
# defaults are intentionally conservative and documented in docs/protocol.md.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "experiment": {
        "name": "experiment",
        "seed": 42,
        "output_dir": "runs",
        "notes": "",
    },
    "data": {
        "dataset": "kadid10k",
        "root": "data/kadid10k",
        "size": 224,
        "split": "holdout",          # holdout | kfold
        "test_frac": 0.2,
        "val_frac": 0.1,            # fraction of ALL reference images
        "fold": 0,
        "kfold": 5,
        "num_workers": 8,
        "augment": "standard",       # standard | light | none
        "norm": "none",              # none (raw [0,1]) | imagenet
        "eval_mode": "center",       # center | resize
        "eval_datasets": ["kadid10k"],
        "tid_root": "data/tid2013",
    },
    "model": {
        "name": "tinycnn",
        "pretrained": False,
        "kwargs": {},
    },
    "train": {
        "epochs": 40,
        "batch_size": 64,
        "lr": 1.0e-3,
        "weight_decay": 5.0e-4,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "warmup_epochs": 2,
        "amp": True,
        "grad_clip": 1.0,
        "label_norm": "zscore",      # zscore | minmax | none
        "loss": {
            "mse": 1.0,
            "plcc": 0.1,
            "rank": 0.0,
            "margin": 0.05,
            "aux_distortion": 0.0,   # optional multi-task head weight
            "aux_level": 0.0,
        },
        "early_stop": {
            "patience": 12,
            "min_delta": 5.0e-4,
            "monitor": "val_srocc",  # val_srocc | val_loss
        },
        "ema": 0.0,
        "drop_last": True,
        "select_best": True,
    },
    "eval": {
        "batch_size": 128,
        "bootstrap": 1000,
        "report_ci": True,
        "tta": False,
        "tta_mode": "flip",  # flip | multicrop | both | none
    },
    "logging": {
        "tensorboard": True,
        "save_predictions": True,
        "save_checkpoint": True,
    },
}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _load_raw(path: str) -> Dict[str, Any]:
    """Load a config, resolving an optional relative ``include`` first."""
    with open(path, "r", encoding="utf-8") as fh:
        if path.endswith(".json"):
            user = json.load(fh) or {}
        else:
            user = yaml.safe_load(fh) or {}
    parent: Dict[str, Any] = {}
    include = user.pop("include", None)
    if include:
        inc_path = include if os.path.isabs(include) else os.path.join(os.path.dirname(path), include)
        parent = _load_raw(inc_path)
    return deep_update(parent, user)


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML (or JSON) config and merge it over DEFAULT_CONFIG."""
    user = _load_raw(path)
    cfg = deep_update(DEFAULT_CONFIG, user)
    cfg["_config_path"] = os.path.abspath(path)
    return cfg


def save_config(cfg: Dict[str, Any], path: str) -> None:
    """Persist the fully-resolved config next to a run's artifacts."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(clean, fh, sort_keys=True)


def set_seed(seed: int) -> None:
    """Seed every RNG that can influence a result."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve 'auto' to CUDA when available, otherwise CPU."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def git_commit() -> str:
    """Return the current git SHA (or 'unknown' outside a checkout)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    """Whether the working tree has uncommitted changes."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
        return bool(out.strip())
    except Exception:
        return False


def environment_info() -> Dict[str, Any]:
    """Metadata stored with every run for auditability."""
    info: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
    }
    if torch.cuda.is_available():
        info["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


def run_dir(cfg: Dict[str, Any]) -> str:
    """Create and return a unique output directory for an experiment."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = cfg["experiment"]["name"]
    path = os.path.join(cfg["experiment"]["output_dir"], f"{stamp}_{name}")
    os.makedirs(path, exist_ok=False)
    return path
