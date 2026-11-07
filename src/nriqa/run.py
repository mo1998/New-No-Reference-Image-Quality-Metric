"""Command-line entry point for a single experiment.

Examples
--------
Train with a config::

    PYTHONPATH=src python -m nriqa.run --config configs/msosnet_kadid.yaml

Override values on the command line::

    PYTHONPATH=src python -m nriqa.run --config configs/msosnet_kadid.yaml \
        --set train.epochs=5 --set data.augment=none --name smoke

Evaluate a checkpoint without retraining::

    PYTHONPATH=src python -m nriqa.run --config configs/msosnet_kadid.yaml \
        --checkpoint runs/.../best.pt --mode eval
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, Dict

import yaml

from .config import load_config, resolve_device, set_seed
from .evaluate import evaluate_cross_dataset, evaluate_predictions, predict_dataframe, save_predictions
from .metrics import summarize
from .models import build_model, count_parameters


def _parse_value(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except Exception:
        return text


def apply_overrides(cfg: Dict[str, Any], overrides) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(value)
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["train", "eval"], default="train")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--name", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="Override a config value, e.g. train.epochs=10")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.overrides)
    if args.name is not None:
        cfg["experiment"]["name"] = args.name
    if args.output_dir is not None:
        cfg["experiment"]["output_dir"] = args.output_dir
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
    if args.fold is not None:
        cfg["data"]["fold"] = args.fold
    set_seed(int(cfg["experiment"]["seed"]))
    device = resolve_device(args.device)

    if args.mode == "train":
        from .train import train

        summary = train(cfg, device=str(device))
        print(json.dumps({k: v for k, v in summary.items() if k not in ("per_distortion",)}, indent=2)[:4000])
        return 0

    if not args.checkpoint:
        ap.error("--checkpoint is required in eval mode")
    from .train import target_statistics  # local import avoids a cycle at module load

    import torch
    from .data import load_dataframe, split_dataframe
    from .evaluate import load_model_from_checkpoint

    model, ckpt = load_model_from_checkpoint(args.checkpoint, device)
    cfg = ckpt["cfg"] if ckpt.get("cfg") else cfg
    target_stats = ckpt.get("target_stats")
    df = load_dataframe(cfg["data"]["dataset"], cfg["data"]["root"])
    _, _, test_df = split_dataframe(
        df,
        mode=cfg["data"]["split"],
        test_frac=cfg["data"]["test_frac"],
        val_frac=cfg["data"]["val_frac"],
        seed=cfg["experiment"]["seed"],
        fold=cfg["data"]["fold"],
        kfold=cfg["data"]["kfold"],
    )
    preds = predict_dataframe(model, test_df, cfg, device, split="test")
    result = evaluate_predictions(preds, target_stats, bootstrap=int(cfg["eval"].get("bootstrap", 0)),
                                  seed=int(cfg["experiment"]["seed"]))
    print("TEST", summarize(result["metrics"]))
    print(json.dumps(result.get("ci", {}), indent=2))
    out_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    save_predictions(preds, os.path.join(out_dir, "predictions_eval.csv"))
    cross = evaluate_cross_dataset(model, cfg, device, target_stats=target_stats)
    print("TID2013", summarize(cross["metrics"]))
    save_predictions(cross["predictions"], os.path.join(out_dir, "predictions_eval_tid2013.csv"))
    with open(os.path.join(out_dir, "eval_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"test": result["metrics"], "test_ci": result.get("ci", {}),
                   "tid2013": cross["metrics"], "tid2013_ci": cross.get("ci", {}),
                   "params": count_parameters(model)}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
