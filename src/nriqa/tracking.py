"""Lightweight local experiment tracking.

No external service is required: every run writes a self-contained directory

    runs/<timestamp>_<name>/
        config.yaml
        environment.json
        metrics.jsonl      # one JSON object per epoch
        summary.json       # final metrics + runtime + parameter counts
        predictions_*.csv  # per-image predictions for error analysis
        best.pt / last.pt
        events.out.tfevents.*   # optional TensorBoard

The JSONL file is append-only and crash-safe, which makes interrupted sweeps
easy to resume and analyse.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

try:  # TensorBoard is optional
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore


class RunLogger:
    def __init__(self, log_dir: str, enable_tensorboard: bool = True):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_path = os.path.join(log_dir, "metrics.jsonl")
        self.summary_path = os.path.join(log_dir, "summary.json")
        self.start_time = time.time()
        self.writer = None
        if enable_tensorboard and SummaryWriter is not None:
            try:
                self.writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb"))
            except Exception:
                self.writer = None

    def log_config(self, cfg: Dict[str, Any], environment: Dict[str, Any]) -> None:
        from .config import save_config

        save_config(cfg, os.path.join(self.log_dir, "config.yaml"))
        with open(os.path.join(self.log_dir, "environment.json"), "w", encoding="utf-8") as fh:
            json.dump(environment, fh, indent=2, sort_keys=True)

    def log_epoch(self, record: Dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("elapsed_s", round(time.time() - self.start_time, 3))
        with open(self.metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        if self.writer is not None:
            step = int(record.get("epoch", 0))
            for key, value in record.items():
                if isinstance(value, (int, float)) and key not in ("epoch", "elapsed_s"):
                    self.writer.add_scalar(key, value, step)

    def log_artifact(self, name: str, payload: Any) -> str:
        path = os.path.join(self.log_dir, name)
        if name.endswith(".json"):
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
        else:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(str(payload))
        return path

    def finish(self, summary: Dict[str, Any]) -> None:
        summary = dict(summary)
        summary.setdefault("wall_time_s", round(time.time() - self.start_time, 3))
        with open(self.summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


# ---------------------------------------------------------------------------
# Aggregate results across runs
# ---------------------------------------------------------------------------
def read_jsonl(path: str):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate_summaries(run_dirs, keys: Optional[list] = None):
    """Collect final summaries from several run directories into a list."""
    import glob

    rows = []
    for pattern in run_dirs:
        for d in sorted(glob.glob(pattern)):
            p = os.path.join(d, "summary.json")
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as fh:
                    row = json.load(fh)
                row["run_dir"] = d
                if keys:
                    row = {k: row.get(k) for k in keys}
                rows.append(row)
    return rows
