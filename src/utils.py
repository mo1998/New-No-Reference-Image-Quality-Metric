"""Shared helpers for keeping things reproducible.

The course rule is that every headline result comes from a fixed seed, so this
module centralises that seed instead of scattering it across scripts.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

# Course-wide seed: every headline result must be reproducible from this value.
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) from one place."""
    # Python's own RNG (random module)
    random.seed(seed)
    # Makes hash-based iteration (dict/set order) deterministic across runs
    os.environ["PYTHONHASHSEED"] = str(seed)
    # NumPy RNG used by data loading / splitting
    np.random.seed(seed)
    # PyTorch CPU RNG
    torch.manual_seed(seed)
    # PyTorch CUDA RNG (no-op when no GPU is available)
    torch.cuda.manual_seed_all(seed)
