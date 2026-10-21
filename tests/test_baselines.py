"""Sanity tests for the full-reference baseline metrics (no dataset needed)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_psnr_capped_for_identical_pair(tmp_path):
    from skimage import io as skio

    from baselines import MAX_PSNR_DB, psnr

    rng = np.random.default_rng(0)
    img = (rng.random((16, 16, 3)) * 255).astype(np.uint8)
    p = tmp_path / 'a.png'
    skio.imsave(str(p), img)
    # identical input -> MSE = 0 -> must clamp to the finite ceiling, not inf
    assert psnr(str(p), str(p)) == MAX_PSNR_DB
