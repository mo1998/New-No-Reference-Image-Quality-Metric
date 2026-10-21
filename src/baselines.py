"""Full-reference baseline metrics for Project 28: PSNR, SSIM, LPIPS.

Conventions (documented for the report):
- PSNR and SSIM are computed on RGB channels: KADID-10k contains a family of
  colour distortions, and a grayscale-only comparison would be blind to them.
- PSNR/SSIM are higher-better; LPIPS is a distance (lower = more similar).
- LPIPS uses the official `lpips` package with the pretrained AlexNet backbone.
"""

from __future__ import annotations

import numpy as np
from skimage import io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import torch
import lpips as lpips_pkg


def _load_rgb(path):
    """Load an image as float RGB in [0, 1]."""
    img = io.imread(path)
    if img.ndim == 2:  # grayscale -> replicate to 3 channels
        img = np.stack([img] * 3, axis=-1)
    img = img[..., :3]
    if img.dtype == np.uint8:
        img = img.astype(np.float64) / 255.0
    return img


MAX_PSNR_DB = 60.0  # ceiling: identical pairs give MSE = 0 -> +inf; 60 dB is already
                    # far below one 8-bit grey level of average error, so clamping
                    # only affects (near-)identical pairs


def psnr(dist_path, ref_path):
    """PSNR in dB on RGB (images normalised to [0, 1]). Higher = better.

    Capped at MAX_PSNR_DB so identical pairs return a finite value (an
    infinite value would poison PLCC/RMSE; SROCC is rank-based and unaffected).
    """
    d, r = _load_rgb(dist_path), _load_rgb(ref_path)
    value = float(peak_signal_noise_ratio(r, d, data_range=1.0))
    return min(value, MAX_PSNR_DB)


def ssim(dist_path, ref_path):
    """Mean SSIM over RGB channels. Higher = better."""
    d, r = _load_rgb(dist_path), _load_rgb(ref_path)
    return float(structural_similarity(r, d, data_range=1.0, channel_axis=2))


class LPIPSMetric:
    """LPIPS (AlexNet) distance between two image files. Lower = more similar."""

    def __init__(self, net='alex', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = lpips_pkg.LPIPS(net=net, verbose=False).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, dist_path, ref_path):
        d = self._tensor(dist_path)
        r = self._tensor(ref_path)
        return float(self.model(d, r).item())

    def _tensor(self, path):
        img = _load_rgb(path)                                   # H,W,3 in [0,1]
        t = torch.from_numpy(img).permute(2, 0, 1).float()      # 3,H,W
        return (t * 2.0 - 1.0).unsqueeze(0).to(self.device)     # [-1,1], 1,3,H,W
