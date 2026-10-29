"""Full-reference baselines: PSNR, SSIM and LPIPS.

PSNR and SSIM run on the RGB channels, since KADID-10k has a whole family of
colour distortions and a grayscale comparison would miss them. PSNR and SSIM are
higher-is-better; LPIPS is a distance, so lower is better. LPIPS uses the
official lpips package with the pretrained AlexNet backbone.
"""

from __future__ import annotations

import numpy as np
from skimage import io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import torch
import lpips as lpips_pkg


def _load_rgb(path):
    """Read an image and return it as float RGB in [0, 1]."""
    # skimage reads PNGs as H x W x C; grayscale images come back as H x W
    img = io.imread(path)
    if img.ndim == 2:  # grayscale -> replicate to 3 channels
        img = np.stack([img] * 3, axis=-1)
    # Drop alpha (or any extra channels) and keep RGB only
    img = img[..., :3]
    # skimage gives uint8 in [0, 255]; scale to [0, 1] float for the metrics
    if img.dtype == np.uint8:
        img = img.astype(np.float64) / 255.0
    return img


MAX_PSNR_DB = 60.0  # ceiling: identical pairs give MSE = 0 -> +inf; 60 dB is already
                    # far below one 8-bit grey level of average error, so clamping
                    # only affects (near-)identical pairs


def psnr(dist_path, ref_path):
    """PSNR in dB, computed on RGB with images normalised to [0, 1].

    Higher is better. Identical pairs would give infinite PSNR, which would
    poison PLCC and RMSE, so the result is capped at MAX_PSNR_DB. SROCC is
    rank-based and would be unaffected, but the cap keeps everything finite.
    """
    d, r = _load_rgb(dist_path), _load_rgb(ref_path)
    # skimage assumes data_range=1.0 for our [0, 1] inputs
    value = float(peak_signal_noise_ratio(r, d, data_range=1.0))
    # Cap infinity so (near-)identical pairs don't poison PLCC/RMSE
    return min(value, MAX_PSNR_DB)


def ssim(dist_path, ref_path):
    """Mean SSIM over the RGB channels. Higher is better."""
    d, r = _load_rgb(dist_path), _load_rgb(ref_path)
    # channel_axis=2 tells SSIM to treat the last axis as colour channels
    return float(structural_similarity(r, d, data_range=1.0, channel_axis=2))


class LPIPSMetric:
    """LPIPS distance between two image files, using the AlexNet backbone.

    Lower means the images look more similar.
    """

    def __init__(self, net='alex', device=None):
        # Prefer GPU when available; otherwise fall back to CPU
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # Official LPIPS with a pretrained AlexNet backbone, frozen for evaluation
        self.model = lpips_pkg.LPIPS(net=net, verbose=False).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, dist_path, ref_path):
        d = self._tensor(dist_path)
        r = self._tensor(ref_path)
        # LPIPS returns a (1, 1, 1, 1) tensor; collapse it to a Python float
        return float(self.model(d, r).item())

    def _tensor(self, path):
        img = _load_rgb(path)                                   # H,W,3 in [0,1]
        t = torch.from_numpy(img).permute(2, 0, 1).float()      # 3,H,W
        # LPIPS expects [-1, 1] inputs in N x C x H x W layout
        return (t * 2.0 - 1.0).unsqueeze(0).to(self.device)     # [-1,1], 1,3,H,W
