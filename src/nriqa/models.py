"""NR-IQA model zoo.

Three families are provided:

1. **From-scratch statistical models** designed for the no-reference setting.
   ``MSOSNet`` combines a fixed multi-scale local-normalisation pyramid with
   learnable band-pass filters, orderless moment pooling and a coarse context
   branch.  ``WaveQNet`` works on a fixed Haar pyramid and mixes sub-band
   statistics with a tiny transformer.
2. **Compact CNN baselines** (``tinycnn``) trained from scratch.
3. **Pretrained-backbone baselines** (MobileNetV3, ResNet-18, EfficientNet-B0)
   used as accuracy/efficiency reference points, not as the proposed method.

All models accept ``[B, 3, H, W]`` tensors in ``[0, 1]`` (unless a pretrained
backbone expects ImageNet normalisation, which is applied by the data
pipeline).  ``forward`` returns a ``[B]`` quality score.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def _local_norm(x: torch.Tensor, k: int = 7, eps: float = 1e-6) -> torch.Tensor:
    """Divisive normalisation: (x - local_mean) / sqrt(local_var + eps)."""
    mu = F.avg_pool2d(x, k, stride=1, padding=k // 2)
    var = F.avg_pool2d(x * x, k, stride=1, padding=k // 2) - mu * mu
    return (x - mu) / torch.sqrt(var.clamp_min(eps))


def _moments(f: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean, std, skewness and kurtosis of each channel, flattened to [B, 4C]."""
    mu = f.mean(dim=(2, 3))
    centred = f - mu[..., None, None]
    var = (centred * centred).mean(dim=(2, 3))
    std = torch.sqrt(var.clamp_min(eps))
    skew = (centred ** 3).mean(dim=(2, 3)) / std.clamp_min(eps) ** 3
    kurt = (centred ** 4).mean(dim=(2, 3)) / std.clamp_min(eps) ** 4
    return torch.cat([mu, std, skew, kurt], dim=1)


def _rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    """Differentiable RGB -> YCbCr using ITU-R BT.601 coefficients."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.564 * (b - y)
    cr = 0.713 * (r - y)
    return torch.cat([y, cb, cr], dim=1)


def _color_statistics(x: torch.Tensor) -> torch.Tensor:
    """Global colour, gradient and saturation descriptors (24 dims)."""
    y = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    mu = y.mean(dim=(2, 3))
    centred = y - mu[..., None, None]
    var = (centred * centred).mean(dim=(2, 3))
    std = torch.sqrt(var.clamp_min(1e-8))
    skew = (centred ** 3).mean(dim=(2, 3)) / std.clamp_min(1e-6) ** 3
    kurt = (centred ** 4).mean(dim=(2, 3)) / std.clamp_min(1e-6) ** 4

    # Channel means/stds in YCbCr
    ycc = _rgb_to_ycbcr(x)
    ycc_mu = ycc.mean(dim=(2, 3))
    ycc_std = ycc.std(dim=(2, 3), unbiased=False)
    # Cross-channel Pearson correlations (3)
    flat = ycc.flatten(2)
    flat = flat - flat.mean(dim=2, keepdim=True)
    denom = flat.std(dim=2, unbiased=False).clamp_min(1e-6)
    corr_ycb = (flat[:, 0] * flat[:, 1]).mean(1) / (denom[:, 0] * denom[:, 1])
    corr_ycr = (flat[:, 0] * flat[:, 2]).mean(1) / (denom[:, 0] * denom[:, 2])
    corr_cbcr = (flat[:, 1] * flat[:, 2]).mean(1) / (denom[:, 1] * denom[:, 2])

    # Gradient energy at two scales (Sobel-like finite differences)
    def _grad_energy(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dx = z[..., :, 1:] - z[..., :, :-1]
        dy = z[..., 1:, :] - z[..., :-1, :]
        g = torch.sqrt(dx[..., :-1, :] ** 2 + dy[..., :, :-1] ** 2 + 1e-8)
        return g.mean(dim=(2, 3)), g.std(dim=(2, 3), unbiased=False)

    g1_mu, g1_std = _grad_energy(y)
    y2 = F.avg_pool2d(y, 2)
    g2_mu, g2_std = _grad_energy(y2)
    lap_mean = (y[..., 1:-1, 1:-1] * 4 - y[..., :-2, 1:-1] - y[..., 2:, 1:-1]
                - y[..., 1:-1, :-2] - y[..., 1:-1, 2:]).var(dim=(2, 3), unbiased=False)

    # Saturation
    mx = x.max(dim=1, keepdim=True).values
    mn = x.min(dim=1, keepdim=True).values
    sat = (mx - mn) / mx.clamp_min(1e-6)
    sat_mu = sat.mean(dim=(2, 3))
    sat_std = sat.std(dim=(2, 3), unbiased=False)

    feats = [
        mu, std, skew, kurt,
        ycc_mu[:, 0:1], ycc_mu[:, 1:2], ycc_mu[:, 2:3],
        ycc_std[:, 0:1], ycc_std[:, 1:2], ycc_std[:, 2:3],
        corr_ycb[:, None], corr_ycr[:, None], corr_cbcr[:, None],
        g1_mu, g1_std, g2_mu, g2_std,
        lap_mean, sat_mu, sat_std,
    ]
    return torch.cat(feats, dim=1)  # 20 dims


# ---------------------------------------------------------------------------
# From-scratch baselines
# ---------------------------------------------------------------------------
class TinyCNN(nn.Module):
    """A compact 4-stage CNN with average+max pooling (from scratch)."""

    def __init__(self, width: int = 32, dropout: float = 0.1, **_):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv2d(3, w, 5, stride=2, padding=2), nn.BatchNorm2d(w), nn.GELU(),
            nn.Conv2d(w, w * 2, 3, stride=2, padding=1), nn.BatchNorm2d(w * 2), nn.GELU(),
            nn.Conv2d(w * 2, w * 4, 3, stride=2, padding=1), nn.BatchNorm2d(w * 4), nn.GELU(),
            nn.Conv2d(w * 4, w * 8, 3, stride=2, padding=1), nn.BatchNorm2d(w * 8), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(w * 16, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        pooled = torch.cat(
            [F.adaptive_avg_pool2d(f, 1).flatten(1), F.adaptive_max_pool2d(f, 1).flatten(1)], dim=1
        )
        return self.head(pooled).squeeze(1)


# ---------------------------------------------------------------------------
# MSOSNet: Multi-Scale Orderless Statistical Network
# ---------------------------------------------------------------------------
class ScaleBranch(nn.Module):
    """Depthwise band-pass feature extractor applied to an MSCN map."""

    def __init__(self, in_ch: int, width: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, width, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.Conv2d(width, width, 5, padding=2, groups=width, bias=False),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class MSOSNet(nn.Module):
    """Multi-Scale Orderless Statistical Network.

    Motivation
    ----------
    Without a reference image, quality is not a property of a particular pixel
    layout but of *how far local image statistics deviate from natural scene
    statistics*.  The architecture therefore discards most spatial arrangement:

    * a fixed multi-scale local-normalisation pyramid yields MSCN-like
      coefficients (the classic NSS representation) with no parameters;
    * a small learnable depthwise band-pass filter per scale selects the
      distortion-sensitive directions;
    * orderless moment pooling (mean/std/skew/kurtosis) plus a coarse
      spatial pyramid summarises each scale;
    * a gated scale-fusion layer learns which scales matter for the image;
    * a tiny colour/gradient descriptor and a coarse context branch keep
      global colour distortions and layout cues.

    Parameters are deliberately concentrated in the small fusion head rather
    than in large convolutional stacks, giving a high accuracy/compute ratio.
    """

    def __init__(
        self,
        widths: Tuple[int, ...] = (16, 24, 32, 40),
        scales: int = 4,
        embed_dim: int = 64,
        context_width: int = 24,
        dropout: float = 0.1,
        num_distortions: int = 25,
        num_levels: int = 5,
        aux_heads: bool = False,
        **_,
    ):
        super().__init__()
        self.scales = scales
        self.widths = widths[:scales]
        self.aux_heads = aux_heads
        self.branches = nn.ModuleList(
            [ScaleBranch(3, w) for w in self.widths]
        )
        # stats dim: 4 moments + 2x2 mean + 2x2 std + 4x4 mean = 4w + 4w + 4w + 16w
        self.stat_dims = [28 * w for w in self.widths]
        self.scale_proj = nn.ModuleList(
            [nn.Sequential(nn.Linear(d, embed_dim), nn.GELU()) for d in self.stat_dims]
        )
        self.scale_score = nn.Sequential(nn.Linear(embed_dim, 32), nn.GELU(), nn.Linear(32, 1))
        self.color_proj = nn.Sequential(nn.Linear(20, 32), nn.GELU())
        # Coarse context branch: 64x64 -> 8x8
        cw = context_width
        self.context = nn.Sequential(
            nn.Conv2d(3, cw, 3, stride=2, padding=1), nn.BatchNorm2d(cw), nn.GELU(),
            nn.Conv2d(cw, cw * 2, 3, stride=2, padding=1), nn.BatchNorm2d(cw * 2), nn.GELU(),
            nn.Conv2d(cw * 2, cw * 4, 3, stride=2, padding=1), nn.BatchNorm2d(cw * 4), nn.GELU(),
        )
        ctx_dim = cw * 8
        self.out_dim = embed_dim + 32 + ctx_dim
        self.head = nn.Sequential(
            nn.LayerNorm(self.out_dim),
            nn.Linear(self.out_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        if aux_heads:
            self.dist_head = nn.Linear(embed_dim, num_distortions)
            self.level_head = nn.Linear(embed_dim, num_levels)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ycc = _rgb_to_ycbcr(x)
        tokens: List[torch.Tensor] = []
        for s in range(self.scales):
            xs = ycc if s == 0 else F.avg_pool2d(ycc, 2 ** s)
            mscn = _local_norm(xs, k=7)
            f = self.branches[s](mscn)
            stats = torch.cat(
                [
                    _moments(f),
                    F.adaptive_avg_pool2d(f, 2).flatten(1),
                    F.adaptive_avg_pool2d(f * f, 2).flatten(1),
                    F.adaptive_avg_pool2d(f, 4).flatten(1),
                ],
                dim=1,
            )
            tokens.append(self.scale_proj[s](stats))
        tok = torch.stack(tokens, dim=1)              # [B, S, D]
        w = torch.softmax(self.scale_score(tok).squeeze(-1), dim=1)
        fused = (tok * w[..., None]).sum(dim=1)        # [B, D]

        color = self.color_proj(_color_statistics(x))
        ctx_in = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        ctx = self.context(ctx_in)
        ctx = torch.cat(
            [F.adaptive_avg_pool2d(ctx, 1).flatten(1), F.adaptive_max_pool2d(ctx, 1).flatten(1)],
            dim=1,
        )
        feat = torch.cat([fused, color, ctx], dim=1)
        return feat, fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat, _ = self.forward_features(x)
        return self.head(feat).squeeze(1)

    def forward_aux(self, x: torch.Tensor):
        feat, fused = self.forward_features(x)
        score = self.head(feat).squeeze(1)
        aux = {}
        if self.aux_heads:
            aux["dist_logits"] = self.dist_head(fused)
            aux["level_logits"] = self.level_head(fused)
        return score, aux


# ---------------------------------------------------------------------------
# WaveQNet: Haar sub-band statistics + tiny transformer
# ---------------------------------------------------------------------------
def _haar2(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """One-level 2D Haar decomposition; returns (LL, [LH, HL, HH])."""
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a + b - c - d) * 0.5
    hl = (a - b + c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll, torch.cat([lh, hl, hh], dim=1)


class WaveQNet(nn.Module):
    """Frequency-domain NR-IQA from fixed Haar sub-bands.

    Distortions such as blur, noise, blocking and ringing have distinct
    sub-band energy signatures.  The network summarises each sub-band with
    robust statistics (including a log-energy map), adds learnable band and
    scale embeddings, and mixes them with a two-layer transformer.
    """

    def __init__(
        self,
        levels: int = 3,
        embed_dim: int = 64,
        depth: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
        num_distortions: int = 25,
        num_levels: int = 5,
        aux_heads: bool = False,
        **_,
    ):
        super().__init__()
        self.levels = levels
        self.aux_heads = aux_heads
        n_bands = levels * 3 + 1
        self.band_embed = nn.Parameter(torch.zeros(1, n_bands, embed_dim))
        self.stat_proj = nn.Linear(7, embed_dim)
        nn.init.trunc_normal_(self.band_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim + 20, 96), nn.GELU(), nn.Dropout(dropout), nn.Linear(96, 1)
        )
        if aux_heads:
            self.dist_head = nn.Linear(embed_dim, num_distortions)
            self.level_head = nn.Linear(embed_dim, num_levels)

    @staticmethod
    def _band_stats(b: torch.Tensor) -> torch.Tensor:
        mu = b.mean(dim=(2, 3))
        centred = b - mu[..., None, None]
        std = centred.pow(2).mean(dim=(2, 3)).sqrt()
        skew = centred.pow(3).mean(dim=(2, 3)) / std.clamp_min(1e-6).pow(3)
        kurt = centred.pow(4).mean(dim=(2, 3)) / std.clamp_min(1e-6).pow(4)
        abs_mean = b.abs().mean(dim=(2, 3))
        log_energy = torch.log1p((b * b).mean(dim=(2, 3)))
        return torch.cat([mu, std, skew, kurt, abs_mean, log_energy, b.flatten(2).amax(dim=2)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        tokens: List[torch.Tensor] = []
        cur = y
        for _ in range(self.levels):
            # Pad odd spatial sizes before the 2x2 decomposition
            if cur.shape[-2] % 2 or cur.shape[-1] % 2:
                cur = F.pad(cur, (0, cur.shape[-1] % 2, 0, cur.shape[-2] % 2), mode="replicate")
            ll, highs = _haar2(cur)
            for i in range(3):
                tokens.append(self._band_stats(highs[:, i : i + 1]))
            cur = ll
        tokens.append(self._band_stats(cur))  # LL at the coarsest level
        t = self.stat_proj(torch.stack(tokens, dim=1)) + self.band_embed
        t = self.encoder(t)
        t = self.norm(t.mean(dim=1))
        color = _color_statistics(x)
        return self.head(torch.cat([t, color], dim=1)).squeeze(1)

    def forward_aux(self, x: torch.Tensor):
        # The auxiliary heads are attached to the pooled representation; this
        # recomputes the encoder cheaply and is only used when aux loss > 0.
        score = self.forward(x)
        aux = {}
        if self.aux_heads:
            # Fall back to zeros is not useful; attach heads in forward instead.
            raise RuntimeError("WaveQNet aux heads are not wired; use MSOSNet for aux training")
        return score, aux


# ---------------------------------------------------------------------------
# Pretrained compact backbones
# ---------------------------------------------------------------------------
class PretrainedBackbone(nn.Module):
    """Wrap a torchvision classifier as a scalar quality regressor."""

    def __init__(self, name: str = "mobilenet_v3_small", pretrained: bool = True,
                 freeze: bool = False, dropout: float = 0.1, **_):
        super().__init__()
        from torchvision import models

        name = name.lower()
        weights = "DEFAULT" if pretrained else None
        if name in ("mobilenet_v3_small", "mobilenetv3_small"):
            m = models.mobilenet_v3_small(weights=weights)
            in_dim = m.classifier[-1].in_features
            m.classifier[-1] = nn.Linear(in_dim, 1)
            self.backbone = m
            self.feature_dim = in_dim
        elif name in ("mobilenet_v3_large",):
            m = models.mobilenet_v3_large(weights=weights)
            in_dim = m.classifier[-1].in_features
            m.classifier[-1] = nn.Linear(in_dim, 1)
            self.backbone = m
            self.feature_dim = in_dim
        elif name in ("resnet18",):
            m = models.resnet18(weights=weights)
            in_dim = m.fc.in_features
            m.fc = nn.Linear(in_dim, 1)
            self.backbone = m
            self.feature_dim = in_dim
        elif name in ("resnet50",):
            m = models.resnet50(weights=weights)
            in_dim = m.fc.in_features
            m.fc = nn.Linear(in_dim, 1)
            self.backbone = m
            self.feature_dim = in_dim
        elif name in ("efficientnet_b0",):
            m = models.efficientnet_b0(weights=weights)
            in_dim = m.classifier[-1].in_features
            m.classifier[-1] = nn.Linear(in_dim, 1)
            self.backbone = m
            self.feature_dim = in_dim
        else:
            raise ValueError(f"Unsupported pretrained backbone: {name}")
        self.freeze = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # The replaced classification head must stay trainable
            for p in self.backbone.get_submodule(self._head_name()).parameters():
                p.requires_grad = True

    def _head_name(self) -> str:
        if hasattr(self.backbone, "classifier"):
            return "classifier"
        return "fc"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(1)




# ---------------------------------------------------------------------------
# Timm foundation backbones (partial fine-tuning)
# ---------------------------------------------------------------------------
class TimmRegressor(nn.Module):
    """A compact foundation backbone with an internal normalisation layer.

    ``freeze_blocks`` freezes the patch embedding and the first N transformer
    blocks, which is the key regulariser when only ~57 reference contents are
    available.  Normalisation statistics come from the backbone's own data
    config, so the data pipeline can keep ``norm: none``.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_siglip_224",
        pretrained: bool = True,
        freeze_blocks: int = 8,
        drop_path: float = 0.1,
        head_hidden: int = 256,
        dropout: float = 0.2,
        dynamic_img_size: bool = False,
        aux_heads: bool = False,
        num_distortions: int = 25,
        num_levels: int = 5,
        distill_dim: int = 0,
        **_,
    ):
        super().__init__()
        import timm

        kwargs: Dict[str, Any] = {
            "pretrained": pretrained,
            "num_classes": 0,
            "drop_path_rate": drop_path,
        }
        if dynamic_img_size:
            kwargs["dynamic_img_size"] = True
        self.backbone = timm.create_model(model_name, **kwargs)
        data_cfg = timm.data.resolve_model_data_config(self.backbone)
        mean = torch.tensor(data_cfg["mean"], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(data_cfg["std"], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("input_mean", mean)
        self.register_buffer("input_std", std)
        self.feat_dim = int(getattr(self.backbone, "num_features", 0)) or int(
            getattr(self.backbone, "embed_dim", 0)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        self.freeze_blocks = int(freeze_blocks)
        self.aux_heads = bool(aux_heads)
        self.distill_dim = int(distill_dim)
        self.feature_proj = (
            nn.Linear(self.feat_dim, self.distill_dim, bias=False)
            if self.distill_dim > 0 else None
        )
        if self.aux_heads:
            self.dist_head = nn.Linear(self.feat_dim, num_distortions)
            self.level_head = nn.Linear(self.feat_dim, num_levels)
        if self.freeze_blocks != 0:
            self._apply_freezing()

    def _apply_freezing(self) -> None:
        frozen_prefixes = ("patch_embed", "pos_embed", "cls_token", "reg_token", "dist_token")
        for name, p in self.backbone.named_parameters():
            p.requires_grad_(True)
            if name.startswith(frozen_prefixes):
                p.requires_grad_(False)
            elif name.startswith("blocks."):
                idx = int(name.split(".")[1])
                if self.freeze_blocks < 0 or idx < self.freeze_blocks:
                    p.requires_grad_(False)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.input_mean) / self.input_std
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x)).squeeze(1)

    def forward_aux(self, x: torch.Tensor):
        f = self.forward_features(x)
        score = self.head(f).squeeze(1)
        aux = {"feat": f}
        if self.feature_proj is not None:
            aux["proj"] = self.feature_proj(f)
        if self.aux_heads:
            aux["dist_logits"] = self.dist_head(f)
            aux["level_logits"] = self.level_head(f)
        return score, aux




# ---------------------------------------------------------------------------
# QStatFusion: statistical + foundation gated fusion
# ---------------------------------------------------------------------------
class QStatFusion(nn.Module):
    """Gated fusion of NSS statistics and a compact foundation backbone.

    The two branches are complementary by construction:

    * the foundation branch (SigLIP/CLIP) provides semantic context and
      high-level degradation patterns but suppresses fine perceptual cues;
    * the MSOSNet branch provides explicit multi-scale local-normalisation
      statistics that are sensitive to blur, noise, blocking and ringing.

    A per-image sigmoid gate decides how much to trust each branch.  The
    foundation branch can be partially frozen exactly like ``TimmRegressor``.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_siglip_224",
        pretrained: bool = True,
        freeze_blocks: int = 8,
        drop_path: float = 0.1,
        stats_widths: Tuple[int, ...] = (16, 24, 32, 40),
        stats_scales: int = 4,
        fusion_dim: int = 256,
        dropout: float = 0.2,
        aux_heads: bool = False,
        num_distortions: int = 25,
        num_levels: int = 5,
        distill_dim: int = 0,
        **_,
    ):
        super().__init__()
        self.foundation = TimmRegressor(
            model_name=model_name,
            pretrained=pretrained,
            freeze_blocks=freeze_blocks,
            drop_path=drop_path,
            head_hidden=256,
            dropout=dropout,
            aux_heads=False,
        )
        self.stats = MSOSNet(
            widths=stats_widths, scales=stats_scales, embed_dim=64, aux_heads=False
        )
        stats_dim = int(self.stats.out_dim)
        self.foundation_proj = nn.Sequential(
            nn.LayerNorm(self.foundation.feat_dim),
            nn.Linear(self.foundation.feat_dim, fusion_dim),
            nn.GELU(),
        )
        self.stats_proj = nn.Sequential(
            nn.LayerNorm(stats_dim),
            nn.Linear(stats_dim, fusion_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim // 2), nn.GELU(),
            nn.Linear(fusion_dim // 2, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, 128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, 1),
        )
        self.aux_heads = bool(aux_heads)
        if self.aux_heads:
            self.dist_head = nn.Linear(fusion_dim, num_distortions)
            self.level_head = nn.Linear(fusion_dim, num_levels)

    def forward_features(self, x: torch.Tensor):
        f_b = self.foundation.forward_features(x)
        f_s_raw, _ = self.stats.forward_features(x)
        f_b = self.foundation_proj(f_b)
        f_s = self.stats_proj(f_s_raw)
        g = torch.sigmoid(self.gate(torch.cat([f_b, f_s], dim=1)))
        fused = g * f_b + (1.0 - g) * f_s
        return fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x)).squeeze(1)

    def forward_aux(self, x: torch.Tensor):
        f = self.forward_features(x)
        score = self.head(f).squeeze(1)
        aux = {"feat": f}
        if self.feature_proj is not None:
            aux["proj"] = self.feature_proj(f)
        if self.aux_heads:
            aux["dist_logits"] = self.dist_head(f)
            aux["level_logits"] = self.level_head(f)
        return score, aux




# ---------------------------------------------------------------------------
# HuggingFace SigLIP2 vision encoder
# ---------------------------------------------------------------------------
class Siglip2Regressor(nn.Module):
    """SigLIP2 vision encoder with a small quality head.

    The text tower is discarded; only the 93 M-parameter vision encoder is kept.
    Input normalisation (mean=std=0.5) is applied inside the model so the data
    pipeline can stay in ``[0,1]``.  The patch embedding and the first
    ``freeze_blocks`` transformer blocks are frozen by default.
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        freeze_blocks: int = 8,
        head_hidden: int = 256,
        dropout: float = 0.2,
        aux_heads: bool = False,
        num_distortions: int = 25,
        num_levels: int = 5,
        distill_dim: int = 0,
        **_,
    ):
        super().__init__()
        import gc

        from transformers import SiglipModel

        full = SiglipModel.from_pretrained(model_name)
        self.vision = full.vision_model
        del full
        gc.collect()

        self.target_size = int(self.vision.config.image_size)
        self.feat_dim = int(self.vision.config.hidden_size)
        self.register_buffer("input_mean", torch.tensor(0.5).view(1, 1, 1, 1))
        self.register_buffer("input_std", torch.tensor(0.5).view(1, 1, 1, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        self.aux_heads = bool(aux_heads)
        self.distill_dim = int(distill_dim)
        self.feature_proj = (
            nn.Linear(self.feat_dim, self.distill_dim, bias=False)
            if self.distill_dim > 0 else None
        )
        if self.aux_heads:
            self.dist_head = nn.Linear(self.feat_dim, num_distortions)
            self.level_head = nn.Linear(self.feat_dim, num_levels)
        self.freeze_blocks = int(freeze_blocks)
        self._apply_freezing()

    def _apply_freezing(self) -> None:
        for p in self.vision.parameters():
            p.requires_grad_(True)
            p.requires_grad_(False) if False else None
        for name, p in self.vision.named_parameters():
            if name.startswith("embeddings"):
                p.requires_grad_(False)
            elif name.startswith("encoder.layers."):
                idx = int(name.split(".")[2])
                if self.freeze_blocks < 0 or idx < self.freeze_blocks:
                    p.requires_grad_(False)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.input_mean) / self.input_std
        return self.vision(pixel_values=x).pooler_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x)).squeeze(1)

    def forward_aux(self, x: torch.Tensor):
        f = self.forward_features(x)
        score = self.head(f).squeeze(1)
        aux = {"feat": f}
        if self.feature_proj is not None:
            aux["proj"] = self.feature_proj(f)
        if self.aux_heads:
            aux["dist_logits"] = self.dist_head(f)
            aux["level_logits"] = self.level_head(f)
        return score, aux


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_MODEL_REGISTRY = {
    "tinycnn": TinyCNN,
    "msosnet": MSOSNet,
    "waveqnet": WaveQNet,
    "mobilenet_v3_small": PretrainedBackbone,
    "mobilenetv3_small": PretrainedBackbone,
    "mobilenet_v3_large": PretrainedBackbone,
    "resnet18": PretrainedBackbone,
    "resnet50": PretrainedBackbone,
    "efficientnet_b0": PretrainedBackbone,
    "timm_regressor": TimmRegressor,
    "qstat_fusion": QStatFusion,
    "siglip2_regressor": Siglip2Regressor,
}


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """Build a model from the ``model`` section of a config."""
    mcfg = cfg["model"]
    name = mcfg["name"].lower()
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}; available: {sorted(_MODEL_REGISTRY)}")
    kwargs = dict(mcfg.get("kwargs") or {})
    if name in ("mobilenet_v3_small", "mobilenetv3_small", "mobilenet_v3_large",
                "resnet18", "resnet50", "efficientnet_b0"):
        kwargs.setdefault("pretrained", bool(mcfg.get("pretrained", True)))
    if name in ("mobilenet_v3_small", "mobilenetv3_small", "mobilenet_v3_large",
                "resnet18", "resnet50", "efficientnet_b0"):
        model = _MODEL_REGISTRY[name](name=name, **kwargs)
    elif name == "timm_regressor":
        model = _MODEL_REGISTRY[name](**kwargs)
    else:
        model = _MODEL_REGISTRY[name](**kwargs)
    return model
