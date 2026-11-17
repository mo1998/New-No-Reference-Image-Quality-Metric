# Architecture specification

This document describes the model families implemented in `src/nriqa/models.py`
and the design rationale for the no-reference setting.  Numerical results and
ablations live in `docs/benchmarks.md` and `docs/final_report.md`.

## 1. MSOSNet — Multi-Scale Orderless Statistical Network

**Parameters:** ~302 K (default configuration).  **Input:** RGB, any size
(trained at 224×224, `[0,1]`).

### Motivation

Without a reference, quality is not a property of a particular spatial layout;
it is a property of how far local image statistics deviate from natural scene
statistics (NSS).  Classic NR-IQA metrics exploit this with handcrafted
multi-scale, locally-normalised coefficients, but use a fixed feature set and a
shallow regressor.  MSOSNet keeps the strong NSS inductive bias and learns
which statistics matter, while discarding most spatial arrangement to reduce
content bias.

### Forward pass

1. **Colour transform.**  RGB → YCbCr with fixed BT.601 coefficients (no
   parameters).  Y carries structure; Cb/Cr carry colour distortions.
2. **Multi-scale local normalisation.**  For scale `s = 0..3`, downsample with
   average pooling by `2^s`, then compute mean/variance-normalised coefficients
   (MSCN):
   \[
   \hat{x}_s = \frac{x_s - \mu_{7\times7}(x_s)}
                    {\sqrt{\sigma^2_{7\times7}(x_s)+\epsilon}}.
   \]
   The normalisation is parameter-free and makes the representation robust to
   content contrast.
3. **Learnable band-pass branch per scale.**  Each scale has a depthwise
   separable block: `1×1 → 5×5 depthwise → 1×1`, BatchNorm + GELU, widths
   `(16, 24, 32, 40)`.  This learns which distortion-sensitive directions to
   keep at each scale (blur, noise, blocking, ringing, …).
4. **Orderless statistical pooling.**  Per scale and channel:
   * first four moments (mean, std, skewness, kurtosis);
   * 2×2 average-pooled means and second moments (coarse layout, 4 bins);
   * 4×4 average-pooled means (16 bins).
   This yields `28 × width` features per scale, with most spatial arrangement
   discarded.
5. **Gated scale fusion.**  Each scale is projected to 64-D; a two-layer
   scoring MLP produces a softmax weight per scale; the fused token is the
   weighted sum.  The gate is content-adaptive: smooth images can rely on
   coarse scales, textured images on fine scales.
6. **Global descriptors.**  A 20-D parameter-free descriptor adds colour
   moments, cross-channel correlations, gradient energy at two scales,
   Laplacian variance and saturation statistics.
7. **Coarse context branch.**  The image is resized to 64×64 and passed through
   three stride-2 convolutions (24/48/96 channels), then average+max pooled.
   This is the only branch that retains coarse layout, and it costs ~30 K
   parameters.
8. **Head.**  LayerNorm → Linear(288→128) → GELU → Dropout → Linear(128→1).
   Optional auxiliary heads predict distortion type (25 classes) and level
   (5 classes) from the fused token.

### Why it is efficient

Almost all computation is depthwise/1×1 convolutions and pooling.  The
parameter budget is concentrated in the 288→128 head, not in a deep ConvNet.
The model has no 3×3 dense convolutions and no attention over pixels.

### Ablation switches

* number of scales (`scales`)
* per-scale widths (`widths`)
* moment vs. spatial-pyramid pooling (code-level flag planned)
* colour/global descriptor on/off (planned)
* coarse context branch on/off (planned)
* auxiliary distortion/level heads on/off (`aux_heads`)

## 2. WaveQNet — Haar sub-band statistics + tiny transformer

**Parameters:** ~76 K.  **Input:** RGB, any even-sized image.

1. Convert to luma.
2. Apply a fixed 2-D Haar decomposition for 3 levels, producing 3 high-frequency
   sub-bands per level plus the coarsest LL band (10 tokens).
3. For each sub-band compute 7 statistics: mean, std, skew, kurtosis, mean
   absolute value, log energy and max absolute value.
4. Project to 64-D, add learned band embeddings, and mix with a 2-layer,
   4-head transformer encoder (`d=64`, FFN 128).
5. Concatenate the pooled token with the 20-D colour descriptor and regress.

The fixed Haar transform is parameter-free and scale-covariant; the transformer
learns cross-band interactions (e.g. noise raises high-frequency energy at all
levels, while blur lowers it).

## 3. TimmRegressor — compact foundation backbone

**Parameters:** 93 M for SigLIP-B/16, 86 M for CLIP-B/16, 22 M for DINOv2-S.

A timm backbone with `num_classes=0`, an internal normalisation buffer (so the
data pipeline can stay in `[0,1]`), and a small LayerNorm→Linear→GELU→Dropout→
Linear head.

* `freeze_blocks=N` freezes the patch embedding and the first N transformer
  blocks.  With only 57–65 reference contents, fine-tuning the last 4 of 12
  blocks is a strong regulariser.
* `freeze_blocks=0` gives full fine-tuning; `freeze_blocks=-1` gives a linear
  probe.
* Optional auxiliary distortion/level heads share the pooled backbone feature.

This family is the **evidence-based hybrid track**: foundation features are
known to carry the strongest quality prior, but the total parameter count is
reported honestly (unlike papers that quote only adapter parameters).

## 4. TinyCNN — from-scratch baseline

A 4-stage CNN (32/64/128/256 channels, stride 2) with average+max pooling and a
128-unit head (~457 K parameters).  It exists to quantify how much of the
performance comes from inductive bias versus pretraining.

## 5. Planned: QStat-Fusion

A gated fusion of the MSOSNet statistical token and a compact foundation token.
The hypothesis is that the two branches are complementary: foundation features
capture semantic/content context and high-level degradation patterns, while the
statistical branch captures low-level NSS deviations.  The fusion gate is
image-adaptive and the total cost is dominated by the backbone.  This model is
only pursued if the ablation shows a consistent gain over either branch alone.

## 6. Design invariants

1. No reference image is ever an input at train or test time.
2. No test content is used for model selection.
3. Total parameters, latency and peak memory are reported, not just trainable
   parameters.
4. Any use of external/foundation pretraining is disclosed in the model name
   and the results table.
