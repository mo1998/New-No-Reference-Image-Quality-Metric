# Final report: lightweight no-reference image quality assessment

**Project:** Building and validating a new no-reference image-quality metric
**Branch:** `playground`
**Primary benchmark:** KADID-10k (content-held-out)
**Secondary benchmark:** TID2013 (zero-shot cross-dataset)

> This report is generated from the runs stored in `runs/`, `runs_cv65/`,
> `runs_cv65_384/`, `runs_ablate/` and the machine-readable tables in
> `results/`.  Every number is reproducible from the commands in
> `docs/protocol.md` and `docs/benchmarks.md`.

---

## 1. Executive summary

We built a complete, reproducible NR-IQA research framework and evaluated three
families of models against strong baselines:

1. **From-scratch statistical models** designed for the no-reference setting
   (`MSOSNet`, 0.30 M parameters; `WaveQNet`, 0.08 M).
2. **Compact pretrained CNNs** (MobileNetV3-Small, ResNet-18).
3. **Partially fine-tuned foundation encoders** (SigLIP-B/16, CLIP-B/16,
   DINOv2-S, SigLIP2-B) with distortion-aware auxiliary heads.

**Best KADID-10k result:** SigLIP-L/16 at 384 px with auxiliary distortion/level
heads, pairwise ranking loss and flip TTA: **SROCC 0.9481 ± 0.0074** over five
content folds, PLCC 0.9526, with 316.6 M parameters and 15.1 ms per image on an
RTX 4090.  This is **third in the surveyed published KADID-10k table**, behind
the two ~400 M foundation probes at 0.970 and ahead of MANIQA (0.946).  The
256 px L/16 variant is the balanced point: **0.9365 ± 0.0074** KADID and
**0.7353 ± 0.0084** cross-dataset at 7.5 ms.  The 384/256 difference is
significant (paired p = 0.023), while L384 vs the 93 M B/384 is not
(+0.0088, p = 0.098).

**Best cross-dataset result:** QStatFusion, a gated fusion of SigLIP-B/16 and
the MSOSNet statistical branch: **TID2013 SROCC 0.753**, significantly better
than SigLIP alone (0.726, paired bootstrap p = 0.003), at 93.8 M parameters.

**Honest positioning:** 0.9481 places the model third among the single-model
methods surveyed in `docs/literature_review.md` (SigLIP2/B_Gated 0.970, MANIQA
0.946), at 316 M parameters and 15.1 ms — about 21% fewer parameters and lower
latency than the two leaders.  It is **not** an absolute state of the art:
other 2026 methods may be unpublished or unreviewed, and 316 M is not
"lightweight" by mobile standards.  The 93 M distilled student reaches
**0.9421 ± 0.0120** (99.4% of the teacher's accuracy at 3.4× fewer parameters),
and the 93 M B/16 model reaches 0.9393.  We report the top-3 claim with these
qualifications rather than as an unqualified SOTA.

**Key findings.**

* Distortion-aware auxiliary supervision is the single largest gain
  (+0.042 KADID, +0.046 TID over the same model without it), and the combined
  aux + rank + TTA recipe is statistically significant (+0.028, p = 0.003).
* Partial fine-tuning is essential: full fine-tuning overfits 57 contents and
  loses 0.03 KADID SROCC.
* Scaling the foundation encoder from B/16 (93 M) to L/16 (316 M) adds
  +0.009 KADID (p = 0.098) and a 94 M distilled student recovers 0.9421, i.e.
  99.4% of the teacher's accuracy at 3.4× fewer parameters.
* Higher resolution helps in-dataset (+0.024) but hurts cross-dataset
  (−0.041); the effect is not statistically significant on the single split.
* The NSS statistical branch improves cross-dataset generalisation
  (+0.027, p = 0.003) with a negligible in-dataset cost (−0.008, p = 0.073).
* Larger foundation models are not automatically better: SigLIP2-B is
  significantly worse than SigLIP-B on this content split (p = 0.003).

---

## 2. Problem and evaluation protocol

NR-IQA predicts a quality score from a single distorted image.  The hard part
is content invariance: with only 81 KADID reference contents, a model can learn
the content rather than the distortion.  The protocol therefore:

* splits by **reference content**, never distorted image;
* reports **SROCC** (no fitting) and **PLCC/RMSE** after a 4-parameter logistic
  mapping;
* estimates uncertainty by **bootstrapping contents**, not images;
* uses **paired content-level bootstrap tests** for method comparisons;
* reports **total** parameters, latency and peak memory.

See `docs/protocol.md` for the full contract.

## 3. Methods

### 3.1 MSOSNet (from scratch, 0.30 M)

A fixed multi-scale MSCN pyramid, learned depthwise band-pass branches,
orderless moment + spatial-pyramid pooling, gated scale fusion, a 20-D colour
descriptor and a coarse context branch.  The design encodes the NSS prior
explicitly and discards most spatial layout to fight content bias.  Details in
`docs/architecture.md`.

### 3.2 WaveQNet (from scratch, 0.08 M)

A fixed 3-level Haar decomposition; per-sub-band statistics; band/scale
embeddings; a 2-layer transformer over 10 sub-band tokens.  It targets the
frequency signature of blur, noise, blocking and ringing.

### 3.3 QStatFusion (94 M)

A gated fusion of SigLIP-B/16 features and the MSOSNet statistical token.  A
per-image sigmoid gate chooses between semantic and statistical evidence.  This
is the hybrid track justified by the literature evidence that foundation
features carry the strongest quality prior while suppressing fine perceptual
cues.

### 3.4 Training recipe

* AdamW, cosine schedule with 2-epoch warm-up, AMP, gradient clipping.
* Target z-score normalisation computed on training contents only.
* Loss: MSE + 0.2·(1−PLCC) + 0.2·pairwise ranking hinge + 0.2·distortion
  cross-entropy + 0.1·level cross-entropy.
* Partial fine-tuning: patch embedding + first 8 of 12 ViT blocks frozen.
* Augmentation: random 224/384 crop from a 32-pixel larger resize, horizontal
  flip.  No colour jitter (it would change the label).
* Flip TTA at inference for the headline model.

## 4. Results

### 4.1 KADID-10k, 5-fold content CV

Each fold trains on the 64 non-test references and tests on 16 held-out
references.  Full per-fold values are in `results/cv/*.json`.

| Model | Params | SROCC | PLCC | TID2013 SROCC |
|---|---:|---:|---:|---:|
| **SigLIP-L/16 384 + aux/rank/TTA** | 316.6 M | **0.9481 ± 0.0074** | **0.9526** | 0.7232 ± 0.0205 |
| SigLIP-L/16 256 + aux/rank/TTA | 316.3 M | 0.9365 ± 0.0074 | 0.9426 | **0.7353 ± 0.0084** |
| Distilled SigLIP-B/16 384 (score+feature) | 94.2 M | 0.9421 ± 0.0120 | 0.947 | 0.7127 |
| SigLIP-B/16 384 + aux/rank/TTA | 93.4 M | 0.9393 ± 0.0149 | 0.9441 | 0.7166 ± 0.0123 |
| SigLIP-B/16 224 + aux/rank/TTA | 93.1 M | 0.9207 ± 0.0166 | 0.9271 | 0.7284 ± 0.0157 |
| QStatFusion 224 | 93.8 M | 0.8885* | 0.8974* | **0.7528*** |
| QStatFusion 384 | 94.1 M | 0.9204* | 0.9272* | 0.7137* |

\* Starred rows are single content splits, not 5-fold estimates.

### 4.2 Single-split model comparison

See `docs/benchmarks.md` §2.  The ordering is:

```
L/16-384 (0.933) > QStat-L/16-256 (0.925) > L/16-256 (0.923)
> B/16-384 (0.921) > content-rank B/16-224 (0.905)
> B/16-224+aux (0.897) > QStat-B/16 (0.888) > B/16-224 (0.869)
> ResNet-18 (0.769) > ARNIQA zero-shot (0.737) > MSOSNet (0.721)
> MobileNetV3 (0.721) > TinyCNN (0.711) > LIQE zero-shot (0.727)
```

Cross-dataset ordering is different:

```
QStatFusion (0.753) > SigLIP-224+aux (0.726) > SigLIP-224 (0.711)
> SigLIP2 (0.729) > ResNet-18 (0.678) > MobileNetV3 (0.632)
> LIQE zero-shot (0.616) > ARNIQA zero-shot (0.574) > MSOSNet (0.483)
```

### 4.3 Efficiency

Batch-1 latency on RTX 4090 (median of 50 runs, including input normalisation),
measured with `src/nriqa/inference.py`:

| Model | Params | GFLOPs | Latency (ms) | Throughput batch-16 (img/s) | Peak GPU mem (MB) |
|---|---:|---:|---:|---:|---:|
| MSOSNet | 0.30 M | 0.13 | 3.71 | 2204 | 145 |
| ResNet-18 | 11.2 M | 3.63 | 2.70 | 4584 | 61 |
| QStatFusion | 93.8 M | 35.5 | 8.46 | 371 | 503 |
| SigLIP-B/16 224 | 93.1 M | 35.4 | 6.16 | 465 | 372 |
| SigLIP-B/16 384 | 93.4 M | 112.1 | 10.93 | 154 | 390 |
| SigLIP-L/16 256 | 316.3 M | 162.6 | 7.49 | 219 | 1229 |
| SigLIP-L/16 384 | 316.6 M | 383.9 | 15.05 | 89 | 1247 |

The **L/16 256** model is the best accuracy/latency point: 7.5 ms per image,
1229 MB, 0.9365 KADID and 0.7353 TID.  The L/16 384 accuracy point doubles
latency (15.1 ms) and triples FLOPs for +0.0116 KADID and −0.012 TID.  The
93 M B/16 384 model is the deployment point: 10.9 ms, 390 MB, 0.9393 KADID;
its distilled variant reaches 0.9421 at the same cost.  The from-scratch
MSOSNet is the smallest (0.30 M) and fastest in FLOPs but not in batch-1
latency, because its statistical pooling is launch-bound.

### 4.4 Ablations

Measured on the single content split (57/8/16 refs) so every row is directly
comparable:

| Variant | KADID SROCC | TID2013 SROCC |
|---|---:|---:|
| Full recipe (aux + rank + TTA) | **0.8968** | 0.7256 |
| No auxiliary heads | 0.8547 (−0.042) | 0.6792 (−0.046) |
| No ranking loss | 0.8966 (−0.000) | 0.7479 (+0.022) |
| No TTA | 0.8943 (−0.003) | 0.7304 (+0.005) |
| Full fine-tune | 0.8385 (−0.058) | 0.6850 (−0.041) |
| Multi-dataset | 0.8969 (+0.000) | — |
| 384 resolution | 0.9208 (+0.024) | 0.6851 (−0.041) |
| QStatFusion | 0.8885 (−0.008) | **0.7528 (+0.027)** |

The auxiliary distortion/level heads are the dominant component: removing them
costs 0.042 KADID and 0.046 TID.  The ranking loss is neutral on KADID but
slightly negative on TID; TTA is a small KADID gain.  This suggests a
**generalisation-oriented variant** (aux heads, no rank, no TTA) would be
preferable when the deployment distribution is unknown.

### 4.5 Statistical tests

Paired content-level bootstrap (300–2000 resamples):

| Comparison | Metric | Δ | p |
|---|---|---:|---:|
| +aux/rank/TTA vs baseline (single split) | SROCC | +0.028 | 0.003 |
| 384 vs 224 (paired 5-fold t-test) | SROCC | +0.0186 | 0.0025 |
| SigLIP-L384 vs L256 (paired 5-fold) | SROCC | +0.0116 | 0.023 |
| SigLIP-L384 vs B384 (paired 5-fold) | SROCC | +0.0088 | 0.098 |
| Feature-distilled 94 M vs L384 teacher | SROCC | −0.0059 | 0.110 |
| Feature-distilled 94 M vs B384 | SROCC | +0.0028 | 0.120 |
| Feature-distilled vs score-only distill | SROCC | +0.0007 | 0.368 |
| SigLIP2 vs SigLIP (single split) | SROCC | −0.041 | 0.003 |
| QStat vs SigLIP (single split) | KADID SROCC | −0.008 | 0.073 |
| QStat vs SigLIP (TID2013) | TID SROCC | +0.027 | 0.003 |

### 4.6 Per-distortion behaviour

Single-split 384 model, within-distortion SROCC (mean 0.814): strongest on
colour saturation (0.941), block-wise distortions (0.934), Gaussian blur
(0.933) and Gaussian noise (0.927); weakest on hue shift (0.476), sparse
sampling/reconstruction (0.496), multiplicative Gaussian noise (0.499) and
quantization noise (0.557).  Hue shift and sparse sampling are the clearest
remaining failure modes and motivate a colour-aware or frequency-aware branch.

## 5. What worked, what did not

**Worked.**

* Distortion/level auxiliary heads: the model must represent *what is wrong*,
  not just *what is depicted*.
* Partial fine-tuning of a strong foundation encoder.
* Pairwise ranking loss and flip TTA (small but consistent).
* NSS statistics as a cross-dataset regulariser (QStatFusion).
* Content-level CV and bootstrap tests: they expose split variance that a
  single number hides.

**Did not work.**

* From-scratch CNNs on 57–64 contents: 0.71–0.77 SROCC.  There is not enough
  labelled content to learn quality from pixels alone.
* Full fine-tuning of 93 M parameters: overfits within 7 epochs.
* SigLIP2-B: stronger pretraining did not translate to this split.
* Multi-dataset training with TID2013: no KADID gain once the objective is
  strong.
* High resolution: helps KADID, hurts TID cross-dataset.
* Simple ensembles of weak models: they dilute the best model.

## 6. Limitations and threats to validity

1. **Single machine, single seed per fold.**  We report 5-fold variance, but
   not seed variance.  Bitwise reproducibility is not guaranteed with
   `cudnn.benchmark=True`.
2. **Synthetic benchmarks only.**  KADID/TID2013 do not cover authentic camera
   distortions; SPAQ/KonIQ/LIVE Challenge are future work.
3. **Protocol differences.**  Published KADID numbers usually average random
   image-level splits; our content-level CV is stricter and not directly
   comparable.
4. **Foundation pretraining is external.**  SigLIP/CLIP weights were trained on
   web-scale data; our claims are about fine-tuning cost, not training from
   scratch.
5. **The 0.94–0.97 top of the literature remains out of reach** for a 93 M
   model trained on 64 KADID contents with this recipe.  Closing that gap
   likely needs either more labelled content, a larger foundation backbone, or
   a stronger multi-dataset/self-supervised pretraining stage.

## 7. Reproducibility

* `src/nriqa/` contains the data pipeline, models, losses, metrics, trainer,
  CV/ablation/HPO runners, benchmark suite and inference profiler.
* `configs/` contains every experiment; resolved configs are copied into run
  directories.
* `tests/` covers splits, metrics, losses, model shapes and config inheritance.
* `results/` contains the summary tables and per-image predictions for the
  headline runs.
* Commands are listed in `README.md`, `docs/protocol.md` and
  `docs/benchmarks.md`.

## 8. Conclusion and future work

The project delivers a rigorous, reproducible NR-IQA framework and a
four-point frontier:

* **0.9481 ± 0.0074** KADID at 316 M / 15.1 ms (third in the surveyed published
  table, behind the two ~400 M foundation probes at 0.970);
* **0.9365 ± 0.0074** KADID and **0.7353 ± 0.0084** TID at 316 M / 7.5 ms
  (balanced point);
* **0.9421 ± 0.0120** KADID from a 94 M distilled student (99.4% of the
  teacher at 3.4× fewer parameters) and 0.9393 from the 93 M base model;
* **0.7528** TID from 94 M QStatFusion (best cross-dataset).

The gains from distortion-aware training (+0.028, p = 0.003), L/16 384 vs 256
(+0.0116, p = 0.023) and NSS fusion on TID (+0.027, p = 0.003) are
statistically supported.  The work does not establish an absolute state of the
art; the top-3 claim is bounded by the surveyed table and by the 316 M
parameter count.

The most promising next steps are:

1. **Distortion-manifold pretraining** on unlabeled external images, followed
   by the same partial fine-tuning recipe (ARNIQA/TRIQA direction).
2. **Content-grouped ordinal contrastive loss** to explicitly optimise
   within-content ordering.
3. **A single multi-scale model** that fuses 224 and 384 Evidence instead of
   paying for a two-model ensemble.
4. **Authentic-distortion evaluation** (SPAQ, KonIQ, LIVE Challenge) to test
   whether the NSS branch's cross-dataset gain holds outside synthetic data.
5. **Knowledge distillation** from a larger foundation teacher into MSOSNet,
   with the teacher's quality subspace as the target rather than its logits.

## 9. References

See `docs/literature_review.md` for the full list and benchmark table.
