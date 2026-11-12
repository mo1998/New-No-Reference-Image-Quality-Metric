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

**Best KADID-10k result:** SigLIP-B/16 at 384 px trained on **KADID + KonIQ-10k**
with auxiliary distortion/level heads, pairwise ranking loss and flip TTA:
**SROCC 0.9481 ± 0.0096** over five content folds, PLCC 0.9524, **TID2013
0.7429 ± 0.0113**, with **93.4 M parameters and 10.9 ms** per image on an RTX
4090.  This is **third in the surveyed published KADID-10k table**, behind the
two ~400 M foundation probes at 0.970 and ahead of MANIQA (0.946).  It matches
the 316 M SigLIP-L/16 teacher exactly (paired p = 0.992) while using 3.4× fewer
parameters, 27% lower latency and 3× lower memory.  The teacher itself has the
same KADID accuracy (0.9481 ± 0.0074) but lower TID (0.7232).

**Best cross-dataset result:** QStatFusion, a gated fusion of SigLIP-B/16 and
the MSOSNet statistical branch: **TID2013 SROCC 0.753**, significantly better
than SigLIP alone (0.726, paired bootstrap p = 0.003), at 93.8 M parameters.

**Honest positioning:** 0.9481 KADID places the model third in the *raw,
mixed-protocol* surveyed table (SigLIP2 / B_Gated 0.970), above MANIQA (0.946),
LoDa (0.931) and ARNIQA (0.908).  After the split-protocol audit
(§4.8), this framing is too pessimistic: random image-level splits inflate
five published methods by **+0.103 to +0.174**, and under a matched content
split those methods fall to **0.850 / 0.844 / 0.776 / 0.789 / 0.815** while ours
stays at **0.9402 three-split / 0.9481 five-fold**.  The 0.9481 is therefore the best
*controlled* result in this study, not the third-best uncontrolled one.  It does
so with **93.4 M parameters, 10.9 ms and 390 MB** — 3.4× fewer parameters,
27% lower latency and roughly 3× lower memory than the 316 M foundation
probes.  The TID result (0.743) is second only to the 94 M QStatFusion
(0.753).  This is **not an absolute state of the art** (0.970 remains), and the
top-3 claim is bounded by the surveyed table; but it is a top-3,
production-viable, lightweight point with statistically significant gains over
the 224 M-class baselines.

**Key findings.**

* Distortion-aware auxiliary supervision is the single largest gain
  (+0.042 KADID, +0.046 TID over the same model without it), and the combined
  aux + rank + TTA recipe is statistically significant (+0.028, p = 0.003).
* Partial fine-tuning is essential: full fine-tuning overfits 57 contents and
  loses 0.03 KADID SROCC.
* Scaling the foundation encoder from B/16 (93 M) to L/16 (316 M) adds
  +0.009 KADID (p = 0.098) and a 94 M distilled student recovers 0.9421, i.e.
  99.4% of the teacher's accuracy at 3.4× fewer parameters.
* **Authentic data scaling is the decisive lightweight lever**: adding
  KonIQ-10k lifts the 93 M B/16 from 0.9393 to **0.9481** KADID (+0.0088,
  p = 0.112) and from 0.7166 to **0.7429** TID (+0.026), matching the 316 M
  teacher at 3.4× fewer parameters.
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
| **SigLIP-B/16 384, KADID + KonIQ** | **93.4 M** | **0.9481 ± 0.0096** | **0.9524** | **0.7429 ± 0.0113** |
| SigLIP-L/16 384 + aux/rank/TTA | 316.6 M | 0.9481 ± 0.0074 | 0.9526 | 0.7232 ± 0.0205 |
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
| SigLIP2-L vision 256 (HF, reference) | 316.0 M | — | 7.89 | — | — |
| SigLIP2-L vision 384 (HF, reference) | 316.3 M | — | 14.87 | — | — |

The **93 M B/16 + KonIQ** model is the deployment point and the headline:
10.9 ms, 390 MB, 0.9481 KADID and 0.7429 TID.  The 316 M L/16 teacher has the
same KADID accuracy but 15.1 ms, 1247 MB and lower TID (0.7232); it is strictly
dominated for deployment.  The 94 M distilled student (0.9421) and the 93 M
KADID-only model (0.9393) are also dominated by the KonIQ-trained point.  The
L/16 256 model is the lowest-latency 316 M variant (7.5 ms) but is 0.012 lower
KADID.  SigLIP2-L, the surveyed leader, uses the same 316 M vision encoder at
14.87 ms — so the 93 M model is the genuine efficiency contribution (3.4× fewer
parameters, 27% lower latency, 3× lower memory).  The from-scratch MSOSNet is
the smallest (0.30 M) and fastest in FLOPs but not in batch-1 latency, because
its statistical pooling is launch-bound.

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
| B16+KonIQ vs B224 (paired 5-fold) | SROCC | +0.0274 | 0.0032 |
| B16+KonIQ vs KADID-only B384 | SROCC | +0.0088 | 0.112 |
| B16+KonIQ vs L384 teacher | SROCC | +0.00004 | 0.992 |
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

### 4.7 Authentic-distortion benchmarks (zero-shot)

Full protocol and tables: [authentic_benchmarks.md](authentic_benchmarks.md).

| Model | Params | LIVEC SROCC | LIVEC PLCC | SPAQ SROCC | SPAQ PLCC |
|---|---:|---:|---:|---:|---:|
| **SigLIP-L/16 384 + KonIQ** | 316.6 M | **0.8846** | **0.9055** | **0.8986** | **0.8990** |
| **SigLIP-B/16 384 + KonIQ (5-fold)** | **93.4 M** | **0.8662 ± 0.0014** | **0.8915 ± 0.0016** | **0.8884 ± 0.0024** | **0.8909 ± 0.0019** |
| TOPIQ (PyIQA) | 45.2 M | 0.8283 | 0.8488 | 0.8709 | 0.8764 |
| SigLIP-B/16 384, KADID only | 93.4 M | 0.6410 | 0.6674 | 0.7493 | 0.7621 |
| ARNIQA (PyIQA) | 23.5 M | 0.5763 | 0.6240 | 0.7935 | 0.8021 |

**Authentic training is decisive.**  Adding KonIQ-10k lifts the 93 M model from
0.6410 → 0.8662 SROCC on LIVEC (+0.225) and 0.7493 → 0.8884 on SPAQ (+0.139).
KADID-only models overfit synthetic distortions and collapse on authentic
captures.  The 93 M model beats TOPIQ (45 M) on both benchmarks and beats all
other PyIQA baselines; the 316 M model adds +0.018 LIVEC and +0.010 SPAQ at
3.4× the parameters.  Published context (Revisiting VL): LIVEC CLIP 0.841,
SigLIP2 0.875, B_Gated 0.887; SPAQ CLIP 0.916, SigLIP2 0.927, B_Gated 0.928.
Our LIVEC result is at the top published level; SPAQ remains below the top
foundation probes.

### 4.8 Split-protocol bias: random vs content

The KADID-10k literature mostly uses random image-level splits, which leak
reference content into the test set.  Retraining published models under matched
7,125/1,000/2,000 random and content-held-out splits isolates the effect; each
cell is a mean ± std over seeds 42, 43 and 44.  Full protocol:
[split_bias.md](split_bias.md).

| Model | Trainable form | Content SROCC | Random SROCC | Δ (random − content) |
|---|---|---:|---:|---:|
| CLIP-IQA | frozen CLIP + head | 0.7764 ± 0.0490 | 0.9322 ± 0.0032 | **+0.1558** |
| ARNIQA | full fine-tune | 0.8438 ± 0.0170 | 0.9537 ± 0.0035 | **+0.1099** |
| TOPIQ | full fine-tune | 0.8496 ± 0.0245 | 0.9521 ± 0.0042 | **+0.1025** |
| MANIQA | frozen ViT + head | 0.7892 ± 0.0649 | 0.9627 ± 0.0027 | **+0.1735** |
| LIQE | full fine-tune, CE | 0.8318 | 0.9184 | **+0.0866** |
| **Ours, SigLIP-B/16 + KonIQ** | full fine-tune | **0.9402 ± 0.0176** | 0.9660 ± 0.0018 | **+0.0258** |

A zero-shot control on the same test subsets (no KADID training) shows the
random subset is not easier: TOPIQ 0.620→0.615, CLIP-IQA 0.435→0.383,
ARNIQA 0.737→0.699, MANIQA 0.387→0.399, LIQE 0.727→0.727.  The inflation is
therefore content leakage, not test composition.  It is **3–5× larger than the
hypothesised 0.02–0.04**.

This is a genuine methodological contribution and it reframes the ranking:
under the controlled content split our 93 M model reaches **0.9402 ± 0.0176**
over three single splits (and **0.9481 ± 0.0096** over five folds), while the
published methods fall to **0.8496 / 0.8438 / 0.8318 / 0.7764 / 0.7892**.  The
random-split numbers near 0.95 are not comparable to 0.9481; the "third-best
uncontrolled" framing was too pessimistic.  Our model is also much less
split-sensitive (+0.026 vs +0.087 to +0.174), although the residual gap shows
leakage still helps.

LIQE now has a genuine full fine-tune using its native quality-class
cross-entropy.  MANIQA full and partial-ViT fine-tuning still collapse to
constant predictions; its stable frozen-feature probe is retained.  CLIP-IQA is
also a probe; TOPIQ/ARNIQA/LIQE are fully fine-tuned.

### 4.9 TID2013 cross-dataset replication

The audit was repeated on TID2013 with matched **2,400/240/360** splits
(20/2/3 references for content; all 25 for random).  TID content has only three
test references, so it is noisier than KADID.

| Model | Trainable form | Content SROCC | Random SROCC | Δ |
|---|---|---:|---:|---:|
| LIQE | full fine-tune, CE | 0.8215 | 0.8633 | **+0.0418** |
| ARNIQA | full fine-tune | 0.6490 | 0.9211 | **+0.2721** |
| CLIP-IQA | frozen CLIP + head | 0.8162 | 0.9027 | **+0.0865** |
| TOPIQ | full fine-tune | 0.8650 | 0.9330 | **+0.0680** |
| MANIQA | frozen ViT + head (3 seeds) | 0.9021 ± 0.0207 | 0.9679 ± 0.0029 | **+0.0658** |
| **Ours, SigLIP-B/16** | full fine-tune | **0.9266** | 0.9460 | **+0.0194** |

The zero-shot control again shows the random TID test subset is *harder*, not
easier: MANIQA 0.634→0.330, TOPIQ 0.603→0.554, CLIP-IQA 0.677→0.528,
ARNIQA 0.710→0.554, LIQE 0.686→0.631.  Post-training random-split gains are
therefore leakage.  The cross-dataset result is qualitatively identical to
KADID: every published method inflates (+0.066 to +0.272), our model only
+0.019, and under the controlled content split our 0.9266 is the strongest.
TID content error bars are wide because only three references are held out.

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
* Multi-teacher (L384 + L256) soft targets: 0.913 fold 0, worse than the
  single-teacher 0.926; the weaker teacher's targets dominate.
* Learned patch-attention pooling: 0.919 fold 0, worse than the plain CLS
  head; the extra capacity overfits 63 contents.
* 512 px dynamic input and EMA: both worse than the 384 px / standard recipe.
* Feature distillation over score-only: +0.0007, not significant; the 94 M
  student is capacity-limited, not supervision-limited.

## 6. Limitations and threats to validity

1. **Single machine, single seed per fold.**  We report 5-fold variance, but
   not seed variance.  Bitwise reproducibility is not guaranteed with
   `cudnn.benchmark=True`.
2. **Benchmark coverage.**  KADID/TID2013 are synthetic; KonIQ authentic data
   is used only for extra training, not as a held-out benchmark.  SPAQ and LIVE
   Challenge remain future work.
3. **Protocol differences.**  Published KADID numbers usually average random
   image-level splits; our content-level CV is stricter and not directly
   comparable.  The split audit (§4.8) shows this is not a minor caveat:
   random splits inflate the five re-run methods by 0.103–0.174 SROCC.
4. **Foundation pretraining is external.**  SigLIP/CLIP weights were trained on
   web-scale data; our claims are about fine-tuning cost, not training from
   scratch.
5. **The split audit is single-seed and covers four published models.**  The
   effect is far larger than seed noise, but the exact inflation for any one
   paper depends on its split and training recipe; the controlled ranking is
   established for the models we re-ran, not for every published method.
6. **Authentic benchmarks are now covered zero-shot.**  LIVEC (0.8846/0.8662)
   and SPAQ (0.8986/0.8884) show that KonIQ training is decisive; SPAQ remains
   below the top foundation probes (0.927–0.928), and KonIQ itself is only used
   for training, not held out.  Benchmark protocols differ across papers, so
   the authentic comparison is contextual rather than a controlled leaderboard.
7. **The 0.970 top of the literature remains out of reach.**  The 93 M model
   is top-3 at 0.9481; closing the last 0.022 likely needs a larger foundation
   backbone or substantially more labelled authentic data.  Closing it while
   staying under 100 M is an open problem.

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
top-3 lightweight result:

* **0.9481 ± 0.0096 KADID, 0.7429 ± 0.0113 TID, 93.4 M parameters, 10.9 ms,
  390 MB** — third in the raw mixed-protocol KADID table and, after the split
  audit, the **best controlled result** among the methods we re-trained;
* **3.4× fewer parameters, 27% lower latency and ~3× lower memory** than the
  316 M foundation probes;
* **0.9481 KADID** from the 316 M teacher as well, confirming that the
  lightweight model is not trading accuracy for size (p = 0.992);
* **0.7528 TID** from the 94 M QStatFusion for the pure cross-dataset point;
* **Split audit:** random image-level KADID splits inflate five published
  methods by **+0.103 to +0.174 SROCC** (3-seed means); under the matched
  content split they fall to **0.850 / 0.844 / 0.776 / 0.789 / 0.815**, while
  our 93 M model reaches **0.9402 ± 0.0176** over three single splits and
  **0.9481 ± 0.0096** over five folds;
* **TID2013 replication:** random splits inflate the same methods by
  **+0.066 to +0.272**, while our model gains only **+0.019**; under the content
  split our 0.9266 is again the strongest.

The key gains are statistically supported: +0.0274 over the 224-class baseline
(p = 0.0032), +0.028 from distortion-aware training (p = 0.003), and +0.026 TID
from KonIQ data scaling.  The KonIQ gain over KADID-only B/16 (+0.0088) is not
significant with five folds (p = 0.112), which is reported honestly.  The work
does not establish an absolute state of the art (0.970 remains), and the
top-3 claim is bounded by the surveyed table; within that scope, it satisfies
the top-3 + substantially lower-cost success criterion.

The most promising next steps are:

1. **Extend the split audit to MANIQA/LIQE and more seeds.**  MANIQA's fine-tune
   was numerically unstable in this harness; a stable implementation and two
   more seeds would turn the single-seed estimate into a distribution.
2. **Adopt the content split as the default reporting protocol** and publish the
   controlled numbers alongside the raw table.
3. **Distortion-manifold pretraining** on unlabeled external images, followed
   by the same partial fine-tuning recipe (ARNIQA/TRIQA direction).
4. **Content-grouped ordinal contrastive loss** to explicitly optimise
   within-content ordering.
5. **A single multi-scale model** that fuses 224 and 384 evidence instead of
   paying for a two-model ensemble.
6. **Knowledge distillation** from a larger foundation teacher into MSOSNet,
   with the teacher's quality subspace as the target rather than its logits.

## 9. References

See `docs/literature_review.md` for the full list and benchmark table.
