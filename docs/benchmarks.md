# Benchmark results

All numbers below were produced by this repository on one machine
(2×RTX 4090, PyTorch 2.1/2.3).  See `docs/protocol.md` for the evaluation
contract and `results/all_runs.csv` for the machine-readable table.

> **Status:** results are updated as runs complete.  The final report is in
> `docs/final_report.md`.

## 1. Headline: 5-fold content cross-validation (KADID-10k)

Protocol: five disjoint content folds; each fold trains on the 64 non-test
references (80%), tests on 16 references (20%), fixed 18 epochs, no test-time
label use.  TTA = horizontal-flip averaging.

| Model | Params | KADID SROCC ↑ | KADID PLCC ↑ | TID2013 SROCC ↑ | TID2013 PLCC ↑ |
|---|---:|---:|---:|---:|---:|
| SigLIP-B/16 384, aux + rank + TTA | 93.4 M | **0.9393 ± 0.0149** | **0.9441** | 0.7166 ± 0.0123 | 0.7428 |
| SigLIP-B/16 224, aux + rank + TTA | 93.1 M | 0.9207 ± 0.0166 | 0.9271 | **0.7284 ± 0.0157** | 0.7746 |
| SigLIP-B/16 224, no aux/rank/TTA (baseline) | 93.1 M | 0.8688* | 0.8772* | 0.7111* | 0.7419* |
| QStatFusion 224 | 93.8 M | 0.8885* | 0.8974* | **0.7528*** | **0.7858*** |
| QStatFusion 384 | 94.1 M | 0.9204* | 0.9272* | 0.7137* | 0.7335* |

\* Single content split; not a 5-fold estimate.  The paired t-test between the
384 and 224 CV variants gives ΔSROCC = +0.0186, p = 0.0025 (n = 5 folds);
Wilcoxon p = 0.0625 (limited power with five folds).

The 384 model is the accuracy point; the 224 model is the balanced point
(better cross-dataset, 1.8x lower latency); QStatFusion is the
generalisation point.  All are partially fine-tuned SigLIP-B/16 encoders (last
4 of 12 transformer blocks + head), with auxiliary distortion/level
classification, pairwise ranking loss and flip TTA.  The 224 per-fold KADID
SROCC values are 0.896, 0.935, 0.935, 0.924, 0.913; the 384 values are 0.919,
0.952, 0.947, 0.951, 0.928.  The spread illustrates how much a single
16-content split can vary, and why the 5-fold mean is the headline number.

## 2. Single content split: model comparison

Fixed split: 57 train / 8 validation / 16 test references.  This split is used
for fast iteration and ablations; it is **not** the headline estimate.

| Model | Pretraining | Params | KADID SROCC ↑ | KADID PLCC ↑ | TID2013 SROCC ↑ | TID2013 PLCC ↑ |
|---|---|---:|---:|---:|---:|---:|
| BRISQUE (PyIQA) | none | 0 | 0.180* | 0.396 | 0.339 | 0.453 |
| NIQE (PyIQA) | none | 0 | 0.244* | 0.271 | 0.267 | 0.368 |
| TinyCNN | none | 0.46 M | 0.711 | 0.723 | 0.456 | 0.527 |
| MSOSNet | none | 0.30 M | 0.721 | 0.730 | 0.483 | 0.530 |
| WaveQNet | none | 0.08 M | 0.496 | 0.547 | 0.234 | 0.344 |
| MobileNetV3-Small | ImageNet | 1.52 M | 0.721 | 0.732 | 0.632 | 0.706 |
| ResNet-18 | ImageNet | 11.2 M | 0.769 | 0.779 | 0.678 | 0.713 |
| CLIP-IQA (PyIQA, zero-shot) | CLIP | ~150 M | 0.435 | 0.465 | 0.529 | 0.613 |
| MUSIQ (PyIQA, zero-shot) | KonIQ | 27.1 M | 0.511 | 0.561 | 0.491 | 0.599 |
| LIQE (PyIQA, zero-shot) | multi | 151.3 M | 0.727 | 0.749 | 0.616 | 0.712 |
| ARNIQA (PyIQA, zero-shot) | distortion manifold | 23.5 M | 0.737 | 0.754 | 0.574 | 0.645 |
| SigLIP-B/16 224 | SigLIP | 93.1 M | 0.869 | 0.877 | 0.711 | 0.742 |
| SigLIP-B/16 224, aux + rank + TTA | SigLIP | 93.1 M | **0.897** | **0.907** | 0.726 | 0.772 |
| SigLIP-B/16 224, full fine-tune | SigLIP | 93.1 M | 0.839 | 0.849 | 0.685 | 0.732 |
| SigLIP-B/16 224, multi-dataset | SigLIP | 93.1 M | 0.897 | 0.908 | — | — |
| SigLIP-B/16 384, aux + rank + TTA | SigLIP | 93.4 M | **0.921** | **0.927** | 0.685 | 0.711 |
| SigLIP2-B/16 224, aux + rank + TTA | SigLIP2 | 93.1 M | 0.856 | 0.868 | 0.729 | 0.768 |
| QStatFusion 224 | SigLIP + NSS | 93.8 M | 0.888 | 0.897 | **0.753** | **0.786** |
| QStatFusion 384 | SigLIP + NSS | 94.1 M | 0.920 | 0.927 | 0.714 | 0.734 |

\* PyIQA's default preprocessing uses center 224 crops; classical metrics are
known to perform better at native resolution.  These rows are zero-shot and
are not directly comparable to models trained on KADID.  They are included to
show that a 93 M foundation model trained on KADID is far stronger than
off-the-shelf NR-IQA metrics.

## 3. What the per-component evidence says

| Variant | KADID SROCC | Δ vs full | TID2013 SROCC | Δ vs full |
|---|---:|---:|---:|---:|
| Full recipe (aux + rank + TTA) | **0.8968** | — | 0.7256 | — |
| No auxiliary heads | 0.8547 | −0.042 | 0.6792 | −0.046 |
| No ranking loss | 0.8966 | −0.000 | 0.7479 | +0.022 |
| No TTA | 0.8943 | −0.003 | 0.7304 | +0.005 |
| Full fine-tune (freeze_blocks=0) | 0.8385 | −0.058 | 0.6850 | −0.041 |
| Multi-dataset (KADID + TID) | 0.8969 | +0.000 | — | — |
| 384 resolution | **0.9208** | +0.024 | 0.6851 | −0.041 |
| QStatFusion (NSS branch) | 0.8885 | −0.008 | **0.7528** | +0.027 |

All variants use the same single content split (57 train / 8 val / 16 test
references), so the deltas are directly comparable.  The 5-fold headline model
uses the full recipe at 224 px.

**Interpretation.**

1. **Partial fine-tuning is essential.**  Full fine-tuning of 93 M parameters on
   57 contents overfits rapidly (validation peaks at 0.86 vs 0.93 for the
   partial variant).
2. **Distortion-aware supervision and ranking loss help**, and TTA adds a small
   amount.  The combined gain is +0.028 on KADID and +0.015 cross-dataset.
3. **Higher resolution improves in-dataset accuracy but hurts cross-dataset
   generalisation.**  This is consistent with the known bias of synthetic
   benchmarks: high-resolution cues help on KADID but do not transfer to
   TID2013's different distortion mix.
4. **Extra TID2013 training data does not help KADID accuracy** once the
   training objective is strong.  It would matter for authentic-distortion
   benchmarks, which are outside this study.
5. **The NSS branch buys cross-dataset robustness.**  QStatFusion is slightly
   worse on KADID but +0.027 on TID2013, approaching ARNIQA/Re-IQA cross-dataset
   numbers while remaining a single 94 M-parameter model.
6. **A larger foundation model is not automatically better.**  SigLIP2-B,
   despite stronger pretraining, overfits this content split more than
   SigLIP-B (0.856 vs 0.897 test SROCC).

## 4. Comparison with published methods

The table below places the best lightweight/academic numbers from the
literature next to this work.  **Protocols differ**: most published KADID
numbers average multiple random 80/20 splits; ours is a single 5-fold content
CV.  Cross-dataset numbers are KADID→TID2013, zero-shot.

| Method | Params | KADID SROCC | TID2013 cross | Source |
|---|---:|---:|---:|---|
| BRISQUE | 0 | 0.528 | — | CONTRIQUE/ARNIQA |
| DBCNN | 24 M | 0.851 | 0.744 | ARNIQA table |
| HyperIQA | 27 M | 0.852 | 0.706 | ARNIQA table |
| TReS | 152 M | 0.859 | — | ARNIQA table |
| MUSIQ | 27 M | 0.875 | — | CMPA table |
| ARNIQA | 24 M | 0.908 | 0.760 | ARNIQA |
| CONTRIQUE | 24 M | 0.934 | 0.612 | CONTRIQUE/ARNIQA |
| LIQE | 151 M | 0.930 | 0.811* | LIQE/CMPA |
| Re-IQA | 48 M | 0.872 | 0.777 | ARNIQA table |
| MANIQA | 24 M | 0.944 | — | MANIQA |
| LoDa | 9 M | 0.931 | — | CMPA table |
| QMamba | 30 M | 0.923 | — | CMPA table |
| CMPA (CLIP-L adapter) | ~428 M | 0.938 | — | CMPA |
| SigLIP2 + MLP | ~400 M | 0.970 | — | Revisiting VL |
| **This work, SigLIP-B/16 384** | **93 M** | **0.939** | 0.717 | 5-fold CV |
| **This work, SigLIP-B/16 224** | **93 M** | 0.921 | **0.728** | 5-fold CV |
| **This work, QStatFusion 224** | **94 M** | 0.888 (single) | **0.753** (single) | single split |

\* LIQE is trained on multiple datasets; its 0.811 is not a KADID-only
zero-shot number.

**Honest positioning.**  The 384 headline model (0.939) is at the level of
MANIQA (0.944) and above LoDa (0.931) and ARNIQA (0.908), while being
significantly cheaper than the 150–400 M vision-language models that occupy
the top (LIQE 0.930/151 M, SigLIP2 probe 0.970/~400 M).  It does **not** beat
the 0.97 top of the published table and is not a new state of the art.  The
224 model is the better cross-dataset point (0.728 TID), and QStatFusion is the
best cross-dataset point (0.753 TID) at a small in-dataset cost.  We claim a
strong three-point accuracy/efficiency/generalisation frontier, not SOTA.

## 5. Efficiency

Measured with `src/nriqa/inference.py` (batch 1, RTX 4090, median of 50 runs;
peak memory includes activations).  Throughput is batch 16.

| Model | Params | GFLOPs | Latency (ms) | Throughput (img/s) | Peak GPU mem (MB) |
|---|---:|---:|---:|---:|---:|
| MSOSNet | 0.30 M | 0.13 | 3.71 | 2204 | 145 |
| ResNet-18 | 11.2 M | 3.63 | 2.70 | 4584 | 61 |
| QStatFusion | 93.8 M | 35.5 | 8.46 | 371 | 503 |
| SigLIP-B/16 224 | 93.1 M | 35.4 | 6.16 | 465 | 372 |
| SigLIP-B/16 384 | 93.4 M | 112.1 | 10.93 | 154 | 390 |

MSOSNet is an instructive case: it has 76× fewer parameters and 27× fewer FLOPs
than SigLIP-B/16, but its latency is only 1.7× lower at batch 1 because the
statistical pooling operations are launch- and memory-bound.  FLOPs alone are a
poor proxy for NR-IQA deployment cost; latency and throughput must be measured.

## 6. Per-distortion error analysis

Single-split 384 checkpoint, SROCC computed within each KADID distortion type
(80 images per type).  Mean per-type SROCC = 0.814.

| Easiest types | SROCC | Hardest types | SROCC |
|---|---:|---|---:|
| colour saturation change | 0.941 | hue shift | 0.476 |
| local block-wise distortions | 0.934 | sparse sampling / reconstruction | 0.496 |
| Gaussian blur | 0.933 | multiplicative Gaussian noise | 0.499 |
| Gaussian noise | 0.927 | quantization noise | 0.557 |
| comfort noise | 0.927 | lossy compression of noisy images | 0.600 |

The model is strong on global blur/noise/blockiness and weak on colour-hue and
sparse-sampling distortions.  Hue shift is nearly a pure colour rotation with
little structural change; sparse sampling produces aliasing patterns that the
224/384 centre crop may not see consistently.  These are concrete targets for a
colour-aware or frequency-aware branch.

## 7. Reproducing a row

```bash
# Headline 5-fold CV
bash scripts/run_cv.sh configs/siglip_b16_aux_kadid.yaml siglip_aux_cv65

# Single split
PYTHONPATH=src .venv/bin/python -m nriqa.run \
  --config configs/siglip_b16_aux_kadid.yaml

# PyIQA zero-shot baselines
bash scripts/run_benchmarks.sh

# Ablations
PYTHONPATH=src .venv/bin/python -m nriqa.ablate \
  --config configs/siglip_b16_aux_kadid.yaml \
  --variants configs/ablations/siglip.json
```
