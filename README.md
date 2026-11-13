# Project 28 — Building and Validating a New No-Reference Image-Quality Metric

**Course:** CSE 681 — Digital Image Processing & Video Processing
**Branch:** `playground`

Build a lightweight **no-reference** image-quality model: predict a quality
score from a single distorted image, with no pristine reference, and validate it
against human MOS labels (KADID-10k, TID2013) using SROCC / PLCC.

## Team

| Code | Name |
|---|---|
| 2400008 | Mohamed Ahmed Fathy Abdelrazik |
| 2501922 | Novert Ahmed Fahim Abdelfattah |
| 2501897 | Farah Hossam Eldin Mahmoud |

## What is in this repository

A complete, reproducible NR-IQA research framework:

* **From-scratch models designed for the no-reference setting**
  * `MSOSNet` (0.30 M): multi-scale MSCN pyramid + learned band-pass filters +
    orderless statistical pooling + gated scale fusion.
  * `WaveQNet` (0.08 M): fixed Haar sub-band statistics + a tiny transformer.
* **Compact pretrained baselines** (MobileNetV3, ResNet-18).
* **Foundation encoders** (CLIP/SigLIP/DINOv2/SigLIP2) with partial fine-tuning,
  distortion-aware auxiliary heads and ranking losses.
* **QStatFusion** (94 M): gated fusion of SigLIP features and the MSOSNet
  statistical branch, aimed at cross-dataset robustness.
* **A rigorous protocol**: content-held-out splits, 5-fold CV, 4-parameter
  logistic PLCC, content-level bootstrap CIs and paired significance tests.
* **Experiment tooling**: YAML configs, run tracking (JSONL/TensorBoard),
  CV/ablation/HPO runners, PyIQA benchmark suite, inference profiler.

## Headline results (KADID-10k, 5-fold content CV)

| Model | Params | SROCC ↑ | PLCC ↑ | TID2013 zero-shot SROCC ↑ | Latency (RTX 4090) |
|---|---:|---:|---:|---:|---:|
| **SigLIP-B/16 384 + KADID+KonIQ** | **93.4 M** | **0.9481 ± 0.0096** | **0.9524** | **0.7429 ± 0.0113** | **10.9 ms** |
| SigLIP-L/16 384 + aux + rank + TTA | 316 M | 0.9481 ± 0.0074 | 0.9526 | 0.7232 ± 0.0205 | 15.1 ms |
| SigLIP-L/16 256 + aux + rank + TTA | 316 M | 0.9365 ± 0.0074 | 0.9426 | **0.7353 ± 0.0084** | 9.4 ms |
| SigLIP-B/16 384 + aux + rank + TTA | 93.4 M | 0.9393 ± 0.0149 | 0.9441 | 0.7166 ± 0.0123 | 10.9 ms |
| SigLIP-B/16 224 + aux + rank + TTA | 93.1 M | 0.9207 ± 0.0166 | 0.9271 | 0.7284 ± 0.0157 | 6.2 ms |
| Distilled SigLIP-B/16 384 (score+feature) | 94.2 M | **0.9421 ± 0.0120** | 0.947 | 0.7127 | 10.9 ms |
| QStatFusion 224 | 93.8 M | 0.8885* | 0.8974* | **0.7528*** | 8.5 ms |
| MSOSNet (from scratch) | 0.30 M | 0.721* | 0.730* | 0.483* | 3.7 ms |
| ResNet-18 (ImageNet) | 11.2 M | 0.769* | 0.779* | 0.678* | 2.7 ms |

\* single content split (57/8/16 refs), used for rapid iteration.  The 384 and
224 rows are 5-fold content CV; see `docs/benchmarks.md` for the full table.

**Honest positioning:** this is a **controlled benchmark audit plus a
lightweight model**, not an absolute state of the art.  Under the content-held-out
protocol the 93 M model reaches **0.9416 ± 0.0137 (five seeds) / 0.9481
(5-fold)**, **statistically tied** with SigLIP2-L (0.9392, p=0.588) while using
3.4× fewer parameters; B_Gated 0.9270, MANIQA 0.8830, TOPIQ 0.8470, LIQE
0.8225, ARNIQA 0.8210 and CLIP-IQA 0.7716.
The random-split numbers near 0.97 are inflated by content leakage; B_Gated is a
controlled reproduction without the paper's LoRA.  See `docs/final_report.md` §4.10 for the controlled
comparison.

**Statistically significant findings** (paired content-level bootstrap):

* Distortion-aware auxiliary heads + ranking + TTA: **+0.028 SROCC, p = 0.003**.
* SigLIP-L/16 384 vs 256 (paired 5-fold): **+0.012 SROCC, p = 0.023**.
* L384 vs 93 M B/384: **+0.009 SROCC, p = 0.098** (not significant with 5 folds).
* NSS branch (QStatFusion) vs SigLIP on TID2013: **+0.027 SROCC, p = 0.003**.
* Removing auxiliary heads costs **0.042** KADID SROCC.
* Full fine-tuning overfits: **−0.058** vs partial fine-tuning.
* SigLIP2-B is significantly worse than SigLIP-B on this split (p = 0.003).

## Split-protocol audit (KADID-10k)

Random image-level splits leak reference content into the test set.  Retraining
published methods under matched 7,125/1,000/2,000 splits (five seeds where
available) shows the inflation directly.  TOPIQ/ARNIQA/LIQE and MANIQA are fine-tuned with their
native/official recipes (LIQE with quality-class CE); CLIP-IQA uses a frozen
probe:

| Model | Content split SROCC | Random split SROCC | Δ |
|---|---:|---:|---:|
| **Ours, SigLIP-B/16 + KonIQ (93 M)** | **0.9416 ± 0.0137 / 0.9481 (5-fold)** | 0.9670 ± 0.0019 | **+0.0253** |
| SigLIP2-L/16 384 (316 M) | 0.9392 ± 0.0105 | 0.9706 ± 0.0016 | **+0.0314** |
| B_Gated reproduction (316 M) | 0.9270 ± 0.0075 | 0.9689 ± 0.0009 | **+0.0418** |
| MANIQA official (136 M) | 0.8830 ± 0.0171 | 0.9704 ± 0.0016 | **+0.0873** |
| TOPIQ | 0.8470 ± 0.0184 | 0.9528 ± 0.0032 | **+0.1058** |
| LIQE (full CE) | 0.8225 ± 0.0587 | 0.9183 ± 0.0021 | **+0.0958** |
| ARNIQA | 0.8210 ± 0.0413 | 0.9545 ± 0.0030 | **+0.1335** |
| CLIP-IQA (probe) | 0.7716 ± 0.0353 | 0.9331 ± 0.0029 | **+0.1615** |

All models now have five seeds. Paired tests: TOPIQ p=0.0003, ARNIQA
p=0.0019, CLIP-IQA p=0.0005, LIQE p=0.024, B_Gated p=0.0003, MANIQA p=0.0003,
SigLIP2 p=0.0036, ours p=0.013. **Ours vs SigLIP2 content: p=0.588 — a
statistical tie**; our advantage is 3.4× fewer parameters and the five-fold
estimate (0.9481). The random-split leaders are inflated by content leakage.

A zero-shot control on the same test subsets shows random is *not* easier
(TOPIQ 0.620→0.615, CLIP-IQA 0.435→0.383, ARNIQA 0.737→0.699, LIQE flat), so
this is content leakage, not test composition.  Under the controlled content
protocol our 93 M model (0.9416 ± 0.0137 over five splits, **0.9481 ± 0.0096**
over five folds) is statistically tied with SigLIP2-L while being 3.4× smaller;
the random-split numbers near 0.97 are not comparable.
TID2013 replication (2,400/240/360, 20/2/3 refs): ours 0.9266→0.9460
(+0.019), TOPIQ 0.8650→0.9330, CLIP-IQA 0.8162→0.9027, ARNIQA
0.6490→0.9211, MANIQA official 0.7819→0.9686, LIQE full CE 0.8215→0.8633.
Full protocol, caveats and reproduction: [split-protocol audit](docs/split_bias.md).

## Authentic-distortion results (zero-shot)

Evaluated without training on either benchmark; the KonIQ-trained models only
see KonIQ during training. Full protocol and tables: [authentic benchmarks](docs/authentic_benchmarks.md).

| Model | Params | LIVEC SROCC | SPAQ SROCC |
|---|---:|---:|---:|
| SigLIP-L/16 384 + KonIQ | 316 M | **0.8846** | **0.8986** |
| SigLIP-B/16 384 + KonIQ (5-fold) | 93.4 M | 0.8662 ± 0.0014 | 0.8884 ± 0.0024 |
| TOPIQ (PyIQA) | 45.2 M | 0.8283 | 0.8709 |
| SigLIP-B/16 384, KADID only | 93.4 M | 0.6410 | 0.7493 |
| ARNIQA (PyIQA) | 23.5 M | 0.5763 | 0.7935 |

KonIQ training is the decisive factor: the 93 M model gains **+0.225 LIVEC**
and **+0.139 SPAQ** over its KADID-only counterpart. Published CLIP/SigLIP2
context: LIVEC 0.841/0.875, SPAQ 0.916/0.927.

## Repository layout

```
├── configs/                     # YAML experiment configs (include: inheritance)
│   ├── ablations/               # ablation grids
│   └── hpo/                     # hyperparameter search spaces
├── data/                        # KADID-10k + TID2013 (not committed)
├── docs/
│   ├── architecture.md          # model specifications
│   ├── authentic_benchmarks.md   # LIVEC + SPAQ zero-shot results
│   ├── split_bias.md            # random vs content split audit
│   ├── benchmarks.md            # all result tables
│   ├── final_report.md          # final report and discussion
│   ├── literature_review.md     # SOTA literature and opportunity analysis
│   └── protocol.md              # evaluation contract
├── notebooks/00_colab_starter.ipynb
├── results/                     # summary tables, CIs, comparisons, inference
├── scripts/                     # one-command CV / benchmark / inference
├── src/
│   ├── baselines.py             # legacy full-reference M1 baselines
│   ├── eval_protocol.py         # legacy M1 evaluation helpers
│   ├── run_m1_baselines.py      # legacy M1 runner
│   ├── utils.py                 # legacy seed helpers
│   └── nriqa/                   # main package
│       ├── data.py              # datasets, content splits, multi-dataset
│       ├── models.py            # MSOSNet, WaveQNet, TimmRegressor, QStatFusion
│       ├── losses.py            # MSE, PLCC, ranking, aux CE
│       ├── metrics.py           # SROCC/PLCC/RMSE, bootstrap, paired tests
│       ├── train.py             # trainer (AMP, EMA, early stop, tracking)
│       ├── evaluate.py          # prediction and cross-dataset evaluation
│       ├── benchmark.py         # PyIQA baseline suite
│       ├── cv.py                # 5-fold content CV
│       ├── ablate.py            # ablation grid
│       ├── hpo.py               # random hyperparameter search
│       ├── compare.py           # paired bootstrap between predictions
│       ├── inference.py         # params/FLOPs/latency/memory
│       ├── probe.py             # frozen-feature linear/MLP probes
│       ├── foundation.py        # feature extraction/caching
│       └── collect_results.py   # aggregate all runs
└── tests/                       # split/metric/loss/model/config tests
```

## Setup

Python 3.9+, PyTorch 2.1+ (CUDA recommended), and the packages in
`requirements.txt`.  Two environments are used in practice:

```bash
# Main environment: torch 2.1, timm 0.9.16, pyiqa
python -m venv .venv --system-site-packages
.venv/bin/pip install -r requirements.txt

# Optional modern environment for SigLIP2 (torch >= 2.2)
python -m venv .venv2
.venv2/bin/pip install torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121
.venv2/bin/pip install numpy pandas scipy pillow pyyaml timm transformers scikit-learn
```

## Datasets

| Dataset | Size | Role | Link |
|---|---|---|---|
| KADID-10k | 10,125 images + MOS | train / test | <http://database.mmsp-kn.de/kadid-10k-database.html> |
| TID2013 | 3,000 images + MOS | cross-dataset / extra training | <http://www.ponomarenko.info/tid2013.htm> |
| KonIQ-10k | 10,073 images + MOS | authentic extra training | <https://doi.org/10.18419/DARUS-2435> |

Expected layout:

```
data/kadid10k/{images/, dmos.csv}
data/tid2013/{distorted_images/, reference_images/, mos_with_names.txt}
data/koniq/{512x384/, koniq10k_scores_and_distributions.tab}
```

## Evaluation protocol

* Split **by reference image (content)**, never by distorted image.
* Report **SROCC** (no fitting) and **PLCC + RMSE** after a 4-parameter logistic
  fit vs MOS.
* 5-fold content CV for headline numbers; single content split for rapid
  iteration.
* Bootstrap **reference contents** for 95% CIs; use paired content-level
  bootstrap tests for method comparisons.
* Fixed seed: `SEED = 42` for every headline result.
* Full details: [`docs/protocol.md`](docs/protocol.md).

## Quick start

```bash
# 1. Fast single-split experiment
PYTHONPATH=src .venv/bin/python -m nriqa.run \
    --config configs/siglip_b16_aux_kadid.yaml

# 2. Headline 5-fold content CV
bash scripts/run_cv.sh configs/siglip_b16_aux_kadid.yaml siglip_aux_cv65

# 3. Pretrained zero-shot baselines (BRISQUE/NIQE/MUSIQ/CLIP-IQA/ARNIQA/LIQE)
bash scripts/run_benchmarks.sh

# 4. Ablation grid
PYTHONPATH=src .venv/bin/python -m nriqa.ablate \
    --config configs/siglip_b16_aux_kadid.yaml \
    --variants configs/ablations/siglip.json

# 5. Inference cost
PYTHONPATH=src .venv/bin/python -m nriqa.inference \
    --checkpoint runs/.../best.pt --size 224

# 6. Compare two prediction files with a paired bootstrap test
PYTHONPATH=src .venv/bin/python -m nriqa.compare \
    --pred A/predictions_test.csv B/predictions_test.csv \
    --labels baseline proposed --reference baseline
```

## Milestones

| # | Week | Deliverable | Status |
|---|---|---|---|
| M1 | 3 | Proposal + literature review + baseline correlation table | Done |
| M2 | 6 | Baseline reproduced + model trained (first checkpoint) + preliminary SROCC/PLCC | Done |
| M3 | 9 | Final model + ablation + cross-dataset check (train KADID → test TID2013) + report/demo | In progress |

## Documentation

* [Literature review](docs/literature_review.md)
* [Architecture specification](docs/architecture.md)
* [Evaluation protocol](docs/protocol.md)
* [Benchmark results](docs/benchmarks.md)
* [Final report](docs/final_report.md)

## Reproducibility notes

* Every run writes its resolved config, environment (Python/PyTorch/CUDA/git
  SHA), per-epoch JSONL metrics, predictions and a summary.
* `torch.backends.cudnn.benchmark = True` favours throughput; exact bitwise
  reproducibility across GPUs is not guaranteed.
* Foundation weights (CLIP/SigLIP/DINOv2/SigLIP2) are downloaded from their
  official sources and are not committed.
* Dataset files are not committed; download links are above.
