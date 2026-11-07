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
| SigLIP-L/16 384 + aux + rank + TTA | 316 M | **0.9481 ± 0.0074** | **0.9526** | 0.7232 ± 0.0205 | 15.1 ms |
| SigLIP-L/16 256 + aux + rank + TTA | 316 M | 0.9365 ± 0.0074 | 0.9426 | **0.7353 ± 0.0084** | 9.4 ms |
| SigLIP-B/16 384 + aux + rank + TTA | 93.4 M | 0.9393 ± 0.0149 | 0.9441 | 0.7166 ± 0.0123 | 10.9 ms |
| SigLIP-B/16 224 + aux + rank + TTA | 93.1 M | 0.9207 ± 0.0166 | 0.9271 | 0.7284 ± 0.0157 | 6.2 ms |
| Distilled SigLIP-B/16 384 (from L/16) | 93.4 M | **0.9414 ± 0.0115** | 0.948 | 0.7132 | 10.9 ms |
| QStatFusion 224 | 93.8 M | 0.8885* | 0.8974* | **0.7528*** | 8.5 ms |
| MSOSNet (from scratch) | 0.30 M | 0.721* | 0.730* | 0.483* | 3.7 ms |
| ResNet-18 (ImageNet) | 11.2 M | 0.769* | 0.779* | 0.678* | 2.7 ms |

\* single content split (57/8/16 refs), used for rapid iteration.  The 384 and
224 rows are 5-fold content CV; see `docs/benchmarks.md` for the full table.

**Honest positioning:** these are strong efficiency points, not a new state of
the art.  Published top methods reach 0.94–0.97 KADID SROCC (MANIQA 0.944,
LIQE 0.930, SigLIP2 probe 0.970) but are 24–400 M parameters.  Our 93 M model is
competitive with the 24–48 M specialised networks and significantly better than
off-the-shelf NR-IQA metrics.  See `docs/final_report.md` §4 for the full
comparison and the limitations.

**Statistically significant findings** (paired content-level bootstrap):

* Distortion-aware auxiliary heads + ranking + TTA: **+0.028 SROCC, p = 0.003**.
* SigLIP-L/16 384 vs 256 (paired 5-fold): **+0.012 SROCC, p = 0.023**.
* L384 vs 93 M B/384: **+0.009 SROCC, p = 0.098** (not significant with 5 folds).
* NSS branch (QStatFusion) vs SigLIP on TID2013: **+0.027 SROCC, p = 0.003**.
* Removing auxiliary heads costs **0.042** KADID SROCC.
* Full fine-tuning overfits: **−0.058** vs partial fine-tuning.
* SigLIP2-B is significantly worse than SigLIP-B on this split (p = 0.003).

## Repository layout

```
├── configs/                     # YAML experiment configs (include: inheritance)
│   ├── ablations/               # ablation grids
│   └── hpo/                     # hyperparameter search spaces
├── data/                        # KADID-10k + TID2013 (not committed)
├── docs/
│   ├── architecture.md          # model specifications
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

Expected layout:

```
data/kadid10k/{images/, dmos.csv}
data/tid2013/{distorted_images/, reference_images/, mos_with_names.txt}
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
