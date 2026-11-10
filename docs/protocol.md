# Evaluation protocol

This document is the contract for every headline number in the project.  It is
deliberately stricter than the common practice in NR-IQA papers, where random
image-level splits can leak content between train and test.

## 1. Data

| Dataset | Distorted images | Reference contents | Distortion types | MOS range |
|---|---:|---:|---:|---|
| KADID-10k | 10,125 | 81 | 25 × 5 levels | 1.0 – 4.93 (higher is better) |
| TID2013 | 3,000 | 25 | 24 × 5 levels | 0.24 – 7.21 (higher is better) |
| KonIQ-10k | 10,073 | 10,073 (unique) | authentic | 1.0 – 4.31 (higher is better) |

KADID-10k is the primary benchmark.  TID2013 is used for (a) zero-shot
cross-dataset evaluation of KADID-only models and (b) additional training data
in the multi-dataset setting.  The two uses are never mixed in one claim.

## 2. Splits

### Single held-out split (pilots)

* Contents are shuffled with `numpy.random.default_rng(seed=42)`.
* 20% of contents are the test set (16 KADID references, 2,000 images).
* 10% are the validation set (8 references, 1,000 images).
* The remaining 57 references (7,125 images) are used for training.
* Validation is used **only** for early stopping/model selection; test is
  touched once per run.

### Five-fold content cross-validation (headline)

* Contents are partitioned into five disjoint folds (`kfold=5`).
* For each fold: one content fold (≈16 refs) is test, one held-out content
  subset is validation, the rest is train.
* Report mean ± sample standard deviation across folds, plus every fold value.
* The final protocol variant (`select_best=false`, `val_frac≈0`) trains on all
  65 non-test references with a fixed epoch budget; this matches the usual
  80/20 training budget and removes selection noise.

### Multi-dataset training

* KADID train contents + **all** TID2013 contents (both are disjoint from the
  KADID test fold).
* TID2013 MOS is linearly mapped to the KADID 1–5 range so one regression head
  is meaningful.  The mapping is monotonic and rank-preserving.
* Models trained this way are **not** used for zero-shot KADID→TID claims;
  their TID evaluation is in-domain.  The config disables the cross-dataset
  report for these runs.
* **KonIQ-10k** is an optional authentic-distortion training set (DaRUS
  doi:10.18419/DARUS-2435, files 72286 images and 72288 labels, CC BY 4.0).
  KonIQ has no KADID distortion/level label, so auxiliary heads are masked for
  KonIQ batches.  KADID and TID test contents are never in KonIQ.

## 3. Metrics

* **SROCC** — Spearman rank correlation, no fitting.  Primary metric.
* **PLCC** — Pearson correlation after a 4-parameter logistic mapping fitted on
  the evaluated split (VQEG convention).
* **RMSE** — root mean squared error after the same logistic mapping.
* **KRCC** — Kendall rank correlation, reported as a robustness check.
* Predictions are always oriented so that higher means better quality.

### Uncertainty and significance

* 95% confidence intervals are obtained by **bootstrapping reference contents**
  (1,000 resamples), not individual images.  This respects the dependence
  between distorted versions of the same content.
* Method comparisons use a **paired content-level bootstrap** (2,000 resamples)
  on the same resamples for both methods.  A difference is called significant
  when the two-sided bootstrap p-value is < 0.05.
* A single split with 16 test contents has a standard error of roughly 0.03 in
  SROCC; differences below ~0.02 should not be over-interpreted without CV.

## 4. Preprocessing

| Model family | Train | Test | Normalisation |
|---|---|---|---|
| From-scratch (MSOSNet, WaveQNet, TinyCNN) | resize short side to `size+32`, random 224 crop, horizontal flip | resize short side `size+32`, center crop 224 | raw `[0,1]` |
| Pretrained CNN (ResNet/MobileNet) | same | same | ImageNet |
| Timm foundation (CLIP/SigLIP/DINOv2) | same | same | backbone-specific (inside the model) |
| Classical (BRISQUE/NIQE) | — | native 512×384 (`eval_mode=full`) | PyIQA default |

Augmentation is deliberately quality-preserving: no colour jitter, no strong
zoom, no rotation that creates new interpolation artifacts.  Random crops are
one-sided (the model always sees at least 70% of the image).

## 5. Model selection and reporting

* Early stopping monitors validation SROCC with a patience of 6–12 epochs.
* Test metrics are computed once from the selected checkpoint.
* Cross-dataset metrics are computed from the same checkpoint on the full
  TID2013 set, with a fresh logistic fit.
* Parameter counts are **total** parameters, not only trainable ones.
* Latency is measured batch=1 on an RTX 4090 after 5 warm-up runs (median of
  30 runs), including the model's input normalisation.
* Peak memory is measured with `torch.cuda.max_memory_allocated()`.
* Every run stores its fully-resolved config, environment (Python/PyTorch/CUDA,
  git SHA, dirty flag), per-epoch metrics, per-image predictions and summary.

## 6. Reproducibility

* `SEED = 42` seeds Python, NumPy, PyTorch CPU/CUDA.
* DataLoader shuffling uses a seeded `torch.Generator`.
* `torch.backends.cudnn.benchmark = True` is used for throughput; exact
  bitwise reproducibility is therefore not guaranteed on different GPUs.
  Reported correlations are stable to ~1e-3 across reruns on the same machine.
* All configurations are YAML and support `include:` inheritance; the resolved
  config is copied into each run directory.

## 7. Known limitations

* KADID and TID2013 are both synthetic-distortion benchmarks.  A model that
  ranks KADID well is not automatically good on authentic distortions
  (SPAQ/KonIQ/LIVE Challenge); this is reported as a limitation.
* A single 5-fold estimate still has finite-sample uncertainty; confidence
  intervals are reported wherever practical.
* Multi-dataset training changes the comparison class and is labelled as such.


## 8. Authentic-distortion evaluation

Two authentic benchmarks are evaluated zero-shot:

* **LIVE Challenge (LIVEC):** 1,169 released images; the seven official training
  images (`t1`–`t7`) are excluded, leaving the standard 1,162-image test set.
  MOS 3.4–92.4.
* **SPAQ:** 11,125 smartphone photographs, MOS 2–96.

No model is trained on either benchmark.  Models that use KonIQ-10k see only
KonIQ during training.  SROCC is reported without fitting; PLCC uses the
4-parameter logistic mapping.  Full tables and caveats:
[authentic_benchmarks.md](authentic_benchmarks.md).
