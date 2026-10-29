# Project 28 — Building and Validating a New No-Reference Image-Quality Metric

**Course:** CSE 681 — Digital Image Processing & Video Processing

Build a lightweight **no-reference** image-quality model: predict a quality score from a
single distorted image, with no pristine reference, and validate it against human MOS
labels (KADID-10k, TID2013) using SROCC / PLCC.

## Team

| Code | Name |
|---|---|
| 2400008 | Mohamed Ahmed Fathy Abdelrazik |
| 2501922 | Novert Ahmed Fahim Abdelfattah |
| 2501897 | Farah Hossam Eldin Mahmoud |

## Milestones

| # | Week | Deliverable | Status |
|---|---|---|---|
| M1 | 3 | Proposal + literature review + baseline correlation table (PSNR/SSIM/LPIPS vs MOS) | Started |
| M2 | 6 | Baseline reproduced + model trained (first checkpoint) + preliminary SROCC/PLCC | Not started |
| M3 | 9 | Final model + ablation + cross-dataset check (train KADID → test TID2013) + report/demo | Not started |

## Datasets

| Dataset | Size | Role | Link |
|---|---|---|---|
| KADID-10k | 10,125 images + MOS | train / test | http://database.mmsp-kn.de/kadid-10k-database.html |
| TID2013 | 3,000 images + MOS | cross-dataset check (M3) | http://www.ponomarenko.info/tid2013.htm |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Requires Python 3.10+.

## Evaluation protocol

- Split **by reference image (content)**, never by distorted image.
- Report **SROCC** (no fitting) and **PLCC + RMSE** (after 4/5-parameter logistic fit) vs MOS.
- Baselines: PSNR, SSIM, LPIPS (DISTS optional). LPIPS/DISTS are distances
  (lower = better) → expected negative correlation with MOS; state the sign convention.
- Fixed seed: `SEED = 42` (`src/utils.py`) for every headline result.

## Repository layout

```
├── data/                          # datasets (not committed) — see Datasets
├── notebooks/
│   └── 00_colab_starter.ipynb    # Colab starter (GPU, Drive, clone, baselines, push)
├── results/
│   ├── .gitkeep
│   ├── m1_baseline_correlations.csv
│   └── m1_baseline_correlations_test_split_raw.csv
├── src/
│   ├── baselines.py              # PSNR / SSIM / LPIPS
│   ├── eval_protocol.py          # content split + SROCC/PLCC/RMSE
│   ├── run_m1_baselines.py       # M1 baseline runner
│   └── utils.py                  # shared seed / reproducibility helpers
└── tests/
    ├── test_baselines.py         # PSNR cap regression test
    └── test_eval_protocol.py     # protocol sanity tests
```

