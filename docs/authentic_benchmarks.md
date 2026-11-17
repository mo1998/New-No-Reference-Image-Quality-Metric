# Authentic-distortion benchmarks: LIVE Challenge and SPAQ

This document evaluates the project's models on the two established
authentic-distortion NR-IQA benchmarks, complementing the synthetic
KADID-10k/TID2013 results.

## Protocol

* **LIVE Challenge (LIVEC):** 1,162 standard test images (the seven official
  training images `t1`–`t7` are excluded). MOS range 3.4–92.4.
* **SPAQ:** 11,125 smartphone photographs. MOS range 2–96.
* **Zero-shot:** no model is trained on LIVEC or SPAQ. Models that use KonIQ-10k
  only see KonIQ during training; LIVEC/SPAQ are held out.
* SROCC (no fitting) and PLCC after a 4-parameter logistic mapping.
* The headline 93 M model is reported as mean ± std over the five content-fold
  checkpoints of the KADID+KonIQ run.
* Images are resized/centre-cropped to the model's native input (384 px for the
  headline models, 224 px for the baselines).
* Published reference numbers are from the Revisiting Vision-Language
  Foundations paper and use their own preprocessing/protocol; treat them as
  context, not an exact head-to-head.

## 1. LIVE Challenge (1,162 images)

| Method | Params | LIVEC SROCC ↑ | LIVEC PLCC ↑ | Source |
|---|---:|---:|---:|---|
| **SigLIP-L/16 384 + KonIQ** | 316.6 M | **0.8846** | **0.9055** | this work |
| **SigLIP-B/16 384 + KonIQ (5-fold)** | **93.4 M** | **0.8662 ± 0.0014** | **0.8915 ± 0.0016** | this work |
| TOPIQ (PyIQA) | 45.2 M | 0.8283 | 0.8488 | PyIQA zero-shot |
| SigLIP-L/16 384, KADID only | 316.6 M | 0.7388 | 0.7842 | this work |
| SigLIP-L/16 256, KADID only | 316.3 M | 0.6780 | 0.7520 | this work |
| MANIQA (PyIQA) | 135.7 M | 0.6560 | 0.7205 | PyIQA zero-shot |
| MUSIQ (PyIQA) | 27.1 M | 0.6477 | 0.7126 | PyIQA zero-shot |
| QStatFusion 224, KADID only | 93.8 M | 0.6432 | 0.6685 | this work |
| SigLIP-B/16 384, KADID only | 93.4 M | 0.6410 | 0.6674 | this work |
| SigLIP-B/16 224, KADID only | 93.1 M | 0.5978 | 0.6509 | this work |
| CLIP-IQA (PyIQA) | ~150 M | 0.5920 | 0.6266 | PyIQA zero-shot |
| ARNIQA (PyIQA) | 23.5 M | 0.5763 | 0.6240 | PyIQA zero-shot |
| BRISQUE | 0 | 0.3296 | 0.3586 | PyIQA zero-shot |
| NIQE | 0 | 0.3122 | 0.3419 | PyIQA zero-shot |

**Published context** (Revisiting VL, CLIVE column): CLIP 0.841, SigLIP2 0.875,
B_Gated 0.887.  The 316 M KonIQ model (0.8846) is between SigLIP2 and B_Gated;
the 93 M model (0.8662) beats TOPIQ and the other PyIQA baselines.

## 2. SPAQ (11,125 images)

| Method | Params | SPAQ SROCC ↑ | SPAQ PLCC ↑ | Source |
|---|---:|---:|---:|---|
| **SigLIP-L/16 384 + KonIQ** | 316.6 M | **0.8986** | **0.8990** | this work |
| **SigLIP-B/16 384 + KonIQ (5-fold)** | **93.4 M** | **0.8884 ± 0.0024** | **0.8909 ± 0.0019** | this work |
| TOPIQ (PyIQA) | 45.2 M | 0.8709 | 0.8764 | PyIQA zero-shot |
| ARNIQA (PyIQA) | 23.5 M | 0.7935 | 0.8021 | PyIQA zero-shot |
| MUSIQ (PyIQA) | 27.1 M | 0.7806 | 0.7912 | PyIQA zero-shot |
| SigLIP-B/16 384, KADID only | 93.4 M | 0.7493 | 0.7621 | this work |
| CLIP-IQA (PyIQA) | ~150 M | 0.6460 | 0.6464 | PyIQA zero-shot |

**Published context** (Revisiting VL, SPAQ column): CLIP 0.916, SigLIP2 0.927,
B_Gated 0.928.  The 316 M KonIQ model (0.8986) and the 93 M model (0.8884) are
below the top foundation probes but above TOPIQ and all other PyIQA baselines.

## 3. What the authentic benchmarks show

1. **Authentic training is decisive.**  The 93 M B/16 improves from
   0.6410 → 0.8662 on LIVEC (+0.225) and from 0.7493 → 0.8884 on SPAQ
   (+0.139) when KonIQ-10k is added to KADID.  KADID-only models overfit
   synthetic distortions and collapse on authentic captures.
2. **The 93 M model is genuinely competitive.**  It beats TOPIQ (45 M) by
   +0.038 SROCC on LIVEC and +0.017 on SPAQ, and it beats every other PyIQA
   baseline, while remaining 10.9 ms / 390 MB.
3. **Capacity still helps on authentic data.**  The 316 M L/16 + KonIQ model is
   +0.018 LIVEC and +0.010 SPAQ over the 93 M model, but costs 3.4× parameters
   and 27% higher latency.
4. **The synthetic-to-authentic gap is the main generalisation bottleneck.**
   The best KADID-only 316 M model reaches 0.7388 LIVEC / 0.7493 SPAQ, i.e.
   roughly 0.15–0.21 below the KonIQ-trained models.

## 4. Caveats

* Published CLIP/SigLIP2 numbers use different preprocessing, prompts/heads and
  possibly different LIVEC variants; the table is context, not a controlled
  leaderboard.
* LIVEC has 1,162 test images after excluding the seven official training
  images; some papers report 1,169 or use a different split.
* SPAQ MOS is on a 1–100 scale; SROCC is scale-invariant, PLCC uses the logistic
  mapping.
* PyIQA baselines are pretrained on KonIQ or multiple datasets; they are not
  retrained for LIVEC/SPAQ.
* All results are zero-shot for LIVEC/SPAQ and single-machine measurements.

## 5. Reproduce

```bash
# LIVEC
PYTHONPATH=src .venv/bin/python -m nriqa.eval_authentic \
  --dataset livec --root data/livec/LIVEC \
  --checkpoint-glob 'runs_cv65_koniq/*/last.pt' --label b16_koniq

# SPAQ
PYTHONPATH=src .venv/bin/python -m nriqa.eval_authentic \
  --dataset spaq --root data/spaq/SPAQ \
  --checkpoint-glob 'runs_cv65_koniq/*/last.pt' --label b16_koniq
```

Data sources: IQA-Toolbox Datasets mirror (`live_challenge.tgz`, `spaq.tgz`),
LIVE Challenge (Ghadiyaram & Bovik, 2016), SPAQ (Fang et al., CVPR 2020).
