# Split-protocol bias on KADID-10k: random vs content-held-out

This audit tests whether the common **random image-level split** inflates
KADID-10k performance relative to our **content-held-out split**.  Under
identical training budgets, random splits inflate published NR-IQA methods by
**+0.103 to +0.174 SROCC** (three-seed means).  The effect is several times
larger than the 0.02–0.04 initially hypothesised.

## Protocol

Both protocols use exactly **7,125 train / 1,000 val / 2,000 test** images.

| Split | Train refs | Val refs | Test refs | Leakage |
|---|---:|---:|---:|---|
| **Content-held-out** | 57 | 8 | 16 | none — test contents never seen in training |
| **Random image-level** | 81 | 81 | 81 | every test image's reference content also appears in training |

* Content split: `data.split=holdout`, 57/8/16 reference contents.
* Random split: `data.split=random`, shuffled images; sizes matched exactly.
* Same architecture, optimizer, schedule and epoch budget for the two splits of
  each model; three seeds (`42`, `43`, `44`) per cell.
* SROCC and PLCC after the 4-parameter logistic mapping.
* Published models are initialised from their official KonIQ/CLIP checkpoints
  and trained on KADID under the two protocols.  This is a controlled
  re-implementation, not the original papers' exact training recipe.
* **Stable probes** are used where the PyIQA implementation is not safely
  end-to-end trainable: CLIP-IQA, MANIQA and LIQE use frozen official features
  with a small trainable head.  TOPIQ and ARNIQA are fully fine-tuned.

## 1. Zero-shot control: is the random test subset simply easier?

Before any KADID training, each pretrained model is evaluated on the two
2,000-image test subsets with **no KADID training at all**.

| Model | Content test SROCC | Random test SROCC | Δ (random − content) |
|---|---:|---:|---:|
| MANIQA | 0.3875 | 0.3991 | +0.0117 |
| TOPIQ | 0.6204 | 0.6149 | −0.0055 |
| CLIP-IQA | 0.4349 | 0.3831 | −0.0518 |
| ARNIQA | 0.7374 | 0.6986 | −0.0388 |
| LIQE | 0.7267 | 0.7267 | +0.0000 |

The random test subset is **not easier**; three of five models score lower on
it.  Test-set composition therefore does not explain the fine-tuned inflation.

## 2. Fine-tuned / probe results, three seeds

Mean ± std over seeds 42, 43 and 44; matched 7,125/1,000/2,000 splits.

| Model | Trainable form | Content SROCC | Random SROCC | **Δ (random − content)** |
|---|---|---:|---:|---:|
| **CLIP-IQA** | frozen CLIP + head | 0.7764 ± 0.0490 | 0.9322 ± 0.0032 | **+0.1558** |
| **ARNIQA** | full fine-tune | 0.8438 ± 0.0170 | 0.9537 ± 0.0035 | **+0.1099** |
| **TOPIQ** | full fine-tune | 0.8496 ± 0.0245 | 0.9521 ± 0.0042 | **+0.1025** |
| **Ours, SigLIP-B/16 + KonIQ** | full fine-tune | **0.9402 ± 0.0176** | 0.9660 ± 0.0018 | **+0.0258** |
| **MANIQA** | frozen ViT + head | 0.7892 ± 0.0649 | 0.9627 ± 0.0027 | **+0.1735** |
| **LIQE** | frozen CLIP + head | 0.8149 ± 0.0041 | 0.9469 ± 0.0014 | **+0.1321** |

Per-seed values: [`multiseed.csv`](../results/split_bias/multiseed.csv);
probe values: [`probe_maniqa.csv`](../results/split_bias/probe_maniqa.csv),
[`probe_liqe.csv`](../results/split_bias/probe_liqe.csv).

**MANIQA full fine-tuning** was not usable in this harness: its PyIQA
implementation stores intermediate ViT outputs in a module-level `save_output`
buffer and produced NaN losses even with AMP off, frozen ViT, lower LR and small
batches.  The probe is the stable form; its content-split variance is high
because the frozen feature set is fixed and the head is small.

## 3. Interpretation

1. **The inflation is content leakage, not test difficulty.**  The zero-shot
   control is flat-to-negative; the fine-tuned/probe gap is +0.103 to +0.174.
2. **The effect is 3–5× larger than the hypothesised 0.02–0.04.**  Random
   image-level KADID numbers are not a mild optimistic bias; they are a
   different, much easier problem.
3. **Our content-split result is the best controlled result in this study.**
   Under the content protocol the 93 M KonIQ model reaches **0.9402 ± 0.0176**
   on three single splits, and **0.9481 ± 0.0096** over five content folds.
   The published methods fall to **0.8496 / 0.8438 / 0.7764 / 0.7892 / 0.8149**.
   The random-split numbers near 0.95 are not comparable to 0.9481.
4. **Our model is much less split-sensitive** (+0.026 vs +0.103 to +0.174).
   KonIQ training and the foundation encoder reduce content memorisation, but
   the residual gap shows leakage still helps.
5. **The "third-best uncontrolled" framing was too pessimistic.**  The 0.9481
   result is third in a table of mixed random-split numbers; under a controlled
   content split it is the strongest result we can reproduce, and the published
   random-split leaders lose 0.10–0.17 SROCC under the same protocol.

## 4. Caveats

* Three seeds per cell, but the content split changes which 16 references are
  held out; this measures split variance, not only training-seed variance.
* The published models are re-trained with our recipe, not the original papers'
  exact schedules; the comparison is about the split protocol, not a claim that
  our re-implementation equals their published numbers.
* CLIP-IQA, MANIQA and LIQE use frozen official features with a trainable head;
  TOPIQ/ARNIQA are fully fine-tuned.  The probe and full fine-tune results
  should not be compared as if they were the same training method.
* MANIQA full fine-tuning is excluded (NaN); its stable probe is included.
* The random split's validation set also leaks, so early stopping is part of the
  leaky protocol rather than an independent control.
* Five-fold content CV (0.9481) is a stronger estimate than the three
  single-split content runs (0.9402); the random runs are single-split, so the
  +0.026 row is a same-protocol comparison, not a five-fold estimate.

## 5. Reproduce

```bash
# full fine-tune under the content split
PYTHONPATH=src .venv/bin/python -m nriqa.run \
  --config configs/split_bias/base_pyiqa.yaml \
  --name topiq_content --output-dir runs_split --device cuda:0 \
  --set model.kwargs.metric_name=topiq_nr --set train.batch_size=32

# random image-level split (sizes matched to 7125/1000/2000)
PYTHONPATH=src .venv/bin/python -m nriqa.run \
  --config configs/split_bias/base_pyiqa.yaml \
  --name topiq_random --output-dir runs_split --device cuda:0 \
  --set data.split=random --set data.test_frac=0.1975308642 \
  --set data.val_frac=0.0987654321 \
  --set model.kwargs.metric_name=topiq_nr --set train.batch_size=32

# zero-shot control
PYTHONPATH=src .venv/bin/python -m nriqa.split_bias

# frozen-feature probes for MANIQA / LIQE
PYTHONPATH=src .venv/bin/python -m nriqa.extract_features \
  --model maniqa --out features/maniqa_kadid.npz --batch-size 8
PYTHONPATH=src .venv/bin/python -m nriqa.feature_probe \
  --features features/maniqa_kadid.npz --model maniqa --seeds 42 43 44
```

Results: [`multiseed_summary.csv`](../results/split_bias/multiseed_summary.csv),
[`zero_shot.csv`](../results/split_bias/zero_shot.csv).
