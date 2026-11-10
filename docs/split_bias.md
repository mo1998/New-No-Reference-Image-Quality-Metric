# Split-protocol bias on KADID-10k: random vs content-held-out

This audit tests whether the common **random image-level split** inflates
KADID-10k performance relative to our **content-held-out split**.  The answer is
yes — by **+0.116 to +0.188 SROCC** for three published NR-IQA architectures
under identical training budgets.  The effect is several times larger than the
0.02–0.04 initially hypothesised.

## Protocol

Both protocols use exactly **7,125 train / 1,000 val / 2,000 test** images.

| Split | Train refs | Val refs | Test refs | Leakage |
|---|---:|---:|---:|---|
| **Content-held-out** | 57 | 8 | 16 | none — test contents never seen in training |
| **Random image-level** | 81 | 81 | 81 | every test image's reference content also appears in training |

* Content split: `data.split=holdout`, 57/8/16 reference contents.
* Random split: `data.split=random`, shuffled images; sizes matched exactly.
* Same architecture, optimizer, learning-rate schedule, epoch budget and early
  stopping for the two splits of each model.
* Evaluation: SROCC and PLCC after the 4-parameter logistic mapping.
* Published models are initialised from their official KonIQ/CLIP checkpoints
  and then trained on KADID under the two protocols; this is a controlled
  re-implementation, not the original papers' exact training recipe.
* One seed (`42`) per cell.

## 1. Zero-shot control: is the random test subset simply easier?

Before training, each pretrained model is evaluated on the two 2,000-image test
subsets with **no KADID training at all**.  If random-split numbers were inflated
by test-set composition, this control would already show it.

| Model | Content test SROCC | Random test SROCC | Δ (random − content) |
|---|---:|---:|---:|
| MANIQA | 0.3875 | 0.3991 | +0.0117 |
| TOPIQ | 0.6204 | 0.6149 | −0.0055 |
| CLIP-IQA | 0.4349 | 0.3831 | −0.0518 |
| ARNIQA | 0.7374 | 0.6986 | −0.0388 |

The random test subset is **not easier**; three of four models score lower on
it.  Test-set composition therefore does not explain the fine-tuned inflation.

## 2. Fine-tuned under both protocols

| Model | Params | Content SROCC | Random SROCC | **Δ (random − content)** |
|---|---:|---:|---:|---:|
| **CLIP-IQA** (frozen CLIP + trainable head) | 102.3 M | 0.7464 | 0.9340 | **+0.1875** |
| **ARNIQA** (full fine-tune) | 23.5 M | 0.8382 | 0.9569 | **+0.1187** |
| **TOPIQ** (full fine-tune) | 45.2 M | 0.8413 | 0.9569 | **+0.1157** |
| **Ours, SigLIP-B/16 + KonIQ** | 93.4 M | 0.9211 | 0.9659 | **+0.0448** |

PLCC follows the same pattern: CLIP-IQA 0.756 → 0.937, ARNIQA 0.854 → 0.960,
TOPIQ 0.850 → 0.960, ours 0.930 → 0.968.

**MANIQA** could not be fine-tuned in this harness: its PyIQA implementation
stores intermediate ViT outputs in a module-level `save_output` buffer; training
produced NaN losses even with AMP off, frozen ViT, lower LR and small batches.
Its zero-shot split control is reported above; it is excluded from the
fine-tuned table rather than reporting an unstable number.

## 3. Interpretation

1. **The inflation is content leakage, not test difficulty.**  The zero-shot
   control is flat-to-negative; the fine-tuned gap is +0.116 to +0.188.
2. **The effect is 3–5× larger than the hypothesised 0.02–0.04.**  Random
   image-level KADID numbers are not a mild optimistic bias; they are a
   different, much easier problem.
3. **Our content-split result is the best controlled result in this study.**
   Under the content protocol, the 93 M KonIQ model reaches **0.9211** on the
   single split (and **0.9481 ± 0.0096** over five content folds), while the
   published models drop to **0.8413 / 0.8382 / 0.7464**.  The published
   random-split numbers near 0.95 are therefore not comparable to 0.9481.
4. **Our model is also much less split-sensitive** (+0.045 vs +0.116 to
   +0.188).  KonIQ training and the foundation encoder reduce content
   memorisation, even though the residual +0.045 shows that leakage still
   helps.
5. **The "third-best uncontrolled" framing was too pessimistic.**  The 0.9481
   result is third in a table of mixed random-split numbers; under a controlled
   content split it is the strongest result we can reproduce, and the published
   random-split leaders lose 0.10–0.19 SROCC under the same protocol.

## 4. Caveats

* One seed and one content/random split; the effect is large, but seed variance
  is not estimated.
* The published models are re-trained with our recipe, not the original papers'
  exact schedules; the comparison is about the split protocol, not a claim that
  our re-implementation equals their published numbers.
* CLIP-IQA uses a frozen official CLIP backbone with a trainable head; its
  prompt-learning variant was not fine-tuned.
* MANIQA fine-tuning is excluded (NaN); its zero-shot control is included.
* The random split's validation set also leaks, so early stopping is part of the
  leaky protocol rather than an independent control.
* Five-fold content CV (0.9481) is a stronger estimate than the single content
  split (0.9211); the random run is single-split, so the +0.045 row is a
  same-protocol comparison, not a five-fold estimate.

## 5. Reproduce

```bash
# fine-tune one model under the content split
PYTHONPATH=src .venv/bin/python -m nriqa.run \
  --config configs/split_bias/base_pyiqa.yaml \
  --name topiq_content --output-dir runs_split --device cuda:0 \
  --set model.kwargs.metric_name=topiq_nr --set train.batch_size=32

# the random image-level split (sizes matched to 7125/1000/2000)
PYTHONPATH=src .venv/bin/python -m nriqa.run \
  --config configs/split_bias/base_pyiqa.yaml \
  --name topiq_random --output-dir runs_split --device cuda:0 \
  --set data.split=random --set data.test_frac=0.1975308642 \
  --set data.val_frac=0.0987654321 \
  --set model.kwargs.metric_name=topiq_nr --set train.batch_size=32

# zero-shot control
PYTHONPATH=src .venv/bin/python -m nriqa.split_bias
```

Results: [`results/split_bias/split_bias_summary.csv`](../results/split_bias/split_bias_summary.csv),
[`results/split_bias/zero_shot.csv`](../results/split_bias/zero_shot.csv).
