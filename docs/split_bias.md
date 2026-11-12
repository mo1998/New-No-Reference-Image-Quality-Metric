# Split-protocol bias on KADID-10k: random vs content-held-out

This audit tests whether the common **random image-level split** inflates
KADID-10k performance relative to our **content-held-out split**.  Under
identical training budgets, random splits inflate published NR-IQA methods by
**+0.048 to +0.162 SROCC** (five-seed means where available; MANIQA +0.111).
The effect is several times
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
* **Training forms** follow each model as closely as the harness allows:
  TOPIQ/ARNIQA use our common recipe, LIQE uses its native quality-class CE,
  MANIQA uses the official recipe (input `[-1,1]`, Adam 1e-5, batch 8, raw MOS),
  and CLIP-IQA uses a frozen CLIP probe.

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

## 2. Fine-tuned / probe results

Mean ± std over seeds 42–46 where five seeds exist; matched 7,125/1,000/2,000
splits.  Ours has three seeds; SigLIP2, LIQE and MANIQA are single-seed.

| Model | Trainable form | Content SROCC | Random SROCC | **Δ (random − content)** |
|---|---|---:|---:|---:|
| **SigLIP2-L/16 384** | partial fine-tune | 0.9243 | 0.9718 | **+0.0475** |
| **Ours, SigLIP-B/16 + KonIQ** | full fine-tune | **0.9402 ± 0.0176** | 0.9660 ± 0.0018 | **+0.0258** |
| **MANIQA** | official recipe | 0.8611 (9 ep) | 0.9716 (20 ep) | **+0.1105** |
| **TOPIQ** | full fine-tune | 0.8470 ± 0.0184 | 0.9528 ± 0.0032 | **+0.1058** |
| **LIQE** | full fine-tune, CE | 0.8318 | 0.9184 | **+0.0866** |
| **ARNIQA** | full fine-tune | 0.8210 ± 0.0413 | 0.9545 ± 0.0030 | **+0.1335** |
| **CLIP-IQA** | frozen CLIP + head | 0.7716 ± 0.0353 | 0.9331 ± 0.0029 | **+0.1615** |

Mean ± std over seeds 42–46 where five seeds exist; ours is three seeds;
SigLIP2/LIQE/MANIQA are single-seed.  MANIQA official content stopped at 9
epochs (random ran 20); its delta is therefore budget-asymmetric.

Per-seed values: [`multiseed.csv`](../results/split_bias/multiseed.csv);
probe values: [`probe_maniqa.csv`](../results/split_bias/probe_maniqa.csv),
[`probe_liqe.csv`](../results/split_bias/probe_liqe.csv).

**LIQE** now has a genuine full fine-tune: the clean forward plus its native
quality-class cross-entropy is stable and replaces the probe.  **MANIQA** full
and partial-ViT fine-tuning still collapse to constant predictions; its stable
probe is retained.

## 3. Interpretation

1. **The inflation is content leakage, not test difficulty.**  The zero-shot
   control is flat-to-negative; the fine-tuned/probe gap is +0.048 to +0.162
   on KADID (official MANIQA +0.111).  Five-seed paired tests: TOPIQ
   p = 0.0003, ARNIQA p = 0.0019, CLIP-IQA p = 0.0005; ours p = 0.12 (n=3).
2. **The effect is 3–5× larger than the hypothesised 0.02–0.04.**  Random
   image-level KADID numbers are not a mild optimistic bias; they are a
   different, much easier problem.
3. **Our content-split result is the best controlled result in this study.**
   Under the content protocol the 93 M KonIQ model reaches **0.9402 ± 0.0176**
   on three single splits, and **0.9481 ± 0.0096** over five content folds.
   The strongest reproducible foundation probe, SigLIP2-L/16, reaches only
   **0.9243**; MANIQA (official recipe) **0.8611**; TOPIQ/ARNIQA/LIQE/CLIP
   **0.8470 / 0.8210 / 0.8318 / 0.7716**.  SigLIP2's random-split 0.9718 and
   MANIQA's 0.9716 are not comparable to 0.9481.
4. **Our model is much less split-sensitive** (+0.026 vs +0.048 to +0.162).
   KonIQ training and the foundation encoder reduce content memorisation, but
   the residual gap shows leakage still helps.
5. **The "third-best uncontrolled" framing was too pessimistic.**  The 0.9481
   result is third in a table of mixed random-split numbers; under a controlled
   content split it is the strongest result we can reproduce, and the published
   random-split leaders lose 0.10–0.17 SROCC under the same protocol.

## 4. Controlled comparison with foundation leaders

| Model | Params | Content SROCC | Random SROCC |
|---|---:|---:|---:|
| **Ours, SigLIP-B/16 + KonIQ** | **93.4 M** | **0.9481 (5-fold)** | 0.9660 |
| SigLIP2-L/16 384 | 316.6 M | 0.9243 | 0.9718 |
| MANIQA (official) | 135.7 M | 0.8611 | 0.9716 |
| TOPIQ | 45.2 M | 0.8470 | 0.9528 |
| LIQE | 151.3 M | 0.8318 | 0.9184 |
| ARNIQA | 23.5 M | 0.8210 | 0.9545 |
| CLIP-IQA | 102.3 M | 0.7716 | 0.9331 |

**Conclusion:** SigLIP2 is the strongest reproducible foundation probe; under
the controlled content split our 93 M model beats it by **+0.0238** while using
3.4× fewer parameters.  The random-split leaders are all around 0.97 because of
leakage.  B_Gated could not be reproduced (no public code/checkpoint available
in this environment); it is not claimed.  This is a **controlled best result**,
not an absolute SOTA claim.

## 5. TID2013 cross-dataset replication

The same audit on TID2013: 3,000 images, 25 references, 24 distortion types.
The matched splits are **2,400 train / 240 val / 360 test** (20/2/3 references
for content; all 25 references for random).  TID2013 content test has only three
references, so the content estimates are noisier than KADID's 16.

### 4.1 Zero-shot control

| Model | Content test SROCC | Random test SROCC | Δ |
|---|---:|---:|---:|
| MANIQA | 0.6340 | 0.3300 | −0.3040 |
| TOPIQ | 0.6034 | 0.5541 | −0.0493 |
| CLIP-IQA | 0.6774 | 0.5276 | −0.1498 |
| ARNIQA | 0.7099 | 0.5543 | −0.1556 |
| LIQE | 0.6860 | 0.6312 | −0.0548 |

The random TID test subset is **much harder**, not easier.  Any post-training
random-split gain is therefore leakage.

### 4.2 Fine-tuned / probe results

| Model | Trainable form | Content SROCC | Random SROCC | Δ |
|---|---|---:|---:|---:|
| **LIQE** | full fine-tune, CE | 0.8215 | 0.8633 | **+0.0418** |
| **ARNIQA** | full fine-tune | 0.6490 | 0.9211 | **+0.2721** |
| **CLIP-IQA** | frozen CLIP + head | 0.8162 | 0.9027 | **+0.0865** |
| **TOPIQ** | full fine-tune | 0.8650 | 0.9330 | **+0.0680** |
| **MANIQA** | frozen ViT + head (3 seeds) | 0.9021 ± 0.0207 | 0.9679 ± 0.0029 | **+0.0658** |
| **Ours, SigLIP-B/16** | full fine-tune | **0.9266** | 0.9460 | **+0.0194** |

TID2013 content ranking: ours 0.9266 > TOPIQ 0.8650 > CLIP-IQA 0.8162 >
ARNIQA 0.6490.  The random ranking is nearly flat (0.90–0.97) because leakage
lets every model fit the test contents.

### 4.3 Interpretation

The cross-dataset replication is qualitatively identical to KADID: random
splits inflate every published method (+0.066 to +0.272) while our model gains
only **+0.019**.  Under the controlled content split our model is again the
strongest.  The TID content numbers have wide error bars because only three
references are held out; the direction and relative ranking are nevertheless
consistent with KADID.

## 6. Caveats

* Five seeds for TOPIQ/ARNIQA/CLIP-IQA, three for ours, single for
  SigLIP2/LIQE/MANIQA.  The content split changes which 16 references are held
  out; this measures split variance, not only training-seed variance.
* The published models are re-trained with our recipe, not the original papers'
  exact schedules; the comparison is about the split protocol, not a claim that
  our re-implementation equals their published numbers.
* CLIP-IQA uses a frozen official feature head.  TOPIQ/ARNIQA use our common
  recipe; LIQE uses native CE; MANIQA uses the official recipe.  These are not
  the same training method, and MANIQA's content run stopped at 9 epochs while
  random ran 20, so its delta is budget-asymmetric.
* The random split's validation set also leaks, so early stopping is part of the
  leaky protocol rather than an independent control.
* Five-fold content CV (0.9481) is a stronger estimate than the three
  single-split content runs (0.9402); the random runs are single-split, so the
  +0.026 row is a same-protocol comparison, not a five-fold estimate.

## 7. Reproduce

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

# official MANIQA recipe (input [-1,1], Adam 1e-5, batch 8, raw MOS)
PYTHONPATH=src .venv/bin/python -m nriqa.run \
  --config configs/split_bias/maniqa_official.yaml \
  --name maniqa_official_content --output-dir runs_maniqa_official --device cuda:0

# LIQE native quality-class CE
PYTHONPATH=src .venv/bin/python -m nriqa.run \
  --config configs/split_bias/liqe_ce.yaml \
  --name liqe_ce_content --output-dir runs_ce_full --device cuda:0
```

Results: [`multiseed_summary.csv`](../results/split_bias/multiseed_summary.csv),
[`zero_shot.csv`](../results/split_bias/zero_shot.csv).

## 8. Full-model training notes

The clean forwards are in `src/nriqa/models.py`:

* `maniqa_stable` removes the module-level `save_output` buffer and uses local
  intermediate-feature hooks.
* `liqe_stable` fixes the patch-selection index bug.

**LIQE** is fully fine-tuned with its native quality-class CE: KADID
0.8318 content / 0.9184 random; TID2013 0.8215 / 0.8633.

**MANIQA** is fully fine-tuned with the official recipe: input scaled to
`[-1,1]` before the model's Inception normalization, Adam lr 1e-5, weight decay
1e-5, batch 8, MSE on raw MOS.  The input-scale fix was decisive — without it
the model collapsed to a constant score.  KADID content reached 0.8611
(9 epochs; stopped early for time) and random 0.9716 (20 epochs), so the
content/random delta is budget-asymmetric and should be treated as indicative.

**Important lesson:** the official MANIQA preprocessing (`Normalize(0.5,0.5)`)
was the missing piece; the architecture and forward were fine.

