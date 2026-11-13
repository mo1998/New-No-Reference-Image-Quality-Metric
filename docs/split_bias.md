# Split-protocol bias on KADID-10k: random vs content-held-out

This audit tests whether the common **random image-level split** inflates
KADID-10k performance relative to our **content-held-out split**.  Under
identical training budgets, random splits inflate published NR-IQA methods by
**+0.025 to +0.162 SROCC** (five-seed means where available; MANIQA +0.104).
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
  each model; seeds `42`–`46` for the main audit models (five seeds).
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

Mean ± std over seeds 42–46; matched 7,125/1,000/2,000 splits.

| Model | Trainable form | Content SROCC | Random SROCC | **Δ (random − content)** |
|---|---|---:|---:|---:|
| **B_Gated + LoRA** | SigLIP2-L + LoRA + gated head | **0.9424 ± 0.0088** | 0.9731 ± 0.0010 | **+0.0307** |
| **Ours, SigLIP-B/16 + KonIQ** | full fine-tune | 0.9416 ± 0.0137 | 0.9670 ± 0.0019 | **+0.0253** |
| **SigLIP2-L/16 384** | partial fine-tune | 0.9392 ± 0.0105 | 0.9706 ± 0.0016 | **+0.0314** |
| **B_Gated (no LoRA)** | SigLIP2-L + gated head | 0.9270 ± 0.0075 | 0.9689 ± 0.0009 | **+0.0418** |
| **MANIQA** | official recipe | 0.8830 ± 0.0171 | 0.9704 ± 0.0016 | **+0.0873** |
| **TOPIQ** | full fine-tune | 0.8470 ± 0.0184 | 0.9528 ± 0.0032 | **+0.1058** |
| **LIQE** | full fine-tune, CE | 0.8225 ± 0.0587 | 0.9183 ± 0.0021 | **+0.0958** |
| **ARNIQA** | full fine-tune | 0.8210 ± 0.0413 | 0.9545 ± 0.0030 | **+0.1335** |
| **CLIP-IQA** | frozen CLIP + head | 0.7716 ± 0.0353 | 0.9331 ± 0.0029 | **+0.1615** |

All cells are five-seed means ± std (seeds 42–46).  **B_Gated + LoRA** follows
the paper's recipe: rank-16 adapters on every `q/k/v/out` projection of the
SigLIP2-L vision encoder (3.4 M trainable on a frozen 320 M backbone) plus the
gated-activation head.  Adding the adapters lifts B_Gated from **0.9270** to
**0.9424** on the content split (paired p = 0.029), i.e. into the same band as
SigLIP2 and our model.  The no-LoRA row is retained to show what the earlier
approximation measured.

Per-seed values: [`multiseed.csv`](../results/split_bias/multiseed.csv);
probe values: [`probe_maniqa.csv`](../results/split_bias/probe_maniqa.csv),
[`probe_liqe.csv`](../results/split_bias/probe_liqe.csv).

**LIQE** now has a genuine full fine-tune: the clean forward plus its native
quality-class cross-entropy is stable and replaces the probe.  **MANIQA** full
and partial-ViT fine-tuning still collapse to constant predictions; its stable
probe is retained.

## 3. Interpretation

1. **The inflation is content leakage, not test difficulty.**  The zero-shot
   control is flat-to-negative; the fine-tuned/probe gap is +0.025 to +0.162
   on KADID.  Five-seed paired tests are significant for TOPIQ (p=0.0003),
   ARNIQA (p=0.0019), CLIP-IQA (p=0.0005), LIQE (p=0.024), B_Gated no-LoRA
   (p=0.0003), B_Gated+LoRA (p=0.0013), MANIQA (p=0.0003), SigLIP2 (p=0.0036)
   and ours (p=0.013).
2. **The effect is 3–5× larger than the hypothesised 0.02–0.04.**  Random
   image-level KADID numbers are not a mild optimistic bias; they are a
   different, much easier problem.
3. **The content-split leaders are statistically tied with each other.**
   Under the content protocol the 93 M KonIQ model reaches **0.9416 ± 0.0137**
   on five single splits, and **0.9481 ± 0.0096** over five content folds.
   B_Gated+LoRA reaches **0.9424 ± 0.0088** and SigLIP2-L/16
   **0.9392 ± 0.0105**.  The paired differences are all inside noise:
   ours − B_Gated+LoRA = −0.0007 (p = 0.77), ours − SigLIP2 = +0.0024
   (p = 0.59).  The remaining models are clearly lower: no-LoRA B_Gated
   **0.9270**, MANIQA **0.8830**, TOPIQ **0.8470**, LIQE **0.8225**,
   ARNIQA **0.8210**, CLIP-IQA **0.7716**.  The random leaders (~0.97) are not
   comparable to the content numbers.
4. **Our model has the smallest measured split gap** (+0.0253, vs +0.0307 for
   B_Gated+LoRA and +0.0314 for SigLIP2), although for these three the
   difference is small in absolute terms; the large gaps belong to the weaker
   baselines (+0.09 to +0.16).  KonIQ training and the fused foundation encoder
   reduce content memorisation, but the residual gap shows leakage still helps.
5. **The honest headline is a controlled *efficiency* result, not a win.**
   Three models sit at 0.939–0.942 on the content split: our 93 M model,
   SigLIP2-L (317 M) and B_Gated+LoRA (320 M, 3.4 M trainable).  Ours matches
   them with **3.4× fewer parameters** and needs no adapter surgery, but it is
   **not** significantly better than either.  The mixed-protocol leaders lose
   0.10–0.17 SROCC under the same protocol, so the published random-split
   ranking does not survive a content-held-out audit.

## 4. Controlled comparison with foundation leaders

| Model | Params | Content SROCC (5 seeds) | Content 5-fold CV | Random SROCC |
|---|---:|---:|---:|---:|
| B_Gated + LoRA | 319.7 M (3.4 M trainable) | **0.9424** | — | 0.9731 |
| **Ours, SigLIP-B/16 + KonIQ** | **93.4 M** | 0.9416 | **0.9481** | 0.9670 |
| SigLIP2-L/16 384 | 316.6 M | 0.9392 | 0.9342 | 0.9706 |
| B_Gated (no LoRA) | 316.5 M | 0.9270 | — | 0.9689 |
| MANIQA (official) | 135.7 M | 0.8830 | — | 0.9704 |
| TOPIQ | 45.2 M | 0.8470 | — | 0.9528 |
| LIQE | 151.3 M | 0.8225 | — | 0.9183 |
| ARNIQA | 23.5 M | 0.8210 | — | 0.9545 |
| CLIP-IQA | 102.3 M | 0.7716 | — | 0.9331 |

The 5-fold column is a *different estimator* of the same content protocol: the
16 test reference contents are rotated so every reference is tested once.  Our
model and SigLIP2 are the only entries with the full five-fold run
(**0.9481 ± 0.0096** vs **0.9342 ± 0.0118**), which is the like-for-like
protocol requested for the foundation comparison.

**Conclusion:** the content split produces a **three-way statistical tie** at
the top — ours **0.9416**, B_Gated+LoRA **0.9424**, SigLIP2 **0.9392** — with
paired p = 0.77 and 0.59 respectively.  Under five-fold content CV our 93 M
model scores **0.9481** against SigLIP2's **0.9342** (+0.0139).  No model in
this audit is significantly better than ours, and ours is not significantly
better than the two foundation-scale models; we can only claim **parity at
3.4× fewer parameters**.  The random leaders are all around 0.97 because of
leakage.  This is a **controlled efficiency result**, not an absolute SOTA
claim.

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
| **MANIQA** | official recipe | 0.7819 (15 ep) | 0.9686 (20 ep) | **+0.1867** |
| **Ours, SigLIP-B/16** | full fine-tune | **0.9266** | 0.9460 | **+0.0194** |

TID2013 content ranking: ours 0.9266 > TOPIQ 0.8650 > LIQE 0.8215 > CLIP-IQA
0.8162 > MANIQA 0.7819 > ARNIQA 0.6490.  The random ranking is nearly flat
(0.86–0.97) because leakage lets every model fit the test contents.

### 4.3 Interpretation

The cross-dataset replication is qualitatively identical to KADID: random
splits inflate every published method (+0.042 to +0.272) while our model gains
only **+0.019**.  MANIQA's official TID content run stopped at 15 epochs; its
3-reference content test is high-variance.  Under the controlled content split our model is again the
strongest.  The TID content numbers have wide error bars because only three
references are held out; the direction and relative ranking are nevertheless
consistent with KADID.

## 6. Caveats

* All eight models have five seeds (42–46).  The content split changes which 16
  references are held out; this measures split variance, not only
  training-seed variance.
* The published models are re-trained with our recipe, not the original papers'
  exact schedules; the comparison is about the split protocol, not a claim that
  our re-implementation equals their published numbers.
* CLIP-IQA uses a frozen official feature head.  TOPIQ/ARNIQA use our common
  recipe; LIQE uses native CE; MANIQA uses the official recipe.  B_Gated+LoRA
  is our reproduction of the paper's adapter + gated-head recipe (the authors
  have not released code); the no-LoRA row is kept as an ablation.  MANIQA's TID
  content run stopped at 15 epochs while random ran 20.  MANIQA KADID ran 20
  epochs on both splits.
* The random split's validation set also leaks, so early stopping is part of the
  leaky protocol rather than an independent control.
* Five-fold content CV (0.9481) is a stronger estimate than the five
  single-split content runs (0.9416); the random runs are single-split, so the
  +0.025 row is a same-protocol comparison, not a five-fold estimate.  The same
  holds for SigLIP2 (0.9342 five-fold vs 0.9392 five single splits).

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

# B_Gated + LoRA (paper recipe; needs the .venv2 transformers stack)
PYTHONPATH=src .venv2/bin/python -m nriqa.run \
  --config configs/split_bias/siglip2_gated_lora.yaml \
  --name bgated_lora_content_s42 --output-dir runs_bgated_lora_s42 \
  --device cuda:0 --set experiment.seed=42

# five-fold content CV on the same folds as our model
PYTHONPATH=src .venv2/bin/python -m nriqa.cv \
  --config configs/siglip2_l16_384_kadid.yaml --k 5 \
  --name siglip2_l16_384_cv65 --python .venv2/bin/python --device cuda:0
```

Results: [`multiseed_summary.csv`](../results/split_bias/multiseed_summary.csv),
[`zero_shot.csv`](../results/split_bias/zero_shot.csv).

## 8. Full-model training notes

The clean forwards are in `src/nriqa/models.py`:

* `maniqa_stable` removes the module-level `save_output` buffer and uses local
  intermediate-feature hooks.
* `liqe_stable` fixes the patch-selection index bug.

**LIQE** is fully fine-tuned with its native quality-class CE: KADID five-seed
0.8225 content / 0.9183 random; TID2013 0.8215 / 0.8633.

**MANIQA** is fully fine-tuned with the official recipe: input scaled to
`[-1,1]` before the model's Inception normalization, Adam lr 1e-5, weight decay
1e-5, batch 8, MSE on raw MOS.  The input-scale fix was decisive — without it
the model collapsed to a constant score.  KADID ran 20 epochs on both splits:
content five-seed 0.8830, random 0.9704 (+0.0873).  TID content 0.7819 (15 epochs,
early-stopped) and random 0.9686 (20 epochs).

**Important lesson:** the official MANIQA preprocessing (`Normalize(0.5,0.5)`)
was the missing piece; the architecture and forward were fine.

