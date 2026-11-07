# Literature review: no-reference image quality assessment (NR-IQA)

**Snapshot:** November 2026. **Primary benchmarks:** KADID-10k (81 reference
contents, 25 synthetic distortion types × 5 levels = 10,125 images) and TID2013
(25 references, 24 distortions × 5 levels = 3,000 images). **Protocol
convention:** SROCC (no fitting), PLCC/RMSE after a 4-parameter logistic fit.

> Numbers below are **as reported by the cited papers**. They are not directly
> comparable across rows: papers differ in train/test splitting (random image
> split vs. content split), number of repetitions, input resolution, and
> whether the backbone was pretrained on external data. Our own measurements,
> run on one fixed content split, are reported separately in
> `docs/benchmarks.md`.

---

## 1. Problem and evaluation pitfalls

NR-IQA predicts a perceptual quality score from a single distorted image. The
central difficulty is not regression capacity but **content invariance**: the
same distortion severity looks different on a smooth sky and on a textured
forest, and a model can learn to recognise the 81 KADID contents instead of the
distortion. Consequences for experimental design:

1. **Split by reference content, never by distorted image.** A random image
   split leaks content across train/test and inflates correlations.
2. **Report variance across splits/folds.** A single 80/20 content split with
   16 test references has a large standard error; differences below ~0.01
   SROCC are usually not meaningful.
3. **Use content-level bootstrap CIs** and paired tests for method comparisons.
4. **Separate in-dataset accuracy from cross-dataset generalisation.** KADID is
   synthetic and visually close to TID2013; a model can exploit dataset-specific
   distortion signatures and still fail on authentic distortions (SPAQ, KonIQ,
   LIVE Challenge).
5. **Report efficiency.** Parameters alone are insufficient: latency, peak
   memory, input resolution and training cost all matter for deployment.

## 2. Taxonomy of current methods

| Family | Representatives | Strength | Weakness |
|---|---|---|---|
| Handcrafted NSS | BRISQUE, NIQE, IL-NIQE, HOSA, CORNIA | Zero training, tiny, interpretable | Weak on synthetic mixtures and authentic distortions |
| CNN regression | DBCNN, HyperIQA, TReS, MUSIQ | Strong in-dataset fit | 24–150 M params; content overfitting; weak cross-dataset |
| Self-supervised / distortion-manifold | CONTRIQUE, Re-IQA, ARNIQA, TRIQA | Learns quality-aware representation without MOS; strong generalisation per parameter | Needs synthetic pretraining; linear head limits calibration |
| Vision-language / foundation probes | CLIP-IQA, LIQE, QPT, CMPA, SigLIP2 probes | Best raw accuracy; semantic quality priors | 150–400 M params; expensive; perceptual cues can be “submerged” by semantics |
| Efficient hybrids | LoDa (9 M), QMamba (30 M), MANIQA (24 M) | Better accuracy/compute than large VLMs | Still 10–30 M; cross-dataset remains ~0.7–0.8 |

## 3. Reported results on KADID-10k and TID2013

### 3.1 In-dataset (train and test on the same dataset)

| Method | Venue | Params | KADID SROCC | KADID PLCC | TID2013 SROCC | TID2013 PLCC | Source |
|---|---|---:|---:|---:|---:|---:|---|
| BRISQUE | TIP 2012 | 0 | 0.528 | 0.567 | 0.604 | 0.694 | ARNIQA / CONTRIQUE tables |
| NIQE | TIP 2013 | 0 | 0.374 | 0.428 | 0.315 | 0.393 | ARNIQA / CONTRIQUE tables |
| DBCNN | TIP 2020 | 24 M | 0.851 | 0.856 | 0.816 | 0.865 | ARNIQA table |
| HyperIQA | CVPR 2020 | 27 M | 0.852 | 0.845 | 0.840 | 0.858 | ARNIQA / CMPA tables |
| TReS | ICCV 2021 | 152 M | 0.859 | 0.858 | 0.863 | 0.883 | ARNIQA table |
| MUSIQ | ICCV 2021 | 27 M | 0.875 | 0.872 | 0.773 | 0.815 | CMPA table |
| CONTRIQUE | CVPR 2022 | 24 M | 0.934 | 0.937 | 0.843 | 0.857 | CONTRIQUE table |
| Re-IQA | CVPR 2023 | 48 M | 0.872 | 0.885 | 0.804 | 0.861 | CMPA table |
| ARNIQA | CVPR 2024 | 24 M | 0.908 | 0.912 | 0.880 | 0.901 | ARNIQA table |
| LIQE | CVPR 2023 | 151 M | 0.930 | 0.931 | — | — | CMPA table |
| MANIQA | CVPR 2022 | 24 M | 0.944 | 0.946 | 0.937 | 0.943 | MANIQA table |
| QMamba | 2024/25 | 30 M | 0.923 | 0.938 | 0.896 | 0.913 | CMPA table |
| LoDa | 2024/25 | 9 M | 0.931 | 0.936 | 0.869 | 0.901 | CMPA table |
| DEIQT | CVPR 2022 | 24 M | 0.889 | 0.887 | 0.892 | 0.908 | CMPA table |
| CMPA (CLIP ViT-L adapter) | arXiv 2026 | 1.8 M trainable / ~428 M total | 0.938 | 0.941 | 0.878 | 0.905 | CMPA table |
| SigLIP2 + MLP | arXiv 2025 | ~400 M | **0.970** | **0.973** | — | — | Revisiting VL table |
| CLIP ViT-L + MLP | arXiv 2025 | ~428 M | 0.965 | 0.968 | — | — | Revisiting VL table |
| DINOv2 + MLP | arXiv 2025 | ~300 M | 0.946 | 0.949 | — | — | Revisiting VL table |

### 3.2 Cross-dataset generalisation (train KADID → test TID2013)

| Method | SROCC | Source |
|---|---:|---|
| HyperIQA | 0.706 | ARNIQA table 5 |
| Su et al. | 0.687 | ARNIQA table 5 |
| CONTRIQUE | 0.612 | ARNIQA table 5 |
| Re-IQA | 0.777 | ARNIQA table 5 |
| ARNIQA | 0.760 | ARNIQA table 5 |
| LIQE (multi-dataset training) | 0.811 | LIQE table 11 (TID2013 column) |

Cross-dataset numbers are protocol-sensitive: ARNIQA trains on KADID alone and
reports 0.760; LIQE is trained on multiple datasets and reports 0.811 on
TID2013. Neither is directly comparable to a KADID-only lightweight model.

## 4. What is actually limiting the state of the art?

1. **Content bias.** With only 81 KADID contents, a 24 M-parameter CNN can fit
   content identity. Heavy crops, flips and small models help, but
   content-invariant representations (orderless statistics, distortion
   classification, contrastive grouping) are underexplored at the <5 M scale.
2. **Semantic vs. perceptual information.** CLIP/SigLIP features are trained to
   be semantically invariant, which suppresses high-frequency distortion cues
   (the “perceptual submergence” noted by CMPA). This is why frozen CLIP CLS
   features are surprisingly weak for quality regression and why adapters,
   prompt learning and multi-layer features are needed.
3. **Distortion coverage.** KADID's 25 synthetic distortions do not cover
   authentic camera noise, compression, motion blur and haze. Models trained
   only on KADID degrade sharply on TID2013 and SPAQ.
4. **Calibration and ranking are different objectives.** MSE fits the MOS scale
   but does not guarantee SROCC; ranking losses and ordinal supervision help,
   yet most papers optimise only L1/MSE.
5. **Evaluation fragmentation.** No standard content split or repetition count.
   Headline differences of 0.005–0.01 are often within split noise, and almost
   no paper reports confidence intervals or paired significance tests.
6. **Efficiency reporting is weak.** FLOPs/latency/peak memory are rarely
   reported, and “trainable parameters” can hide a 400 M frozen backbone.

## 5. Opportunity map for a lightweight model

| Opportunity | Why it is promising | Risk |
|---|---|---|
| **Multi-scale orderless NSS + learned band-pass filters** | Quality is encoded in local statistics; discarding spatial layout removes content bias. Extremely cheap (<1 M params). | May saturate below VLM accuracy; localised artifacts need coarse layout cues. |
| **Wavelet/sub-band statistics + tiny transformer** | Blur, noise, blocking and ringing have distinct sub-band signatures; fixed Haar transform is parameter-free. | Sub-band statistics alone may miss semantic masking. |
| **Distortion-manifold pretraining + ordinal contrastive learning** | Uses unlabeled references and ordered distortion levels; learns quality direction without MOS. Strong generalisation per parameter (ARNIQA/TRIQA evidence). | Needs careful synthetic pipeline and external images; training cost. |
| **Content-invariant ordinal triplet loss** | Directly optimises the property that matters (ordering across distortion levels of the same content). | Batch construction and negative sampling; risk of shortcut learning. |
| **Foundation feature distillation into a statistical student** | Evidence shows foundation features carry the strongest quality prior; distilling a low-dimensional quality subspace could retain most accuracy at <5 M params. | This is distillation, which the brief discourages unless justified; teacher licensing/compute. |
| **Hybrid: frozen compact foundation + NSS branch + gated fusion** | Complementary semantic and distortion cues; can outperform either branch alone. | Total parameters dominated by the backbone; must report total, not just trainable, cost. |
| **Multi-crop / multi-scale test-time aggregation** | Recovers global and local artifacts without retraining; cheap at inference if batched. | Latency multiplies; gains are usually small. |

## 6. Design implications for this project

The evidence supports a two-track strategy:

* **Track A (primary, from first principles):** `MSOSNet` and `WaveQNet`, which
  encode NSS and sub-band priors explicitly and use orderless pooling to fight
  content bias. These are the genuinely novel, lightweight models.
* **Track B (evidence-based hybrid):** a compact foundation branch (CLIP
  ViT-B/16 or DINOv2-S) fused with the NSS branch, or distilled into it. This
  is justified only if Track A plateaus well below the accuracy needed for a
  top-3 claim, and it must be reported with **total** parameter count and
  latency, not just trainable parameters.

The next section of the project turns these hypotheses into controlled
experiments; see `docs/protocol.md` for the exact evaluation contract.

## 7. References

- Mittal et al., *No-Reference Image Quality Assessment in the Spatial Domain*
  (BRISQUE), IEEE TIP 2012.
- Mittal et al., *Making a “Completely Blind” Image Quality Analyzer* (NIQE),
  IEEE SPL 2013.
- Zhang et al., *Dual CNN for No-Reference Image Quality Assessment* (DBCNN),
  IEEE TIP 2020. <https://arxiv.org/abs/1905.03786>
- Su et al., *Content-Variant Reference Image Quality Assessment* /
  HyperIQA, CVPR 2020. <https://arxiv.org/abs/2005.02193>
- Golestaneh et al., *TReS: Transformer for No-Reference Image Quality
  Assessment*, ICCV 2021. <https://arxiv.org/abs/2011.09954>
- Ke et al., *MUSIQ: Multi-scale Image Quality Transformer*, ICCV 2021.
  <https://arxiv.org/abs/2108.05997>
- Madhusudana et al., *Image Quality Assessment using Contrastive Learning*
  (CONTRIQUE), CVPR 2022. <https://arxiv.org/abs/2110.13266>
- Saha et al., *Re-IQA: Unsupervised Learning for Image Quality Assessment in
  the Wild*, CVPR 2023.
- Zhang et al., *LIQE: Blind Image Quality Assessment via Vision-Language
  Correspondence*, CVPR 2023. <https://arxiv.org/abs/2303.14968>
- Yang et al., *MANIQA: Multi-dimension Attention Network for No-Reference
  Image Quality Assessment*, CVPRW 2022. <https://arxiv.org/abs/2204.08958>
- Agnolucci et al., *ARNIQA: Learning Distortion Manifold for Image Quality
  Assessment*, CVPR 2024. <https://arxiv.org/abs/2310.14918>
- Sureddi et al., *TRIQA: Image Quality Assessment by Contrastive Pretraining
  on Ordered Distortion Triplets*, 2025. <https://arxiv.org/abs/2507.12687>
- *Revisiting Vision Language Foundations for No-Reference Image Quality
  Assessment*, 2025. <https://arxiv.org/abs/2509.17374>
- *Bridging the Perceptual Gap: Residual-Enhanced Downscaling and
  Manifold-Aware Perception Alignment for NR-IQA* (CMPA), 2026.
  <https://arxiv.org/abs/2609.16664>
- *DR.Experts: Differential Refinement of Distortion-Aware Experts for BIQA*,
  2026. <https://arxiv.org/abs/2602.09531>
- Wang et al., *Exploring CLIP for Assessing the Look and Feel of Images*
  (CLIP-IQA), AAAI 2023. <https://arxiv.org/abs/2207.12396>
