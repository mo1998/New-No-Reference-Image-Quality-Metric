# Literature shortlist — Project 28 (No-Reference IQA)

Candidates for the M1 "Related Work" section. Rule from the rubric: **>=4 references,
>=2 from 2023–2026, each correctly summarized**. All entries below were verified against
publisher/arXiv pages on 2026-10-10. BibTeX: `milestones/references.bib`.

**How to read this list — citation roles.** The course's lightweight/compute constraint
applies to (a) our own model and (b) everything we RUN in our experiments (PSNR, SSIM,
LPIPS, our network). It does **not** apply to cited work: related methods can use heavy
backbones, diffusion, or LLMs — we cite them for positioning and motivation, we do not
implement them. Each entry below states its role explicitly.

## From the course catalog (core)

| Ref | Venue / Year | What it is | Role in our project |
|---|---|---|---|
| LPIPS (Zhang et al.) | CVPR 2018 | Learned full-reference perceptual metric (AlexNet-class features) | **Run** as baseline; light enough for our budget |
| DISTS (Ding et al.) | TPAMI 2022 | FR metric unifying structure + texture | Optional baseline (run-slot) |
| Siniukov et al. | arXiv 2022 | Shows differentiable FR metrics can be gamed / diverge from subjective scores | Cite — motivation for NR framing |
| SPIPS (Lao & Yu) | arXiv 2025 | Hybrid metric mixing deep semantics + low-level details + classical IQA | Cite — recent anchor |

## Strong method candidates

1. **CLIP-IQA** — Wang, Chan, Loy. *Exploring CLIP for Assessing the Look and Feel of
   Images.* AAAI 2023. https://arxiv.org/abs/2207.12396
   Frozen vision-language model + antonym prompts; scores quality zero-shot; fine-tunable
   variant too. **Role:** cite — the "frozen backbone, tiny adaptation" philosophy.

2. **ARNIQA** — Agnolucci et al. *Learning Distortion Manifold for Image Quality
   Assessment.* WACV 2024 (oral). https://arxiv.org/abs/2310.14918 · github.com/miccunifi/ARNIQA
   Self-supervised distortion manifold; frozen encoder + simple linear regressor.
   **Role:** cite — closest published design to our plan (frozen features + light head).

3. **LAR-IQA** — Jamshidi Avanaki et al. *A Lightweight, Accurate, and Robust NR-IQA
   Model.* arXiv 2024 (ECCV AIM challenge). https://arxiv.org/abs/2408.17057
   MobileNetV3 dual-branch + small head, ~21M params, 5.7x faster than prior SOTA;
   evaluates **KADID → TID2013 cross-dataset** (our stretch goal). **Role:** cite — the
   lightweight-deployment corner we position against.

4. **Revisiting Vision-Language Foundations for NR-IQA** — Yadav, Huy, Liu. WACV 2026.
   https://arxiv.org/abs/2509.17374
   Six frozen backbones, identical lightweight MLP head; activation choice matters;
   resource-efficient baselines. **Role:** cite — newest anchor for "frozen backbone +
   small head".

## Added 2025–2026 (verified)

5. **A-FINE** — Chen, Wu, Ma, Zhang. *Toward Generalized Image Quality Assessment:
   Relaxing the Perfect Reference Quality Assumption.* CVPR 2025, pp. 12742–12752.
   https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Toward_Generalized_Image_Quality_Assessment_Relaxing_the_Perfect_Reference_Quality_CVPR_2025_paper.html
   **FULL-REFERENCE metric** (it still needs a reference image — it only relaxes the
   assumption that the reference is *perfect* quality). The diffusion model appears only
   while **constructing the DiffIQA dataset** (~180k images) — A-FINE itself is not a
   diffusion model. **Role:** cite as motivation — recent CVPR evidence that the FR
   paradigm (our baselines' paradigm) has structural limits. **We do not implement it.**

6. **IQA Survey** — Ma, Shi, Lu, Xie, Chao, Sui. *A Survey on Image Quality Assessment:
   Insights, Analysis, and Future Outlook.* arXiv 2025. https://arxiv.org/abs/2502.08540
   **Role:** cite — structures the Related Work intro. (Preprint — marked as such.)

7. **MA-CLIP (Magnitude-Aware CLIP)** — Liao et al. *Beyond Cosine Similarity: Magnitude-
   Aware CLIP for NR-IQA.* **AAAI 2026**, vol. 40(9), pp. 6934–6942.
   doi:10.1609/aaai.v40i9.37627 · arXiv:2511.09948 · github.com/zhix000/MA-CLIP
   Training-free: CLIP image features' *magnitude* correlates with quality; fused with
   prompt similarity. **About "lightweight":** it is training-free (zero training compute)
   and the released code defaults to a CLIP **ResNet-50** backbone (not a huge ViT —
   optional ViT-B/32, RN101); inference-only. It is still larger than our target network,
   and it does not train on KADID. **Role:** cite — freshest training-free method; the
   released code supports KADID inference, so it is a candidate *optional* comparison
   baseline at M3 if we want a 2026 reference point.

8. *(optional, ultra-fresh)* **MOSAIQ-500k / MOSAIQ-Bench** — Yang, Wang, Xu, Zhou, Wang.
   arXiv:2609.20247, Sept 2026. Unifies and score-realigns IQA datasets; splits **by
   reference image** — independent confirmation our content split is standard. **Role:**
   cite (optional; preprint, 3 weeks old).

9. *(optional architecture inspiration)* **TOPIQ** — Chen et al., IEEE TIP 2024,
   doi:10.1109/TIP.2024.3378466. Semantics-guided attention to distortion regions.
   **Role:** cite if used.

## Rubric check

- Total usable references: 12+ (requirement: >=4) ✔
- From 2023–2026: CLIP-IQA 2023, ARNIQA 2024, LAR-IQA 2024, TOPIQ 2024, A-FINE 2025,
  survey 2025, SPIPS 2025, WACV-2026, MA-CLIP 2026, MOSAIQ 2026 (optional)
  → **9 confirmed recent** (requirement: >=2) ✔

## Positioning paragraph (draft for the Related Work section)

> Recent NR-IQA work converges on two recipes: (i) reuse a large pretrained backbone and
> attach a lightweight adaptation — prompts for CLIP [CLIP-IQA; MA-CLIP], or a small head
> on frozen features [ARNIQA; WACV-2026] — and (ii) design explicitly lightweight
> end-to-end models for deployment [LAR-IQA]. Surveys and the newest CVPR work
> [survey 2025; A-FINE 2025] agree that FR metrics are structurally limited and that
> generalization, not in-dataset accuracy, is the open problem. Our project sits at the
> intersection under a strict laptop-class compute budget: we freeze a pretrained
> backbone, train a small regression head on KADID-10k with a content-based split, and
> validate with the standard SROCC/PLCC protocol against full-reference baselines
> (PSNR, SSIM, LPIPS) plus a cross-dataset check on TID2013.
