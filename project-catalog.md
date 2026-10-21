# Digital Image Processing — Team Project Catalog

### [COURSE CODE] — Digital Image Processing · [TERM] · Instructor: [INSTRUCTOR NAME]

**30 projects** across 5 categories, emphasizing image coding/compression and allowing
pretrained AI/ML components as building blocks within a classical image-processing
framework. Team size 3 · duration 9 weeks · milestones at Weeks 3/6/9 (M1 15% / M2 35% /
M3 50%).

> This is the **companion catalog** to `project-instructions.md`, which covers team
> formation, compute policy, the milestone framework, student roles, LaTeX report
> templates, grading rubrics, the publication bonus track, and a dataset quick-reference
> table. Read that document first for the rules of the road — this one is just the list
> of projects to pick from. Section references below (e.g. "§9.3") point into
> `project-instructions.md`.

---

## Table of Contents

- [A. Learned & Hybrid Image Compression (10 projects)](#a-learned--hybrid-image-compression-10-projects) — Projects 1–10
- [B. Compression-Adjacent / Video / Security (5 projects)](#b-compression-adjacent--video--security-5-projects) — Projects 11–15
- [C. Restoration & Enhancement with AI (6 projects)](#c-restoration--enhancement-with-ai-6-projects) — Projects 16–21
- [D. Segmentation, Features & Color with AI (5 projects)](#d-segmentation-features--color-with-ai-5-projects) — Projects 22–26
- [E. Inpainting, Forensics & Metrics (4 projects)](#e-inpainting-forensics--metrics-4-projects) — Projects 27–30

Each entry gives objectives, references, a public dataset, baseline, compute tier, and a
3-milestone task table (Student A / B / C, per `project-instructions.md` §5).

---

## Project Catalog (30 Projects)

Every entry gives: objectives, 2–4 grounded references (real, recent where the sub-field
moves fast), a public dataset with approximate size, the baseline your results should be
compared against, a compute tier, and a 3-milestone task table. Treat the listed references
as a **starting point, not the whole literature review** — part of the M1 grade is finding
and correctly summarizing additional 2023–2026 work yourselves.

---

### A. Learned & Hybrid Image Compression (10 projects)

#### 1. Lightweight Learned Image Compression for Laptop-Class Training

**Objectives**
- Reproduce a compact hyperprior-based learned codec (reduced channel width) that trains on a laptop in the project window.
- Benchmark rate–distortion (PSNR / MS-SSIM vs. bits-per-pixel) against JPEG and published ELIC operating points.
- Characterize the accuracy/complexity trade-off as model width and latent channel count shrink.

**References**
- Ballé, Minnen, Singh, Hwang, Johnston, "Variational Image Compression with a Scale Hyperprior," *ICLR* 2018.
- Minnen, Ballé, Toderici, "Joint Autoregressive and Hierarchical Priors for Learned Image Compression," *NeurIPS* 2018.
- He, Yang, Peng, Wang, "ELIC: Efficient Learned Image Compression with Unevenly Grouped Space-Channel Contextual Adaptive Coding," *CVPR* 2022. [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/He_ELIC_Efficient_Learned_Image_Compression_With_Unevenly_Grouped_Space-Channel_Contextual_CVPR_2022_paper.pdf)
- [CompressAI](https://github.com/InterDigitalInc/CompressAI) — open-source reference implementations and pretrained models.

**Dataset:** DIV2K train+val (800+100 2K images, ~3.5 GB) — [link](https://data.vision.ee.ethz.ch/cvl/DIV2K/), train on random 256×256 crops. Evaluate on Kodak (24 images) — [link](http://r0k.us/graphics/kodak/) and Tecnick — [link](https://testimages.org/).

**Baseline to beat:** standard JPEG at matched bpp; stretch goal: approach published small-ELIC operating points. **Compute tier:** GPU-light (CPU feasible with small crops).

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | DIV2K pipeline + crop sampler; survey hyperprior literature | Scope reduced-width architecture; sketch training config | Define RD-curve protocol (bpp via entropy estimate, PSNR/MS-SSIM) | Proposal + feasibility test (1 training step runs) |
| 2 | 6 | JPEG baseline RD curve on Kodak | Trained small hyperprior codec, first checkpoint | Preliminary RD curve of your codec vs. JPEG | Progress report with one RD plot |
| 3 | 9 | Final JPEG/Tecnick comparison table | Final model + ablation on latent width | Full RD curves, qualitative crops, decode-time table | Final report, repo, demo |

---

#### 2. Low-Complexity Context Modeling for Neural Codecs

**Objectives**
- Implement 2–3 lightweight entropy-context designs (checkerboard / channel-wise / spatial-competition) on a shared hyperprior backbone.
- Measure decode speed vs. RD performance for each.

**References**
- He et al., "Checkerboard Context Model for Efficient Learned Image Compression," *CVPR* 2021.
- "Spatial Competition for Low-Complexity Learned Image Compression," *arXiv:2605.13243* (2026). [[paper]](https://arxiv.org/abs/2605.13243)
- "What Matters in Practical Learned Image Compression," *arXiv:2605.05148* (2026). [[paper]](https://arxiv.org/abs/2605.05148)

**Dataset:** DIV2K (train), Kodak + Tecnick (eval) — links above.

**Baseline:** serial autoregressive context model (Minnen et al. 2018) RD-performance/speed. **Compute tier:** GPU-light.

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | DIV2K/Kodak pipeline; summarize 3 context-model papers | Implement shared hyperprior backbone skeleton | Define decode-time + RD measurement harness | Proposal with architecture diagram |
| 2 | 6 | Reproduce serial-autoregressive baseline | Implement checkerboard context variant | Timing harness working on 1 variant | Progress report, 1 context model working |
| 3 | 9 | Final baseline numbers | All context variants implemented | Full speed-vs-RD comparison table/plot | Final report, repo, demo |

---

#### 3. Single-Image / Single-Video Overfit Neural Compression (Implicit Neural Codec)

**Objectives**
- Implement a C3-style implicit neural representation trained (overfit) per image/clip — no large training set or long training run needed.
- Study RD performance vs. overfitting time and parameter count.

**References**
- "C3: High-Performance and Low-Complexity Neural Compression from a Single Image or Video" — [project page](https://c3-neural-compression.github.io/).
- Dupont, Goliński, Alizadeh, Teh, Doucet, "COIN: COmpression with Implicit Neural representations," 2021.

**Dataset:** Kodak (images); one short clip from UVG — [link](http://ultravideo.fi/#testsequences) for the video variant.

**Baseline:** JPEG / HEVC-intra at matched bpp for the same image/clip. **Compute tier:** CPU-OK — the most laptop-friendly compression project on the list.

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | JPEG/HEVC-intra reference numbers on 5 Kodak images | Implicit-network architecture (SIREN/Fourier-feature MLP) | Overfitting-time vs. quality measurement plan | Proposal + 1-image overfit proof-of-concept |
| 2 | 6 | Full Kodak baseline table | Overfit codec working on all 24 Kodak images | RD curve, parameter-count sweep | Progress report, RD plot |
| 3 | 9 | UVG clip baseline (HEVC/x264) | Video variant of the implicit codec | Full image+video RD comparison, timing | Final report, repo, demo |

---

#### 4. Rate–Distortion–Perception Controllable Compression

**Objectives**
- Build a codec whose decoder is steerable along the distortion↔perception trade-off at inference time (one model, adjustable knob).
- Reproduce, qualitatively, the trade-off characterized by Blau & Michaeli.

**References**
- Blau & Michaeli, "Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff," *ICML* 2019. [[paper]](http://proceedings.mlr.press/v97/blau19a/blau19a.pdf)
- "Controllable Distortion-Perception Tradeoff Through Latent Diffusion for Neural Image Compression," *arXiv:2412.11379*. [[paper]](https://arxiv.org/html/2412.11379v1)

**Dataset:** DIV2K (train), Kodak + CLIC validation — [link](https://clic2025.compression.cc/tasks/).

**Baseline:** PSNR-optimized hyperprior codec (no perceptual term) at matched bpp. **Compute tier:** GPU-light.

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | DIV2K/Kodak/CLIC pipeline; summarize Blau & Michaeli | Design the steerable-knob conditioning mechanism | Plan LPIPS/PSNR trade-off curve protocol | Proposal with trade-off sketch |
| 2 | 6 | PSNR-only baseline codec trained | Knob-conditioned codec, first working checkpoint | Preliminary trade-off curve (2–3 knob settings) | Progress report, trade-off plot |
| 3 | 9 | Final baseline RD numbers | Full knob sweep trained and stable | Full LPIPS-vs-PSNR frontier, qualitative crops | Final report, repo, demo |

---

#### 5. Diffusion-Prior Extreme Low-Bitrate Compression

**Objectives**
- Build an extreme-compression pipeline (< 0.05 bpp) transmitting a compact latent/semantic code, reconstructed with a small **pretrained, frozen** diffusion prior.
- Evaluate fidelity/realism trade-off with LPIPS/FID on a small set.

**References**
- "Diffusion-based Extreme Image Compression with Compressed Feature Initialization," *arXiv:2410.02640*. [[paper]](https://arxiv.org/html/2410.02640v1)
- "Single-step Diffusion for Image Compression at Ultra-Low Bitrates," *arXiv:2506.16572*. [[paper]](https://arxiv.org/pdf/2506.16572)

**Dataset:** CLIC validation (professional subset), Kodak.

**Baseline:** VVC-intra or a standard learned codec at the same (very low) bpp. **Compute tier:** GPU-light — use a small public pretrained diffusion checkpoint as a frozen prior; do **not** train diffusion from scratch.

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | VVC-intra/learned-codec reference at target low bpp | Select & wire up frozen pretrained diffusion prior | Plan LPIPS/FID evaluation on small set | Proposal + frozen-prior smoke test |
| 2 | 6 | Full baseline numbers at target bpp | Latent encoder trained to drive the frozen prior | Preliminary LPIPS/FID vs. baseline | Progress report, qualitative samples |
| 3 | 9 | Final baseline comparison table | Final pipeline, failure-case analysis | Full LPIPS/FID table, qualitative grid | Final report, repo, demo |

---

#### 6. JPEG-Compatible Learned Post-Filter

**Objectives**
- Train a lightweight restoration network that post-processes standard JPEG output to close the RD gap to learned codecs, while remaining fully JPEG-decodable by any standard viewer.

**References**
- "Compression-Aware One-Step Diffusion Model for JPEG Artifact Removal," *ICCV* 2025. [[paper]](https://arxiv.org/pdf/2502.09873)
- "SODiff: Semantic-Oriented Diffusion Model for JPEG Compression Artifacts Removal," *arXiv:2508.07346*. [[paper]](https://arxiv.org/html/2508.07346)

**Dataset:** DIV2K compressed with libjpeg at several quality factors; standard LIVE1/Classic5 JPEG-artifact test sets (mirrored via [BasicSR](https://github.com/XPixelGroup/BasicSR)).

**Baseline:** raw JPEG at the same file size vs. JPEG + your post-filter. **Compute tier:** CPU-OK for a small model / GPU-light for the full version.

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | JPEG-QF sweep pipeline + baseline PSNR/SSIM table | Post-filter architecture (small CNN/one-step diffusion) | Define equal-file-size comparison protocol | Proposal + architecture sketch |
| 2 | 6 | LIVE1/Classic5 baseline numbers | Post-filter trained at one QF | Preliminary before/after comparison | Progress report, 1 QF result |
| 3 | 9 | Final baseline table across QFs | Post-filter working across QF range | Full RD-at-equal-size comparison, crops | Final report, repo, demo |

---

#### 7. Learned Entropy Coding for Classical Transform Coding

**Objectives**
- Keep the classical JPEG/JPEG2000 transform (DCT or wavelet) but replace the fixed entropy-coding stage with a small learned context model + arithmetic coder.
- Measure the bitrate reduction at equal reconstruction quality.

**References**
- Minnen, Ballé, Toderici, "Joint Autoregressive and Hierarchical Priors for Learned Image Compression," *NeurIPS* 2018 (context-modeling methodology).
- Mentzer, Agustsson, Tschannen, Timofte, Van Gool, "Practical Full Resolution Learned Lossless Image Compression," *CVPR* 2019.

**Dataset:** Kodak, DIV2K.

**Baseline:** standard libjpeg / OpenJPEG entropy stage at equal quantization. **Compute tier:** CPU-OK.

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | Extract DCT/wavelet coefficients from libjpeg/OpenJPEG | Design learned context model for coefficient stream | Define equal-quality bitrate comparison protocol | Proposal + extracted-coefficient demo |
| 2 | 6 | Baseline entropy-stage bitrate table | Learned context model trained on DIV2K coefficients | Preliminary bitrate comparison on Kodak | Progress report, 1 comparison table |
| 3 | 9 | Final libjpeg/OpenJPEG baseline numbers | Final learned entropy coder, stable arithmetic coding | Full bitrate-reduction table at matched quality | Final report, repo, demo |

---

#### 8. Image Coding for Machines (ICM): Compressing for a Frozen Vision Model

**Objectives**
- Design a codec/bit-allocation strategy optimized for a downstream frozen classifier's or detector's accuracy, not pixel fidelity.
- Quantify bitrate saved at equal task accuracy vs. a human-oriented codec.

**References**
- "Image Coding for Machines via Feature-Preserving Rate-Distortion Optimization," *arXiv:2504.02216*. [[paper]](https://arxiv.org/abs/2504.02216)
- Curated reading list: [Awesome_VCM](https://github.com/lingyzhu0101/Awesome_VCM).

**Dataset:** COCO val2017 subset — [link](https://cocodataset.org/) with a frozen pretrained classifier/detector (e.g., torchvision ResNet-50 or a small YOLO).

**Baseline:** a standard learned/JPEG codec optimized for PSNR, evaluated on the same downstream task. **Compute tier:** GPU-light (inference-only on the frozen task model).

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | COCO subset + frozen classifier/detector wired up | Design task-aware rate-distortion loss | Define task-accuracy-vs-bpp protocol | Proposal + frozen-model smoke test |
| 2 | 6 | PSNR-optimized-codec task-accuracy baseline | Task-aware codec, first trained checkpoint | Preliminary accuracy-vs-bpp curve | Progress report, 1 curve |
| 3 | 9 | Final baseline accuracy-vs-bpp table | Final task-aware codec, ablation on loss weighting | Full comparison, bits-saved-at-equal-accuracy number | Final report, repo, demo |

---

#### 9. Remote-Sensing Image Compression for Edge Devices

**Objectives**
- Build and evaluate a low-complexity learned codec tailored to aerial/satellite image statistics.
- Compare to JPEG2000, still the incumbent in many remote-sensing pipelines.

**References**
- "Towards an Efficient Remote Sensing Image Compression Network with Visual State Space Model," *Remote Sensing* 2025. [[DOI]](https://doi.org/10.3390/rs17030425)
- "Map-Assisted Remote-Sensing Image Compression at Extremely Low Bitrates," 2025. [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0924271625001030)

**Dataset:** UC Merced Land Use (2,100 images, 256×256, ~317 MB) — [link](http://weegee.vision.ucmerced.edu/datasets/landuse.html), or a subsampled NWPU-RESISC45.

**Baseline:** JPEG2000 at matched bpp (OpenJPEG/Pillow). **Compute tier:** CPU-OK / GPU-light.

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | UC Merced pipeline; JPEG2000 reference numbers | Scope lightweight codec architecture for RS statistics | Define RD comparison protocol | Proposal + JPEG2000 baseline table |
| 2 | 6 | Full JPEG2000 RD curve | Trained small codec, first checkpoint | Preliminary RD curve vs. JPEG2000 | Progress report, RD plot |
| 3 | 9 | Final JPEG2000 numbers across classes | Final codec + per-land-use-class breakdown | Full RD comparison, qualitative crops | Final report, repo, demo |

---

#### 10. Diagnostic-Region-Priority Medical Image Compression

**Objectives**
- Build an ROI-aware learned codec allocating more bits to diagnostically relevant regions (e.g., lesion/organ masks) and fewer to background.
- Quantify the quality-vs-bitrate gain specifically within the ROI.

**References**
- "Deep Learning Based Medical Image Compression Using Cross Attention Learning and Wavelet Transform," *Scientific Reports* 2025. [[paper]](https://www.nature.com/articles/s41598-025-23582-y)

**Dataset:** a public de-identified subset from [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/) or the [NIH ChestX-ray14 sample](https://www.kaggle.com/datasets/nih-chest-xrays/data) — use a few hundred images.

**Baseline:** uniform-quality JPEG2000 at the same average bpp. **Compute tier:** CPU-OK / GPU-light.

> **Ethics note:** even de-identified public medical data has specific usage terms — read and
> cite them exactly; no attempt at re-identification; see the companion *project-instructions.md*, §9.3 (Dataset Licensing & Ethics).

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | Dataset access + license review; ROI mask source (manual or existing annotations) | Scope ROI-aware bit-allocation mechanism | Define ROI-specific vs. global quality metrics | Proposal + license/ethics statement |
| 2 | 6 | Uniform-JPEG2000 baseline, ROI metrics computed | ROI-aware codec, first checkpoint | Preliminary ROI-vs-background quality comparison | Progress report, 1 comparison |
| 3 | 9 | Final baseline table | Final codec + bit-allocation ablation | Full ROI-vs-global RD comparison | Final report, repo, demo |

---

### B. Compression-Adjacent / Video / Security (5 projects)

#### 11. Lightweight Neural Video Compression for a Single Clip

**Objectives**
- Implement an ultra-lightweight implicit/overfit neural video codec for a handful of short clips.
- Compare RD and encode/decode speed to x264/x265 at matched bitrate.

**References**
- "Ultra-lightweight Neural Video Representation Compression," *arXiv:2512.04019*. [[paper]](https://arxiv.org/abs/2512.04019)
- "Towards Practical Real-Time Neural Video Compression," *CVPR* 2025. [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Jia_Towards_Practical_Real-Time_Neural_Video_Compression_CVPR_2025_paper.pdf)

**Dataset:** UVG dataset (7 standard 1080p sequences) — [link](http://ultravideo.fi/#testsequences); use short (2–5 s) or downsampled clips.

**Baseline:** x264 (libx264) at matched bitrate via ffmpeg. **Compute tier:** GPU-light.

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | ffmpeg/x264 pipeline + clip selection | Implicit video-codec architecture sketch | Define RD + encode/decode-time protocol | Proposal + x264 reference numbers |
| 2 | 6 | Full x264 RD curve on all clips | Overfit codec working on 1 clip | Preliminary RD comparison, 1 clip | Progress report, RD plot |
| 3 | 9 | Final x264 baseline across clips | Codec working on all clips, parameter sweep | Full RD + timing comparison table | Final report, repo, demo |

---

#### 12. Robust Invisible Watermarking vs. Generative-AI Removal Attacks

**Objectives**
- Implement a learned invisible-watermark encoder/decoder (StegaStamp-style).
- Implement/apply a generative-AI removal attack and measure watermark robustness after (a) compression and (b) diffusion-based regeneration.

**References**
- Tancik, Mildenhall, Ng, "StegaStamp: Invisible Hyperlinks in Physical Photographs," *CVPR* 2020.
- Zhao et al., "Invisible Image Watermarks Are Provably Removable Using Generative AI," *NeurIPS* 2024 — code: [WatermarkAttacker](https://github.com/xuandongzhao/watermarkattacker).
- Xu et al., "InvisMark: Invisible and Robust Watermarking for AI-generated Image Provenance," *WACV* 2025. [[paper]](https://openaccess.thecvf.com/content/WACV2025/papers/Xu_InvisMark_Invisible_and_Robust_Watermarking_for_AI-Generated_Image_Provenance_WACV_2025_paper.pdf)

**Dataset:** a 500–1,000 image subset of COCO or DIV2K as cover images.

**Baseline:** StegaStamp bit-accuracy/robustness under JPEG compression + the WatermarkAttacker pipeline, vs. your improved scheme. **Compute tier:** CPU-OK / GPU-light.

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | Cover-image subset; reproduce StegaStamp encoder/decoder | Scope watermark-robustness improvement idea | Set up WatermarkAttacker removal pipeline | Proposal + working StegaStamp demo |
| 2 | 6 | StegaStamp bit-accuracy baseline (clean + JPEG) | Improved watermark scheme, first checkpoint | Bit-accuracy under removal attack, baseline numbers | Progress report, 1 robustness table |
| 3 | 9 | Final baseline robustness table | Final scheme, ablation on embedding strength | Full robustness comparison (compression + removal attack) | Final report, repo, demo |

---

#### 13. Compressed-Domain Image Classification

**Objectives**
- Train a classifier operating directly on JPEG DCT coefficients (or a learned codec's latent), without full pixel-domain decoding.
- Measure accuracy/speed trade-off vs. decode-then-classify.

**References**
- Gueguen, Sergeev, Kadlec, Liu, Yosinski, "Faster Neural Networks Straight from JPEG," *NeurIPS* 2018.

**Dataset:** CIFAR-10/100 (small, standard, public) re-encoded as JPEG, or a 20-class ImageNet subset.

**Baseline:** a standard ResNet trained on decoded RGB pixels, matched for parameter count. **Compute tier:** CPU-OK / GPU-light.

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | JPEG-DCT extraction pipeline; dataset subset chosen | Design DCT-domain classifier architecture | Define accuracy/speed comparison protocol | Proposal + extracted-DCT demo |
| 2 | 6 | RGB-ResNet baseline trained, accuracy recorded | DCT-domain classifier, first trained checkpoint | Preliminary accuracy + speed comparison | Progress report, 1 comparison |
| 3 | 9 | Final RGB baseline (accuracy + decode+infer time) | Final DCT-domain classifier | Full accuracy-vs-speed table, ablation on which DCT bands matter | Final report, repo, demo |

---

#### 14. Robustness of Learned Compression to Corrupted/Adversarial Inputs

**Objectives**
- Stress-test a pretrained learned codec against common corruptions (noise, blur, geometric shift) and small adversarial perturbations.
- Characterize failure modes (rate spikes, artifacts) vs. a classical codec under the same corruptions.

**References**
- Team-driven: identify 2–3 current (2024–2026) codec-robustness papers during M1 — this is intentionally open so the literature review itself is graded.
- Michaelis et al., [imagecorruptions](https://github.com/bethgelab/imagecorruptions) library (methodology reference for the corruption suite).

**Dataset:** Kodak/CLIC + the `imagecorruptions` suite applied on top.

**Baseline:** JPEG/JPEG2000 under the same corruptions. **Compute tier:** GPU-light (uses a pretrained codec from CompressAI; no large training needed).

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | Wire up `imagecorruptions` + pretrained CompressAI codec | Survey & select 2–3 robustness-relevant papers | Define rate-spike / artifact-severity metrics | Proposal + corruption-suite demo |
| 2 | 6 | JPEG/JPEG2000-under-corruption baseline table | Reproduce 1 robustness finding from the literature | Preliminary rate/quality-vs-corruption-severity curves | Progress report, 1 corruption type analyzed |
| 3 | 9 | Final classical-codec corruption table | Full corruption + adversarial sweep on learned codec | Full comparison, failure-mode gallery | Final report, repo, demo |

---

#### 15. Perceptual Quality-Aware Bit Allocation Using a Learned IQA Loss

**Objectives**
- Replace a codec's MSE/MS-SSIM training loss with a learned perceptual metric (LPIPS/DISTS).
- Study how resulting bit allocation and visual quality change at equal bpp.

**References**
- Zhang, Isola, Efros, Shechtman, Wang, "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (LPIPS), *CVPR* 2018.
- Ding, Ma, Wang, Simoncelli, "Image Quality Assessment: Unifying Structure and Texture Similarity" (DISTS), *TPAMI* 2020.

**Dataset:** DIV2K (train), Kodak (eval).

**Baseline:** the same codec architecture trained with MSE only. **Compute tier:** GPU-light.

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | DIV2K/Kodak pipeline; MSE-only codec scoped | Wire up LPIPS/DISTS as a differentiable training loss | Define bpp-matching + visual-comparison protocol | Proposal + LPIPS-loss smoke test |
| 2 | 6 | MSE-only codec trained, RD curve recorded | LPIPS-trained codec, first checkpoint | Preliminary bpp-matched visual comparison | Progress report, side-by-side crops |
| 3 | 9 | Final MSE-only RD numbers | Final LPIPS/DISTS-trained codec | Full bpp-matched comparison, bit-allocation heatmaps | Final report, repo, demo |

---

### C. Restoration & Enhancement with AI (6 projects)

#### 16. Lightweight Real-Time Super-Resolution for Edge/Laptop Deployment

**Objectives**
- Build and benchmark an efficient ×2/×4 SR network under an explicit parameter/FLOP budget, following NTIRE/AIM efficiency-challenge methodology.
- Measure PSNR/SSIM vs. CPU runtime.

**References**
- "The Tenth NTIRE 2025 Efficient Super-Resolution Challenge Report," *arXiv:2504.10686*. [[paper]](https://arxiv.org/html/2504.10686v1)
- "Efficient Perceptual Image Super Resolution: AIM 2025 Study and Benchmark," *arXiv:2510.12765*. [[paper]](https://arxiv.org/html/2510.12765v1)

**Dataset:** DIV2K (train/val); test on Set5, Set14, BSD100, Urban100, Manga109 (mirrored via [BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md)).

**Baseline:** bicubic upsampling and a classic lightweight network (FSRCNN or ESPCN) at matched budget. **Compute tier:** CPU-OK (small patches) / GPU-light.

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | DIV2K + 5 benchmark sets pipeline; FSRCNN baseline scoped | Design efficient SR architecture within FLOP budget | Define PSNR/SSIM + CPU-runtime measurement protocol | Proposal + bicubic baseline numbers |
| 2 | 6 | FSRCNN baseline trained on all 5 test sets | Efficient SR network, first checkpoint | Preliminary PSNR/SSIM-vs-runtime comparison | Progress report, 1 benchmark set result |
| 3 | 9 | Final FSRCNN numbers, all test sets | Final model + FLOP-budget ablation | Full PSNR/SSIM/runtime table across all 5 sets | Final report, repo, demo |

---

#### 17. Self-Supervised Real-World Image Denoising Without Paired Data

**Objectives**
- Implement a self-supervised denoiser (no clean/noisy pairs required).
- Evaluate on real sensor noise vs. a classical filter and a supervised baseline.

**References**
- "Unsupervised Real-World Denoising: Sparsity is All You Need," *arXiv:2503.21377*. [[paper]](https://arxiv.org/html/2503.21377v1)
- "Blind-Spot Guided Diffusion for Self-supervised Real-World Denoising," *arXiv:2509.16091*. [[paper]](https://arxiv.org/html/2509.16091v1)

**Dataset:** SIDD-Small/Medium — [link](https://www.eecs.yorku.ca/~kamel/sidd/); DND benchmark — [link](https://noise.visinf.tu-darmstadt.de/).

**Baseline:** a classical filter (BM3D or non-local means); supervised DnCNN baseline if time allows. **Compute tier:** GPU-light.

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | SIDD/DND pipeline; BM3D baseline scoped | Design self-supervised denoising objective | Define PSNR/SSIM evaluation on real noise | Proposal + BM3D baseline numbers |
| 2 | 6 | Full BM3D baseline table | Self-supervised denoiser, first checkpoint | Preliminary comparison on SIDD validation | Progress report, 1 comparison |
| 3 | 9 | Final classical + (if time) supervised baseline | Final self-supervised model | Full SIDD + DND comparison table | Final report, repo, demo |

---

#### 18. Lightweight One-Step Diffusion for JPEG Artifact Removal

**Objectives**
- Distill/train a small one-step (or few-step) diffusion model to remove JPEG blocking/ringing artifacts at low quality factors.
- Trade off runtime against a full multi-step diffusion restorer.

**References**
- "Compression-Aware One-Step Diffusion Model for JPEG Artifact Removal," *ICCV* 2025. [[paper]](https://arxiv.org/pdf/2502.09873)

**Dataset:** DIV2K (JPEG-compressed at QF 10–30 with libjpeg); standard LIVE1/Classic5 JPEG test sets.

**Baseline:** a classical deblocking filter and a plain CNN restorer (ARCNN-style) at the same QF. **Compute tier:** GPU-light.

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | QF-sweep JPEG pipeline; ARCNN baseline scoped | Select/wire up small pretrained diffusion backbone | Define PSNR/SSIM/LPIPS + runtime protocol | Proposal + ARCNN baseline numbers |
| 2 | 6 | Full ARCNN baseline across QFs | One-step diffusion restorer, first checkpoint | Preliminary comparison at 1 QF | Progress report, 1 QF result |
| 3 | 9 | Final classical + ARCNN baseline table | Final restorer, runtime-vs-quality ablation | Full comparison across QFs, qualitative crops | Final report, repo, demo |

---

#### 19. Blind Image Deblurring with a Compact Transformer

**Objectives**
- Build a parameter-budgeted transformer or hybrid CNN-transformer for single-image blind motion deblurring.
- Compare accuracy/efficiency against the course's classical Wiener-filter pipeline (Week 7).

**References**
- Kupyn, Martyniuk, Wu, Wang, "DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better," *ICCV* 2019 (heavier reference point).
- Team-driven: survey 2–3 recent lightweight deblurring transformers during M1.

**Dataset:** GoPro deblurring dataset (standard, public — use a subsampled split), or a synthetically blurred DIV2K subset for a lighter variant.

**Baseline:** classical Wiener/Lucy–Richardson deconvolution (course Week 7 material). **Compute tier:** GPU-light.

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | GoPro subset pipeline; Wiener-filter baseline from Week 7 lab | Design compact transformer/hybrid architecture | Define PSNR/SSIM + parameter-count protocol | Proposal + Wiener baseline numbers |
| 2 | 6 | Full Wiener baseline on GoPro subset | Compact model, first trained checkpoint | Preliminary comparison | Progress report, 1 comparison |
| 3 | 9 | Final classical baseline table | Final model + parameter-budget ablation | Full comparison, qualitative crops | Final report, repo, demo |

---

#### 20. Compressed-Sensing Image Reconstruction with a Learned Unrolled Optimizer

**Objectives**
- Implement an unrolled optimization network (ISTA-Net-style) reconstructing images from far fewer linear measurements than Nyquist sampling requires.
- Compare reconstruction quality/speed to classical CS solvers.

**References**
- Zhang & Ghanem, "ISTA-Net: Interpretable Optimization-Inspired Deep Network for Image Compressive Sensing," *CVPR* 2018 — code: [ISTA-Net-PyTorch](https://github.com/jianzhangcs/ISTA-Net-PyTorch).
- "Multi-stage Iterative Compressed Sensing Framework Based on DIFF Transformer and ISTA for Remote Sensing Images," *Scientific Reports* 2025. [[paper]](https://www.nature.com/articles/s41598-025-16727-6)

**Dataset:** Set11 (standard 11-image CS benchmark, bundled with ISTA-Net code) + BSD500 for training.

**Baseline:** classical CS solvers (TVAL3 or a simple iterative-shrinkage baseline) at the same sampling ratio. **Compute tier:** CPU-OK / GPU-light.

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | BSD500/Set11 pipeline; classical CS-solver baseline scoped | Implement unrolled-optimizer architecture skeleton | Define PSNR-vs-sampling-ratio protocol | Proposal + classical-solver baseline numbers |
| 2 | 6 | Full classical baseline across sampling ratios | Unrolled network trained at 1 sampling ratio | Preliminary comparison at that ratio | Progress report, 1 comparison |
| 3 | 9 | Final baseline table across ratios | Final network across sampling ratios, ablation on unroll depth | Full comparison, timing | Final report, repo, demo |

---

#### 21. Diagnostic-Preserving Denoising for Low-Dose Medical Images

**Objectives**
- Train a denoiser for simulated low-dose X-ray/CT noise.
- Evaluate both by pixel-fidelity metrics and by whether a frozen diagnostic classifier's predictions are preserved after denoising.

**References**
- Team-driven: survey 2024–2026 low-dose medical denoising literature during M1.
- Methodology reference: "Deep Learning Based Medical Image Compression Using Cross Attention Learning and Wavelet Transform," *Scientific Reports* 2025 (ROI-relevance evaluation idea).

**Dataset:** NIH ChestX-ray14 public sample — [link](https://www.kaggle.com/datasets/nih-chest-xrays/data), with synthetically added low-dose-style noise (paired real low/high-dose public data is scarce — state this limitation explicitly in the report).

**Baseline:** a classical denoising filter (BM3D/median) at the same noise level. **Compute tier:** CPU-OK / GPU-light.

> **Ethics note:** same as Project 10 — respect the dataset's license, no re-identification attempts (see *project-instructions.md*, §9.3).

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | ChestX-ray14 sample + license review; noise-simulation pipeline | Design denoiser architecture | Select & freeze a pretrained diagnostic classifier for the preservation metric | Proposal + license/ethics statement |
| 2 | 6 | Classical-filter baseline, both pixel and classifier-preservation metrics | Denoiser, first checkpoint | Preliminary pixel + classifier-preservation comparison | Progress report, 1 comparison |
| 3 | 9 | Final baseline table | Final denoiser + noise-level ablation | Full comparison on both metric families | Final report, repo, demo |

---

### D. Segmentation, Features & Color with AI (5 projects)

#### 22. Adapting SAM2 to a New Domain with a Lightweight Adapter

**Objectives**
- Keep Meta's pretrained SAM2 encoder frozen; train only a small adapter/decoder head to specialize it to a new domain (medical, satellite, or course-collected images).
- Measure IoU/Dice gain over zero-shot SAM2 prompting.

**References**
- Ravi et al., "SAM 2: Segment Anything in Images and Videos," *arXiv:2408.00714* / *ICLR* 2025 — code: [sam2](https://github.com/facebookresearch/sam2).
- "SAM2-UNet: Segment Anything 2 Makes Strong Encoder for Natural and Medical Image Segmentation," 2025. [[paper]](https://link.springer.com/article/10.1007/s44267-025-00106-w)

**Dataset:** BSDS500 (general), or a small public [ISIC](https://www.isic-archive.com/) skin-lesion subset (medical).

**Baseline:** zero-shot SAM2 with point/box prompts (no fine-tuning). **Compute tier:** CPU-OK for inference / GPU-light for adapter training — the frozen encoder is what keeps this laptop-feasible.

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | Domain dataset + masks prepared; zero-shot SAM2 wired up | Design adapter/decoder head architecture | Define IoU/Dice evaluation protocol | Proposal + zero-shot SAM2 baseline numbers |
| 2 | 6 | Full zero-shot baseline table | Adapter trained, first checkpoint | Preliminary IoU/Dice comparison | Progress report, 1 comparison |
| 3 | 9 | Final zero-shot numbers | Final adapter + ablation on adapter size | Full IoU/Dice comparison, qualitative masks | Final report, repo, demo |

---

#### 23. Prompt-Driven Interactive Segmentation as a Dataset-Annotation Tool

**Objectives**
- Build a small interactive labeling tool around SAM2 (click/box prompts → mask).
- Measure annotation-time savings vs. manual polygon labeling; quantify mask quality vs. ground truth.

**References**
- Ravi et al., "SAM 2: Segment Anything in Images and Videos" (as above).

**Dataset:** a few hundred images from COCO, scored against existing COCO ground-truth masks.

**Baseline:** manual polygon annotation time, or the stock SAM2 demo UI. **Compute tier:** CPU-OK (inference-only).

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | COCO subset + GT masks prepared | Design minimal click/box-prompt UI around SAM2 | Define time-to-annotate + mask-IoU protocol | Proposal + SAM2 inference demo |
| 2 | 6 | Manual-annotation time measured on a sample | Working interactive tool, first version | Preliminary IoU-vs-GT comparison | Progress report, working tool demo |
| 3 | 9 | Final manual-baseline timing | Polished tool with usability fixes | Full time-savings + IoU comparison, small user test | Final report, repo, demo |

---

#### 24. Learned Image Colorization with a Lightweight Diffusion/GAN Head

**Objectives**
- Build a compact colorization model conditioned on grayscale input.
- Compare against the classic Zhang et al. CNN colorizer with pixel metrics and a perceptual/user-preference study.

**References**
- Zhang, Isola, Efros, "Colorful Image Colorization," *ECCV* 2016.
- "Enhancing Image Colorization with Semantic-Guided Diffusion and Lightweight Fine-Tuning," 2025/2026. [[paper]](https://link.springer.com/article/10.1007/s00371-026-04577-w)

**Dataset:** 5,000–10,000 images from COCO or Places365, grayscaled as input, color originals as ground truth.

**Baseline:** Zhang et al. 2016 classic colorization CNN. **Compute tier:** GPU-light.

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | Grayscale/color-pair pipeline; Zhang-2016 baseline scoped | Design lightweight colorization head | Define pixel-metric + user-preference-study protocol | Proposal + baseline colorization samples |
| 2 | 6 | Zhang-2016 baseline trained/evaluated | Lightweight model, first checkpoint | Preliminary pixel-metric comparison | Progress report, side-by-side samples |
| 3 | 9 | Final baseline numbers | Final model + ablation | Full comparison + small user-preference study | Final report, repo, demo |

---

#### 25. Content-Aware Pseudo-Color Mapping for Scientific/Medical Grayscale Images

**Objectives**
- Extend the course's Week-10 pseudo-color/intensity-slicing material with a learned, saliency-aware colormap highlighting important regions better than a fixed colormap (jet/viridis).
- Evaluate with a small user study or a downstream-task proxy.

**References**
- Course material, Week 10 (HSI pseudo-color processing) — classical baseline methodology.
- Team-driven: survey 2023–2026 saliency-guided visualization papers during M1.

**Dataset:** a public thermal dataset sample, or the NIH ChestX-ray14 sample (above).

**Baseline:** standard intensity-slicing / fixed LUT pseudo-color (course Week 10 method). **Compute tier:** CPU-OK.

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | Dataset pipeline; fixed-LUT pseudo-color baseline (Week 10 method) | Design saliency-aware colormap mechanism | Define user-study or downstream-task proxy protocol | Proposal + fixed-LUT baseline samples |
| 2 | 6 | Full fixed-LUT baseline gallery | Saliency-aware colormap, first version | Preliminary comparison / proxy-task numbers | Progress report, side-by-side samples |
| 3 | 9 | Final baseline gallery | Final colormap method | Full comparison + user study / proxy-task results | Final report, repo, demo |

---

#### 26. Compact Deep Feature Descriptors vs. Classical SIFT/ORB

**Objectives**
- Train or fine-tune a small learned local-feature descriptor.
- Benchmark matching accuracy/speed against the course's classical Harris/SIFT/ORB pipeline (Week 9) on an image-matching/registration task.

**References**
- DeTone, Malisiewicz, Rabinovich, "SuperPoint: Self-Supervised Interest Point Detection and Description," *CVPRW* 2018.
- Course material, Week 9 (Harris/SIFT/ORB).

**Dataset:** [HPatches](https://github.com/hpatches/hpatches-dataset) (standard public local-feature benchmark), or a self-collected small image-pair set.

**Baseline:** OpenCV SIFT/ORB + RANSAC matching (already built in the course's Week 9 lab). **Compute tier:** CPU-OK / GPU-light.

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | HPatches pipeline; SIFT/ORB baseline from Week 9 lab | Design/select compact descriptor architecture | Define matching-accuracy + speed protocol | Proposal + SIFT/ORB baseline numbers |
| 2 | 6 | Full SIFT/ORB baseline table | Descriptor trained/fine-tuned, first checkpoint | Preliminary matching comparison | Progress report, 1 comparison |
| 3 | 9 | Final classical baseline table | Final descriptor + speed ablation | Full accuracy/speed comparison, registration demo | Final report, repo, demo |

---

### E. Inpainting, Forensics & Metrics (4 projects)

#### 27. Diffusion-Based Image Inpainting on a Resource-Constrained Budget

**Objectives**
- Use a small pretrained public diffusion inpainting checkpoint with a lightweight fine-tuned mask-conditioning head.
- Evaluate on a structured-vs-random mask benchmark.

**References**
- "Towards Enhanced Image Inpainting: Mitigating Unwanted Object Insertion and Preserving Color Consistency," *CVPR* 2025. [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Towards_Enhanced_Image_Inpainting_Mitigating_Unwanted_Object_Insertion_and_Preserving_CVPR_2025_paper.pdf)

**Dataset:** [Places2](http://places2.csail.mit.edu/) — a random 2,000–5,000 image subset, **not** the full dataset — with synthetic masks.

**Baseline:** classical exemplar-based inpainting (OpenCV `cv2.inpaint`, Telea/Navier–Stokes methods). **Compute tier:** GPU-light (frozen pretrained backbone + small trainable head).

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | Places2 subset + mask generator; `cv2.inpaint` baseline scoped | Select/wire up frozen pretrained diffusion inpainting backbone | Define structured-vs-random-mask evaluation protocol | Proposal + classical-inpaint baseline samples |
| 2 | 6 | Full classical baseline on both mask types | Mask-conditioning head trained, first checkpoint | Preliminary comparison on random masks | Progress report, side-by-side samples |
| 3 | 9 | Final classical baseline on both mask types | Final pipeline, failure-case analysis | Full comparison both mask types, LPIPS/FID | Final report, repo, demo |

---

#### 28. Building and Validating a New No-Reference Image-Quality Metric

**Objectives**
- Design a lightweight no-reference IQA model.
- Train/calibrate against human MOS data; benchmark correlation (SROCC/PLCC) against LPIPS/DISTS and classical metrics.

**References**
- Zhang et al. (LPIPS) and Ding et al. (DISTS), as above.
- "Scene Perceived Image Perceptual Score (SPIPS)," *arXiv:2504.17234*. [[paper]](https://arxiv.org/html/2504.17234)
- "Applicability Limitations of Differentiable Full-Reference Image-Quality Metrics," *arXiv:2212.05499*. [[paper]](https://arxiv.org/pdf/2212.05499)

**Dataset:** [KADID-10k](http://database.mmsp-kn.de/kadid-10k-database.html) or [TID2013](http://www.ponomarenko.info/tid2013.htm) — small, public, with MOS labels.

**Baseline:** PSNR/SSIM and pretrained LPIPS correlation with MOS on the same set. **Compute tier:** CPU-OK / GPU-light.

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | KADID-10k/TID2013 pipeline; PSNR/SSIM/LPIPS-vs-MOS baseline correlations | Design lightweight no-reference IQA architecture | Define SROCC/PLCC evaluation protocol | Proposal + baseline correlation table |
| 2 | 6 | Full baseline correlation table | IQA model trained, first checkpoint | Preliminary SROCC/PLCC for your metric | Progress report, 1 correlation number |
| 3 | 9 | Final baseline correlations | Final IQA model + ablation | Full SROCC/PLCC comparison table, cross-dataset generalization check | Final report, repo, demo |

---

#### 29. Compression-Robust Detection of AI-Generated/Edited Images

**Objectives**
- Train a detector distinguishing real vs. AI-generated/edited images.
- Study how detection accuracy degrades after re-compression (JPEG at various QFs).

**References**
- Zhao et al., "Invisible Image Watermarks Are Provably Removable Using Generative AI," *NeurIPS* 2024 (adjacent tooling).
- Team-driven: survey 2024–2026 AI-image-detection literature during M1 — a fast-moving area, so the literature review is explicitly part of the grade here.

**Dataset:** a subset of COCO as "real" images, plus a small public set of diffusion-generated images released alongside detector papers (document exact provenance and license of any generated-image set used).

**Baseline:** a simple frequency-domain (DCT statistics) classical detector vs. your learned detector, both before/after re-compression. **Compute tier:** CPU-OK / GPU-light.

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | Real/generated dataset assembled + license documented | Implement DCT-statistics classical detector | Define accuracy-vs-QF evaluation protocol | Proposal + classical detector baseline numbers |
| 2 | 6 | Full classical baseline across QFs | Learned detector, first checkpoint | Preliminary accuracy-vs-QF comparison | Progress report, 1 comparison |
| 3 | 9 | Final classical baseline table | Final detector + ablation | Full accuracy-degradation-vs-QF comparison | Final report, repo, demo |

---

#### 30. Rate–Distortion Benchmarking Suite: Classical vs. Learned Codecs

**Objectives**
- Build a rigorous, reusable benchmarking pipeline (no novel model required) comparing classical codecs (JPEG, JPEG2000, WebP, AVIF) against 2–3 pretrained learned codecs (via CompressAI) under one common, documented protocol.
- Produce a public leaderboard-style report; this is a systems/rigor-focused capstone for a team that wants depth on experimental methodology rather than a new model.

**References**
- [CompressAI](https://github.com/InterDigitalInc/CompressAI).
- CLIC challenge protocol and leaderboard — [link](https://clic2025.compression.cc/tasks/).
- "Measuring the Progress of Learned Image Compression." [[article]](https://www.mabyduck.com/blog/challenge-on-learned-image-compression-2025/)

**Dataset:** Kodak, Tecnick, and the CLIC validation set.

**Baseline:** none needed to beat — the contribution is rigor, reproducibility, and breadth of comparison. Bonus credit available for actually submitting results to the open CLIC validation leaderboard. **Compute tier:** CPU-OK (inference/evaluation only, pretrained codecs).

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | Kodak/Tecnick/CLIC pipeline assembled | Install & wire up 2–3 CompressAI pretrained codecs | Design the common bpp-matching + metric protocol | Proposal + protocol specification document |
| 2 | 6 | Classical-codec (JPEG/JPEG2000/WebP/AVIF) results | Learned-codec results running end-to-end | Preliminary comparison table, 1 dataset | Progress report, 1 comparison table |
| 3 | 9 | Final classical-codec results, all datasets | Final learned-codec results, all datasets | Full leaderboard-style report, decode-time table; optional CLIC submission | Final report, repo, demo |

---

*End of catalog. See `project-instructions.md` for team formation, compute policy, milestones, roles, LaTeX templates, rubrics, the bonus track, and the dataset quick-reference table.*
