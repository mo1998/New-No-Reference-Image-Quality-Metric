# Project Journal — Project 28: No-Reference Image-Quality Metric

One entry per build step. Format: **what we did and why → exact commands → how we verified
it → what to say if the instructor asks**. This doubles as raw material for the M1 report
and for the final reproducibility statement (course §9.3).

---

## 2026-10-09 — Step 0: Repository scaffold

**What & why.** Created the team repository the course requires from Week 1
(instructions §9.3). Private GitHub repo so course material and unfinished results stay
team-only.

**Commands.**
```
git init -b main
git config user.name "mo1998"
git config user.email "mo1998@users.noreply.github.com"
git add -A
git commit -m "Initial commit: repo scaffold for Project 28 (no-reference IQA)"
gh repo create mo1998/dip-project-28-nr-iqa --private --source=. --push
```

**What it contains.** `README.md` (project overview, milestones, datasets, evaluation
protocol), `requirements.txt` (starter dependency list), `.gitignore` (excludes `data/`,
model checkpoints, venv), `src/utils.py` (`set_seed(42)` — the fixed seed the course
requires), `data/` and `results/` folders, plus the two course documents.

**How we verified.** `gh repo view` → visibility `PRIVATE`, default branch `main`;
`git status` clean; repo at https://github.com/mo1998/dip-project-28-nr-iqa

**Instructor Q&A.**
- *Why one private repo per team?* Course §9.3 requires it from Week 1. Commit history is
  partial evidence of individual contribution if the §9.5 peer evaluation is disputed, and
  it is graded under M2 "Experimental rigor" and M3 "Reproducibility".
- *Why does the repo already contain `set_seed(42)`?* The course requires a fixed random
  seed for the headline result; putting it in a shared utility means every script uses the
  same one from day one.

---

## 2026-10-10 — Step 1: Local Python environment (`.venv` + pinned versions)

**What & why.** Created the isolated project environment the course requires (§3: standard
stack, pinned versions). A virtual environment keeps the project's package versions
independent of system Python so every teammate and the instructor can reproduce the setup
from `requirements.txt`.

**Commands.**
```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# smoke test
.\.venv\Scripts\python.exe -c "import torch, lpips, pyiqa; ..."
# pin exact versions of everything that got installed
cmd /c ".venv\Scripts\python.exe -m pip freeze > requirements.lock.txt"
```

**Installed (headline versions).** torch 2.14.1+cpu, torchvision 0.29.1+cpu, numpy 2.2.6,
scipy 1.15.3, lpips 0.1.4, pyiqa 0.1.16, scikit-image 0.25.2, opencv-python 5.0.0.93,
jupyterlab 4.6.4. Full list: `requirements.lock.txt`.

**How we verified.** Import smoke test passes: `lpips import OK`, `pyiqa import OK`;
`torch.cuda.is_available()` is `False` — expected, this machine has no NVIDIA GPU
(Colab is our GPU option; see Step 2).

**Instructor Q&A.**
- *Why a virtual environment instead of installing system-wide?* Isolation + reproducibility:
  deleting `.venv` leaves no trace, and anyone can rebuild it from `requirements.txt`.
- *Why CPU PyTorch?* No GPU on this machine and Project 28 is tagged CPU-OK; the course's
  reference machine allows CPU-only. GPU runs happen on Colab as a supplement.
- *How are versions pinned?* `requirements.lock.txt` records the exact version of every
  installed package (`pip freeze`); `requirements.txt` stays the human-edited list of
  direct dependencies.

---

## 2026-10-10 — Step 2: Colab starter notebook (GPU supplement)

**What & why.** The team will use free Google Colab (T4-class GPU) for GPU-heavy runs —
e.g., LPIPS over the full 10k-image set and CNN training. Course §3 explicitly allows free
cloud notebooks as a *supplement*: the project must still complete without them, which the
local CPU setup from Step 1 guarantees.

**Rules baked into the notebook.**
- Colab VMs are wiped between sessions → datasets and checkpoints go to Google Drive
  (`/content/drive/MyDrive/dip-p28/`); code lives in GitHub; results of record are committed
  to `results/`.
- The repo is private → the notebook clones it with a fine-grained read-only token stored in
  Colab Secrets as `GH_TOKEN` (never printed, never committed; the clone cell scrubs it from
  the git remote afterwards).
- Same fixed seed (42) and same dependency list as the local environment.

**File.** `notebooks/00_colab_starter.ipynb` — cells: GPU check → Drive mount → token clone
→ dependency install → seed.

**How we verify.** Open the notebook on colab.research.google.com (upload from the repo) and
run all cells; expected output: GPU name printed, Drive mounted, repo cloned, `seed set: 42`.

**Instructor Q&A.**
- *Is using Colab allowed?* Yes — §3: allowed as a supplement, not a requirement; the
  pipeline stays CPU-feasible and nothing is stored only in Colab.
- *Why a GitHub token?* The repo is private until the M3 deliverable (which requires a
  public code link, §9.4); the token is read-only, kept in Colab's secret store, and can be
  revoked any time.

---

## 2026-10-10 — Step 3: KADID-10k dataset (download + verification)

**What & why.** Downloaded the official KADID-10k archive (~3.1 GB) and verified it before
any use. Storage: the zip persists in Drive; each session copies it to `/content` and
unzips locally.

**Issue 1 - expired TLS certificate.** The authors' host (datasets.vqa.mmsp-kn.de) serves
an expired certificate, so wget refused (exit 5, SSL verification failure). We downloaded
with `--no-check-certificate` and compensated with content checks: (a) exact byte size
3,067,408,471, (b) zip CRC test (`unzip -t`), (c) SHA-256 of `dmos.csv` compared against
the authors' OSF mirror (osf.io/xkqjh, valid TLS) - **matched**.

**Issue 2 - file count.** Expected 10,206 PNGs, found 10,338 = 81 refs + 10,125 distorted
+ 132 macOS `._*` metadata files (harmless; loaders filter them out).

**Dataset facts (for the report).** Layout: `kadid10k/images/` (refs `Ixx.png`, distorted
`Ixx_yy_zz.png`) + `kadid10k/dmos.csv` with columns `dist_img, ref_img, dmos, var`.
10,125 labelled distorted images; DMOS in [1, 5], higher = better quality. Citation:
Lin, Hosu & Saupe, *KADID-10k*, QoMEX 2019.

**How we verified.** Script prints: refs=81, distorted=10,125, DMOS rows=10,125,
sha256 matches=True (run in the Colab notebook, committed there).

**Instructor Q&A.**
- *Why bypass certificate verification?* The certificate was expired server-side; we
  replaced transport authentication with content authentication (exact size + CRC +
  published-hash cross-check against the authors' own mirror). Documented for
  reproducibility.
- *Which files does the project use?* The 10,125 distorted images with their DMOS labels;
  the 81 pristine references are needed for the full-reference baselines (PSNR/SSIM/LPIPS).

---

## 2026-10-10 — Step 4: M1 baseline correlations (PSNR / SSIM / LPIPS vs DMOS)

**What & why.** Computed the three full-reference baselines on the M1 test split - the
table the catalog specifies as the M1 deliverable. Code: `src/run_m1_baselines.py` using
`src/baselines.py` + `src/eval_protocol.py`. Protocol: content split by reference image
(65 train / 16 test refs, seed 42), metrics on RGB, correlations per VQEG convention
(SROCC raw; PLCC/RMSE after a 4-parameter logistic mapping), LPIPS sign-flipped
(distance -> quality direction).

**Results (test split, 2,000 distorted images; commit `58868ea`).**

| Metric | SROCC | PLCC | RMSE |
|---|---|---|---|
| PSNR | 0.667 | 0.670 | 0.803 |
| SSIM | 0.627 | 0.616 | 0.852 |
| LPIPS | 0.808 | 0.807 | 0.639 |

Raw per-image scores: `results/m1_baseline_correlations_test_split_raw.csv`.

**Bugs found & fixed (each with a regression test).**
1. Logistic fit overflowed on PSNR's dB scale -> inputs standardised (z-score) + finite
   guard with identity fallback.
2. (Near-)identical image pairs -> PSNR = +inf -> nan PLCC. Fix: PSNR capped at 60 dB
   (64/2,000 pairs hit the ceiling) + all non-finite scores sanitized before correlation.
3. Colab push failures, three stacked causes: (a) duplicated `git` in the command builder;
   (b) `$repo.git` is not substituted by the notebook shell - the dot breaks the variable
   match, so bash ate `$repo` as empty and origin became `https://github.com/.git` ->
   every push returned 404; fixed by cloning via `subprocess` (no shell interpolation)
   and re-validating the origin URL before pushing; (c) non-fast-forward after remote
   moved on -> push cell now runs `pull --rebase` first.

**Local tests.** `tests/test_eval_protocol.py` + `tests/test_baselines.py` - 5 pass
(content-split leak check, monotone recovery, PSNR-scale stability, non-finite sanitation,
60 dB cap).

**Instructor Q&A.**
- *Why is LPIPS (a deep method) acceptable as a baseline?* It is a pretrained
  full-reference metric used as a published reference point; the project's own
  contribution is the no-reference model. The catalog lists it as a baseline.
- *Why split by content?* Splitting by distorted image would leak pristine content
  across train/test and inflate every correlation.
- *Why is PSNR capped at 60 dB?* Identical pairs give MSE = 0 -> infinite PSNR; 60 dB is
  already below one 8-bit grey level of average error; capped pairs are counted and
  reported.
- *Why PLCC after a logistic fit?* Metric outputs and DMOS are on different scales; the
  logistic mapping is the standard VQEG remedy. SROCC is rank-based and needs no fit.

---

## 2026-10-10 — Step 5: M1 report draft (LaTeX)

**What & why.** Assembled the Milestone 1 report draft `milestones/m1_proposal.tex` from the
course template (§6.1) with everything produced so far: problem framing, related work (11
references with explicit roles), dataset & baseline plan with the measured baseline table,
proposed frozen-backbone + lightweight-head method, feasibility (CPU-OK / GPU-light),
MVP/stretch split, four risks with mitigations, and the per-student work breakdown. The
shared bibliography moved to `milestones/references.bib` (one file for all three
milestones).

**How we verify.** All `\citep` keys are checked against the `.bib` (see the key-check
below); balanced LaTeX environments verified by script. The PDF compile runs on Overleaf
(pdflatex -> bibtex -> pdflatex -> pdflatex) - LaTeX is not installed on the laptop.

**Instructor Q&A.**
- *Why frozen features + a light head?* Frozen features carry quality information
  (ARNIQA, WACV-2026); training stays feasible on the course machine and overfitting on
  8k training images is controlled.
- *Why is LPIPS acceptable though it is deep?* It is a required catalog baseline; its
  AlexNet backbone runs on CPU and the computation is one-off.
- *What is novel?* Not a new architecture: a rigorously validated, budget-conscious
  instantiation with a leakage-controlled protocol and a cross-dataset check.

**Remaining before submission.** Fill the `\todo` markers (names, date, course code),
team edit pass, compile to PDF on Overleaf, and verify the baseline numbers against
`results/m1_baseline_correlations.csv`.
