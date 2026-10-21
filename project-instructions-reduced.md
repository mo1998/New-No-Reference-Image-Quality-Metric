# Digital Image Processing — Team Project Instructions

### [CSE 681] — Digital Image Processing & Video Processing· [F26] · Instructor: [Hazem Abbas & Mahmoud Khalil]

**Team size:** 3 students · **Duration:** 9 weeks · **Milestones:** 3 (Weeks 3, 6, 9)

> This document covers logistics, compute policy, the milestone framework, grading rubrics,
> the publication bonus track, and LaTeX report templates. **The 30-project catalog is a
> companion document: `project-catalog.md`** — hand both out together. Placeholders in
> `[LIKE THIS]` follow the same convention as the rest of the course repository and must be
> filled in before distribution.

---

## Table of Contents

1. [Overview & Learning Goals](#1-overview--learning-goals)
2. [Team Formation & Project Selection](#2-team-formation--project-selection)
3. [Compute & Software Policy](#3-compute--software-policy)
4. [The 3-Milestone Framework](#4-the-3-milestone-framework)
5. [Student Roles](#5-student-roles)
6. [LaTeX Report Templates](#6-latex-report-templates)
7. [Milestone Rubrics](#7-milestone-rubrics)
8. [Publication Bonus Track](#8-publication-bonus-track)
9. [Things You May Have Missed](#9-things-you-may-have-missed)
10. [Dataset Quick-Reference Table](#10-dataset-quick-reference-table)

**→ See `project-catalog.md` for the 30 project entries** (objectives, references, datasets,
baselines, compute tiers, and per-student milestone tasks).

---

## 1. Overview & Learning Goals

Each team of three reproduces, adapts, or extends a piece of **recent, publicly verifiable**
image-processing research, with a deliberate tilt toward **image coding/compression**, since
that is where this course can make the most distinctive contribution relative to a generic
computer-vision project course. Projects may use pretrained AI/ML components (frozen
backbones, small fine-tuned heads, pretrained diffusion priors) as building blocks within a
classical image-processing framework — this is encouraged, not just permitted, and mirrors
how the field actually works in 2026.

By the end of the project, every team will have:

- Reproduced a credible baseline from the literature on public data.
- Implemented and evaluated a specific, scoped extension or comparison.
- Benchmarked against a stated state-of-the-art (SOTA) reference point using a documented,
  fair protocol.
- Produced a short paper-style report and a working, reproducible code repository.

## 2. Team Formation & Project Selection

- **Teams of 3**, self-formed by default; instructor assigns any leftover students by Oct 12. 2026.
- **Project signup** opens immediately for one week (before Milestone 1). Use a
  shared signup sheet against the `project-catalog.md` list; cap **at most 2 teams per
  project title** so the class gets breadth of results to compare and discuss in the final
  session.
- **Propose-your-own-topic option**: a team may propose a project outside the catalog if it
  (a) has at least one 2023–2026 reference paper, (b) uses a public dataset, and (c) fits the
  laptop-class compute budget in Section 3. Submit the proposal as part of Milestone 1 — it
  is graded against the same rubric as a catalog project, with the literature-review and
  feasibility criteria carrying extra weight since nobody has pre-vetted it.
- Two teams may not select the **same project** without instructor approval (to avoid
  temptation to collaborate past the team boundary); two teams working independently on the
  **same catalog entry** is fine and often instructive for the final session.

## 3. Compute & Software Policy

No HPC cluster, shared lab GPU, or paid cloud compute is required or assumed. Design point:

> **Reference machine:** 8-core CPU, 16 GB RAM, optional entry-level consumer/laptop GPU
> (e.g., RTX 3050/4050-class or Apple Silicon) with 4–8 GB VRAM. Every project in the catalog
> is solvable on this machine within the 9-week window using the dataset-subsampling
> guidance given in each entry.

- Each catalog entry is tagged **CPU-OK** (trains/runs fine with no GPU, just patience) or
  **GPU-light** (noticeably faster with an entry-level GPU, but not blocked without one).
- **Free cloud notebooks (Google Colab, Kaggle Kernels) are allowed as a supplement**, not a
  requirement — they are not "high-performance computing" in the sense this policy excludes
  (that exclusion targets shared HPC clusters / paid multi-GPU rental, not a free T4 for an
  afternoon). Do not architect a project that is infeasible without one.
- **Always subsample.** Nobody needs the full 118k-image COCO `train2017` split or the full
  Places2 training set for a 9-week course project. Each catalog entry says what subset size
  is reasonable; teams may subsample further and must state the exact subset used (image
  count, selection method, random seed) in the report for reproducibility.
- **Use pretrained/frozen components wherever the literature does.** If a project's reference
  paper uses a frozen foundation model (SAM2, a pretrained diffusion prior, a pretrained
  classifier) as part of its pipeline, your team should too — training such a model from
  scratch is explicitly out of scope and not expected.
- Standard stack: Python 3.10+, PyTorch, NumPy/SciPy, OpenCV/scikit-image,
  [CompressAI](https://github.com/InterDigitalInc/CompressAI) for compression baselines,
  Jupyter for exploration. Pin versions in a `requirements.txt` or `environment.yml`
  (required at every milestone — see Section 9.4).

## 4. The 3-Milestone Framework

| Milestone | Week | Theme | Weight of project grade |
|---|---|---|---|
| **M1 — Proposal & Literature Review** | 3 | Scope the problem, survey ≥4 recent references, identify dataset & baseline, assess feasibility | 15% |
| **M2 — Implementation & Preliminary Results** | 6 | Working baseline reproduction + first results from your own method/extension | 35% |
| **M3 — Final Report, Results & Presentation** | 9 | Full comparison vs. SOTA baseline, ablations, final report, live demo/presentation | 50% |

*(The instructor may shift these by ±1 week to align with the course calendar; the relative
spacing — roughly a third of the project at each checkpoint — should be preserved.)*

**At Milestone 1, every team must state two things explicitly, to be graded against at M2/M3:**

1. A **minimum-viable deliverable (MVP)** — the smallest version of the project that still
   produces a valid, reportable comparison against the baseline.
2. A **stretch goal** — what the team will attempt if the MVP is on track by Week 5–6.

This is the single most useful thing a team can do to avoid a Week-8 crisis, and M2 grading
explicitly rewards having hit the MVP on schedule over having attempted (and only partially
completed) the stretch goal.

## 5. Student Roles

Each catalog project's milestone table suggests a 3-way division of labor. Roles are a
*default division of labor*, not a mandatory assignment — teams may reassign freely, but
should say so in the report's contribution statement (Section 9.5 covers the required
peer-evaluation mechanism that keeps this honest).

| Role | Typical responsibilities |
|---|---|
| **Student A — Data & Baseline Lead** | Dataset acquisition/subsampling, preprocessing pipeline, reproducing the published baseline, owns the literature-review table |
| **Student B — Model/Algorithm Lead** | Core method implementation, training loop, the project's main technical contribution |
| **Student C — Evaluation & Integration Lead** | Metrics implementation, benchmark protocol, ablations, figures/tables, demo packaging |

All three students co-author the report and are expected to understand the whole pipeline,
not just their own slice — Milestone 3's presentation is unscripted Q&A, and each student is
individually examinable on any part of the project.

---

## 6. LaTeX Report Templates

Three standalone, self-contained `article`-class templates — one per milestone. They are
intentionally **not** dependent on the course repository's `shared/preamble.tex`, so any
team can compile them directly on Overleaf or locally with nothing beyond a standard TeX
Live install. All three use `natbib` with the `unsrtnat` style for author–year citations
(`\citep`/`\citet`), consistent with the rest of the course's documents, and compile cleanly
under `pdflatex` (verified: `pdflatex → bibtex → pdflatex → pdflatex`, zero errors, zero
undefined references, against a dummy two-entry `references.bib`).

Each team maintains **one shared `references.bib`** across all three milestones (add entries
as you find them during the literature review — don't recreate the bibliography each time).

> **Build sequence (same as the rest of the course):**
> ```
> pdflatex report.tex
> bibtex report
> pdflatex report.tex
> pdflatex report.tex
> ```

### 6.1 Milestone 1 — Proposal & Literature Review Template

```latex
\documentclass[11pt]{article}

% ============================================================
% Milestone 1 -- Project Proposal & Literature Review Template
% DIP Team Project
% ============================================================
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[round]{natbib}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=blue!50!black, citecolor=blue!50!black, urlcolor=blue!50!black}
\usepackage{enumitem}
\usepackage{xcolor}

\newcommand{\todo}[1]{\textcolor{red}{\textbf{[TODO: #1]}}}

\title{\LARGE Project Proposal \& Literature Review \\[4pt]
\large \todo{Project Title}}
\author{
  \todo{Student A Full Name} \and
  \todo{Student B Full Name} \and
  \todo{Student C Full Name}
}
\date{Milestone 1 --- \todo{Date} \\{} [COURSE CODE] --- Digital Image Processing}

\begin{document}
\maketitle

\begin{abstract}
\todo{3--5 sentences: what problem are you tackling, why does it matter, what is your
planned approach, and what public dataset/baseline will you use. Write this last.}
\end{abstract}

\section{Problem Statement \& Motivation}
\label{sec:motivation}
\todo{What image-processing problem are you solving? Why is it interesting or useful?
How does it connect to material from the course (name the week/topic)? 1 paragraph.}

\section{Related Work}
\label{sec:related}
\todo{Summarize at least 4 references, at least 2 of which should be from 2023--2026.
Use the table below as a starting organizer, then write 1--2 paragraphs of prose
connecting them to your plan.}

\begin{table}[h]
\centering
\caption{Key references.}
\label{tab:refs}
\begin{tabular}{@{}p{0.30\linewidth}p{0.12\linewidth}p{0.45\linewidth}@{}}
\toprule
\textbf{Reference} & \textbf{Year} & \textbf{Relevance to this project} \\
\midrule
\citep{example2024} & 2024 & \todo{one sentence} \\
\citep{example2018}  & 2018 & \todo{one sentence} \\
\bottomrule
\end{tabular}
\end{table}

\section{Dataset \& Baseline Plan}
\label{sec:data}
\todo{Name the public dataset(s), give exact size/subset you will use, and the license.
State exactly which baseline (classical or prior learned method) you will reproduce
and compare against.}

\begin{table}[h]
\centering
\caption{Dataset summary.}
\label{tab:data}
\begin{tabular}{@{}lll@{}}
\toprule
\textbf{Dataset} & \textbf{Subset used} & \textbf{Link} \\
\midrule
\todo{name} & \todo{e.g., 800 train / 100 val crops} & \todo{URL} \\
\bottomrule
\end{tabular}
\end{table}

\section{Proposed Method}
\label{sec:method}
\todo{Sketch your planned approach: architecture diagram or pseudocode, what is novel
or extended relative to the baseline, and why you expect it to work.}

\section{Feasibility \& Compute Plan}
\label{sec:feasibility}
\todo{State your compute tier (CPU-OK / GPU-light), expected training time per
experiment, and confirm this fits a laptop-class machine within the project window.}

\subsection{Minimum Viable Deliverable (MVP)}
\todo{The smallest version of this project that still produces a valid, reportable
comparison against the baseline.}

\subsection{Stretch Goal}
\todo{What you will attempt if the MVP is on track by Week 5--6.}

\section{Risk Assessment}
\label{sec:risk}
\begin{itemize}[leftmargin=*]
  \item \todo{Risk 1 and its mitigation / fallback.}
  \item \todo{Risk 2 and its mitigation / fallback.}
\end{itemize}

\section{Work Breakdown}
\label{sec:workbreakdown}
\begin{table}[h]
\centering
\caption{Milestone task allocation (see also the project catalog entry for this
project).}
\label{tab:tasks}
\begin{tabular}{@{}p{0.12\linewidth}p{0.26\linewidth}p{0.26\linewidth}p{0.26\linewidth}@{}}
\toprule
\textbf{Milestone} & \textbf{Student A} & \textbf{Student B} & \textbf{Student C} \\
\midrule
M1 (Wk 3) & \todo{task} & \todo{task} & \todo{task} \\
M2 (Wk 6) & \todo{task} & \todo{task} & \todo{task} \\
M3 (Wk 9) & \todo{task} & \todo{task} & \todo{task} \\
\bottomrule
\end{tabular}
\end{table}

\bibliographystyle{unsrtnat}
\bibliography{references}

\end{document}
```

### 6.2 Milestone 2 — Progress Report Template

```latex
\documentclass[11pt]{article}

% ============================================================
% Milestone 2 -- Progress Report Template
% DIP Team Project
% ============================================================
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[round]{natbib}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=blue!50!black, citecolor=blue!50!black, urlcolor=blue!50!black}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{caption}

\newcommand{\todo}[1]{\textcolor{red}{\textbf{[TODO: #1]}}}

\title{\LARGE Progress Report \\[4pt]
\large \todo{Project Title}}
\author{
  \todo{Student A Full Name} \and
  \todo{Student B Full Name} \and
  \todo{Student C Full Name}
}
\date{Milestone 2 --- \todo{Date} \\{} [COURSE CODE] --- Digital Image Processing}

\begin{document}
\maketitle

\begin{abstract}
\todo{3--5 sentences: recap the problem, state what is implemented and working now,
and headline your single most important preliminary result.}
\end{abstract}

\section{Recap}
\label{sec:recap}
\todo{1 paragraph: problem, baseline, dataset -- carried over from M1, updated if scope
changed.}

\section{Implementation Status}
\label{sec:status}
\todo{What is built and working right now. Be concrete: architecture as actually
implemented (not as planned), training configuration, any deviation from the M1 plan
and why.}

\subsection{Baseline Reproduction}
\todo{Confirm the baseline is reproduced and working; report its numbers.}

\subsection{Your Method}
\todo{Status of your extension/contribution: what runs end-to-end, what doesn't yet.}

\section{Preliminary Results}
\label{sec:results}
\todo{At least one quantitative table or rate-distortion/accuracy curve comparing your
method (even partially trained) to the baseline, plus 2--3 qualitative examples.}

\begin{table}[h]
\centering
\caption{Preliminary quantitative comparison.}
\label{tab:prelim}
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Method} & \textbf{Metric 1} & \textbf{Metric 2} & \textbf{Notes} \\
\midrule
Baseline        & \todo{val} & \todo{val} & \todo{} \\
Ours (partial)  & \todo{val} & \todo{val} & \todo{} \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[h]
\centering
% \includegraphics[width=0.8\linewidth]{figures/preliminary_result.png}
\caption{\todo{Describe what this figure actually shows -- not what you expected it to
show. If the qualitative result is weaker or different than hoped, say so and explain
your current hypothesis.}}
\label{fig:prelim}
\end{figure}

\section{Challenges \& Deviations from the M1 Plan}
\label{sec:challenges}
\todo{Be specific and honest. What did not work as planned? What did you change and
why? This section is graded on candor and problem-solving, not on everything having
gone smoothly.}

\section{Updated Work Breakdown \& MVP Status}
\label{sec:workbreakdown}
\todo{State explicitly whether the M1 minimum-viable deliverable has been reached. If
not, what is the revised plan to reach it before M3?}

\begin{table}[h]
\centering
\caption{Updated milestone task allocation.}
\label{tab:tasks}
\begin{tabular}{@{}p{0.12\linewidth}p{0.26\linewidth}p{0.26\linewidth}p{0.26\linewidth}@{}}
\toprule
\textbf{Milestone} & \textbf{Student A} & \textbf{Student B} & \textbf{Student C} \\
\midrule
M2 (Wk 6) -- completed & \todo{what was done} & \todo{what was done} & \todo{what was done} \\
M3 (Wk 9) -- planned   & \todo{task} & \todo{task} & \todo{task} \\
\bottomrule
\end{tabular}
\end{table}

\section{Next Steps}
\label{sec:next}
\todo{Concrete plan for Weeks 7--9.}

\bibliographystyle{unsrtnat}
\bibliography{references}

\end{document}
```

### 6.3 Milestone 3 — Final Report Template

```latex
\documentclass[11pt]{article}

% ============================================================
% Milestone 3 -- Final Report Template
% DIP Team Project
% ============================================================
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[round]{natbib}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=blue!50!black, citecolor=blue!50!black, urlcolor=blue!50!black}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{caption}
\usepackage{subcaption}

\newcommand{\todo}[1]{\textcolor{red}{\textbf{[TODO: #1]}}}

\title{\LARGE \todo{Project Title} \\[4pt]
\large Final Report}
\author{
  \todo{Student A Full Name} \and
  \todo{Student B Full Name} \and
  \todo{Student C Full Name}
}
\date{Milestone 3 --- \todo{Date} \\{} [COURSE CODE] --- Digital Image Processing}

\begin{document}
\maketitle

\begin{abstract}
\todo{150--250 words. Problem, approach, dataset, headline quantitative result vs.
the stated baseline/SOTA reference, and one-sentence takeaway. Write this last.}
\end{abstract}

\section{Introduction}
\label{sec:intro}
\todo{What problem, why it matters, what you did, and a one-paragraph summary of your
findings (reviewers should be able to read the intro and abstract alone and know what
you found).}

\section{Related Work}
\label{sec:related}
\todo{Position your work relative to the references identified at M1 (and any added
since). Group by theme, not by chronology. At least 2 references should be from
2023--2026.}

\section{Method}
\label{sec:method}
\todo{Describe your final approach precisely enough that another team could
reimplement it: architecture, loss function(s), training procedure, hyperparameters.
Use an algorithm block or architecture figure if it helps.}

\begin{figure}[h]
\centering
% \includegraphics[width=0.85\linewidth]{figures/architecture.png}
\caption{\todo{Architecture / pipeline diagram.}}
\label{fig:arch}
\end{figure}

\section{Experimental Setup}
\label{sec:setup}

\subsection{Dataset}
\todo{Exact subset used (count, selection method, random seed), license, and any
preprocessing.}

\subsection{Baseline(s)}
\todo{Exact baseline(s) you compare against, and confirmation that your reproduction
matches published numbers (or an explanation of any gap).}

\subsection{Metrics \& Protocol}
\todo{Every metric used, computed how, and the exact comparison protocol (e.g.,
bpp-matching procedure for rate--distortion comparisons, train/val/test split, number
of seeds).}

\section{Results}
\label{sec:results}

\begin{table}[h]
\centering
\caption{Main quantitative comparison against the baseline.}
\label{tab:main}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Method} & \textbf{Metric 1} & \textbf{Metric 2} & \textbf{Metric 3} & \textbf{Compute/Runtime} \\
\midrule
Baseline      & \todo{val} & \todo{val} & \todo{val} & \todo{val} \\
Ours          & \todo{val} & \todo{val} & \todo{val} & \todo{val} \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[h]
\centering
% \includegraphics[width=0.48\linewidth]{figures/qual1.png}
% \includegraphics[width=0.48\linewidth]{figures/qual2.png}
\caption{\todo{Qualitative comparison. Describe exactly what is shown, including any
failure cases -- do not caption a figure with what you expected it to show.}}
\label{fig:qual}
\end{figure}

\subsection{Ablation Studies}
\label{sec:ablation}
\todo{At least one ablation isolating the effect of your main design choice. Table or
plot required, not just prose.}

\section{Discussion \& Limitations}
\label{sec:discussion}
\todo{What worked, what didn't, and why (to the best of your understanding). Be
explicit about limitations -- scope, dataset size, compute budget, anything that would
change if this were scaled up. A paper that honestly analyzes its limitations is
graded higher than one that overclaims.}

\section{Conclusion}
\label{sec:conclusion}
\todo{2--4 sentences: what you built, what you found, what you would do next with more
time/compute.}

\section*{Individual Contributions}
\label{sec:contributions}
\todo{One sentence per student stating their primary contributions, consistent with
the peer-evaluation form submitted separately.}

\bibliographystyle{unsrtnat}
\bibliography{references}

\appendix
\section{Additional Results}
\label{sec:appendix}
\todo{Optional: extra qualitative examples, full hyperparameter tables, or results
that did not fit in the main body.}

\end{document}
```

---

## 7. Milestone Rubrics

These three rubrics apply **uniformly across all 30 catalog projects** (and any approved
custom topic) — no per-project rubric is needed, since "did you reproduce a credible
baseline, implement a scoped extension, and benchmark it fairly" is the same question
regardless of which image-processing problem a team picked.

**Weighted project grade** = 0.15 × M1 + 0.35 × M2 + 0.50 × M3 (see Section 4).

### 7.1 Milestone 1 Rubric — Proposal & Literature Review (100 pts)

| Criterion | Points | What earns full marks |
|---|---|---|
| Problem motivation & scope clarity | 15 | Clear statement of the problem, why it matters, and explicit connection to a course topic |
| Literature review | 25 | ≥4 references correctly summarized, ≥2 from 2023–2026, understanding demonstrated (not just citation-dropping) |
| Dataset & baseline identification | 15 | Public, correctly licensed, feasible dataset; baseline is a real, citable method |
| Feasibility & compute plan | 15 | Realistic for the stated compute tier and 9-week window; subsampling plan stated |
| Task breakdown & team role clarity | 10 | All 3 milestones' tasks assigned per student, consistent with Section 5 |
| Risk assessment & MVP/stretch split | 10 | At least 2 real risks with mitigations; MVP is genuinely minimal and still valid |
| Writing & LaTeX quality | 10 | Uses the Section 6.1 template correctly, proper `\citep`/`\citet`, clean tables/figures |

### 7.2 Milestone 2 Rubric — Implementation & Preliminary Results (100 pts)

| Criterion | Points | What earns full marks |
|---|---|---|
| Baseline reproduction correctness | 20 | Baseline runs end-to-end and its numbers are plausible/consistent with the literature |
| Core method implementation | 25 | The team's own method/extension runs end-to-end, even if not fully tuned |
| Preliminary quantitative results | 20 | At least one real comparison table/curve, correct metrics, computed correctly |
| Experimental rigor | 15 | Clean train/val/test split, fixed seed, `requirements.txt`/`environment.yml` present |
| Progress vs. M1 plan | 10 | MVP status stated explicitly; deviations explained, not hidden |
| Report quality | 10 | Uses the Section 6.2 template; figures/tables captioned accurately (not aspirationally) |

### 7.3 Milestone 3 Rubric — Final Report, Results & Presentation (100 pts)

| Criterion | Points | What earns full marks |
|---|---|---|
| Technical correctness & completeness | 20 | Method fully and precisely described; reproducible from the report alone |
| Quantitative comparison vs. baseline/SOTA | 20 | Fair, documented protocol; results support the claims made |
| Ablation studies / analysis depth | 15 | ≥1 ablation isolating the main design choice's effect |
| Reproducibility | 10 | Public code repo, README, fixed seeds, environment file, documented hyperparameters |
| Final report writing quality | 15 | Uses the Section 6.3 template; honest limitations section; correct citations |
| Presentation / demo | 10 | Clear 8–10 min presentation + live Q&A; working demo if applicable |
| Individual contribution clarity | 10 | Consistent with the peer-evaluation form (Section 9.5); every student can answer questions on the whole pipeline |

---

## 8. Publication Bonus Track

Some teams will produce work that is genuinely competitive beyond the course. This track
rewards that **without** lowering the bar for the standard rubrics above — bonus points are
additive, never a substitute for meeting the core milestone criteria.

### 8.1 Bonus Scorecard

| # | Criterion | Met? |
|---|---|---|
| 1 | **Novelty beyond reproduction** — a new method, combination, or analysis, not just a re-run of existing code | ☐ |
| 2 | **Matches or improves a stated recent SOTA baseline** on a public benchmark under a fair, documented protocol — *or* delivers a rigorous, well-analyzed negative/surprising result | ☐ |
| 3 | **At least one ablation study** isolating the contribution's effect | ☐ |
| 4 | **Full reproducibility** — public code repo, README, fixed seeds, environment file, documented hyperparameters | ☐ |
| 5 | **Venue-standard writing** — honest related-work positioning, explicit limitations, correct citations throughout | ☐ |
| 6 | **Clean academic-integrity record** — proper attribution of pretrained/frozen components, AI-tool usage disclosed per Section 9.1 | ☐ |

**Grading mechanics:**
- **≥5 of 6 criteria met:** up to **+5 percentage points** added to the project's final
  weighted score (capped at 100).
- **All 6 criteria met, and the instructor judges the work genuinely submission-ready:**
  the team is offered co-mentored submission to one of the venues below, with the instructor
  as corresponding/senior author per standard supervisory-authorship norms. This is offered,
  not automatic — **pre-approval from the instructor/TA is required before the M3 deadline**,
  so there are no last-minute surprises about whether a claim of novelty holds up.

### 8.2 Realistic Venues for gradudate Team Work

Calibrate ambitions honestly: a 9-week, 3-person gradudate project is very unlikely to
land in a top-tier main conference track (CVPR/ICCV/NeurIPS main), but several **realistic
and genuinely prestigious** paths exist:

- **Challenge leaderboards (the most concrete "beat SOTA" path).** [CLIC](https://clic2025.compression.cc/) (image/video
  compression), and the CVPR workshop challenges [NTIRE](https://cvlai.net/ntire/) and AIM
  (restoration/super-resolution/efficiency) explicitly accept external submissions, publish
  public leaderboards, and are realistic for strong student work — several catalog projects
  (1, 2, 16, 30) are deliberately scoped to make a CLIC/NTIRE-style submission plausible.
- **IEEE workshop/mid-tier venues directly relevant to image coding:** ICIP, PCS (Picture
  Coding Symposium), VCIP — all realistic targets for a focused compression project.
- **Workshop tracks** at CVPR/ICCV/ECCV/WACV (separate from the brutally competitive main
  tracks, and where a lot of genuinely good applied work is published).
- **BMVC or WACV main track** — more competitive, but not out of reach for an unusually
  strong result with a real ablation and honest comparison.
- **Student/gradudate research venues:** your institution's gradudate research
  symposium, NCUR, regional IEEE student paper competitions.
- **arXiv preprint** — always available, costs nothing, and is the right first step for any
  project meeting criteria 1–5 even if it doesn't clear the bar for a venue submission.

---

## 9. Things You May Have Missed

A few items that the project brief doesn't cover but that will come up in practice.

### 9.1 AI-Tool Usage & Citation Policy

The brief explicitly invites "embedding AI/ML tools as algorithms" — that needs a companion
policy on *how* AI tools may be used, since there's a real difference between the two:

- **Expected and encouraged:** using pretrained/frozen foundation models (SAM2, pretrained
  diffusion priors, pretrained classifiers) as building blocks, exactly as the literature
  does. Cite the model/checkpoint used.
- **Allowed with disclosure:** using an LLM (Claude, ChatGPT, Copilot, etc.) for code
  scaffolding, debugging help, or editing prose. Add a one-line disclosure in the report
  (e.g., "Code scaffolding for the training loop was AI-assisted; all results were run and
  verified by the team").
- **Not allowed:** having an AI tool generate reported results, write the technical content
  of the report without the team understanding it, or fabricate/embellish numbers. This
  extends the existing course policy that "plagiarism detection tools will be applied to all
  code submissions" to project code and reports.
- **[INSTRUCTOR: confirm this matches your institution's AI-use policy before distributing —
  adjust the disclosure threshold if your policy is stricter or more permissive.]**


### 9.2 Dataset Licensing & Ethics

"Publicly available" is not the same as "unrestricted." Before a team commits to a dataset:

- Read the dataset's actual license (research-only vs. redistribution-permitted vs.
  commercial-use-permitted) and cite it correctly in the report.
- **Face datasets** (CelebA, FFHQ, etc.): research-use-only in essentially all cases; no
  re-identification claims; no display of identifiable individuals in a public demo beyond
  what the license permits.
- **Medical data** (TCIA, NIH ChestX-ray14, ISIC, etc.): even de-identified public datasets
  carry specific usage terms (`project-catalog.md` Projects 10, 21, 25 flag this explicitly)
  — no attempt to re-identify subjects, and results should not be framed as clinically
  validated.
- **Generated-image datasets** (for Project 29): document exact provenance, since licensing
  for AI-generated content varies by the model that produced it.

### 9.3 Version Control & Reproducibility Infrastructure

Required from Week 1, not just at M3:

- One GitHub/GitLab **repository per team**, created before M1. Commit history is partial
  evidence of individual contribution if a peer-evaluation discrepancy needs resolving.
- `README.md` with setup instructions, `requirements.txt` or `environment.yml`, and a fixed
  random seed for the headline result.
- This is graded explicitly under M2 "Experimental rigor" and M3 "Reproducibility" (Section 7).



### 9.4 Logistics Checklist

- **Team formation + topic signup:** by end of Week 2 (Section 2).

- **Academic integrity:** the course's existing policy ("strictly individual" for exams,
  plagiarism detection on code) extends to project code and reports; collaboration *within*
  a team is expected, collaboration *across* teams on the same deliverable is not.
- **Final deliverable packaging (due at M3):** final report Latex and PDF (compiled from the Section
  6.3 template), public code repository link, `README.md`, a live or ≤5-minute recorded demo,
  and the presentation slide deck.
- **Final presentation format:** 8–10 minute presentation + 5 minutes of unscripted Q&A,
  [POSTER SESSION OR SCHEDULED TALKS — INSTRUCTOR TO CONFIRM FORMAT].

---

## 10. Dataset Quick-Reference Table

A consolidated list of every public dataset referenced across the 30 catalog projects, for
convenience. Project numbers refer to `project-catalog.md`.

| Dataset | Approx. size | Used in projects | Link |
|---|---|---|---|
| Kodak | 24 images | 1, 2, 3, 4, 5, 7, 15, 30 | [link](http://r0k.us/graphics/kodak/) |
| Tecnick / TESTIMAGES | ~100 images | 1, 9, 30 | [link](https://testimages.org/) |
| CLIC (validation sets) | varies, small | 4, 5, 30 | [link](https://clic2025.compression.cc/tasks/) |
| DIV2K | 900 images, ~3.5 GB | 1, 2, 4, 6, 7, 15, 16, 18 | [link](https://data.vision.ee.ethz.ch/cvl/DIV2K/) |
| BSD500 | 500 images | 20, 22 | [link](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) |
| Set5/Set14/BSD100/Urban100/Manga109 | small (5–109 images each) | 16 | [mirror](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md) |
| LIVE1 / Classic5 | small | 6, 18 | mirrored via [BasicSR](https://github.com/XPixelGroup/BasicSR) |
| SIDD (Small/Medium) | few GB | 17 | [link](https://www.eecs.yorku.ca/~kamel/sidd/) |
| DND benchmark | 50 images | 17 | [link](https://noise.visinf.tu-darmstadt.de/) |
| GoPro deblurring | ~3 GB (subsample) | 19 | standard public mirror (search "GoPro Large deblurring dataset") |
| Set11 (CS benchmark) | 11 images | 20 | bundled with [ISTA-Net-PyTorch](https://github.com/jianzhangcs/ISTA-Net-PyTorch) |
| COCO (val2017) | ~1 GB (val split) | 8, 12, 13, 23, 24, 29 | [link](https://cocodataset.org/) |
| UC Merced Land Use | 2,100 images, ~317 MB | 9 | [link](http://weegee.vision.ucmerced.edu/datasets/landuse.html) |
| NWPU-RESISC45 | 31,500 images | 9 (optional, subsample) | search "NWPU-RESISC45 dataset" (official page / Kaggle mirror) |
| TCIA (The Cancer Imaging Archive) | varies | 10 | [link](https://www.cancerimagingarchive.net/) |
| NIH ChestX-ray14 (sample) | subsample a few hundred–thousand | 10, 21, 25 | [link](https://www.kaggle.com/datasets/nih-chest-xrays/data) |
| UVG | 7 sequences | 3, 11 | [link](http://ultravideo.fi/#testsequences) |
| HPatches | moderate, public | 26 | [link](https://github.com/hpatches/hpatches-dataset) |
| Places2 | large — **use a 2,000–5,000 image subset** | 24, 27 | [link](http://places2.csail.mit.edu/) |
| ISIC skin-lesion archive | varies, subsample | 22 | [link](https://www.isic-archive.com/) |
| KADID-10k | 10,125 images + MOS | 28 | [link](http://database.mmsp-kn.de/kadid-10k-database.html) |
| TID2013 | 3,000 images + MOS | 28 | [link](http://www.ponomarenko.info/tid2013.htm) |
| CIFAR-10/100 | small, standard | 13 | standard public mirror (`torchvision.datasets`) |

---

*End of instructions. See `project-catalog.md` for the 30 project entries. Questions or
scope changes to any catalog entry should be raised with the instructor/TA before the
Milestone 1 deadline. *
