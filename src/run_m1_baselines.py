"""Runner for the M1 deliverable: full-reference baseline correlations vs DMOS.

Runs PSNR, SSIM and LPIPS on a content-based train/test split of KADID-10k
(split by reference image, fixed seed 42) and reports SROCC, PLCC and RMSE
against the human DMOS labels. Writes a summary table to
results/m1_baseline_correlations.csv and per-image scores to
results/m1_baseline_correlations_<subset>_raw.csv.

Usage:
    python src/run_m1_baselines.py --data-root /content/kadid10k/kadid10k \
        --out results/m1_baseline_correlations.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# Make the sibling modules importable whether this is run as a script or imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import MAX_PSNR_DB, LPIPSMetric, psnr, ssim  # noqa: E402
from eval_protocol import (                    # noqa: E402
    SEED,
    correlation_report,
    orient_higher_is_better,
    split_by_reference,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', required=True,
                    help='folder containing images/ and dmos.csv')
    ap.add_argument('--out', default='results/m1_baseline_correlations.csv')
    ap.add_argument('--test-frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--full', action='store_true',
                    help='also evaluate on all 10,125 images (slower)')
    args = ap.parse_args()

    # KADID-10k layout: images/ holds the PNGs, dmos.csv holds the human labels
    img_dir = os.path.join(args.data_root, 'images')
    df = pd.read_csv(os.path.join(args.data_root, 'dmos.csv'))

    # Split by reference (content) so no image content leaks across train/test
    train_refs, test_refs = split_by_reference(df['ref_img'], args.test_frac, args.seed)
    test_mask = df['ref_img'].isin(test_refs)
    print(f'refs: {len(train_refs)} train / {len(test_refs)} test '
          f'(content split, seed {args.seed})')
    print(f'images: {test_mask.sum()} test distorted / {len(df)} total')

    # By default only the held-out test split is reported; --full adds all images
    subsets = {'test_split': df[test_mask]}
    if args.full:
        subsets['all_images'] = df

    # LPIPS model is expensive to build; create once and reuse for every pair
    lpips_metric = LPIPSMetric()
    print('LPIPS device:', lpips_metric.device)

    summary_rows = []
    for subset_name, sub in subsets.items():
        print(f'--- {subset_name}: {len(sub)} images ---')
        psnr_vals, ssim_vals, lpips_vals = [], [], []
        for _, row in tqdm(sub.iterrows(), total=len(sub)):
            dp = os.path.join(img_dir, row['dist_img'])
            rp = os.path.join(img_dir, row['ref_img'])
            psnr_vals.append(psnr(dp, rp))
            ssim_vals.append(ssim(dp, rp))
            lpips_vals.append(lpips_metric(dp, rp))

        # Report how many pairs hit the PSNR ceiling (near-identical images)
        n_capped = sum(1 for v in psnr_vals if v >= MAX_PSNR_DB)
        if n_capped:
            print(f'PSNR hit the {MAX_PSNR_DB:.0f} dB ceiling for {n_capped} pair(s) '
                  f'(near-identical images)')

        mos = sub['dmos'].to_numpy()
        for name, vals, higher_better in (
            ('PSNR', psnr_vals, True),
            ('SSIM', ssim_vals, True),
            ('LPIPS', lpips_vals, False),
        ):
            # LPIPS is lower-better, so flip its sign before correlating
            scores = np.asarray(vals) if higher_better else orient_higher_is_better(vals)
            rep = correlation_report(scores, mos)
            summary_rows.append({'subset': subset_name, 'metric': name, **rep,
                                 'n': len(sub)})
            print(f'{name}: SROCC={rep["srocc"]:.4f}  PLCC={rep["plcc"]:.4f}  '
                  f'RMSE={rep["rmse"]:.4f}')

        # Keep the per-image scores for later error analysis / figures
        raw = sub[['dist_img', 'ref_img', 'dmos']].copy()
        raw['psnr'] = psnr_vals
        raw['ssim'] = ssim_vals
        raw['lpips'] = lpips_vals
        raw_path = args.out.replace('.csv', f'_{subset_name}_raw.csv')
        raw.to_csv(raw_path, index=False)
        print('saved raw scores:', raw_path)

    out_df = pd.DataFrame(summary_rows)
    # Make sure the destination folder exists before writing the summary
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print()
    print(out_df.to_string(index=False))
    print('saved:', args.out)


if __name__ == '__main__':
    main()
