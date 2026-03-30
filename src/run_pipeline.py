#!/usr/bin/env python3
"""Master pipeline: runs all analyses end-to-end.

Usage: python3 src/run_pipeline.py [--task MMN|P3] [--skip-preprocessing]
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import time

# Set up paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from preprocessing.preprocess import preprocess_all, find_subject_ids
from surprise.estimators import compute_all_surprise
from encoding.encoding_analysis import (
    run_full_encoding_analysis, extract_erp_features,
    extract_time_frequency_features
)
from decoding.decoding_analysis import run_full_decoding_analysis

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REGRESSORS_DIR = PROJECT_DIR / "data" / "regressors"
RESULTS_DIR = PROJECT_DIR / "results"

import mne
mne.set_log_level('WARNING')


def step1_preprocess(task="MMN"):
    """Step 1: Preprocess all subjects."""
    print("\n" + "=" * 70)
    print(f"STEP 1: PREPROCESSING — Task: {task}")
    print("=" * 70)
    qc = preprocess_all(task)
    return qc


def step2_compute_surprise(task="MMN"):
    """Step 2: Compute surprise regressors for all subjects."""
    print("\n" + "=" * 70)
    print(f"STEP 2: SURPRISE MODELS — Task: {task}")
    print("=" * 70)

    REGRESSORS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "aim1").mkdir(parents=True, exist_ok=True)

    # Find preprocessed subjects
    seq_files = sorted(PROCESSED_DIR.glob(f"*_task-{task}_sequence.npy"))
    print(f"Found {len(seq_files)} subjects with stimulus sequences")

    all_results = {}
    correlation_data = []

    for seq_file in seq_files:
        sub_id = seq_file.name.split('_task')[0]
        print(f"\n  {sub_id}: Computing surprise regressors...")

        sequence = np.load(str(seq_file))
        print(f"    Sequence: {len(sequence)} trials, "
              f"{sum(sequence)} deviants ({100*sum(sequence)/len(sequence):.1f}%)")

        # Compute all surprise models
        results = compute_all_surprise(
            sequence,
            window_sizes=[10, 20, 50],
            hazard_rates=[1/200, 1/50, 1/100, 1/500]
        )

        # Save regressor table
        reg_df = pd.DataFrame({
            'trial': results['trial'],
            'stimulus': results['stimulus'],
            'static_shannon': results['static_shannon'],
            'adaptive_shannon_w10': results['adaptive_shannon_w10'],
            'adaptive_shannon_w20': results['adaptive_shannon_w20'],
            'adaptive_shannon_w50': results['adaptive_shannon_w50'],
            'bayesian_surprise': results['bayesian_surprise'],
            'posterior_entropy': results['posterior_entropy'],
            'changepoint_surprise': results['changepoint_surprise'],
            'changepoint_prob': results['changepoint_prob'],
            'run_length_mean': results['run_length_mean'],
            'estimated_volatility': results['estimated_volatility'],
        })

        reg_file = REGRESSORS_DIR / f"{sub_id}_task-{task}_regressors.csv"
        reg_df.to_csv(str(reg_file), index=False)
        print(f"    Saved: {reg_file.name}")

        # Collect correlation data
        surprise_cols = ['static_shannon', 'adaptive_shannon_w20',
                         'bayesian_surprise', 'changepoint_surprise']
        corr = reg_df[surprise_cols].corr()
        correlation_data.append({
            'subject': sub_id,
            **{f'r_{c1}_{c2}': corr.loc[c1, c2]
               for c1 in surprise_cols for c2 in surprise_cols if c1 < c2}
        })

        all_results[sub_id] = {
            'n_trials': len(sequence),
            'n_deviants': int(sum(sequence)),
            'static_shannon_std': float(results['static_shannon'].std()),
            'adaptive_shannon_w20_std': float(results['adaptive_shannon_w20'].std()),
            'bayesian_surprise_std': float(results['bayesian_surprise'].std()),
            'changepoint_surprise_std': float(results['changepoint_surprise'].std()),
        }

    # Save summary
    summary_df = pd.DataFrame(correlation_data)
    summary_df.to_csv(str(RESULTS_DIR / "aim1" / f"regressor_correlations_{task}.csv"),
                      index=False)

    # Print group-level correlation summary
    print(f"\n{'='*50}")
    print("GROUP-LEVEL REGRESSOR CORRELATIONS (mean ± std)")
    print(f"{'='*50}")
    for col in summary_df.columns:
        if col.startswith('r_'):
            vals = summary_df[col]
            print(f"  {col}: {vals.mean():.3f} ± {vals.std():.3f}")

    # Save Aim 1 results
    with open(str(RESULTS_DIR / "aim1" / f"surprise_summary_{task}.json"), 'w') as f:
        json.dump(all_results, f, indent=2)

    return all_results


def step2b_extract_features(task="MMN"):
    """Step 2b: Extract ERP and time-frequency features."""
    print("\n" + "=" * 70)
    print(f"STEP 2b: FEATURE EXTRACTION — Task: {task}")
    print("=" * 70)

    epo_files = sorted(PROCESSED_DIR.glob(f"*_task-{task}_epo.fif"))
    print(f"Found {len(epo_files)} preprocessed subjects")

    all_tf_rows = []

    for epo_file in epo_files:
        sub_id = epo_file.name.split('_task')[0]
        print(f"\n  {sub_id}: Extracting time-frequency features...")

        try:
            epochs = mne.read_epochs(str(epo_file), preload=True, verbose=False)

            # Extract TF features
            tf_feats = extract_time_frequency_features(epochs)

            for i in range(len(epochs)):
                row = {'subject': sub_id, 'trial': i}
                for col, vals in tf_feats.items():
                    row[col] = vals[i]
                all_tf_rows.append(row)

        except Exception as e:
            print(f"    ERROR: {e}")

    if all_tf_rows:
        tf_df = pd.DataFrame(all_tf_rows)
        tf_file = RESULTS_DIR / "aim2" / f"tf_features_{task}.csv"
        (RESULTS_DIR / "aim2").mkdir(parents=True, exist_ok=True)
        tf_df.to_csv(str(tf_file), index=False)
        print(f"\n  TF features saved: {tf_file}")


def step3_encoding(task="MMN"):
    """Step 3: Encoding analysis (Aim 2)."""
    print("\n" + "=" * 70)
    print(f"STEP 3: ENCODING ANALYSIS — Task: {task}")
    print("=" * 70)
    return run_full_encoding_analysis(task)


def step4_decoding(task="MMN"):
    """Step 4: Decoding analysis (Aim 3)."""
    print("\n" + "=" * 70)
    print(f"STEP 4: DECODING ANALYSIS — Task: {task}")
    print("=" * 70)
    return run_full_decoding_analysis(task)


def main():
    parser = argparse.ArgumentParser(description="Run ERP CORE surprise benchmark")
    parser.add_argument('--task', default='MMN', choices=['MMN', 'P3'],
                        help='Task paradigm (default: MMN)')
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip preprocessing if already done')
    parser.add_argument('--skip-tf', action='store_true',
                        help='Skip time-frequency extraction')
    args = parser.parse_args()

    start_time = time.time()

    print(f"\n{'#' * 70}")
    print(f"SURPRISE EEG BENCHMARK PIPELINE")
    print(f"Task: {args.task}")
    print(f"{'#' * 70}")

    # Step 1: Preprocessing
    if not args.skip_preprocessing:
        step1_preprocess(args.task)
    else:
        print("\nSkipping preprocessing (--skip-preprocessing)")

    # Step 2: Surprise models
    step2_compute_surprise(args.task)

    # Step 2b: Time-frequency features
    if not args.skip_tf:
        step2b_extract_features(args.task)
    else:
        print("\nSkipping TF extraction (--skip-tf)")

    # Step 3: Encoding analysis
    step3_encoding(args.task)

    # Step 4: Decoding analysis
    step4_decoding(args.task)

    elapsed = time.time() - start_time
    print(f"\n{'#' * 70}")
    print(f"PIPELINE COMPLETE — {elapsed/60:.1f} minutes")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()
