#!/usr/bin/env python3
"""Missing analyses for surprise-EEG benchmark project.

1. Simulation-based power analysis
2. Time-frequency feature extraction (theta/delta power)
3. Cross-validated out-of-sample prediction (leave-one-subject-out)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats, signal
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_AIM1 = PROJECT_DIR / "results" / "aim1"
RESULTS_AIM2 = PROJECT_DIR / "results" / "aim2"


# ============================================================================
# 1. SIMULATION-BASED POWER ANALYSIS
# ============================================================================

def run_power_analysis(n_subjects=39, trials_per_subject=900,
                       effect_sizes=(0.10, 0.15, 0.20, 0.25),
                       n_simulations=1000, alpha=0.05, seed=42):
    """Simulation-based power analysis for surprise-ERP correlations.

    Uses effect sizes from Mars et al. (2008): r ~ 0.15-0.25 for
    surprise-ERP correlations at single-trial level.

    Approach: for each simulation, generate data for N subjects with ~T trials
    each, fit OLS per subject, then do a one-sample t-test on the subject-level
    betas. This mirrors the two-stage summary statistics approach.
    """
    print("=" * 60)
    print("1. SIMULATION-BASED POWER ANALYSIS")
    print("=" * 60)

    rng = np.random.RandomState(seed)
    total_obs = n_subjects * trials_per_subject
    print(f"  N = {n_subjects} subjects, ~{trials_per_subject} trials/subject")
    print(f"  Total observations: {total_obs}")
    print(f"  Simulations per effect size: {n_simulations}")
    print(f"  Alpha: {alpha}")
    print()

    results = {
        'parameters': {
            'n_subjects': n_subjects,
            'trials_per_subject': trials_per_subject,
            'total_observations': total_obs,
            'n_simulations': n_simulations,
            'alpha': alpha,
        },
        'power_results': {}
    }

    # For each effect size r, compute the corresponding beta
    # In a simple regression y = beta*x + noise, with x ~ N(0,1) and noise ~ N(0, sigma^2),
    # r = beta / sqrt(beta^2 + sigma^2). Setting sigma=1:
    # beta = r / sqrt(1 - r^2)

    for r_target in effect_sizes:
        beta_surprise = r_target / np.sqrt(1 - r_target**2)
        beta_stimulus = 0.5  # Fixed stimulus effect
        noise_sd = 1.0

        print(f"  Effect size r = {r_target:.2f} (beta_surprise = {beta_surprise:.4f})...")

        sig_count = 0
        t0 = time.time()

        for sim_i in range(n_simulations):
            subject_betas = np.zeros(n_subjects)

            for subj in range(n_subjects):
                # Vary trials per subject slightly (like real data)
                n_trials = rng.randint(
                    int(trials_per_subject * 0.8),
                    int(trials_per_subject * 1.2)
                )

                # Generate stimulus (binary) and surprise (continuous)
                stimulus = rng.binomial(1, 0.2, size=n_trials).astype(float)
                surprise = rng.randn(n_trials)

                # Add subject-level random intercept and slope
                subj_intercept = rng.randn() * 0.3
                subj_slope_var = rng.randn() * (beta_surprise * 0.3)

                # Generate EEG amplitude
                noise = rng.randn(n_trials) * noise_sd
                eeg = (subj_intercept
                       + beta_stimulus * stimulus
                       + (beta_surprise + subj_slope_var) * surprise
                       + noise)

                # Fit OLS for this subject: eeg ~ stimulus + surprise
                X = np.column_stack([np.ones(n_trials), stimulus, surprise])
                coeffs, _, _, _ = np.linalg.lstsq(X, eeg, rcond=None)
                subject_betas[subj] = coeffs[2]  # surprise beta

            # One-sample t-test on subject-level betas
            t_stat, p_val = stats.ttest_1samp(subject_betas, 0)
            if p_val < alpha:
                sig_count += 1

        power = sig_count / n_simulations
        elapsed = time.time() - t0

        results['power_results'][f'r_{r_target:.2f}'] = {
            'r': r_target,
            'beta_surprise': float(beta_surprise),
            'power': power,
            'n_significant': sig_count,
            'n_simulations': n_simulations,
            'elapsed_seconds': round(elapsed, 1),
        }

        print(f"    Power = {power*100:.1f}% ({sig_count}/{n_simulations} significant)")
        print(f"    Time: {elapsed:.1f}s")

    # Summary
    print()
    print("  SUMMARY:")
    for key, res in results['power_results'].items():
        print(f"    r = {res['r']:.2f}: {res['power']*100:.1f}% power")

    # Save
    RESULTS_AIM1.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_AIM1 / "power_analysis.json"
    with open(str(output_path), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {output_path}")

    return results


# ============================================================================
# 2. TIME-FREQUENCY FEATURE EXTRACTION
# ============================================================================

def extract_tf_features_bandpass(epochs_data, sfreq, times):
    """Extract theta and delta power using bandpass filter + Hilbert envelope.

    This is much faster than Morlet wavelets.

    Parameters
    ----------
    epochs_data : ndarray, shape (n_epochs, n_channels, n_times)
    sfreq : float
    times : ndarray

    Returns
    -------
    theta_power : ndarray, shape (n_epochs,)
    delta_power : ndarray, shape (n_epochs,)
    """
    n_epochs, n_channels, n_times = epochs_data.shape

    # Post-stimulus window mask (0-500ms)
    post_mask = (times >= 0.0) & (times <= 0.5)

    # Average across channels first (faster)
    mean_data = np.mean(epochs_data, axis=1)  # (n_epochs, n_times)

    # Theta band (4-8 Hz)
    nyq = sfreq / 2.0
    theta_low, theta_high = 4.0 / nyq, min(8.0 / nyq, 0.99)
    try:
        b_theta, a_theta = signal.butter(3, [theta_low, theta_high], btype='band')
        theta_filtered = signal.filtfilt(b_theta, a_theta, mean_data, axis=1)
        theta_envelope = np.abs(signal.hilbert(theta_filtered, axis=1))
        theta_power = np.mean(theta_envelope[:, post_mask] ** 2, axis=1)
    except Exception as e:
        print(f"    WARNING: Theta extraction failed: {e}")
        theta_power = np.full(n_epochs, np.nan)

    # Delta band (1-4 Hz)
    delta_low, delta_high = 1.0 / nyq, min(4.0 / nyq, 0.99)
    try:
        b_delta, a_delta = signal.butter(3, [delta_low, delta_high], btype='band')
        delta_filtered = signal.filtfilt(b_delta, a_delta, mean_data, axis=1)
        delta_envelope = np.abs(signal.hilbert(delta_filtered, axis=1))
        delta_power = np.mean(delta_envelope[:, post_mask] ** 2, axis=1)
    except Exception as e:
        print(f"    WARNING: Delta extraction failed: {e}")
        delta_power = np.full(n_epochs, np.nan)

    return theta_power, delta_power


def run_time_frequency_extraction(task="MMN", max_subjects=None):
    """Extract theta and delta power from preprocessed epoch files.

    Parameters
    ----------
    task : str
        "MMN" or "P3"
    max_subjects : int or None
        If set, only process this many subjects (for testing).
    """
    import mne

    print(f"\n{'=' * 60}")
    print(f"2. TIME-FREQUENCY FEATURE EXTRACTION (task={task})")
    print("=" * 60)

    epo_files = sorted(PROCESSED_DIR.glob(f"*_task-{task}_epo.fif"))
    print(f"  Found {len(epo_files)} preprocessed epoch files")

    if max_subjects is not None:
        epo_files = epo_files[:max_subjects]
        print(f"  Processing first {max_subjects} subjects (test mode)")

    all_rows = []
    t0 = time.time()

    for i, epo_file in enumerate(epo_files):
        sub_id = epo_file.name.split('_task')[0]
        print(f"  [{i+1}/{len(epo_files)}] {sub_id}...", end=" ", flush=True)

        try:
            epochs = mne.read_epochs(str(epo_file), preload=True, verbose=False)
            data = epochs.get_data()
            sfreq = epochs.info['sfreq']
            times_arr = epochs.times

            theta_power, delta_power = extract_tf_features_bandpass(
                data, sfreq, times_arr
            )

            n_epochs = len(epochs)
            for trial_idx in range(n_epochs):
                all_rows.append({
                    'subject': sub_id,
                    'trial': trial_idx,
                    'theta_power': float(theta_power[trial_idx]),
                    'delta_power': float(delta_power[trial_idx]),
                })

            print(f"{n_epochs} trials, theta={np.nanmean(theta_power):.2e}, "
                  f"delta={np.nanmean(delta_power):.2e}")

            # Free memory
            del epochs, data
        except Exception as e:
            print(f"FAILED: {e}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    if not all_rows:
        print("  No data extracted!")
        return None

    tf_df = pd.DataFrame(all_rows)
    RESULTS_AIM2.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_AIM2 / f"tf_features_{task}.csv"
    tf_df.to_csv(str(output_path), index=False)
    print(f"  Saved {len(tf_df)} rows to {output_path}")
    print(f"  Subjects: {tf_df['subject'].nunique()}")
    print(f"  Theta power: mean={tf_df['theta_power'].mean():.2e}, "
          f"std={tf_df['theta_power'].std():.2e}")
    print(f"  Delta power: mean={tf_df['delta_power'].mean():.2e}, "
          f"std={tf_df['delta_power'].std():.2e}")

    return tf_df


# ============================================================================
# 3. CROSS-VALIDATED OUT-OF-SAMPLE PREDICTION
# ============================================================================

def run_cross_validated_prediction(task="MMN"):
    """Leave-one-subject-out cross-validation for surprise model comparison.

    For each surprise model:
    - Train on N-1 subjects
    - Predict ERP amplitude on held-out subject
    - Compute MSE per fold
    - Compare aggregate MSE across models
    """
    print(f"\n{'=' * 60}")
    print(f"3. CROSS-VALIDATED OUT-OF-SAMPLE PREDICTION (task={task})")
    print("=" * 60)

    # Load analysis dataframe
    if task == "MMN":
        dv_col = "mmn_amplitude"
    else:
        dv_col = "p3b_amplitude"

    analysis_path = RESULTS_AIM2 / f"analysis_df_{task}.csv"
    if not analysis_path.exists():
        print(f"  ERROR: {analysis_path} not found!")
        return None

    df = pd.read_csv(str(analysis_path))
    print(f"  Loaded {len(df)} rows, {df['subject'].nunique()} subjects")

    surprise_models = ['static_shannon', 'adaptive_shannon_w20',
                       'bayesian_surprise', 'changepoint_surprise']
    available_models = [m for m in surprise_models if m in df.columns]

    subjects = sorted(df['subject'].unique())
    n_subjects = len(subjects)
    print(f"  Surprise models: {available_models}")
    print(f"  DV: {dv_col}")
    print(f"  Leave-one-subject-out CV ({n_subjects} folds)")

    results = {
        'task': task,
        'dv': dv_col,
        'n_subjects': n_subjects,
        'n_observations': len(df),
        'models': {}
    }

    # Baseline model: stimulus only
    model_configs = {'baseline': ['stimulus']}
    for m in available_models:
        model_configs[m] = ['stimulus', m]

    for model_name, predictors in model_configs.items():
        print(f"\n  Model: {model_name} (predictors: {predictors})")

        fold_mses = []
        fold_r2s = []

        for subj in subjects:
            # Split
            train_mask = df['subject'] != subj
            test_mask = df['subject'] == subj

            train_df = df[train_mask]
            test_df = df[test_mask]

            # Prepare features
            X_train = train_df[predictors].values
            y_train = train_df[dv_col].values
            X_test = test_df[predictors].values
            y_test = test_df[dv_col].values

            # Standardize features using training stats
            train_mean = X_train.mean(axis=0)
            train_std = X_train.std(axis=0) + 1e-10
            X_train_z = (X_train - train_mean) / train_std
            X_test_z = (X_test - train_mean) / train_std

            # Add intercept
            X_train_z = np.column_stack([np.ones(len(X_train_z)), X_train_z])
            X_test_z = np.column_stack([np.ones(len(X_test_z)), X_test_z])

            # Fit OLS
            coeffs, _, _, _ = np.linalg.lstsq(X_train_z, y_train, rcond=None)
            y_pred = X_test_z @ coeffs

            # Compute MSE
            mse = np.mean((y_test - y_pred) ** 2)
            fold_mses.append(float(mse))

            # Compute R^2
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            fold_r2s.append(float(r2))

        mean_mse = np.mean(fold_mses)
        std_mse = np.std(fold_mses)
        mean_r2 = np.mean(fold_r2s)
        std_r2 = np.std(fold_r2s)

        results['models'][model_name] = {
            'predictors': predictors,
            'mean_mse': mean_mse,
            'std_mse': std_mse,
            'median_mse': float(np.median(fold_mses)),
            'mean_r2': mean_r2,
            'std_r2': std_r2,
            'fold_mses': fold_mses,
            'fold_r2s': fold_r2s,
        }

        print(f"    Mean MSE: {mean_mse:.4e} (+/- {std_mse:.4e})")
        print(f"    Mean R^2: {mean_r2:.4f} (+/- {std_r2:.4f})")

    # Compare models to baseline
    print(f"\n  MODEL COMPARISON (vs baseline):")
    baseline_mses = np.array(results['models']['baseline']['fold_mses'])
    baseline_mean = results['models']['baseline']['mean_mse']

    for model_name in available_models:
        model_mses = np.array(results['models'][model_name]['fold_mses'])
        model_mean = results['models'][model_name]['mean_mse']

        # Paired t-test on fold MSEs (lower is better)
        t_stat, p_val = stats.ttest_rel(model_mses, baseline_mses)
        mse_reduction = (baseline_mean - model_mean) / baseline_mean * 100

        results['models'][model_name]['vs_baseline'] = {
            'mse_reduction_pct': float(mse_reduction),
            't_stat': float(t_stat),
            'p_val': float(p_val),
        }

        direction = "better" if model_mean < baseline_mean else "worse"
        print(f"    {model_name}: MSE {direction} by {abs(mse_reduction):.3f}%, "
              f"t={t_stat:.3f}, p={p_val:.4f}")

    # Save
    RESULTS_AIM2.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_AIM2 / f"cross_validated_prediction_{task}.json"
    with open(str(output_path), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {output_path}")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("MISSING ANALYSES FOR SURPRISE-EEG BENCHMARK")
    print("=" * 60)
    print()

    # ---- 1. Power Analysis ----
    power_results = run_power_analysis(
        n_subjects=39,
        trials_per_subject=900,
        effect_sizes=(0.10, 0.15, 0.20, 0.25),
        n_simulations=1000,
        alpha=0.05,
        seed=42,
    )

    # ---- 2. Time-Frequency Features ----
    # Test with 5 subjects first
    print("\n\n  --- Testing TF extraction on 5 subjects first ---")
    tf_test = run_time_frequency_extraction(task="MMN", max_subjects=5)

    if tf_test is not None and len(tf_test) > 0:
        print("\n  --- Test passed! Running on all subjects ---")
        tf_mmn = run_time_frequency_extraction(task="MMN", max_subjects=None)
        tf_p3 = run_time_frequency_extraction(task="P3", max_subjects=None)
    else:
        print("\n  Test failed or no data. Skipping full TF extraction.")
        tf_mmn = None
        tf_p3 = None

    # ---- 3. Cross-Validated Prediction ----
    cv_mmn = run_cross_validated_prediction(task="MMN")
    cv_p3 = run_cross_validated_prediction(task="P3")

    # ---- Final Summary ----
    print("\n\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    # Power analysis summary
    print("\n1. POWER ANALYSIS:")
    if power_results:
        for key, res in power_results['power_results'].items():
            print(f"   With N=39 and ~900 trials/subject, "
                  f"we have {res['power']*100:.1f}% power to detect r={res['r']:.2f}")

    # TF features summary
    print("\n2. TIME-FREQUENCY FEATURES:")
    for task_name, tf_data in [("MMN", tf_mmn), ("P3", tf_p3)]:
        if tf_data is not None:
            print(f"   {task_name}: {len(tf_data)} trials, "
                  f"{tf_data['subject'].nunique()} subjects")
            print(f"     Theta: mean={tf_data['theta_power'].mean():.2e}")
            print(f"     Delta: mean={tf_data['delta_power'].mean():.2e}")
        else:
            print(f"   {task_name}: Not available")

    # CV prediction summary
    print("\n3. CROSS-VALIDATED PREDICTION:")
    for task_name, cv_data in [("MMN", cv_mmn), ("P3", cv_p3)]:
        if cv_data is not None:
            print(f"   {task_name}:")
            baseline_mse = cv_data['models']['baseline']['mean_mse']
            print(f"     Baseline MSE: {baseline_mse:.4e}")
            best_model = None
            best_mse = baseline_mse
            for m in ['static_shannon', 'adaptive_shannon_w20',
                      'bayesian_surprise', 'changepoint_surprise']:
                if m in cv_data['models']:
                    m_mse = cv_data['models'][m]['mean_mse']
                    vs = cv_data['models'][m].get('vs_baseline', {})
                    pct = vs.get('mse_reduction_pct', 0)
                    p = vs.get('p_val', 1)
                    print(f"     {m}: MSE={m_mse:.4e} ({pct:+.3f}%, p={p:.4f})")
                    if m_mse < best_mse:
                        best_mse = m_mse
                        best_model = m
            if best_model:
                print(f"     Best model: {best_model}")
            else:
                print(f"     No model beat baseline")

    print("\n\nDone! All results saved.")


if __name__ == "__main__":
    main()
