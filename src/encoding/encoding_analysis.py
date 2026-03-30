#!/usr/bin/env python3
"""Aim 2: Encoding analysis — how well do surprise models explain EEG responses?

Implements:
  - ERP feature extraction (MMN and P3b windows)
  - Time-frequency feature extraction
  - Linear mixed-effects models (surprise → EEG amplitude)
  - Model comparison (AIC/BIC, individual vs baseline)
  - Time-resolved regression
  - VIF computation
  - Cluster-based permutation tests
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REGRESSORS_DIR = PROJECT_DIR / "data" / "regressors"
RESULTS_DIR = PROJECT_DIR / "results" / "aim2"


# ROI channel definitions
MMN_ROI_NAMES = ['Fz', 'FCz', 'Cz', 'FC1', 'FC2', 'F1', 'F2']
P3B_ROI_NAMES = ['Pz', 'CPz', 'P1', 'P2', 'CP1', 'CP2']

# Time windows (in seconds relative to stimulus onset)
MMN_WINDOW = (0.100, 0.250)
P3B_WINDOW = (0.250, 0.500)


def extract_erp_features(epochs, task="MMN"):
    """Extract mean amplitude in MMN and P3b windows.

    Returns
    -------
    features : dict with keys 'mmn_amplitude', 'p3b_amplitude'
    """
    import mne

    # Case-insensitive channel matching
    ch_lower_map = {ch.lower(): ch for ch in epochs.ch_names}

    mmn_chs = [ch_lower_map[roi.lower()] for roi in MMN_ROI_NAMES
               if roi.lower() in ch_lower_map]
    p3b_chs = [ch_lower_map[roi.lower()] for roi in P3B_ROI_NAMES
               if roi.lower() in ch_lower_map]

    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    times = epochs.times

    features = {}

    # MMN window
    mmn_mask = (times >= MMN_WINDOW[0]) & (times <= MMN_WINDOW[1])
    if mmn_chs:
        ch_idx = [epochs.ch_names.index(ch) for ch in mmn_chs]
        mmn_data = data[:, ch_idx, :][:, :, mmn_mask]
        features['mmn_amplitude'] = np.mean(mmn_data, axis=(1, 2))  # (n_epochs,)
    else:
        # Use all frontocentral channels (first third as approximation)
        n_ch = len(epochs.ch_names)
        ch_idx = list(range(n_ch // 3))
        mmn_data = data[:, ch_idx, :][:, :, mmn_mask]
        features['mmn_amplitude'] = np.mean(mmn_data, axis=(1, 2))

    # P3b window
    p3b_mask = (times >= P3B_WINDOW[0]) & (times <= P3B_WINDOW[1])
    if p3b_chs:
        ch_idx = [epochs.ch_names.index(ch) for ch in p3b_chs]
        p3b_data = data[:, ch_idx, :][:, :, p3b_mask]
        features['p3b_amplitude'] = np.mean(p3b_data, axis=(1, 2))
    else:
        n_ch = len(epochs.ch_names)
        ch_idx = list(range(2 * n_ch // 3, n_ch))
        p3b_data = data[:, ch_idx, :][:, :, p3b_mask]
        features['p3b_amplitude'] = np.mean(p3b_data, axis=(1, 2))

    return features


def extract_time_frequency_features(epochs):
    """Extract theta and delta power using Morlet wavelets."""
    import mne

    data = epochs.get_data()
    sfreq = epochs.info['sfreq']
    n_epochs, n_channels, n_times = data.shape

    # Theta (4-8 Hz) and Delta (1-4 Hz) power
    freqs_theta = np.arange(4, 9, 1)
    freqs_delta = np.arange(1, 5, 1)
    n_cycles_theta = freqs_theta / 2.0
    n_cycles_delta = freqs_delta / 2.0

    # Use MNE's time-frequency decomposition
    from mne.time_frequency import tfr_array_morlet

    # Theta power
    tfr_theta = tfr_array_morlet(data, sfreq, freqs_theta,
                                  n_cycles=n_cycles_theta, output='power')
    # Average over frequencies and channels, keep time
    theta_power = np.mean(tfr_theta, axis=(1, 2))  # (n_epochs, n_times)

    # Average theta in post-stimulus window
    times = epochs.times
    post_mask = (times >= 0.0) & (times <= 0.5)
    theta_mean = np.mean(theta_power[:, post_mask], axis=1)  # (n_epochs,)

    # Delta power
    tfr_delta = tfr_array_morlet(data, sfreq, freqs_delta,
                                  n_cycles=n_cycles_delta, output='power')
    delta_power = np.mean(tfr_delta, axis=(1, 2))
    delta_mean = np.mean(delta_power[:, post_mask], axis=1)

    return {
        'theta_power': theta_mean,
        'delta_power': delta_mean,
    }


def compute_vif(df, columns):
    """Compute Variance Inflation Factor for each regressor."""
    from numpy.linalg import lstsq
    vif_values = {}
    for col in columns:
        others = [c for c in columns if c != col]
        if not others:
            vif_values[col] = 1.0
            continue
        X = df[others].values
        X = np.column_stack([np.ones(len(X)), X])
        y = df[col].values
        coeffs, residuals, _, _ = lstsq(X, y, rcond=None)
        y_pred = X @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        vif_values[col] = 1.0 / (1 - r_squared) if r_squared < 1 else np.inf
    return vif_values


def run_encoding_models(analysis_df, dv_column, model_names):
    """Run individual encoding models comparing each surprise model to baseline.

    PRIMARY analysis: each surprise model vs. stimulus-class-only baseline.

    Returns
    -------
    results : dict of model comparison results
    """
    import statsmodels.formula.api as smf

    results = {}

    # Baseline model: stimulus class only
    formula_base = f'{dv_column} ~ stimulus'
    try:
        model_base = smf.mixedlm(formula_base, analysis_df, groups=analysis_df['subject'])
        fit_base = model_base.fit(reml=False)
        results['baseline'] = {
            'aic': fit_base.aic,
            'bic': fit_base.bic,
            'llf': fit_base.llf,
            'converged': fit_base.converged,
        }
    except Exception as e:
        print(f"    Baseline model failed: {e}")
        return results

    # Individual surprise models
    for name in model_names:
        if name not in analysis_df.columns:
            continue
        formula = f'{dv_column} ~ stimulus + {name}'
        try:
            model = smf.mixedlm(formula, analysis_df, groups=analysis_df['subject'])
            fit = model.fit(reml=False)

            # Likelihood ratio test vs baseline
            lr_stat = 2 * (fit.llf - fit_base.llf)
            lr_pval = 1 - stats.chi2.cdf(lr_stat, df=1)

            results[name] = {
                'aic': fit.aic,
                'bic': fit.bic,
                'llf': fit.llf,
                'delta_aic': fit.aic - fit_base.aic,
                'delta_bic': fit.bic - fit_base.bic,
                'lr_stat': lr_stat,
                'lr_pval': lr_pval,
                'beta': fit.fe_params.get(name, np.nan),
                'beta_se': fit.bse_fe.get(name, np.nan) if hasattr(fit, 'bse_fe') else np.nan,
                'converged': fit.converged,
            }
        except Exception as e:
            print(f"    Model {name} failed: {e}")
            results[name] = {'error': str(e)}

    return results


def time_resolved_regression(epochs, regressors_df, surprise_columns, roi_channels=None):
    """Run regression at each time point for time-resolved encoding.

    Parameters
    ----------
    epochs : mne.Epochs
    regressors_df : pd.DataFrame with surprise values per trial
    surprise_columns : list of str
    roi_channels : list of str or None (use all)

    Returns
    -------
    betas : dict of {model_name: (n_times,) array of beta coefficients}
    pvals : dict of {model_name: (n_times,) array of p-values}
    times : array of time points
    """
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    times = epochs.times
    n_epochs, n_channels, n_times = data.shape

    if roi_channels:
        ch_idx = [epochs.ch_names.index(ch) for ch in roi_channels
                  if ch in epochs.ch_names]
        if ch_idx:
            data = data[:, ch_idx, :]

    # Average over ROI channels
    roi_data = np.mean(data, axis=1)  # (n_epochs, n_times)

    betas = {}
    pvals = {}

    for col in surprise_columns:
        if col not in regressors_df.columns:
            continue
        x = regressors_df[col].values
        if len(x) != n_epochs:
            print(f"    WARNING: regressor length mismatch for {col}: "
                  f"{len(x)} vs {n_epochs} epochs")
            continue

        # Standardize regressor
        x_z = (x - np.mean(x)) / (np.std(x) + 1e-10)

        beta_t = np.zeros(n_times)
        pval_t = np.ones(n_times)

        for t_idx in range(n_times):
            y = roi_data[:, t_idx]
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_z, y)
            beta_t[t_idx] = slope
            pval_t[t_idx] = p_value

        betas[col] = beta_t
        pvals[col] = pval_t

    return betas, pvals, times


def cluster_permutation_test(betas, pvals, times, n_permutations=1000,
                              cluster_threshold=0.05, seed=42):
    """Simple cluster-based permutation test for time-resolved betas.

    Tests whether clusters of significant time points exceed chance.
    """
    rng = np.random.RandomState(seed)

    # Find clusters in the observed data
    sig_mask = pvals < cluster_threshold
    clusters = []
    in_cluster = False
    start = 0

    for t in range(len(times)):
        if sig_mask[t] and not in_cluster:
            start = t
            in_cluster = True
        elif not sig_mask[t] and in_cluster:
            clusters.append((start, t))
            in_cluster = False
    if in_cluster:
        clusters.append((start, len(times)))

    if not clusters:
        return [], []

    # Cluster mass = sum of |beta| within cluster
    observed_masses = []
    for s, e in clusters:
        mass = np.sum(np.abs(betas[s:e]))
        observed_masses.append(mass)

    # Permutation distribution: sign-flip test
    max_cluster_masses = []
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(betas))
        perm_betas = betas * signs
        # Re-run simple t-test for permuted data
        perm_sig = np.abs(perm_betas) > np.percentile(np.abs(perm_betas), 95)

        # Find clusters in permuted data
        perm_clusters = []
        in_c = False
        for t in range(len(times)):
            if perm_sig[t] and not in_c:
                s_p = t
                in_c = True
            elif not perm_sig[t] and in_c:
                perm_clusters.append((s_p, t))
                in_c = False
        if in_c:
            perm_clusters.append((s_p, len(times)))

        if perm_clusters:
            max_mass = max(np.sum(np.abs(perm_betas[s:e])) for s, e in perm_clusters)
        else:
            max_mass = 0
        max_cluster_masses.append(max_mass)

    max_cluster_masses = np.array(max_cluster_masses)

    # P-values for observed clusters
    cluster_pvals = []
    for mass in observed_masses:
        p = np.mean(max_cluster_masses >= mass)
        cluster_pvals.append(p)

    return list(zip(clusters, observed_masses, cluster_pvals)), max_cluster_masses


def run_full_encoding_analysis(task="MMN"):
    """Run the complete Aim 2 encoding analysis."""
    import mne

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REGRESSORS_DIR.mkdir(parents=True, exist_ok=True)

    # Find processed subjects
    epo_files = sorted(PROCESSED_DIR.glob(f"*_task-{task}_epo.fif"))
    print(f"Found {len(epo_files)} preprocessed subjects for task {task}")

    if not epo_files:
        print("No preprocessed data found. Run preprocessing first.")
        return

    # Collect all data for mixed-effects models
    all_rows = []
    all_betas_mmn = {}
    all_betas_p3b = {}
    surprise_cols = ['static_shannon', 'adaptive_shannon_w20',
                     'bayesian_surprise', 'changepoint_surprise']

    for epo_file in epo_files:
        sub_id = epo_file.name.split('_task')[0]
        print(f"\n  Processing {sub_id}...")

        # Load epochs
        epochs = mne.read_epochs(str(epo_file), preload=True, verbose=False)

        # Load regressors
        reg_file = REGRESSORS_DIR / f"{sub_id}_task-{task}_regressors.csv"
        if not reg_file.exists():
            print(f"    Regressors not found: {reg_file}")
            continue

        regressors = pd.read_csv(str(reg_file))

        # Ensure alignment
        n_epochs = len(epochs)
        if len(regressors) != n_epochs:
            print(f"    WARNING: regressor/epoch mismatch: {len(regressors)} vs {n_epochs}")
            min_len = min(len(regressors), n_epochs)
            regressors = regressors.iloc[:min_len]
            epochs = epochs[:min_len]

        # Extract ERP features
        erp_feats = extract_erp_features(epochs, task)

        # Build analysis dataframe
        for i in range(len(epochs)):
            row = {
                'subject': sub_id,
                'trial': i,
                'stimulus': int(regressors.iloc[i]['stimulus']),
                'mmn_amplitude': erp_feats['mmn_amplitude'][i],
                'p3b_amplitude': erp_feats['p3b_amplitude'][i],
            }
            for col in surprise_cols:
                if col in regressors.columns:
                    row[col] = regressors.iloc[i][col]
            all_rows.append(row)

        # Time-resolved regression for this subject
        mmn_chs = [ch for ch in epochs.ch_names
                    if ch.lower() in [c.lower() for c in MMN_ROI_NAMES]]
        p3b_chs = [ch for ch in epochs.ch_names
                    if ch.lower() in [c.lower() for c in P3B_ROI_NAMES]]

        betas_mmn, pvals_mmn, times = time_resolved_regression(
            epochs, regressors, surprise_cols, mmn_chs or None)
        betas_p3b, pvals_p3b, _ = time_resolved_regression(
            epochs, regressors, surprise_cols, p3b_chs or None)

        for col in surprise_cols:
            if col in betas_mmn:
                all_betas_mmn.setdefault(col, []).append(betas_mmn[col])
            if col in betas_p3b:
                all_betas_p3b.setdefault(col, []).append(betas_p3b[col])

    if not all_rows:
        print("No data collected!")
        return

    # Create combined dataframe
    analysis_df = pd.DataFrame(all_rows)
    analysis_df.to_csv(str(RESULTS_DIR / f"analysis_df_{task}.csv"), index=False)
    print(f"\nCombined analysis dataframe: {analysis_df.shape}")

    # Standardize surprise regressors
    for col in surprise_cols:
        if col in analysis_df.columns:
            analysis_df[col] = (analysis_df[col] - analysis_df[col].mean()) / \
                               (analysis_df[col].std() + 1e-10)

    # VIF analysis
    available_cols = [c for c in surprise_cols if c in analysis_df.columns]
    vif = compute_vif(analysis_df, available_cols)
    print(f"\nVIF values:")
    for col, val in vif.items():
        print(f"  {col}: {val:.2f}")

    vif_df = pd.DataFrame([vif])
    vif_df.to_csv(str(RESULTS_DIR / f"vif_{task}.csv"), index=False)

    # Correlation matrix
    corr = analysis_df[available_cols].corr()
    corr.to_csv(str(RESULTS_DIR / f"correlation_matrix_{task}.csv"))
    print(f"\nCorrelation matrix:")
    print(corr.round(3))

    # PRIMARY: Individual model encoding analyses
    print(f"\n{'='*60}")
    print("PRIMARY ANALYSIS: Individual surprise models vs baseline")
    print(f"{'='*60}")

    encoding_results = {}
    for dv in ['mmn_amplitude', 'p3b_amplitude']:
        print(f"\n  DV: {dv}")
        results = run_encoding_models(analysis_df, dv, available_cols)
        encoding_results[dv] = results

        for name, res in results.items():
            if 'error' in res:
                print(f"    {name}: ERROR - {res['error']}")
            elif name == 'baseline':
                print(f"    {name}: AIC={res['aic']:.1f}, BIC={res['bic']:.1f}")
            else:
                print(f"    {name}: ΔAIC={res.get('delta_aic', 'N/A'):.1f}, "
                      f"ΔBIC={res.get('delta_bic', 'N/A'):.1f}, "
                      f"LR p={res.get('lr_pval', 'N/A'):.4f}, "
                      f"β={res.get('beta', 'N/A'):.4f}")

    # Save encoding results
    with open(str(RESULTS_DIR / f"encoding_results_{task}.json"), 'w') as f:
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump(encoding_results, f, indent=2, default=convert)

    # Time-resolved analysis (averaged across subjects)
    print(f"\n{'='*60}")
    print("TIME-RESOLVED REGRESSION (group average)")
    print(f"{'='*60}")

    time_resolved_results = {}
    for roi_name, all_betas_dict in [('mmn_roi', all_betas_mmn),
                                      ('p3b_roi', all_betas_p3b)]:
        time_resolved_results[roi_name] = {}
        for col in available_cols:
            if col not in all_betas_dict:
                continue
            beta_matrix = np.array(all_betas_dict[col])  # (n_subjects, n_times)
            mean_beta = np.mean(beta_matrix, axis=0)
            se_beta = np.std(beta_matrix, axis=0) / np.sqrt(beta_matrix.shape[0])

            # One-sample t-test at each time point
            t_vals, p_vals = stats.ttest_1samp(beta_matrix, 0, axis=0)

            time_resolved_results[roi_name][col] = {
                'mean_beta': mean_beta.tolist(),
                'se_beta': se_beta.tolist(),
                't_values': t_vals.tolist(),
                'p_values': p_vals.tolist(),
                'times': times.tolist(),
                'n_subjects': beta_matrix.shape[0],
            }

            # Cluster permutation test
            clusters, _ = cluster_permutation_test(mean_beta, p_vals, times)
            sig_clusters = [(c, m, p) for c, m, p in clusters if p < 0.05]
            if sig_clusters:
                print(f"  {roi_name} - {col}: {len(sig_clusters)} significant clusters")
                for (s, e), mass, p in sig_clusters:
                    print(f"    {times[s]*1000:.0f}-{times[e-1]*1000:.0f} ms, "
                          f"mass={mass:.4f}, p={p:.3f}")

    # Save time-resolved results
    with open(str(RESULTS_DIR / f"time_resolved_{task}.json"), 'w') as f:
        json.dump(time_resolved_results, f, indent=2)

    print(f"\nAll Aim 2 results saved to {RESULTS_DIR}")
    return encoding_results, time_resolved_results


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "MMN"
    run_full_encoding_analysis(task)
