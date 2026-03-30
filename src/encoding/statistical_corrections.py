#!/usr/bin/env python3
"""Statistical corrections for Aim 2 encoding analysis.

Addresses:
  a) Bonferroni-Holm correction for multiple comparisons
  b) Re-fit mixed-effects models with better convergence settings
  c) Additional effect sizes: Cohen's d, partial R-squared, beta CIs
  d) Bootstrap CIs for delta-AIC values
  e) Improved cluster permutation test reporting
  f) Saves corrected results to encoding_results_corrected_{task}.json
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
RESULTS_DIR = PROJECT_DIR / "results" / "aim2"

SURPRISE_COLS = [
    'static_shannon',
    'adaptive_shannon_w20',
    'bayesian_surprise',
    'changepoint_surprise',
]

DV_COLUMNS = {
    'MMN': 'mmn_amplitude',
    'P3':  'p3b_amplitude',
}

# Map task to the ERP window label used in reporting
WINDOW_LABELS = {
    'MMN': 'MMN window (100-250 ms)',
    'P3':  'P3b window (250-500 ms)',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert_for_json(obj):
    """Make numpy types JSON-serializable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def bonferroni_holm(pvals):
    """Apply Bonferroni-Holm step-down correction.

    Parameters
    ----------
    pvals : dict  {model_name: p_value}

    Returns
    -------
    corrected : dict  {model_name: corrected_p_value}
    """
    names = list(pvals.keys())
    raw = np.array([pvals[n] for n in names])
    m = len(raw)

    # Sort ascending
    order = np.argsort(raw)
    sorted_pvals = raw[order]

    corrected_sorted = np.zeros(m)
    for i in range(m):
        corrected_sorted[i] = sorted_pvals[i] * (m - i)

    # Enforce monotonicity (each corrected p must be >= the previous)
    for i in range(1, m):
        corrected_sorted[i] = max(corrected_sorted[i], corrected_sorted[i - 1])

    # Cap at 1.0
    corrected_sorted = np.minimum(corrected_sorted, 1.0)

    # Map back to original order
    corrected = {}
    for i, idx in enumerate(order):
        corrected[names[idx]] = float(corrected_sorted[i])

    return corrected


# ---------------------------------------------------------------------------
# (a) Multiple-comparison correction
# ---------------------------------------------------------------------------

def apply_corrections(original_results, task):
    """Apply Bonferroni-Holm correction to LR-test p-values.

    Correction is applied across the 4 model-vs-baseline comparisons
    within each DV (ERP window) separately.
    """
    dv = DV_COLUMNS[task]
    print(f"\n--- (a) Bonferroni-Holm correction for {task} ({dv}) ---")

    corrected_results = {}

    for dv_key in ['mmn_amplitude', 'p3b_amplitude']:
        if dv_key not in original_results:
            continue

        dv_data = original_results[dv_key]
        # Collect uncorrected p-values for the 4 surprise models
        uncorrected = {}
        for model in SURPRISE_COLS:
            if model in dv_data and 'lr_pval' in dv_data[model]:
                uncorrected[model] = dv_data[model]['lr_pval']

        if not uncorrected:
            corrected_results[dv_key] = {}
            continue

        corrected_p = bonferroni_holm(uncorrected)

        print(f"  DV = {dv_key}  (n_comparisons = {len(uncorrected)})")
        for model in SURPRISE_COLS:
            if model in uncorrected:
                sig_raw = '*' if uncorrected[model] < 0.05 else ''
                sig_cor = '*' if corrected_p[model] < 0.05 else ''
                print(f"    {model:30s}  p_raw={uncorrected[model]:.6f}{sig_raw:2s}  "
                      f"p_holm={corrected_p[model]:.6f}{sig_cor}")

        corrected_results[dv_key] = {
            'uncorrected': uncorrected,
            'holm_corrected': corrected_p,
        }

    return corrected_results


# ---------------------------------------------------------------------------
# (b) Re-fit mixed-effects models with better convergence
# ---------------------------------------------------------------------------

def refit_models(analysis_df, task):
    """Re-fit LME models with improved convergence settings.

    - Scale DV to micro-volt range (*1e6)
    - Use method='powell' with maxiter=500; fallback to 'lbfgs'
    """
    import statsmodels.formula.api as smf

    print(f"\n--- (b) Re-fitting mixed-effects models for {task} ---")

    # Work on a copy and scale amplitudes to micro-volts
    df = analysis_df.copy()
    for col in ['mmn_amplitude', 'p3b_amplitude']:
        if col in df.columns:
            df[col] = df[col] * 1e6  # V -> uV

    # Z-score surprise columns (they may already be z-scored in the saved CSV,
    # but the user said so -- we still re-standardize for safety)
    for col in SURPRISE_COLS:
        if col in df.columns:
            mu, sd = df[col].mean(), df[col].std()
            if sd > 0:
                df[col] = (df[col] - mu) / sd

    refit_results = {}

    for dv_key in ['mmn_amplitude', 'p3b_amplitude']:
        if dv_key not in df.columns:
            continue

        print(f"\n  DV = {dv_key} (scaled to uV)")
        refit_results[dv_key] = {}

        # --- Baseline model ---
        formula_base = f'{dv_key} ~ stimulus'
        fit_base = _fit_lme(smf, formula_base, df, label='baseline')
        if fit_base is None:
            print("    Baseline model failed to converge with any method -- skipping DV.")
            continue

        refit_results[dv_key]['baseline'] = {
            'aic': fit_base.aic,
            'bic': fit_base.bic,
            'llf': fit_base.llf,
            'converged': fit_base.converged,
            'method_used': getattr(fit_base, '_method_used', 'unknown'),
        }
        print(f"    baseline: AIC={fit_base.aic:.1f}  converged={fit_base.converged}")

        # --- Individual surprise models ---
        for model_name in SURPRISE_COLS:
            if model_name not in df.columns:
                continue

            formula = f'{dv_key} ~ stimulus + {model_name}'
            fit = _fit_lme(smf, formula, df, label=model_name)
            if fit is None:
                refit_results[dv_key][model_name] = {'error': 'all methods failed'}
                print(f"    {model_name}: FAILED")
                continue

            lr_stat = 2 * (fit.llf - fit_base.llf)
            lr_stat = max(lr_stat, 0.0)  # guard against numerical noise
            lr_pval = 1 - stats.chi2.cdf(lr_stat, df=1)

            beta = fit.fe_params.get(model_name, np.nan)
            beta_se = fit.bse_fe.get(model_name, np.nan) if hasattr(fit, 'bse_fe') else np.nan

            refit_results[dv_key][model_name] = {
                'aic': fit.aic,
                'bic': fit.bic,
                'llf': fit.llf,
                'delta_aic': fit.aic - fit_base.aic,
                'delta_bic': fit.bic - fit_base.bic,
                'lr_stat': lr_stat,
                'lr_pval': lr_pval,
                'beta': beta,
                'beta_se': beta_se,
                'converged': fit.converged,
                'method_used': getattr(fit, '_method_used', 'unknown'),
            }

            print(f"    {model_name:30s}: dAIC={fit.aic - fit_base.aic:+.2f}  "
                  f"LR p={lr_pval:.6f}  beta={beta:.4f}  "
                  f"converged={fit.converged}  method={getattr(fit, '_method_used', '?')}")

    return refit_results


def _fit_lme(smf, formula, df, label=''):
    """Try fitting LME with powell, then lbfgs, then default (lbfgs with loose tol)."""
    methods = [
        ('powell', {'maxiter': 500}),
        ('lbfgs', {'maxiter': 500}),
        ('lbfgs', {'maxiter': 2000}),
    ]
    for method, kwargs in methods:
        try:
            model = smf.mixedlm(formula, df, groups=df['subject'])
            fit = model.fit(reml=False, method=method, maxiter=kwargs['maxiter'])
            fit._method_used = method
            if fit.converged:
                return fit
            # Keep result even if not converged -- we may still use the last
            last_fit = fit
        except Exception:
            last_fit = None

    # Return best attempt even if not converged
    if last_fit is not None:
        return last_fit
    return None


# ---------------------------------------------------------------------------
# (c) Additional effect sizes
# ---------------------------------------------------------------------------

def compute_effect_sizes(analysis_df, refit_results, task):
    """Compute Cohen's d, partial R-squared, and 95% CIs for betas."""
    print(f"\n--- (c) Effect sizes for {task} ---")

    df = analysis_df.copy()
    # Scale to uV for consistency with refitted models
    for col in ['mmn_amplitude', 'p3b_amplitude']:
        if col in df.columns:
            df[col] = df[col] * 1e6

    effect_sizes = {}

    for dv_key in ['mmn_amplitude', 'p3b_amplitude']:
        if dv_key not in df.columns:
            continue

        print(f"\n  DV = {dv_key}")
        effect_sizes[dv_key] = {}

        # --- Cohen's d: deviant vs standard ---
        if 'stimulus' in df.columns:
            std_vals = df.loc[df['stimulus'] == 0, dv_key].values
            dev_vals = df.loc[df['stimulus'] == 1, dv_key].values
            if len(std_vals) > 1 and len(dev_vals) > 1:
                pooled_sd = np.sqrt(
                    ((len(std_vals) - 1) * np.var(std_vals, ddof=1) +
                     (len(dev_vals) - 1) * np.var(dev_vals, ddof=1)) /
                    (len(std_vals) + len(dev_vals) - 2)
                )
                cohens_d = (np.mean(dev_vals) - np.mean(std_vals)) / pooled_sd if pooled_sd > 0 else 0.0
                effect_sizes[dv_key]['cohens_d_deviant_vs_standard'] = cohens_d
                print(f"    Cohen's d (deviant - standard) = {cohens_d:.4f}")
            else:
                effect_sizes[dv_key]['cohens_d_deviant_vs_standard'] = None
                print(f"    Cohen's d: insufficient data for deviant/standard split")

        # --- Partial R-squared and beta CIs for each surprise model ---
        if dv_key not in refit_results:
            continue

        baseline_info = refit_results[dv_key].get('baseline', {})
        baseline_llf = baseline_info.get('llf', None)

        for model_name in SURPRISE_COLS:
            model_info = refit_results[dv_key].get(model_name, {})
            if 'error' in model_info:
                continue

            # Partial R-squared approximation:
            # R2_partial = 1 - exp(-LR_stat / N)   (McFadden-like)
            # Better: use the log-likelihood based pseudo R2
            model_llf = model_info.get('llf', None)
            n = len(df)

            if baseline_llf is not None and model_llf is not None:
                lr_stat = model_info.get('lr_stat', 0)
                # Cox-Snell-like partial R2
                partial_r2 = 1 - np.exp(-lr_stat / n)
            else:
                partial_r2 = None

            # 95% CI for beta
            beta = model_info.get('beta', np.nan)
            beta_se = model_info.get('beta_se', np.nan)
            if not (np.isnan(beta) or np.isnan(beta_se)):
                ci_lower = beta - 1.96 * beta_se
                ci_upper = beta + 1.96 * beta_se
            else:
                ci_lower = ci_upper = None

            effect_sizes[dv_key][model_name] = {
                'partial_r2': partial_r2,
                'beta': beta,
                'beta_se': beta_se,
                'beta_ci_95_lower': ci_lower,
                'beta_ci_95_upper': ci_upper,
            }

            print(f"    {model_name:30s}: partial_R2={partial_r2:.6f}  "
                  f"beta={beta:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")

    return effect_sizes


# ---------------------------------------------------------------------------
# (d) Bootstrap CIs for delta-AIC
# ---------------------------------------------------------------------------

def bootstrap_delta_aic(analysis_df, task, n_boot=1000, seed=42):
    """Compute bootstrap 95% CIs for group-level delta-AIC.

    Strategy: for each bootstrap resample of subjects, re-compute the
    sum of subject-level AIC contributions (log-likelihood contributions)
    to obtain a group delta-AIC.
    """
    import statsmodels.formula.api as smf

    print(f"\n--- (d) Bootstrap CIs for delta-AIC ({task}, n_boot={n_boot}) ---")

    rng = np.random.RandomState(seed)

    df = analysis_df.copy()
    for col in ['mmn_amplitude', 'p3b_amplitude']:
        if col in df.columns:
            df[col] = df[col] * 1e6

    for col in SURPRISE_COLS:
        if col in df.columns:
            mu, sd = df[col].mean(), df[col].std()
            if sd > 0:
                df[col] = (df[col] - mu) / sd

    subjects = df['subject'].unique()
    n_subjects = len(subjects)

    bootstrap_results = {}

    for dv_key in ['mmn_amplitude', 'p3b_amplitude']:
        if dv_key not in df.columns:
            continue

        print(f"\n  DV = {dv_key}")
        bootstrap_results[dv_key] = {}

        # Pre-compute subject-level log-likelihoods for baseline and each model
        subj_llf_base = {}
        subj_llf_models = {m: {} for m in SURPRISE_COLS}

        formula_base = f'{dv_key} ~ stimulus'

        for subj in subjects:
            subj_df = df[df['subject'] == subj]
            if len(subj_df) < 5:
                continue

            # Baseline: simple OLS per subject (we use OLS for subject-level
            # because single-subject LME is degenerate)
            try:
                import statsmodels.api as sm
                X_base = pd.get_dummies(subj_df['stimulus'], drop_first=True, dtype=float)
                X_base = sm.add_constant(X_base)
                y = subj_df[dv_key].values
                ols_base = sm.OLS(y, X_base).fit()
                subj_llf_base[subj] = ols_base.llf
            except Exception:
                continue

            for model_name in SURPRISE_COLS:
                if model_name not in subj_df.columns:
                    continue
                try:
                    X_m = pd.get_dummies(subj_df['stimulus'], drop_first=True, dtype=float)
                    X_m[model_name] = subj_df[model_name].values
                    X_m = sm.add_constant(X_m)
                    ols_m = sm.OLS(y, X_m).fit()
                    subj_llf_models[model_name][subj] = ols_m.llf
                except Exception:
                    pass

        # Compute observed group delta-AIC
        valid_subjects = sorted(set(subj_llf_base.keys()))

        for model_name in SURPRISE_COLS:
            model_subjs = [s for s in valid_subjects if s in subj_llf_models[model_name]]
            if len(model_subjs) < 3:
                print(f"    {model_name}: too few subjects for bootstrap")
                continue

            # Observed delta-AIC (sum of subject-level delta-AIC)
            # subject-level delta-AIC = -2*(llf_model - llf_base) + 2*(1)
            # extra parameter = 1 (the surprise beta)
            obs_delta_aics = []
            for s in model_subjs:
                d_aic_s = -2 * (subj_llf_models[model_name][s] - subj_llf_base[s]) + 2
                obs_delta_aics.append(d_aic_s)
            obs_delta_aics = np.array(obs_delta_aics)
            obs_group_delta_aic = np.sum(obs_delta_aics)

            # Bootstrap
            boot_delta_aics = np.zeros(n_boot)
            for b in range(n_boot):
                boot_idx = rng.choice(len(model_subjs), size=len(model_subjs), replace=True)
                boot_delta_aics[b] = np.sum(obs_delta_aics[boot_idx])

            ci_lower = np.percentile(boot_delta_aics, 2.5)
            ci_upper = np.percentile(boot_delta_aics, 97.5)

            bootstrap_results[dv_key][model_name] = {
                'observed_group_delta_aic': obs_group_delta_aic,
                'boot_ci_95_lower': ci_lower,
                'boot_ci_95_upper': ci_upper,
                'boot_mean': np.mean(boot_delta_aics),
                'boot_sd': np.std(boot_delta_aics),
                'n_subjects': len(model_subjs),
            }

            # Negative delta-AIC = model is better than baseline
            print(f"    {model_name:30s}: group dAIC={obs_group_delta_aic:+.2f}  "
                  f"95% CI=[{ci_lower:+.2f}, {ci_upper:+.2f}]")

    return bootstrap_results


# ---------------------------------------------------------------------------
# (e) Cluster permutation test with proper reporting
# ---------------------------------------------------------------------------

def improved_cluster_permutation(task, n_permutations=5000, seed=42):
    """Re-run cluster permutation test with better p-value reporting.

    - Uses 5000 permutations for precision
    - Reports p < 0.001 instead of p = 0.000
    - Includes minimum possible p = 1/(n_permutations+1)
    """
    print(f"\n--- (e) Improved cluster permutation tests for {task} ---")

    time_resolved_file = RESULTS_DIR / f"time_resolved_{task}.json"
    if not time_resolved_file.exists():
        print(f"    Time-resolved results not found: {time_resolved_file}")
        return {}

    with open(str(time_resolved_file), 'r') as f:
        tr_data = json.load(f)

    rng = np.random.RandomState(seed)
    cluster_results = {}

    for roi_name in ['mmn_roi', 'p3b_roi']:
        if roi_name not in tr_data:
            continue

        cluster_results[roi_name] = {}
        print(f"\n  ROI: {roi_name}")

        for model_name in SURPRISE_COLS:
            if model_name not in tr_data[roi_name]:
                continue

            info = tr_data[roi_name][model_name]
            mean_beta = np.array(info['mean_beta'])
            p_values = np.array(info['p_values'])
            times = np.array(info['times'])

            # Run cluster-based permutation test
            cluster_threshold = 0.05
            sig_mask = p_values < cluster_threshold

            # Find clusters in observed data
            clusters = _find_clusters(sig_mask)

            if not clusters:
                cluster_results[roi_name][model_name] = {
                    'n_clusters': 0,
                    'significant_clusters': [],
                }
                continue

            # Observed cluster masses
            observed_masses = [np.sum(np.abs(mean_beta[s:e])) for s, e in clusters]

            # Permutation distribution (sign-flip)
            max_cluster_masses = np.zeros(n_permutations)
            for perm in range(n_permutations):
                signs = rng.choice([-1, 1], size=len(mean_beta))
                perm_betas = mean_beta * signs
                perm_sig = np.abs(perm_betas) > np.percentile(np.abs(perm_betas), 95)
                perm_clusters = _find_clusters(perm_sig)
                if perm_clusters:
                    max_cluster_masses[perm] = max(
                        np.sum(np.abs(perm_betas[s:e])) for s, e in perm_clusters
                    )

            # Compute p-values with proper floor
            min_p = 1.0 / (n_permutations + 1)
            sig_clusters = []
            for (s, e), mass in zip(clusters, observed_masses):
                p_raw = np.mean(max_cluster_masses >= mass)
                # Floor at minimum possible p
                p_report = max(p_raw, min_p)

                cluster_info = {
                    'start_ms': float(times[s] * 1000),
                    'end_ms': float(times[e - 1] * 1000),
                    'start_idx': int(s),
                    'end_idx': int(e),
                    'mass': float(mass),
                    'p_value': float(p_report),
                    'p_value_formatted': _format_p(p_report),
                    'significant': p_report < 0.05,
                    'n_permutations': n_permutations,
                }
                sig_clusters.append(cluster_info)

                if p_report < 0.05:
                    print(f"    {model_name:30s}: {times[s]*1000:.0f}-{times[e-1]*1000:.0f} ms  "
                          f"mass={mass:.4f}  {_format_p(p_report)}")

            cluster_results[roi_name][model_name] = {
                'n_clusters': len(clusters),
                'significant_clusters': [c for c in sig_clusters if c['significant']],
                'all_clusters': sig_clusters,
            }

    return cluster_results


def _find_clusters(mask):
    """Find contiguous runs of True in a boolean mask."""
    clusters = []
    in_cluster = False
    start = 0
    for t in range(len(mask)):
        if mask[t] and not in_cluster:
            start = t
            in_cluster = True
        elif not mask[t] and in_cluster:
            clusters.append((start, t))
            in_cluster = False
    if in_cluster:
        clusters.append((start, len(mask)))
    return clusters


def _format_p(p):
    """Format p-value with proper reporting (avoid p = 0.000)."""
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    elif p < 0.05:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.3f}"


# ---------------------------------------------------------------------------
# (f) Assemble and save corrected results
# ---------------------------------------------------------------------------

def run_corrections(task):
    """Run all statistical corrections for a given task and save results."""
    print(f"\n{'='*70}")
    print(f"  STATISTICAL CORRECTIONS FOR TASK: {task}")
    print(f"{'='*70}")

    # Load original results
    orig_file = RESULTS_DIR / f"encoding_results_{task}.json"
    if not orig_file.exists():
        print(f"Original results not found: {orig_file}")
        return

    with open(str(orig_file), 'r') as f:
        original_results = json.load(f)

    # Load analysis dataframe
    df_file = RESULTS_DIR / f"analysis_df_{task}.csv"
    if not df_file.exists():
        print(f"Analysis dataframe not found: {df_file}")
        return

    analysis_df = pd.read_csv(str(df_file))
    print(f"Loaded analysis_df: {analysis_df.shape[0]} rows, "
          f"{analysis_df['subject'].nunique()} subjects")

    # (a) Bonferroni-Holm correction
    correction_results = apply_corrections(original_results, task)

    # (b) Re-fit models with better convergence
    refit_results = refit_models(analysis_df, task)

    # Apply Holm correction to the refitted p-values as well
    refit_correction = {}
    for dv_key in ['mmn_amplitude', 'p3b_amplitude']:
        if dv_key not in refit_results:
            continue
        uncorrected = {}
        for model in SURPRISE_COLS:
            if model in refit_results[dv_key] and 'lr_pval' in refit_results[dv_key][model]:
                uncorrected[model] = refit_results[dv_key][model]['lr_pval']
        if uncorrected:
            refit_correction[dv_key] = {
                'uncorrected': uncorrected,
                'holm_corrected': bonferroni_holm(uncorrected),
            }

    # (c) Effect sizes
    effect_sizes = compute_effect_sizes(analysis_df, refit_results, task)

    # (d) Bootstrap CIs for delta-AIC
    bootstrap_results = bootstrap_delta_aic(analysis_df, task,
                                             n_boot=1000, seed=42)

    # (e) Cluster permutation tests
    cluster_results = improved_cluster_permutation(task,
                                                    n_permutations=5000, seed=42)

    # (f) Assemble final output
    corrected_output = {
        'task': task,
        'window': WINDOW_LABELS.get(task, task),
        'original_results': original_results,
        'holm_correction_original': correction_results,
        'refitted_models': refit_results,
        'holm_correction_refitted': refit_correction,
        'effect_sizes': effect_sizes,
        'bootstrap_delta_aic': bootstrap_results,
        'cluster_permutation': cluster_results,
    }

    out_file = RESULTS_DIR / f"encoding_results_corrected_{task}.json"
    with open(str(out_file), 'w') as f:
        json.dump(corrected_output, f, indent=2, default=_convert_for_json)

    print(f"\nCorrected results saved to {out_file}")
    return corrected_output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tasks = sys.argv[1:] if len(sys.argv) > 1 else ['MMN', 'P3']
    for task in tasks:
        run_corrections(task)
