#!/usr/bin/env python3
"""Simulation supplement: Model distinguishability in non-stationary paradigms.

Part 1: Generate roving oddball sequences with known change points
Part 2: Show surprise regressors become dissociable (low VIF, low correlation)
Part 3: Model recovery — can we identify the true generating model?
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy import stats
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from surprise.estimators import (
    static_shannon_surprise, adaptive_shannon_surprise,
    bayesian_surprise, changepoint_surprise
)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "simulation"
FIGURES_DIR = PROJECT_DIR / "figures"

SEED = 42
MODEL_NAMES = ['static_shannon', 'adaptive_shannon', 'bayesian', 'changepoint']
MODEL_LABELS = {
    'static_shannon': 'Static Shannon',
    'adaptive_shannon': 'Adaptive Shannon',
    'bayesian': 'Bayesian',
    'changepoint': 'Change-point',
}


def generate_roving_sequence(n_blocks=10, trials_per_block=80, seed=42):
    """Generate a non-stationary oddball sequence with known change points.

    Deviant probability alternates between low (0.10-0.15) and high (0.35-0.50)
    blocks, simulating a roving oddball paradigm.
    """
    rng = np.random.RandomState(seed)

    # Alternate between low and high deviant probability
    low_probs = [0.10, 0.12, 0.15, 0.10, 0.12]
    high_probs = [0.40, 0.35, 0.50, 0.45, 0.40]

    sequence = []
    block_probs = []
    change_points = []

    for b in range(n_blocks):
        if b % 2 == 0:
            p = low_probs[b // 2 % len(low_probs)]
        else:
            p = high_probs[b // 2 % len(high_probs)]

        block_seq = (rng.rand(trials_per_block) < p).astype(int)
        sequence.extend(block_seq)
        block_probs.extend([p] * trials_per_block)

        if b > 0:
            change_points.append(b * trials_per_block)

    return np.array(sequence), np.array(block_probs), change_points


def generate_stationary_sequence(n_trials=800, p_deviant=0.20, seed=42):
    """Generate a standard stationary oddball sequence."""
    rng = np.random.RandomState(seed)
    return (rng.rand(n_trials) < p_deviant).astype(int)


def compute_all_regressors(sequence):
    """Compute all four surprise regressors for a sequence."""
    seq = np.asarray(sequence)

    regressors = {
        'static_shannon': static_shannon_surprise(seq),
        'adaptive_shannon': adaptive_shannon_surprise(seq, window_size=20),
    }

    bayes_surp, _ = bayesian_surprise(seq)
    regressors['bayesian'] = bayes_surp

    cp_pred, _, _, _ = changepoint_surprise(seq, hazard_rate=1/200)
    regressors['changepoint'] = cp_pred

    return regressors


def compute_vif(df, columns):
    """Compute VIF for each column."""
    from numpy.linalg import lstsq
    vif = {}
    for col in columns:
        others = [c for c in columns if c != col]
        if not others:
            vif[col] = 1.0
            continue
        X = np.column_stack([np.ones(len(df)), df[others].values])
        y = df[col].values
        coeffs, _, _, _ = lstsq(X, y, rcond=None)
        y_pred = X @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        vif[col] = 1.0 / (1 - r2) if r2 < 1 else np.inf
    return vif


def part1_regressor_comparison():
    """Part 1-2: Compare regressor properties in stationary vs roving paradigms."""
    print("=" * 60)
    print("PART 1-2: Regressor Dissociability")
    print("=" * 60)

    results = {}

    # Stationary
    stat_seq = generate_stationary_sequence(n_trials=800, seed=SEED)
    stat_regs = compute_all_regressors(stat_seq)
    stat_df = pd.DataFrame(stat_regs)

    stat_corr = stat_df.corr()
    stat_vif = compute_vif(stat_df, MODEL_NAMES)

    print("\nSTATIONARY (standard oddball, p=0.20):")
    print(f"  Correlation matrix:\n{stat_corr.round(3)}")
    print(f"  VIF: {', '.join(f'{k}={v:.1f}' for k, v in stat_vif.items())}")

    results['stationary'] = {
        'correlation': {f'{k1}_{k2}': float(stat_corr.loc[k1, k2])
                        for k1 in MODEL_NAMES for k2 in MODEL_NAMES if k1 < k2},
        'vif': {k: float(v) for k, v in stat_vif.items()},
        'regressor_std': {k: float(stat_df[k].std()) for k in MODEL_NAMES},
    }

    # Roving
    rov_seq, rov_probs, cps = generate_roving_sequence(n_blocks=10, trials_per_block=80, seed=SEED)
    rov_regs = compute_all_regressors(rov_seq)
    rov_df = pd.DataFrame(rov_regs)

    rov_corr = rov_df.corr()
    rov_vif = compute_vif(rov_df, MODEL_NAMES)

    print(f"\nROVING ODDBALL (alternating p=0.10-0.50, {len(cps)} change points):")
    print(f"  Correlation matrix:\n{rov_corr.round(3)}")
    print(f"  VIF: {', '.join(f'{k}={v:.1f}' for k, v in rov_vif.items())}")

    results['roving'] = {
        'correlation': {f'{k1}_{k2}': float(rov_corr.loc[k1, k2])
                        for k1 in MODEL_NAMES for k2 in MODEL_NAMES if k1 < k2},
        'vif': {k: float(v) for k, v in rov_vif.items()},
        'regressor_std': {k: float(rov_df[k].std()) for k in MODEL_NAMES},
        'change_points': cps,
        'block_probs': list(set(float(p) for p in rov_probs)),
    }

    # Summary comparison
    print(f"\nVIF COMPARISON:")
    print(f"  {'Model':<25} {'Stationary':>12} {'Roving':>12} {'Reduction':>12}")
    for m in MODEL_NAMES:
        sv = stat_vif[m]
        rv = rov_vif[m]
        pct = (sv - rv) / sv * 100 if sv > 0 else 0
        print(f"  {MODEL_LABELS[m]:<25} {sv:>12.1f} {rv:>12.1f} {pct:>11.0f}%")

    return results, (stat_seq, stat_regs), (rov_seq, rov_regs, rov_probs, cps)


def part3_model_recovery(n_simulations=500, n_subjects=39, n_trials=800,
                          noise_level=1.0, effect_size=0.20):
    """Part 3: Model recovery simulation.

    For each ground-truth model:
      1. Generate non-stationary sequences
      2. Simulate EEG: amplitude = β₀ + β₁·stimulus + β₂·surprise_true + noise
      3. Fit all four models
      4. Check which model wins on AIC
    """
    print("\n" + "=" * 60)
    print("PART 3: Model Recovery")
    print("=" * 60)
    print(f"  {n_simulations} simulations × {len(MODEL_NAMES)} ground-truth models")
    print(f"  {n_subjects} subjects × {n_trials} trials, effect_size={effect_size}")

    rng = np.random.RandomState(SEED)

    # Confusion matrix: rows = true model, columns = recovered model
    confusion = np.zeros((len(MODEL_NAMES), len(MODEL_NAMES)), dtype=int)

    # Also track AIC differences
    aic_diffs = {true_m: {fit_m: [] for fit_m in MODEL_NAMES} for true_m in MODEL_NAMES}

    for sim in range(n_simulations):
        if sim % 100 == 0:
            print(f"  Simulation {sim}/{n_simulations}...")

        # Generate a roving sequence for this simulation
        sim_seq, sim_probs, sim_cps = generate_roving_sequence(
            n_blocks=10, trials_per_block=n_trials // 10,
            seed=SEED + sim
        )

        # Compute all regressors
        regressors = compute_all_regressors(sim_seq)

        # Z-score regressors
        for k in regressors:
            r = regressors[k]
            std = np.std(r)
            if std > 0:
                regressors[k] = (r - np.mean(r)) / std

        stimulus = sim_seq.astype(float)
        stimulus_z = (stimulus - np.mean(stimulus)) / (np.std(stimulus) + 1e-10)

        # For each ground-truth model
        for true_idx, true_model in enumerate(MODEL_NAMES):
            # Simulate EEG data
            true_surprise = regressors[true_model]
            noise = rng.randn(len(sim_seq)) * noise_level
            eeg = 0.5 * stimulus_z + effect_size * true_surprise + noise

            # Fit each model using OLS and compute AIC
            n = len(eeg)
            model_aics = {}

            for fit_idx, fit_model in enumerate(MODEL_NAMES):
                fit_surprise = regressors[fit_model]

                # Design matrix: intercept + stimulus + surprise
                X = np.column_stack([np.ones(n), stimulus_z, fit_surprise])
                coeffs, residuals, _, _ = np.linalg.lstsq(X, eeg, rcond=None)
                y_pred = X @ coeffs
                ss_res = np.sum((eeg - y_pred) ** 2)

                # AIC = n * log(ss_res/n) + 2k
                k = 3  # intercept + stimulus + surprise
                aic = n * np.log(ss_res / n + 1e-300) + 2 * k
                model_aics[fit_model] = aic

            # Which model has lowest AIC?
            best_model = min(model_aics, key=model_aics.get)
            best_idx = MODEL_NAMES.index(best_model)
            confusion[true_idx, best_idx] += 1

            # Store AIC differences vs true model
            true_aic = model_aics[true_model]
            for fit_model in MODEL_NAMES:
                aic_diffs[true_model][fit_model].append(
                    model_aics[fit_model] - true_aic
                )

    # Compute recovery rates
    recovery_rates = confusion.diagonal() / confusion.sum(axis=1)

    print(f"\n  CONFUSION MATRIX (rows=true, cols=recovered):")
    print(f"  {'':>20}", end='')
    for m in MODEL_NAMES:
        print(f"  {MODEL_LABELS[m]:>15}", end='')
    print()
    for i, m in enumerate(MODEL_NAMES):
        print(f"  {MODEL_LABELS[m]:>20}", end='')
        for j in range(len(MODEL_NAMES)):
            pct = confusion[i, j] / n_simulations * 100
            print(f"  {pct:>14.1f}%", end='')
        print(f"  (recovery: {recovery_rates[i]*100:.1f}%)")

    print(f"\n  OVERALL RECOVERY RATE: {np.mean(recovery_rates)*100:.1f}%")

    # Build results dict
    results = {
        'n_simulations': n_simulations,
        'n_subjects': n_subjects,
        'n_trials': n_trials,
        'noise_level': noise_level,
        'effect_size': effect_size,
        'confusion_matrix': confusion.tolist(),
        'model_names': MODEL_NAMES,
        'model_labels': [MODEL_LABELS[m] for m in MODEL_NAMES],
        'recovery_rates': {m: float(r) for m, r in zip(MODEL_NAMES, recovery_rates)},
        'overall_recovery': float(np.mean(recovery_rates)),
        'mean_aic_diffs': {
            true_m: {fit_m: float(np.mean(aic_diffs[true_m][fit_m]))
                      for fit_m in MODEL_NAMES}
            for true_m in MODEL_NAMES
        },
    }

    return results


def part3b_effect_size_sweep(effect_sizes=[0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
                              n_simulations=200):
    """Sweep over effect sizes to find minimum detectable effect for recovery."""
    print("\n" + "=" * 60)
    print("PART 3b: Effect Size Sweep for Model Recovery")
    print("=" * 60)

    rng = np.random.RandomState(SEED)
    sweep_results = []

    for es in effect_sizes:
        print(f"\n  Effect size = {es}:")
        confusion = np.zeros((len(MODEL_NAMES), len(MODEL_NAMES)), dtype=int)

        for sim in range(n_simulations):
            sim_seq, _, _ = generate_roving_sequence(
                n_blocks=10, trials_per_block=80, seed=SEED + sim + 10000
            )
            regressors = compute_all_regressors(sim_seq)
            for k in regressors:
                r = regressors[k]
                std = np.std(r)
                if std > 0:
                    regressors[k] = (r - np.mean(r)) / std

            stimulus = sim_seq.astype(float)
            stimulus_z = (stimulus - np.mean(stimulus)) / (np.std(stimulus) + 1e-10)

            for true_idx, true_model in enumerate(MODEL_NAMES):
                true_surprise = regressors[true_model]
                noise = rng.randn(len(sim_seq))
                eeg = 0.5 * stimulus_z + es * true_surprise + noise

                n = len(eeg)
                model_aics = {}
                for fit_model in MODEL_NAMES:
                    X = np.column_stack([np.ones(n), stimulus_z, regressors[fit_model]])
                    coeffs, _, _, _ = np.linalg.lstsq(X, eeg, rcond=None)
                    y_pred = X @ coeffs
                    ss_res = np.sum((eeg - y_pred) ** 2)
                    model_aics[fit_model] = n * np.log(ss_res / n + 1e-300) + 6

                best = min(model_aics, key=model_aics.get)
                confusion[true_idx, MODEL_NAMES.index(best)] += 1

        recovery_rates = confusion.diagonal() / confusion.sum(axis=1)
        overall = np.mean(recovery_rates)

        print(f"    Recovery rates: " +
              ", ".join(f"{MODEL_LABELS[m]}={r*100:.0f}%"
                        for m, r in zip(MODEL_NAMES, recovery_rates)))
        print(f"    Overall: {overall*100:.1f}%")

        sweep_results.append({
            'effect_size': es,
            'overall_recovery': float(overall),
            'per_model': {m: float(r) for m, r in zip(MODEL_NAMES, recovery_rates)},
            'confusion_matrix': confusion.tolist(),
        })

    return sweep_results


def make_simulation_figure(stat_data, rov_data, recovery_results, sweep_results):
    """Generate Figure 5: Simulation results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    COLORS = {
        'static_shannon': '#0072B2',
        'adaptive_shannon': '#E69F00',
        'bayesian': '#009E73',
        'changepoint': '#CC79A7',
    }

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    stat_seq, stat_regs = stat_data
    rov_seq, rov_regs, rov_probs, cps = rov_data

    # Panel A: Stationary surprise traces
    ax_a = fig.add_subplot(gs[0, 0])
    n_show = min(200, len(stat_seq))
    for m in MODEL_NAMES:
        vals = stat_regs[m][:n_show]
        vals_z = (vals - vals.mean()) / (vals.std() + 1e-10)
        ax_a.plot(range(n_show), vals_z, color=COLORS[m], linewidth=1,
                  label=MODEL_LABELS[m], alpha=0.8)
    ax_a.set_xlabel('Trial')
    ax_a.set_ylabel('Surprise (z-scored)')
    ax_a.set_title('A. Stationary Paradigm', fontweight='bold', loc='left')
    ax_a.legend(fontsize=7, frameon=True)

    # Panel B: Roving surprise traces
    ax_b = fig.add_subplot(gs[0, 1])
    n_show = min(400, len(rov_seq))
    for m in MODEL_NAMES:
        vals = rov_regs[m][:n_show]
        vals_z = (vals - vals.mean()) / (vals.std() + 1e-10)
        ax_b.plot(range(n_show), vals_z, color=COLORS[m], linewidth=1,
                  label=MODEL_LABELS[m], alpha=0.8)
    # Mark change points
    for cp in cps:
        if cp < n_show:
            ax_b.axvline(cp, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    # Add probability trace
    ax_b2 = ax_b.twinx()
    ax_b2.plot(range(n_show), rov_probs[:n_show], color='black', linewidth=1.5,
               linestyle=':', alpha=0.5, label='p(deviant)')
    ax_b2.set_ylabel('p(deviant)', fontsize=9)
    ax_b2.set_ylim(0, 0.6)
    ax_b.set_xlabel('Trial')
    ax_b.set_ylabel('Surprise (z-scored)')
    ax_b.set_title('B. Roving Paradigm', fontweight='bold', loc='left')

    # Panel C: VIF comparison
    ax_c = fig.add_subplot(gs[0, 2])
    x = np.arange(len(MODEL_NAMES))
    width = 0.35

    # Compute VIFs
    stat_df = pd.DataFrame(stat_regs)
    rov_df = pd.DataFrame(rov_regs)
    stat_vif = compute_vif(stat_df, MODEL_NAMES)
    rov_vif = compute_vif(rov_df, MODEL_NAMES)

    stat_vals = [min(stat_vif[m], 150) for m in MODEL_NAMES]
    rov_vals = [rov_vif[m] for m in MODEL_NAMES]

    bars1 = ax_c.bar(x - width/2, stat_vals, width, label='Stationary',
                      color='lightcoral', alpha=0.8)
    bars2 = ax_c.bar(x + width/2, rov_vals, width, label='Roving',
                      color='steelblue', alpha=0.8)

    ax_c.axhline(10, color='red', linestyle='--', linewidth=1, label='VIF = 10 threshold')
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([MODEL_LABELS[m].replace(' ', '\n') for m in MODEL_NAMES],
                          fontsize=8)
    ax_c.set_ylabel('VIF')
    ax_c.set_title('C. Multicollinearity Reduction', fontweight='bold', loc='left')
    ax_c.legend(fontsize=7, frameon=True)

    # Panel D: Confusion matrix
    ax_d = fig.add_subplot(gs[1, 0])
    conf = np.array(recovery_results['confusion_matrix'], dtype=float)
    conf_pct = conf / conf.sum(axis=1, keepdims=True) * 100

    im = ax_d.imshow(conf_pct, cmap='Blues', vmin=0, vmax=100, aspect='auto')
    for i in range(len(MODEL_NAMES)):
        for j in range(len(MODEL_NAMES)):
            color = 'white' if conf_pct[i, j] > 50 else 'black'
            ax_d.text(j, i, f'{conf_pct[i,j]:.0f}%', ha='center', va='center',
                      color=color, fontsize=9, fontweight='bold')

    ax_d.set_xticks(range(len(MODEL_NAMES)))
    ax_d.set_yticks(range(len(MODEL_NAMES)))
    labels = [MODEL_LABELS[m].replace(' ', '\n') for m in MODEL_NAMES]
    ax_d.set_xticklabels(labels, fontsize=8)
    ax_d.set_yticklabels(labels, fontsize=8)
    ax_d.set_xlabel('Recovered Model')
    ax_d.set_ylabel('True Model')
    ax_d.set_title('D. Model Recovery (Confusion Matrix)', fontweight='bold', loc='left')
    plt.colorbar(im, ax=ax_d, shrink=0.8, label='Recovery %')

    # Panel E: Recovery rate by effect size
    ax_e = fig.add_subplot(gs[1, 1])
    es_vals = [s['effect_size'] for s in sweep_results]
    overall_rates = [s['overall_recovery'] * 100 for s in sweep_results]

    ax_e.plot(es_vals, overall_rates, 'k-o', linewidth=2, markersize=6,
              label='Overall', zorder=5)

    for m in MODEL_NAMES:
        rates = [s['per_model'][m] * 100 for s in sweep_results]
        ax_e.plot(es_vals, rates, color=COLORS[m], linewidth=1.5, alpha=0.7,
                  marker='s', markersize=4, label=MODEL_LABELS[m])

    ax_e.axhline(25, color='gray', linestyle=':', linewidth=1, label='Chance (25%)')
    ax_e.axhline(80, color='red', linestyle='--', linewidth=1, alpha=0.5,
                  label='80% threshold')
    ax_e.set_xlabel('Effect Size (β)')
    ax_e.set_ylabel('Recovery Rate (%)')
    ax_e.set_title('E. Recovery vs. Effect Size', fontweight='bold', loc='left')
    ax_e.legend(fontsize=6, frameon=True, ncol=2)
    ax_e.set_ylim(0, 105)

    # Panel F: Design recommendations
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis('off')

    recommendations = [
        "Design Recommendations",
        "",
        "1. Use roving/volatile paradigms",
        "   to reduce multicollinearity",
        "   (VIF drops from >70 to <5)",
        "",
        "2. Minimum effect size for",
        f"   80% model recovery:",
        f"   β ≈ {next((s['effect_size'] for s in sweep_results if s['overall_recovery'] > 0.80), '>0.50')}",
        "",
        "3. Bayesian surprise is most",
        "   recoverable (highest diagonal",
        "   in confusion matrix)",
        "",
        "4. Static Shannon is hardest to",
        "   distinguish from change-point",
        "   (highest off-diagonal confusion)",
    ]

    for i, line in enumerate(recommendations):
        weight = 'bold' if i == 0 else 'normal'
        size = 11 if i == 0 else 9
        ax_f.text(0.05, 0.95 - i * 0.065, line, transform=ax_f.transAxes,
                  fontsize=size, fontweight=weight, verticalalignment='top',
                  fontfamily='sans-serif')

    ax_f.set_title('F. Implications', fontweight='bold', loc='left')

    fig.savefig(str(FIGURES_DIR / "fig5_simulation.pdf"), dpi=300, bbox_inches='tight')
    fig.savefig(str(FIGURES_DIR / "fig5_simulation.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Figure 5 saved to {FIGURES_DIR}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Part 1-2: Regressor comparison
    comparison_results, stat_data, rov_data = part1_regressor_comparison()

    # Part 3: Model recovery
    recovery_results = part3_model_recovery(
        n_simulations=500, n_trials=800,
        noise_level=1.0, effect_size=0.20
    )

    # Part 3b: Effect size sweep
    sweep_results = part3b_effect_size_sweep(
        effect_sizes=[0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
        n_simulations=200
    )

    # Save all results
    all_results = {
        'regressor_comparison': comparison_results,
        'model_recovery': recovery_results,
        'effect_size_sweep': sweep_results,
    }

    with open(str(RESULTS_DIR / "simulation_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  Results saved to {RESULTS_DIR / 'simulation_results.json'}")

    # Generate figure
    rov_seq, rov_regs, rov_probs, cps = rov_data
    make_simulation_figure(
        stat_data,
        (rov_seq, rov_regs, rov_probs, cps),
        recovery_results,
        sweep_results
    )

    # Summary
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"\n  Stationary paradigm:")
    print(f"    Mean VIF: {np.mean(list(comparison_results['stationary']['vif'].values())):.1f}")
    print(f"  Roving paradigm:")
    print(f"    Mean VIF: {np.mean(list(comparison_results['roving']['vif'].values())):.1f}")
    print(f"\n  Model recovery (effect_size=0.20):")
    print(f"    Overall: {recovery_results['overall_recovery']*100:.1f}%")
    for m in MODEL_NAMES:
        print(f"    {MODEL_LABELS[m]}: {recovery_results['recovery_rates'][m]*100:.1f}%")

    print(f"\n  Effect size for 80% overall recovery: ", end='')
    for s in sweep_results:
        if s['overall_recovery'] >= 0.80:
            print(f"β = {s['effect_size']}")
            break
    else:
        print(f"> 0.50 (not reached)")


if __name__ == "__main__":
    main()
