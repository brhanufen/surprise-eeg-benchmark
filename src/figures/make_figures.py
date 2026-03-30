#!/usr/bin/env python3
"""Generate all publication-ready figures.

Figure 1: Study schematic and surprise model hierarchy
Figure 2: ERP replication and quality control
Figure 3: Surprise encoding results
Figure 4: Decoding benchmark
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REGRESSORS_DIR = PROJECT_DIR / "data" / "regressors"

# Publication style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color-blind friendly palette (IBM Design / Wong 2011)
COLORS = {
    'static_shannon': '#0072B2',      # blue
    'adaptive_shannon_w20': '#E69F00', # orange
    'bayesian_surprise': '#009E73',    # teal/green
    'changepoint_surprise': '#CC79A7', # pink/magenta
    'standard': '#56B4E9',             # sky blue
    'deviant': '#D55E00',              # vermillion
}

MODEL_LABELS = {
    'static_shannon': 'Static Shannon',
    'adaptive_shannon_w20': 'Adaptive Shannon',
    'bayesian_surprise': 'Bayesian',
    'changepoint_surprise': 'Change-point',
}


def figure1_schematic(task="MMN"):
    """Figure 1: Study schematic and surprise model hierarchy."""
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: Task paradigm schematic
    ax_a = fig.add_subplot(gs[0, 0])
    # Simple oddball sequence visualization
    np.random.seed(42)
    seq = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    for i, s in enumerate(seq):
        color = COLORS['deviant'] if s == 1 else COLORS['standard']
        ax_a.bar(i, 1, color=color, edgecolor='white', linewidth=0.5)
    ax_a.set_xlabel('Trial')
    ax_a.set_ylabel('')
    ax_a.set_yticks([])
    ax_a.set_title('A. Oddball Paradigm', fontweight='bold', loc='left')
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS['standard'], label='Standard'),
                       Patch(facecolor=COLORS['deviant'], label='Deviant')]
    ax_a.legend(handles=legend_elements, loc='upper right', frameon=True)

    # Panel B: Model hierarchy
    ax_b = fig.add_subplot(gs[0, 1])
    models = ['Static\nShannon', 'Adaptive\nShannon', 'Bayesian', 'Change-\npoint']
    equations = [r'$-\log p_{global}$', r'$-\log p_w$',
                 r'$D_{KL}$', r'$P(cp_t)$']
    colors = [COLORS['static_shannon'], COLORS['adaptive_shannon_w20'],
              COLORS['bayesian_surprise'], COLORS['changepoint_surprise']]

    for i, (model, eq, col) in enumerate(zip(models, equations, colors)):
        ax_b.add_patch(plt.Rectangle((i * 2.5, 0.2), 2, 0.6, facecolor=col,
                                      alpha=0.3, edgecolor=col, linewidth=2))
        ax_b.text(i * 2.5 + 1, 0.55, model, ha='center', va='center',
                  fontweight='bold', fontsize=9)
        ax_b.text(i * 2.5 + 1, 0.35, eq, ha='center', va='center', fontsize=8)
        if i < 3:
            ax_b.annotate('', xy=((i+1)*2.5, 0.5), xytext=(i*2.5+2, 0.5),
                          arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax_b.set_xlim(-0.3, 10.3)
    ax_b.set_ylim(0, 1)
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    ax_b.set_title('B. Surprise Model Hierarchy', fontweight='bold', loc='left')
    ax_b.text(5, 0.05, r'Increasing complexity $\rightarrow$',
              ha='center', fontsize=9, style='italic')

    # Panel C: Example surprise traces
    ax_c = fig.add_subplot(gs[1, 0])

    # Load one subject's regressors
    reg_files = sorted(REGRESSORS_DIR.glob(f"*_task-{task}_regressors.csv"))
    if reg_files:
        reg_df = pd.read_csv(str(reg_files[0]))
        n_show = min(200, len(reg_df))

        for col, label in MODEL_LABELS.items():
            if col in reg_df.columns:
                vals = reg_df[col].values[:n_show]
                # Normalize for visualization
                vals_norm = (vals - vals.mean()) / (vals.std() + 1e-10)
                ax_c.plot(range(n_show), vals_norm, color=COLORS[col],
                          alpha=0.8, linewidth=1, label=label)

        # Mark deviants
        deviant_trials = np.where(reg_df['stimulus'].values[:n_show] == 1)[0]
        for dt in deviant_trials:
            ax_c.axvline(dt, color='gray', alpha=0.15, linewidth=0.5)

        ax_c.set_xlabel('Trial')
        ax_c.set_ylabel('Surprise (z-scored)')
        ax_c.legend(loc='upper right', frameon=True, fontsize=8)
    ax_c.set_title('C. Example Surprise Traces', fontweight='bold', loc='left')

    # Panel D: Pipeline flowchart
    ax_d = fig.add_subplot(gs[1, 1])
    steps = ['ERP CORE\nData', 'Preprocessing\n(MNE)', 'Feature\nExtraction',
             'Encoding\n(Aim 2)', 'Decoding\n(Aim 3)']
    for i, step in enumerate(steps):
        y = 0.85 - i * 0.18
        ax_d.add_patch(plt.Rectangle((0.2, y-0.06), 0.6, 0.12,
                                      facecolor='lightsteelblue', edgecolor='steelblue',
                                      linewidth=1.5, zorder=2))
        ax_d.text(0.5, y, step, ha='center', va='center', fontsize=9, zorder=3)
        if i < len(steps) - 1:
            ax_d.annotate('', xy=(0.5, y-0.06), xytext=(0.5, y-0.12),
                          arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.5))

    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    ax_d.set_xticks([])
    ax_d.set_yticks([])
    ax_d.set_title('D. Analysis Pipeline', fontweight='bold', loc='left')

    fig.savefig(str(FIGURES_DIR / f"fig1_schematic_{task}.pdf"))
    fig.savefig(str(FIGURES_DIR / f"fig1_schematic_{task}.png"))
    plt.close(fig)
    print(f"  Figure 1 saved")


def figure2_erp_replication(task="MMN"):
    """Figure 2: ERP replication and quality control."""
    import mne

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    epo_files = sorted(PROCESSED_DIR.glob(f"*_task-{task}_epo.fif"))
    if not epo_files:
        print("  No preprocessed data for Figure 2")
        return

    # Collect evoked responses
    all_standard = []
    all_deviant = []

    for epo_file in epo_files:
        try:
            epochs = mne.read_epochs(str(epo_file), preload=True, verbose=False)
            seq_file = epo_file.with_name(epo_file.name.replace('_epo.fif', '_sequence.npy'))
            if seq_file.exists():
                sequence = np.load(str(seq_file))
                n = min(len(sequence), len(epochs))
                standard_idx = np.where(sequence[:n] == 0)[0]
                deviant_idx = np.where(sequence[:n] == 1)[0]

                if len(standard_idx) > 0:
                    std_evoked = epochs[standard_idx].average()
                    all_standard.append(std_evoked)
                if len(deviant_idx) > 0:
                    dev_evoked = epochs[deviant_idx].average()
                    all_deviant.append(dev_evoked)
        except Exception as e:
            print(f"  Error loading {epo_file.name}: {e}")
            continue

    if not all_standard or not all_deviant:
        print("  Insufficient data for Figure 2")
        return

    # Grand averages
    grand_standard = mne.grand_average(all_standard)
    grand_deviant = mne.grand_average(all_deviant)
    times = grand_standard.times * 1000  # Convert to ms

    # Get channel data
    std_data = grand_standard.data * 1e6  # Convert to µV
    dev_data = grand_deviant.data * 1e6
    ch_names = grand_standard.ch_names

    # Find ROI channels
    MMN_ROI_NAMES = ['Fz', 'FCz', 'Cz', 'FC1', 'FC2', 'F1', 'F2']
    P3B_ROI_NAMES = ['Pz', 'CPz', 'P1', 'P2', 'CP1', 'CP2']

    mmn_idx = [i for i, ch in enumerate(ch_names)
               if ch.upper() in [c.upper() for c in MMN_ROI_NAMES]
               or ch.lower() in [c.lower() for c in MMN_ROI_NAMES]]
    p3b_idx = [i for i, ch in enumerate(ch_names)
               if ch.upper() in [c.upper() for c in P3B_ROI_NAMES]
               or ch.lower() in [c.lower() for c in P3B_ROI_NAMES]]

    if not mmn_idx:
        mmn_idx = list(range(min(5, len(ch_names))))
    if not p3b_idx:
        p3b_idx = list(range(max(0, len(ch_names)-5), len(ch_names)))

    # Panel A: Fronto-central (MMN) waveforms
    ax_a = fig.add_subplot(gs[0, 0])
    std_mmn = np.mean(std_data[mmn_idx, :], axis=0)
    dev_mmn = np.mean(dev_data[mmn_idx, :], axis=0)

    # Compute SEM across subjects
    std_all = np.array([s.data[mmn_idx, :].mean(0) * 1e6 for s in all_standard])
    dev_all = np.array([d.data[mmn_idx, :].mean(0) * 1e6 for d in all_deviant])
    std_sem = np.std(std_all, axis=0) / np.sqrt(len(all_standard))
    dev_sem = np.std(dev_all, axis=0) / np.sqrt(len(all_deviant))

    ax_a.plot(times, std_mmn, color=COLORS['standard'], linewidth=2, label='Standard')
    ax_a.fill_between(times, std_mmn - std_sem, std_mmn + std_sem,
                       color=COLORS['standard'], alpha=0.2)
    ax_a.plot(times, dev_mmn, color=COLORS['deviant'], linewidth=2, label='Deviant')
    ax_a.fill_between(times, dev_mmn - dev_sem, dev_mmn + dev_sem,
                       color=COLORS['deviant'], alpha=0.2)

    ax_a.axvspan(100, 250, alpha=0.1, color='gray', label='MMN window')
    ax_a.axhline(0, color='black', linewidth=0.5)
    ax_a.axvline(0, color='black', linewidth=0.5, linestyle='--')
    ax_a.set_xlabel('Time (ms)')
    ax_a.set_ylabel('Amplitude (µV)')
    ax_a.set_title('A. Fronto-Central (MMN ROI)', fontweight='bold', loc='left')
    ax_a.legend(loc='upper right', frameon=True)
    ax_a.invert_yaxis()  # ERP convention: negative up

    # Panel B: Parietal (P3b) waveforms
    ax_b = fig.add_subplot(gs[0, 1])
    std_p3b = np.mean(std_data[p3b_idx, :], axis=0)
    dev_p3b = np.mean(dev_data[p3b_idx, :], axis=0)

    std_all_p3 = np.array([s.data[p3b_idx, :].mean(0) * 1e6 for s in all_standard])
    dev_all_p3 = np.array([d.data[p3b_idx, :].mean(0) * 1e6 for d in all_deviant])
    std_sem_p3 = np.std(std_all_p3, axis=0) / np.sqrt(len(all_standard))
    dev_sem_p3 = np.std(dev_all_p3, axis=0) / np.sqrt(len(all_deviant))

    ax_b.plot(times, std_p3b, color=COLORS['standard'], linewidth=2, label='Standard')
    ax_b.fill_between(times, std_p3b - std_sem_p3, std_p3b + std_sem_p3,
                       color=COLORS['standard'], alpha=0.2)
    ax_b.plot(times, dev_p3b, color=COLORS['deviant'], linewidth=2, label='Deviant')
    ax_b.fill_between(times, dev_p3b - dev_sem_p3, dev_p3b + dev_sem_p3,
                       color=COLORS['deviant'], alpha=0.2)

    ax_b.axvspan(250, 500, alpha=0.1, color='gray', label='P3b window')
    ax_b.axhline(0, color='black', linewidth=0.5)
    ax_b.axvline(0, color='black', linewidth=0.5, linestyle='--')
    ax_b.set_xlabel('Time (ms)')
    ax_b.set_ylabel('Amplitude (µV)')
    ax_b.set_title('B. Parietal (P3b ROI)', fontweight='bold', loc='left')
    ax_b.legend(loc='upper right', frameon=True)

    # Panel C: Difference waveform (deviant - standard) for both ROIs
    ax_c = fig.add_subplot(gs[1, 0])
    diff_mmn = dev_mmn - std_mmn
    diff_p3b = dev_p3b - std_p3b

    # Compute SEM for difference waves
    diff_all_mmn = dev_all - std_all
    diff_all_p3 = dev_all_p3 - std_all_p3
    diff_sem_mmn = np.std(diff_all_mmn, axis=0) / np.sqrt(diff_all_mmn.shape[0])
    diff_sem_p3 = np.std(diff_all_p3, axis=0) / np.sqrt(diff_all_p3.shape[0])

    ax_c.plot(times, diff_mmn, color=COLORS['static_shannon'], linewidth=2,
              label='Fronto-central')
    ax_c.fill_between(times, diff_mmn - diff_sem_mmn, diff_mmn + diff_sem_mmn,
                       color=COLORS['static_shannon'], alpha=0.2)
    ax_c.plot(times, diff_p3b, color=COLORS['changepoint_surprise'], linewidth=2,
              label='Parietal')
    ax_c.fill_between(times, diff_p3b - diff_sem_p3, diff_p3b + diff_sem_p3,
                       color=COLORS['changepoint_surprise'], alpha=0.2)

    ax_c.axvspan(100, 250, alpha=0.08, color='blue')
    ax_c.axvspan(250, 500, alpha=0.08, color='red')
    ax_c.axhline(0, color='black', linewidth=0.5)
    ax_c.axvline(0, color='black', linewidth=0.5, linestyle='--')
    ax_c.set_xlabel('Time (ms)')
    ax_c.set_ylabel('Difference Amplitude (µV)')
    ax_c.set_title('C. Difference Waveforms (Deviant \u2212 Standard)',
                    fontweight='bold', loc='left')
    ax_c.legend(loc='lower right', frameon=True)
    ax_c.invert_yaxis()

    # Panel D: QC summary
    ax_d = fig.add_subplot(gs[1, 1])
    qc_file = RESULTS_DIR / "aim1" / f"qc_summary_{task}.csv"
    if qc_file.exists():
        qc_df = pd.read_csv(str(qc_file))
        subjects = qc_df['subject'].values
        x = np.arange(len(subjects))
        width = 0.35

        ax_d.bar(x - width/2, qc_df['n_standards'], width,
                 label='Standards', color=COLORS['standard'], alpha=0.8)
        ax_d.bar(x + width/2, qc_df['n_deviants'], width,
                 label='Deviants', color=COLORS['deviant'], alpha=0.8)

        ax_d.set_xlabel('Subject')
        ax_d.set_ylabel('N Epochs (after rejection)')
        ax_d.set_xticks(x[::5])
        ax_d.set_xticklabels([s.replace('sub-', '') for s in subjects[::5]],
                              rotation=45)
        ax_d.legend(frameon=True)
    ax_d.set_title('D. Epoch Counts After Rejection', fontweight='bold', loc='left')

    fig.savefig(str(FIGURES_DIR / f"fig2_erp_replication_{task}.pdf"))
    fig.savefig(str(FIGURES_DIR / f"fig2_erp_replication_{task}.png"))
    plt.close(fig)
    print(f"  Figure 2 saved")


def figure3_encoding(task="MMN"):
    """Figure 3: Surprise encoding results."""
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Load time-resolved results
    tr_file = RESULTS_DIR / "aim2" / f"time_resolved_{task}.json"
    enc_file = RESULTS_DIR / "aim2" / f"encoding_results_{task}.json"

    if not tr_file.exists() or not enc_file.exists():
        print("  Encoding results not found for Figure 3")
        return

    with open(str(tr_file)) as f:
        tr_results = json.load(f)
    with open(str(enc_file)) as f:
        enc_results = json.load(f)

    surprise_cols = ['static_shannon', 'adaptive_shannon_w20',
                     'bayesian_surprise', 'changepoint_surprise']

    # Panel A: Time-resolved regression coefficients (MMN ROI)
    ax_a = fig.add_subplot(gs[0, 0])
    if 'mmn_roi' in tr_results:
        for col in surprise_cols:
            if col in tr_results['mmn_roi']:
                r = tr_results['mmn_roi'][col]
                times = np.array(r['times']) * 1000
                beta = np.array(r['mean_beta'])
                se = np.array(r['se_beta'])
                pvals = np.array(r['p_values'])

                ax_a.plot(times, beta, color=COLORS[col], linewidth=1.5,
                          label=MODEL_LABELS[col])
                ax_a.fill_between(times, beta - 1.96*se, beta + 1.96*se,
                                   color=COLORS[col], alpha=0.15)

                # Mark significant clusters
                sig_mask = pvals < 0.05
                sig_times = times[sig_mask]
                if len(sig_times) > 0:
                    ax_a.scatter(sig_times, beta[sig_mask],
                                 color=COLORS[col], s=2, alpha=0.5)

    ax_a.axhline(0, color='black', linewidth=0.5)
    ax_a.axvline(0, color='black', linewidth=0.5, linestyle='--')
    ax_a.axvspan(100, 250, alpha=0.08, color='gray')
    ax_a.set_xlabel('Time (ms)')
    ax_a.set_ylabel('Beta coefficient')
    ax_a.set_title('A. Time-Resolved Regression (Fronto-Central)',
                    fontweight='bold', loc='left')
    ax_a.legend(loc='upper right', frameon=True, fontsize=8)

    # Panel B: Model comparison (MMN window)
    ax_b = fig.add_subplot(gs[0, 1])
    if 'mmn_amplitude' in enc_results:
        models = []
        delta_aics = []
        colors = []
        for col in surprise_cols:
            if col in enc_results['mmn_amplitude']:
                r = enc_results['mmn_amplitude'][col]
                if 'delta_aic' in r:
                    models.append(MODEL_LABELS[col])
                    delta_aics.append(r['delta_aic'])
                    colors.append(COLORS[col])

        if models:
            bars = ax_b.bar(range(len(models)), delta_aics, color=colors, alpha=0.8)
            ax_b.set_xticks(range(len(models)))
            ax_b.set_xticklabels(models, rotation=30, ha='right')
            ax_b.set_ylabel('ΔAIC (vs. baseline)')
            ax_b.axhline(0, color='black', linewidth=0.5)

            # Add significance stars
            for i, col in enumerate(surprise_cols):
                if col in enc_results['mmn_amplitude']:
                    r = enc_results['mmn_amplitude'][col]
                    if 'lr_pval' in r and r['lr_pval'] < 0.05:
                        ax_b.text(i, delta_aics[i] - 0.5, '*',
                                  ha='center', fontsize=14, fontweight='bold')

    ax_b.set_title('B. Model Comparison (MMN Window)', fontweight='bold', loc='left')

    # Panel C: Model comparison (P3b window)
    ax_c = fig.add_subplot(gs[1, 0])
    if 'p3b_amplitude' in enc_results:
        models = []
        delta_aics = []
        colors = []
        for col in surprise_cols:
            if col in enc_results['p3b_amplitude']:
                r = enc_results['p3b_amplitude'][col]
                if 'delta_aic' in r:
                    models.append(MODEL_LABELS[col])
                    delta_aics.append(r['delta_aic'])
                    colors.append(COLORS[col])

        if models:
            ax_c.bar(range(len(models)), delta_aics, color=colors, alpha=0.8)
            ax_c.set_xticks(range(len(models)))
            ax_c.set_xticklabels(models, rotation=30, ha='right')
            ax_c.set_ylabel('ΔAIC (vs. baseline)')
            ax_c.axhline(0, color='black', linewidth=0.5)

            for i, col in enumerate(surprise_cols):
                if col in enc_results['p3b_amplitude']:
                    r = enc_results['p3b_amplitude'][col]
                    if 'lr_pval' in r and r['lr_pval'] < 0.05:
                        ax_c.text(i, delta_aics[i] - 0.5, '*',
                                  ha='center', fontsize=14, fontweight='bold')

    ax_c.set_title('C. Model Comparison (P3b Window)', fontweight='bold', loc='left')

    # Panel D: Variance explained
    ax_d = fig.add_subplot(gs[1, 1])
    # Compute R² improvement for each model
    for dv_name, dv_label in [('mmn_amplitude', 'MMN'), ('p3b_amplitude', 'P3b')]:
        if dv_name not in enc_results:
            continue
        models_names = []
        r2_improvements = []
        for col in surprise_cols:
            if col in enc_results[dv_name]:
                r = enc_results[dv_name][col]
                if 'llf' in r and 'llf' in enc_results[dv_name].get('baseline', {}):
                    base_llf = enc_results[dv_name]['baseline']['llf']
                    # McFadden's pseudo-R² improvement
                    r2_imp = 1 - r['llf'] / base_llf if base_llf != 0 else 0
                    models_names.append(MODEL_LABELS[col])
                    r2_improvements.append(abs(r2_imp))

    if models_names:
        ax_d.bar(range(len(models_names)), r2_improvements,
                 color=[COLORS[c] for c in surprise_cols[:len(models_names)]], alpha=0.8)
        ax_d.set_xticks(range(len(models_names)))
        ax_d.set_xticklabels(models_names, rotation=30, ha='right')
        ax_d.set_ylabel('ΔPseudo-R² (vs. baseline)')

    ax_d.set_title('D. Variance Explained', fontweight='bold', loc='left')

    fig.savefig(str(FIGURES_DIR / f"fig3_encoding_{task}.pdf"))
    fig.savefig(str(FIGURES_DIR / f"fig3_encoding_{task}.png"))
    plt.close(fig)
    print(f"  Figure 3 saved")


def figure4_decoding(task="MMN"):
    """Figure 4: Decoding benchmark."""
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 3, wspace=0.35)

    dec_file = RESULTS_DIR / "aim3" / f"decoding_results_{task}.json"
    if not dec_file.exists():
        print("  Decoding results not found for Figure 4")
        return

    with open(str(dec_file)) as f:
        dec_results = json.load(f)

    fs_colors = {
        'ERP-only': '#1f77b4',
        'ERP+TF': '#ff7f0e',
        'ERP+surprise': '#2ca02c',
        'Full': '#d62728',
    }

    # Panel A: ROC curves (cross-subject)
    ax_a = fig.add_subplot(gs[0, 0])
    if 'cross_subject' in dec_results:
        for fs_name in ['ERP-only', 'ERP+TF', 'ERP+surprise', 'Full']:
            if fs_name in dec_results['cross_subject']:
                r = dec_results['cross_subject'][fs_name]
                if 'roc_fpr' in r and 'roc_tpr' in r:
                    ax_a.plot(r['roc_fpr'], r['roc_tpr'],
                              color=fs_colors.get(fs_name, 'gray'),
                              linewidth=2,
                              label=f"{fs_name} (AUC={r['roc_auc']:.3f})")

    ax_a.plot([0, 1], [0, 1], 'k--', linewidth=0.5, label='Chance')
    ax_a.set_xlabel('False Positive Rate')
    ax_a.set_ylabel('True Positive Rate')
    ax_a.set_title('A. ROC Curves (Cross-Subject)', fontweight='bold', loc='left')
    ax_a.legend(loc='lower right', frameon=True, fontsize=8)
    ax_a.set_xlim([-0.01, 1.01])
    ax_a.set_ylim([-0.01, 1.01])

    # Panel B: Feature ablation barplot
    ax_b = fig.add_subplot(gs[0, 1])
    if 'cross_subject' in dec_results:
        names = []
        aucs = []
        colors = []
        for fs_name in ['ERP-only', 'ERP+TF', 'ERP+surprise', 'Full']:
            if fs_name in dec_results['cross_subject']:
                r = dec_results['cross_subject'][fs_name]
                names.append(fs_name)
                aucs.append(r['roc_auc'])
                colors.append(fs_colors.get(fs_name, 'gray'))

        if names:
            bars = ax_b.bar(range(len(names)), aucs, color=colors, alpha=0.8)
            ax_b.set_xticks(range(len(names)))
            ax_b.set_xticklabels(names, rotation=30, ha='right')
            ax_b.set_ylabel('ROC-AUC')
            ax_b.axhline(0.5, color='black', linewidth=0.5, linestyle='--',
                          label='Chance')
            ax_b.set_ylim([0.4, 1.0])

    ax_b.set_title('B. Feature Ablation (Cross-Subject)', fontweight='bold', loc='left')

    # Panel C: Within-subject vs cross-subject
    ax_c = fig.add_subplot(gs[0, 2])
    if 'within_subject' in dec_results and 'cross_subject' in dec_results:
        for fs_name in ['ERP-only', 'ERP+surprise']:
            ws = dec_results['within_subject'].get(fs_name, {})
            cs = dec_results['cross_subject'].get(fs_name, {})
            if 'mean_auc' in ws and 'roc_auc' in cs:
                x_pos = ['ERP-only', 'ERP+surprise'].index(fs_name)
                ax_c.bar(x_pos * 2, cs['roc_auc'], 0.8,
                         color=fs_colors[fs_name], alpha=0.6, label=f'{fs_name} (cross)')
                ax_c.bar(x_pos * 2 + 0.9, ws['mean_auc'], 0.8,
                         color=fs_colors[fs_name], alpha=1.0, label=f'{fs_name} (within)')

        ax_c.set_xticks([0.45, 2.45])
        ax_c.set_xticklabels(['ERP-only', 'ERP+surprise'])
        ax_c.set_ylabel('ROC-AUC')
        ax_c.axhline(0.5, color='black', linewidth=0.5, linestyle='--')
        ax_c.legend(loc='upper right', frameon=True, fontsize=7)

    ax_c.set_title('C. Cross vs Within Subject', fontweight='bold', loc='left')

    fig.savefig(str(FIGURES_DIR / f"fig4_decoding_{task}.pdf"))
    fig.savefig(str(FIGURES_DIR / f"fig4_decoding_{task}.png"))
    plt.close(fig)
    print(f"  Figure 4 saved")


def make_all_figures(task="MMN"):
    """Generate all figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating figures for task {task}...")
    figure1_schematic(task)
    figure2_erp_replication(task)
    figure3_encoding(task)
    figure4_decoding(task)
    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "MMN"
    make_all_figures(task)
