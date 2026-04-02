#!/usr/bin/env python3
"""Generate tables for the manuscript as formatted text files."""

import json
import pandas as pd
import numpy as np
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
RESULTS = PROJ / "results"
OUT = PROJ / "manuscript"


def table1_preprocessing():
    """Table 1: Preprocessing parameters."""
    rows = [
        ["Parameter", "Specification"],
        ["High-pass filter", "0.1 Hz (zero-phase FIR, Hamming window)"],
        ["Low-pass filter", "30 Hz (zero-phase FIR, Hamming window)"],
        ["Sampling rate", "256 Hz (downsampled from 1024 Hz)"],
        ["Reference", "Average reference"],
        ["Artifact rejection (ICA)", "FastICA, 15 components, automatic EOG detection"],
        ["Artifact rejection (epochs)", "Amplitude threshold: ±150 µV"],
        ["Epoch window", "−200 to 800 ms (stimulus-locked)"],
        ["Baseline correction", "−200 to 0 ms"],
        ["MMN ROI channels", "Fz, FCz, Cz, FC3, FC4"],
        ["P3b ROI channels", "Pz, CPz, P3, P4, CP1, CP2"],
    ]
    return rows


def table2_encoding(task="MMN"):
    """Table 2: Encoding model comparison results."""
    enc_file = RESULTS / "aim2" / f"encoding_results_{task}.json"
    corr_file = RESULTS / "aim2" / f"encoding_results_corrected_{task}.json"

    # Try corrected first, fall back to original
    use_file = corr_file if corr_file.exists() else enc_file
    if not use_file.exists():
        return None

    with open(str(use_file)) as f:
        enc = json.load(f)

    rows = [["Model", "Window", "ΔAIC", "ΔBIC", "LRT p", "p (corrected)", "β", "β 95% CI"]]

    for dv, window_name in [("mmn_amplitude", "MMN"), ("p3b_amplitude", "P3b")]:
        if dv not in enc:
            continue
        for model in ["static_shannon", "adaptive_shannon_w20",
                       "bayesian_surprise", "changepoint_surprise"]:
            if model not in enc[dv] or "error" in enc[dv][model]:
                continue
            r = enc[dv][model]

            model_label = {
                "static_shannon": "Static Shannon",
                "adaptive_shannon_w20": "Adaptive Shannon (w=20)",
                "bayesian_surprise": "Bayesian",
                "changepoint_surprise": "Change-point",
            }[model]

            daic = f"{r.get('delta_aic', 'N/A'):.1f}"
            dbic = f"{r.get('delta_bic', 'N/A'):.1f}"
            pval = f"{r.get('lr_pval', float('nan')):.4f}"
            pcorr = f"{r.get('lr_pval_corrected', r.get('lr_pval', float('nan'))):.4f}"
            beta = f"{r.get('beta', float('nan')):.6f}"

            # CI
            b = r.get('beta', 0)
            se = r.get('beta_se', float('nan'))
            if not np.isnan(se) and se > 0:
                ci = f"[{b - 1.96*se:.6f}, {b + 1.96*se:.6f}]"
            else:
                ci = "N/A"

            rows.append([model_label, window_name, daic, dbic, pval, pcorr, beta, ci])

    return rows


def table3_decoding(task="MMN"):
    """Table 3: Decoding results."""
    dec_file = RESULTS / "aim3" / f"decoding_results_{task}.json"
    if not dec_file.exists():
        return None

    with open(str(dec_file)) as f:
        dec = json.load(f)

    rows = [["Feature Set", "Evaluation", "ROC-AUC", "PR-AUC", "Balanced Acc.", "Brier Score"]]

    for fs_name in ["ERP-only", "ERP+TF", "ERP+surprise", "Full"]:
        # Cross-subject
        if "cross_subject" in dec and fs_name in dec["cross_subject"]:
            r = dec["cross_subject"][fs_name]
            rows.append([
                fs_name, "Cross-subject",
                f"{r['roc_auc']:.3f}", f"{r['pr_auc']:.3f}",
                f"{r['balanced_accuracy']:.3f}", f"{r['brier_score']:.3f}"
            ])
        # Within-subject
        if "within_subject" in dec and fs_name in dec["within_subject"]:
            r = dec["within_subject"][fs_name]
            rows.append([
                fs_name, "Within-subject",
                f"{r['mean_auc']:.3f} ± {r['std_auc']:.3f}", "—", "—", "—"
            ])

    return rows


def format_table(rows, title=""):
    """Format a table as text."""
    if not rows:
        return "Table not available.\n"

    # Compute column widths
    n_cols = len(rows[0])
    widths = [max(len(str(row[i])) for row in rows) for i in range(n_cols)]

    lines = []
    if title:
        lines.append(title)
        lines.append("=" * sum(widths + [3 * (n_cols - 1)]))

    for i, row in enumerate(rows):
        line = " | ".join(str(row[j]).ljust(widths[j]) for j in range(n_cols))
        lines.append(line)
        if i == 0:
            lines.append("-+-".join("-" * w for w in widths))

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(format_table(table1_preprocessing(), "Table 1. Preprocessing Parameters"))
    print()

    t2 = table2_encoding("MMN")
    if t2:
        print(format_table(t2, "Table 2. Encoding Model Comparison"))
    print()

    t3 = table3_decoding("MMN")
    if t3:
        print(format_table(t3, "Table 3. Decoding Results (MMN Paradigm)"))

    # Save tables
    with open(str(OUT / "tables.txt"), "w") as f:
        f.write(format_table(table1_preprocessing(), "Table 1. Preprocessing Parameters"))
        f.write("\n\n")
        if t2:
            f.write(format_table(t2, "Table 2. Encoding Model Comparison"))
        f.write("\n\n")
        if t3:
            f.write(format_table(t3, "Table 3. Decoding Results (MMN Paradigm)"))

    print("\nTables saved to manuscript/tables.txt")
