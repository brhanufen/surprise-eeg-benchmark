#!/usr/bin/env python3
"""Aim 3: Decoding benchmark — does surprise improve classification?

Implements:
  - Binary classification: standard vs. deviant
  - Feature sets: (i) ERP-only, (ii) ERP+TF, (iii) ERP+surprise, (iv) full
  - Cross-subject CV (PRIMARY): leave-5-subjects-out
  - Within-subject CV (SECONDARY): 5-fold stratified
  - Metrics: ROC-AUC, PR-AUC, balanced accuracy, calibration
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, roc_curve,
                             precision_recall_curve, brier_score_loss)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REGRESSORS_DIR = PROJECT_DIR / "data" / "regressors"
RESULTS_DIR = PROJECT_DIR / "results" / "aim3"


# Feature set definitions
SURPRISE_COLS = ['static_shannon', 'adaptive_shannon_w20',
                 'bayesian_surprise', 'changepoint_surprise']
ERP_COLS = ['mmn_amplitude', 'p3b_amplitude']
TF_COLS = ['theta_power', 'delta_power']


def residualize_surprise(df, surprise_cols):
    """Residualize surprise regressors against stimulus type.

    Static Shannon and adaptive Shannon are deterministic functions of
    stimulus type (standard=0/deviant=1). Including them raw in a classifier
    that predicts stimulus type leads to perfect (trivial) classification.

    For a meaningful test of H2, we use CONTEXTUAL surprise: the
    trial-to-trial variation in surprise BEYOND what stimulus type alone
    predicts. This is computed by regressing each surprise regressor on
    stimulus type and using the residuals.

    Bayesian and change-point surprise vary within stimulus class, so
    their residuals retain meaningful variance. Static Shannon residuals
    will be near-zero (as expected).
    """
    from scipy.stats import linregress
    residualized = {}
    for col in surprise_cols:
        if col not in df.columns:
            continue
        x = df['stimulus'].values.astype(float)
        y = df[col].values
        slope, intercept, _, _, _ = linregress(x, y)
        resid = y - (slope * x + intercept)
        residualized[f'{col}_resid'] = resid
    return residualized


def build_feature_sets(df):
    """Build the four feature sets for ablation study.

    Surprise features are residualized against stimulus type to prevent
    trivial label leakage (see blueprint v4, Section 5.6).
    """
    feature_sets = {}

    # (i) ERP-only
    erp_cols = [c for c in ERP_COLS if c in df.columns]
    if erp_cols:
        feature_sets['ERP-only'] = erp_cols

    # (ii) ERP + time-frequency
    tf_cols = [c for c in TF_COLS if c in df.columns]
    if erp_cols and tf_cols:
        feature_sets['ERP+TF'] = erp_cols + tf_cols

    # (iii) ERP + contextual surprise (residualized)
    surp_resid_cols = [f'{c}_resid' for c in SURPRISE_COLS if f'{c}_resid' in df.columns]
    if erp_cols and surp_resid_cols:
        feature_sets['ERP+surprise'] = erp_cols + surp_resid_cols

    # (iv) Full model
    all_cols = erp_cols + tf_cols + surp_resid_cols
    all_cols = [c for c in all_cols if c in df.columns]
    if all_cols:
        feature_sets['Full'] = all_cols

    return feature_sets


def cross_subject_cv(df, feature_cols, n_leave_out=5, seed=42):
    """Leave-N-subjects-out cross-validation (PRIMARY evaluation).

    Parameters
    ----------
    df : DataFrame with columns: subject, stimulus, and feature columns
    feature_cols : list of feature column names
    n_leave_out : number of subjects to leave out per fold
    seed : random seed

    Returns
    -------
    results : dict with metrics
    """
    subjects = df['subject'].unique()
    n_subjects = len(subjects)

    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)

    # Create folds
    n_folds = max(1, n_subjects // n_leave_out)
    folds = np.array_split(subjects, n_folds)

    all_y_true = []
    all_y_prob = []
    all_y_pred = []
    fold_aucs = []

    for fold_idx, test_subjects in enumerate(folds):
        train_mask = ~df['subject'].isin(test_subjects)
        test_mask = df['subject'].isin(test_subjects)

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, 'stimulus'].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, 'stimulus'].values

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        # Pipeline: scale + logistic regression
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=seed,
                                       class_weight='balanced'))
        ])
        pipe.fit(X_train, y_train)

        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = pipe.predict(X_test)

        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob)
        all_y_pred.extend(y_pred)

        fold_auc = roc_auc_score(y_test, y_prob)
        fold_aucs.append(fold_auc)

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_y_pred = np.array(all_y_pred)

    results = {
        'roc_auc': float(roc_auc_score(all_y_true, all_y_prob)),
        'pr_auc': float(average_precision_score(all_y_true, all_y_prob)),
        'balanced_accuracy': float(balanced_accuracy_score(all_y_true, all_y_pred)),
        'brier_score': float(brier_score_loss(all_y_true, all_y_prob)),
        'fold_aucs': [float(a) for a in fold_aucs],
        'fold_auc_mean': float(np.mean(fold_aucs)),
        'fold_auc_std': float(np.std(fold_aucs)),
        'n_folds': len(folds),
        'n_test_subjects_per_fold': n_leave_out,
    }

    # ROC curve points
    fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
    results['roc_fpr'] = fpr.tolist()
    results['roc_tpr'] = tpr.tolist()

    # PR curve points
    precision, recall, _ = precision_recall_curve(all_y_true, all_y_prob)
    results['pr_precision'] = precision.tolist()
    results['pr_recall'] = recall.tolist()

    return results


def within_subject_cv(df, feature_cols, n_folds=5, seed=42):
    """Within-subject 5-fold stratified CV (SECONDARY evaluation).

    Returns per-subject AUC values.
    """
    subjects = df['subject'].unique()
    subject_results = {}

    for sub in subjects:
        sub_df = df[df['subject'] == sub]
        X = sub_df[feature_cols].values
        y = sub_df['stimulus'].values

        if len(np.unique(y)) < 2 or len(y) < 20:
            continue

        try:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            y_true_all = []
            y_prob_all = []

            for train_idx, test_idx in skf.split(X, y):
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', LogisticRegression(C=1.0, max_iter=1000,
                                               random_state=seed,
                                               class_weight='balanced'))
                ])
                pipe.fit(X[train_idx], y[train_idx])
                y_prob = pipe.predict_proba(X[test_idx])[:, 1]
                y_true_all.extend(y[test_idx])
                y_prob_all.extend(y_prob)

            y_true_all = np.array(y_true_all)
            y_prob_all = np.array(y_prob_all)

            subject_results[sub] = {
                'roc_auc': float(roc_auc_score(y_true_all, y_prob_all)),
                'pr_auc': float(average_precision_score(y_true_all, y_prob_all)),
                'n_trials': len(y),
            }
        except Exception as e:
            subject_results[sub] = {'error': str(e)}

    # Aggregate
    valid_aucs = [r['roc_auc'] for r in subject_results.values()
                  if 'roc_auc' in r]

    aggregate = {
        'mean_auc': float(np.mean(valid_aucs)) if valid_aucs else 0,
        'std_auc': float(np.std(valid_aucs)) if valid_aucs else 0,
        'median_auc': float(np.median(valid_aucs)) if valid_aucs else 0,
        'n_subjects': len(valid_aucs),
        'per_subject': subject_results,
    }

    return aggregate


def run_full_decoding_analysis(task="MMN"):
    """Run the complete Aim 3 decoding analysis."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load the analysis dataframe from Aim 2
    analysis_file = PROJECT_DIR / "results" / "aim2" / f"analysis_df_{task}.csv"
    if not analysis_file.exists():
        print(f"Analysis dataframe not found: {analysis_file}")
        print("Run Aim 2 encoding analysis first.")
        return

    df = pd.read_csv(str(analysis_file))
    print(f"Loaded analysis dataframe: {df.shape}")
    print(f"Subjects: {df['subject'].nunique()}")
    print(f"Standards: {sum(df['stimulus']==0)}, Deviants: {sum(df['stimulus']==1)}")

    # Residualize surprise regressors against stimulus type
    # This prevents trivial label leakage (surprise = f(stimulus_type))
    surp_cols = [c for c in SURPRISE_COLS if c in df.columns]
    residuals = residualize_surprise(df, surp_cols)
    for col, vals in residuals.items():
        df[col] = vals
    print(f"Residualized surprise columns: {list(residuals.keys())}")

    # Try to add time-frequency features
    tf_file = PROJECT_DIR / "results" / "aim2" / f"tf_features_{task}.csv"
    if tf_file.exists():
        tf_df = pd.read_csv(str(tf_file))
        for col in TF_COLS:
            if col in tf_df.columns:
                df[col] = tf_df[col]

    # Build feature sets
    feature_sets = build_feature_sets(df)
    print(f"\nFeature sets:")
    for name, cols in feature_sets.items():
        print(f"  {name}: {cols}")

    all_results = {}

    # Cross-subject CV (PRIMARY)
    print(f"\n{'='*60}")
    print("PRIMARY: Cross-Subject Leave-5-Out CV")
    print(f"{'='*60}")

    cross_sub_results = {}
    for fs_name, fs_cols in feature_sets.items():
        print(f"\n  Feature set: {fs_name}")
        results = cross_subject_cv(df, fs_cols, n_leave_out=5)
        cross_sub_results[fs_name] = results
        print(f"    ROC-AUC: {results['roc_auc']:.3f} "
              f"(fold mean: {results['fold_auc_mean']:.3f} ± {results['fold_auc_std']:.3f})")
        print(f"    PR-AUC: {results['pr_auc']:.3f}")
        print(f"    Balanced accuracy: {results['balanced_accuracy']:.3f}")
        print(f"    Brier score: {results['brier_score']:.3f}")

    all_results['cross_subject'] = cross_sub_results

    # Within-subject CV (SECONDARY)
    print(f"\n{'='*60}")
    print("SECONDARY: Within-Subject 5-Fold Stratified CV")
    print(f"{'='*60}")

    within_sub_results = {}
    for fs_name, fs_cols in feature_sets.items():
        print(f"\n  Feature set: {fs_name}")
        results = within_subject_cv(df, fs_cols)
        within_sub_results[fs_name] = results
        print(f"    Mean AUC: {results['mean_auc']:.3f} ± {results['std_auc']:.3f}")
        print(f"    Median AUC: {results['median_auc']:.3f}")
        print(f"    N subjects: {results['n_subjects']}")

    all_results['within_subject'] = within_sub_results

    # Statistical comparison: ERP-only vs ERP+surprise (key test for H2)
    print(f"\n{'='*60}")
    print("KEY TEST (H2): ERP-only vs ERP+surprise")
    print(f"{'='*60}")

    if 'ERP-only' in cross_sub_results and 'ERP+surprise' in cross_sub_results:
        erp_aucs = cross_sub_results['ERP-only']['fold_aucs']
        surp_aucs = cross_sub_results['ERP+surprise']['fold_aucs']

        # Paired test on fold AUCs
        if len(erp_aucs) == len(surp_aucs) and len(erp_aucs) > 1:
            from scipy.stats import wilcoxon, ttest_rel
            try:
                t_stat, t_pval = ttest_rel(surp_aucs, erp_aucs)
                print(f"  Paired t-test: t={t_stat:.3f}, p={t_pval:.4f}")
            except Exception:
                pass
            try:
                w_stat, w_pval = wilcoxon(np.array(surp_aucs) - np.array(erp_aucs))
                print(f"  Wilcoxon test: W={w_stat:.3f}, p={w_pval:.4f}")
            except Exception:
                pass

        delta_auc = (cross_sub_results['ERP+surprise']['roc_auc'] -
                     cross_sub_results['ERP-only']['roc_auc'])
        print(f"  ΔAUC (surprise - ERP-only): {delta_auc:+.4f}")

    all_results['task'] = task
    all_results['n_subjects'] = int(df['subject'].nunique())
    all_results['n_trials'] = int(len(df))

    # Save results
    output_file = RESULTS_DIR / f"decoding_results_{task}.json"
    with open(str(output_file), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll Aim 3 results saved to {output_file}")
    return all_results


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "MMN"
    run_full_decoding_analysis(task)
