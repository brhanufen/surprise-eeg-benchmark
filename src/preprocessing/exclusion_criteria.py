#!/usr/bin/env python3
"""Subject exclusion criteria based on QC metrics.

Reads qc_summary_MMN.csv and qc_summary_P3.csv from results/aim1/,
applies minimum epoch count thresholds, and saves lists of excluded
subjects.

Thresholds
----------
- MMN: minimum 100 epochs after rejection
- P3:  minimum 30 epochs after rejection (fewer trials by design)
"""

import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "aim1"

# Minimum epoch counts after artifact rejection
MIN_EPOCHS = {
    "MMN": 100,
    "P3": 30,
}


def identify_exclusions(task: str) -> pd.DataFrame:
    """Identify subjects that fail QC for a given task.

    Parameters
    ----------
    task : str
        "MMN" or "P3".

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [subject, task, n_epochs_after_rejection,
        rejection_rate, reason] for each excluded subject.
    """
    qc_file = RESULTS_DIR / f"qc_summary_{task}.csv"
    if not qc_file.exists():
        raise FileNotFoundError(f"QC summary not found: {qc_file}")

    qc = pd.read_csv(qc_file)
    min_epochs = MIN_EPOCHS[task]

    excluded_rows = []
    for _, row in qc.iterrows():
        reasons = []
        n_after = row["n_epochs_after_rejection"]
        rej_rate = row["rejection_rate"]

        if n_after < min_epochs:
            reasons.append(
                f"epoch count {int(n_after)} < minimum {min_epochs}"
            )

        if rej_rate > 0.80:
            reasons.append(f"rejection rate {rej_rate:.1%} > 80%")

        if reasons:
            excluded_rows.append(
                {
                    "subject": row["subject"],
                    "task": task,
                    "n_epochs_after_rejection": int(n_after),
                    "rejection_rate": round(rej_rate, 4),
                    "reason": "; ".join(reasons),
                }
            )

    return pd.DataFrame(excluded_rows)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for task in ("MMN", "P3"):
        print(f"\n{'='*60}")
        print(f"Exclusion check: {task}  (min epochs = {MIN_EPOCHS[task]})")
        print(f"{'='*60}")

        try:
            excl = identify_exclusions(task)
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            continue

        if excl.empty:
            print("  No subjects excluded.")
        else:
            print(f"  {len(excl)} subject(s) excluded:")
            for _, row in excl.iterrows():
                print(f"    {row['subject']}: {row['reason']}")

        out_file = RESULTS_DIR / f"excluded_subjects_{task}.csv"
        excl.to_csv(out_file, index=False)
        print(f"  Saved: {out_file}")


if __name__ == "__main__":
    main()
