#!/usr/bin/env python3
"""EEG preprocessing pipeline for ERP CORE data.

Follows the v4 blueprint specifications:
  - High-pass: 0.1 Hz (zero-phase FIR)
  - Low-pass: 30 Hz (zero-phase FIR)
  - Reference: Average reference
  - Artifact rejection: ICA for eye artifacts; autoreject for epoch rejection
  - Epoch window: -200 to 800 ms
  - Baseline correction: -200 to 0 ms
  - Downsample to 256 Hz if needed
"""

import os
import sys
import json
import numpy as np
import mne
from pathlib import Path

# Suppress excessive MNE output
mne.set_log_level('WARNING')

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results" / "aim1"


def find_subject_ids(task="MMN"):
    """Find all available subject IDs for a given task."""
    subjects = []
    for d in sorted(RAW_DIR.iterdir()):
        if d.is_dir() and d.name.startswith("sub-"):
            ses_dir = d / f"ses-{task}" / "eeg"
            set_file = ses_dir / f"{d.name}_ses-{task}_task-{task}_eeg.set"
            if set_file.exists():
                subjects.append(d.name)
    return subjects


def load_raw(subject_id, task="MMN"):
    """Load raw EEG data from EEGLAB .set format."""
    ses_dir = RAW_DIR / subject_id / f"ses-{task}" / "eeg"
    set_file = ses_dir / f"{subject_id}_ses-{task}_task-{task}_eeg.set"

    if not set_file.exists():
        raise FileNotFoundError(f"File not found: {set_file}")

    raw = mne.io.read_raw_eeglab(str(set_file), preload=True)
    return raw


def load_events_from_tsv(subject_id, task="MMN"):
    """Load events from BIDS events.tsv file."""
    ses_dir = RAW_DIR / subject_id / f"ses-{task}" / "eeg"
    events_file = ses_dir / f"{subject_id}_ses-{task}_task-{task}_events.tsv"

    if not events_file.exists():
        return None

    import pandas as pd
    df = pd.read_csv(events_file, sep='\t')
    return df


def preprocess_subject(subject_id, task="MMN"):
    """Run full preprocessing pipeline for one subject.

    Returns
    -------
    epochs : mne.Epochs
        Cleaned epochs.
    event_sequence : np.ndarray
        Binary stimulus sequence (0=standard, 1=deviant) for all retained epochs.
    qc : dict
        Quality control metrics.
    """
    print(f"  Loading raw data...")
    raw = load_raw(subject_id, task)

    # Set standard 10-20 montage for channel positions
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='warn')
    except Exception as e:
        print(f"  Montage warning: {e}")

    # Get original sampling rate
    orig_sfreq = raw.info['sfreq']
    print(f"  Original sfreq: {orig_sfreq} Hz, n_channels: {len(raw.ch_names)}")

    # 1. Filter: bandpass 0.1 - 30 Hz (zero-phase FIR)
    print(f"  Filtering (0.1 - 30 Hz)...")
    raw.filter(l_freq=0.1, h_freq=30.0, method='fir', phase='zero-double',
               fir_window='hamming', fir_design='firwin')

    # 2. Downsample to 256 Hz if needed
    if raw.info['sfreq'] > 256:
        print(f"  Downsampling to 256 Hz...")
        raw.resample(256)

    # 3. Set average reference
    print(f"  Setting average reference...")
    raw.set_eeg_reference('average', projection=False)

    # 4. Find events
    print(f"  Finding events...")
    events = mne.events_from_annotations(raw)
    events_array = events[0]
    event_id_map = events[1]

    print(f"  Event IDs found: {event_id_map}")

    # Define event IDs based on task
    # ERP CORE event codes (verified from task-*_events.json):
    #   MMN: 80=standard, 70=deviant, 180=first standards stream
    #   P3:  11-55 are stimuli (tens=block target letter, units=trial stimulus)
    #        Target = tens digit matches units digit (11,22,33,44,55)
    #        Non-target = all other combinations
    #        201=correct response, 202=error response (not stimuli)

    if task == "MMN":
        event_id = {}
        for name, code in event_id_map.items():
            try:
                num = int(name)
                if num == 80:
                    event_id['standard'] = code
                elif num == 70:
                    event_id['deviant'] = code
                # Skip 180 (first standards) and 1 (STATUS)
            except ValueError:
                pass

    elif task == "P3":
        event_id = {}
        for name, code in event_id_map.items():
            try:
                num = int(name)
                if 11 <= num <= 55 and num % 10 != 0:
                    tens = num // 10
                    units = num % 10
                    if tens == units:
                        event_id[f'deviant/{name}'] = code  # Target
                    else:
                        event_id[f'standard/{name}'] = code  # Non-target
                # Skip 201 (correct response) and 202 (error response)
            except ValueError:
                pass

    print(f"  Using event_id: {event_id}")

    if not event_id:
        print(f"  WARNING: Could not determine event IDs! Available: {event_id_map}")
        return None, None, None

    # 5. Epoch the data
    print(f"  Epoching (-0.2 to 0.8 s)...")
    try:
        epochs = mne.Epochs(raw, events_array, event_id=event_id,
                           tmin=-0.2, tmax=0.8,
                           baseline=(-0.2, 0.0),
                           preload=True,
                           reject=None,  # Will use autoreject instead
                           detrend=None)
    except Exception as e:
        print(f"  ERROR during epoching: {e}")
        return None, None, None

    n_epochs_before = len(epochs)
    print(f"  {n_epochs_before} epochs created")

    # 6. ICA for eye artifact removal
    print(f"  Running ICA...")
    try:
        ica = mne.preprocessing.ICA(n_components=15, method='fastica',
                                     random_state=42, max_iter=500)
        ica.fit(epochs)

        # Auto-detect EOG components
        # Try to find EOG channels
        eog_chs = [ch for ch in raw.ch_names if 'eog' in ch.lower() or
                    ch.lower() in ['veog', 'heog', 'vp1', 'vp2', 'hp1', 'hp2',
                                   'fp1', 'fp2', 'fpz']]

        if eog_chs:
            eog_indices, eog_scores = ica.find_bads_eog(epochs, ch_name=eog_chs[0],
                                                         threshold=3.0)
        else:
            # Use frontal channels as proxy for EOG
            eog_indices, eog_scores = ica.find_bads_eog(epochs, threshold=3.0)

        if eog_indices:
            ica.exclude = eog_indices[:2]  # Remove at most 2 components
            print(f"  Removing {len(ica.exclude)} ICA components: {ica.exclude}")
            ica.apply(epochs)
        else:
            print(f"  No EOG components detected, skipping ICA artifact removal")
    except Exception as e:
        print(f"  ICA failed: {e}, proceeding without ICA")

    # 7. Autoreject for epoch rejection
    print(f"  Running autoreject...")
    try:
        from autoreject import AutoReject
        ar = AutoReject(n_interpolate=[1, 4, 8], random_state=42, n_jobs=1,
                        verbose=False)
        epochs_clean = ar.fit_transform(epochs)
        n_rejected = n_epochs_before - len(epochs_clean)
        print(f"  Autoreject: {n_rejected} epochs rejected, {len(epochs_clean)} remaining")
        epochs = epochs_clean
    except Exception as e:
        print(f"  Autoreject failed: {e}")
        # Fallback: simple amplitude threshold
        epochs.drop_bad(reject=dict(eeg=150e-6))
        print(f"  Fallback rejection: {len(epochs)} epochs remaining")

    # 8. Build binary stimulus sequence for retained epochs
    event_sequence = np.zeros(len(epochs), dtype=int)
    for i, event_code in enumerate(epochs.events[:, 2]):
        is_deviant = False
        for key, val in event_id.items():
            if val == event_code and ('deviant' in key.lower() or 'target' in key.lower()):
                is_deviant = True
                break
        event_sequence[i] = 1 if is_deviant else 0

    # QC metrics
    n_standards = np.sum(event_sequence == 0)
    n_deviants = np.sum(event_sequence == 1)
    qc = {
        'subject': subject_id,
        'task': task,
        'n_epochs_before_rejection': n_epochs_before,
        'n_epochs_after_rejection': len(epochs),
        'n_rejected': n_epochs_before - len(epochs),
        'rejection_rate': (n_epochs_before - len(epochs)) / n_epochs_before,
        'n_standards': int(n_standards),
        'n_deviants': int(n_deviants),
        'sfreq': epochs.info['sfreq'],
        'n_channels': len(epochs.ch_names),
    }

    print(f"  QC: {n_standards} standards, {n_deviants} deviants, "
          f"{qc['rejection_rate']:.1%} rejected")

    return epochs, event_sequence, qc


def preprocess_all(task="MMN"):
    """Preprocess all subjects for a given task."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    subjects = find_subject_ids(task)
    print(f"Found {len(subjects)} subjects for task {task}")

    all_qc = []
    for i, sub_id in enumerate(subjects):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(subjects)}] Processing {sub_id} - {task}")
        print(f"{'='*60}")

        try:
            epochs, event_sequence, qc = preprocess_subject(sub_id, task)

            if epochs is None:
                print(f"  SKIPPED: preprocessing failed")
                continue

            # Save processed epochs
            out_file = PROCESSED_DIR / f"{sub_id}_task-{task}_epo.fif"
            epochs.save(str(out_file), overwrite=True)
            print(f"  Saved: {out_file.name}")

            # Save event sequence
            seq_file = PROCESSED_DIR / f"{sub_id}_task-{task}_sequence.npy"
            np.save(str(seq_file), event_sequence)

            all_qc.append(qc)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save QC summary
    if all_qc:
        import pandas as pd
        qc_df = pd.DataFrame(all_qc)
        qc_file = RESULTS_DIR / f"qc_summary_{task}.csv"
        qc_df.to_csv(str(qc_file), index=False)
        print(f"\n{'='*60}")
        print(f"QC Summary saved to {qc_file}")
        print(qc_df.to_string())

    return all_qc


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "MMN"
    preprocess_all(task)
