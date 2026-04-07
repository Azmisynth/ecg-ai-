"""
preprocess.py — Clean ECG signals, extract median beats, and normalise.

Pipeline per record:
  1. Load raw 12-lead signal (100 Hz, 10 s → 1000 samples per lead).
  2. Per-lead: bandpass filter + baseline wander removal via NeuroKit2.
  3. Detect R-peaks on lead II; segment individual beats.
  4. Compute the median beat across all detected beats (robust to noise).
  5. Z-score normalise each lead's median beat independently.

Output: numpy array of shape (12, BEAT_LEN) per record.
"""

import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk
from pathlib import Path
from typing import Optional

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ptbxl_small"

# ── constants ────────────────────────────────────────────────────────────────
FS = 100           # sampling rate (Hz)
BEAT_LEN = 100     # samples per median beat window (1 s at 100 Hz)
HALF = BEAT_LEN // 2

LEAD_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


# ── per-lead cleaning ─────────────────────────────────────────────────────────
def clean_lead(signal: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Apply NeuroKit2 ECG cleaning to a single-lead signal.
    Uses a 0.5–40 Hz bandpass + powerline filter.
    """
    cleaned = nk.ecg_clean(signal, sampling_rate=fs, method="biosppy")
    return cleaned.astype(np.float32)


# ── R-peak detection ──────────────────────────────────────────────────────────
def detect_rpeaks(signal: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Detect R-peaks in a cleaned single-lead signal.
    Returns array of sample indices.
    """
    _, info = nk.ecg_peaks(signal, sampling_rate=fs, method="neurokit")
    peaks = info["ECG_R_Peaks"]
    return peaks.astype(int)


# ── median beat extraction ────────────────────────────────────────────────────
def median_beat(signal: np.ndarray, rpeaks: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract a fixed-length window around each R-peak and return the
    element-wise median across all beats.

    Returns None if fewer than 2 valid beats are found.
    """
    beats = []
    for r in rpeaks:
        start = r - HALF
        end = r + HALF
        if start < 0 or end > len(signal):
            continue
        beats.append(signal[start:end])

    if len(beats) < 2:
        return None

    beats_arr = np.stack(beats, axis=0)   # (n_beats, BEAT_LEN)
    return np.median(beats_arr, axis=0).astype(np.float32)


# ── z-score normalisation ─────────────────────────────────────────────────────
def normalise(beat: np.ndarray) -> np.ndarray:
    """Z-score normalise a 1-D beat array. Returns zeros if std == 0."""
    std = beat.std()
    if std == 0:
        return np.zeros_like(beat)
    return ((beat - beat.mean()) / std).astype(np.float32)


# ── full record pipeline ──────────────────────────────────────────────────────
def process_record(record_path: Path) -> Optional[np.ndarray]:
    """
    Run the full preprocessing pipeline on one WFDB record.

    Returns:
        np.ndarray of shape (12, BEAT_LEN), or None on failure.
    """
    try:
        rec = wfdb.rdrecord(str(record_path))
    except Exception as e:
        print(f"  [error] Could not read {record_path}: {e}")
        return None

    raw = rec.p_signal  # (1000, 12) — samples × leads
    if raw is None or raw.shape[1] < 12:
        return None

    # Use lead II (index 1) for R-peak detection across all leads
    lead_ii = clean_lead(raw[:, 1])
    try:
        rpeaks = detect_rpeaks(lead_ii)
    except Exception as e:
        print(f"  [warn] R-peak detection failed for {record_path.name}: {e}")
        return None

    beats_per_lead = []
    for lead_idx in range(12):
        cleaned = clean_lead(raw[:, lead_idx])
        beat = median_beat(cleaned, rpeaks)
        if beat is None:
            # Fallback: use central BEAT_LEN samples of cleaned signal
            mid = len(cleaned) // 2
            beat = cleaned[mid - HALF: mid + HALF]
        beats_per_lead.append(normalise(beat))

    return np.stack(beats_per_lead, axis=0)  # (12, BEAT_LEN)


# ── batch processing ──────────────────────────────────────────────────────────
def build_dataset(
    df: pd.DataFrame,
    data_dir: Path = DATA_DIR,
    n: int = 100,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Process *n* records and return:
        X        — float32 array (N, 12, BEAT_LEN)
        y        — int array    (N,)  binary label: 1=NORM, 0=abnormal
        ecg_ids  — list of ecg_id values that succeeded
    """
    # Build a simple binary label from SCP codes
    scp_df = pd.read_csv(data_dir / "scp_statements.csv", index_col=0)
    norm_codes = set(scp_df[scp_df["diagnostic_class"] == "NORM"].index.tolist())
    # Also add the literal NORM code
    norm_codes.add("NORM")

    subset = df.head(n)
    X_list, y_list, ids = [], [], []

    for ecg_id, row in subset.iterrows():
        record_path = data_dir / row["filename_lr"]
        features = process_record(record_path)
        if features is None:
            continue

        # Label: 1 if any NORM code present, else 0
        codes = row["scp_codes"]  # dict already eval'd in load_data.py
        label = int(bool(set(codes.keys()) & norm_codes))

        X_list.append(features)
        y_list.append(label)
        ids.append(ecg_id)

        if len(ids) % 10 == 0:
            print(f"  Processed {len(ids)} / {n} records …")

    X = np.stack(X_list, axis=0).astype(np.float32)  # (N, 12, BEAT_LEN)
    y = np.array(y_list, dtype=np.int64)
    print(f"  Dataset: {X.shape}, labels distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y, ids


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from load_data import download_metadata

    print("=== ECG Preprocessing ===")
    df = download_metadata()
    X, y, ids = build_dataset(df, n=100)

    out_dir = DATA_DIR.parent
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    print(f"\nSaved X.npy {X.shape} and y.npy {y.shape} to {out_dir}")
    print("Proceed to model.py / train.py when ready.")
