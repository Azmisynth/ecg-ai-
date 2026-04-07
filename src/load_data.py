"""
load_data.py — Download a 100-record subset of PTB-XL from PhysioNet.

PTB-XL full dataset is ~2.6 GB. This script downloads only the first
100 records (plus the metadata files) by fetching individual files via
the PhysioNet wfdb API so nothing large hits disk unnecessarily.
"""

import os
import wfdb
import pandas as pd
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ptbxl_small"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PHYSIONET_DB = "ptb-xl"   # wfdb.dl_files appends the version automatically
N_RECORDS = 100          # how many ECG records to fetch
SAMPLING_RATE = 100      # use 100 Hz version (smaller files than 500 Hz)


def download_metadata() -> pd.DataFrame:
    """Download and save the PTB-XL metadata CSV files."""
    meta_files = ["ptbxl_database.csv", "scp_statements.csv"]
    for fname in meta_files:
        dest = DATA_DIR / fname
        if dest.exists():
            print(f"  [skip] {fname} already exists")
            continue
        print(f"  Downloading {fname} …")
        wfdb.dl_files(
            db=PHYSIONET_DB,
            dl_dir=str(DATA_DIR),
            files=[fname],
        )
    df = pd.read_csv(DATA_DIR / "ptbxl_database.csv", index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(lambda x: eval(x))
    return df


def download_records(df: pd.DataFrame, n: int = N_RECORDS) -> list[Path]:
    """Download the first *n* ECG records (100 Hz, .dat + .hea pairs)."""
    subset = df.head(n)
    downloaded: list[Path] = []

    for ecg_id, row in subset.iterrows():
        # filename_lr  →  100 Hz path, e.g. "records100/00000/00001_lr"
        rel_path: str = row["filename_lr"]
        for ext in (".dat", ".hea"):
            fname = rel_path + ext
            dest = DATA_DIR / fname
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            wfdb.dl_files(
                db=PHYSIONET_DB,
                dl_dir=str(DATA_DIR),
                files=[fname],
            )
        downloaded.append(DATA_DIR / rel_path)
        if (len(downloaded)) % 10 == 0:
            print(f"  Downloaded {len(downloaded)}/{n} records …")

    print(f"  Done — {len(downloaded)} records in {DATA_DIR}")
    return downloaded


def load_record(record_path: Path) -> wfdb.Record:
    """Read a single WFDB record from disk."""
    return wfdb.rdrecord(str(record_path))


if __name__ == "__main__":
    print("=== PTB-XL small download ===")
    print(f"Target directory : {DATA_DIR}")
    print(f"Records to fetch : {N_RECORDS}  (100 Hz)")

    df = download_metadata()
    print(f"Metadata loaded  : {len(df)} total records in database")

    paths = download_records(df, n=N_RECORDS)

    # Quick sanity check — load the first record
    rec = load_record(paths[0])
    print(f"\nSample record    : {rec.record_name}")
    print(f"Leads            : {rec.sig_name}")
    print(f"Signal shape     : {rec.p_signal.shape}  (samples × leads)")
    print(f"Sampling rate    : {rec.fs} Hz")
    print("\nAll done. Proceed with preprocess.py.")
