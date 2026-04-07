"""
train.py — Build dataset, train ECGResNet, report per-class AUC.

Dataset notes
-------------
* Uses the FULL cleaned 10-second signal (1000 samples @ 100 Hz), not
  the median-beat representation from preprocess.py.
* On first run it downloads + cleans records and caches X_full.npy /
  y_multilabel.npy so subsequent runs are fast.

Labels (multi-label, one-hot encoded)
--------------------------------------
  Index 0 — NORM  (normal sinus rhythm)
  Index 1 — AFIB  (atrial fibrillation)
  Index 2 — STD   (ST depression)
  Index 3 — STE   (ST elevation)
"""

import sys
import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
SRC_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from model import ECGResNet, LABELS, N_CLASSES

DATA_DIR   = PROJECT_ROOT / "data"
PTBXL_DIR  = DATA_DIR / "ptbxl_small"
CKPT_DIR   = PROJECT_ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

X_CACHE = DATA_DIR / "X_full.npy"
Y_CACHE = DATA_DIR / "y_multilabel.npy"

# ── SCP code → label index mapping ───────────────────────────────────────────
# PTB-XL rhythm / diagnostic SCP codes for our 4 classes
SCP_MAP = {
    "NORM": 0,
    "AFIB": 1,
    "AFLT": 1,   # atrial flutter — grouped with AFIB
    "STD_": 2,   # ST depression (underscore variant used in PTB-XL)
    "STD":  2,
    "NDT":  2,   # non-diagnostic T-wave → ST depression proxy
    "STE_": 3,   # ST elevation
    "STE":  3,
    "AMI":  3,   # acute MI — grouped with STE
    "IMI":  3,   # inferior MI
    "LMI":  3,   # lateral MI
    "ALMI": 3,
    "IPLMI":3,
    "IPMI": 3,
}

FS       = 100    # sampling rate
N_SAMPS  = 1000   # 10 seconds
N_RECORDS = 100


# ── per-lead signal cleaning ──────────────────────────────────────────────────

def clean_signal(raw: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Clean a single-lead ECG signal with NeuroKit2.
    raw : (N_SAMPS,) float
    Returns cleaned float32 array of same length.
    """
    try:
        cleaned = nk.ecg_clean(raw, sampling_rate=fs, method="biosppy")
        # Z-score normalise
        std = cleaned.std()
        if std > 0:
            cleaned = (cleaned - cleaned.mean()) / std
        return cleaned.astype(np.float32)
    except Exception:
        # Fallback: just normalise the raw signal
        std = raw.std()
        if std > 0:
            raw = (raw - raw.mean()) / std
        return raw.astype(np.float32)


# ── dataset construction ──────────────────────────────────────────────────────

def build_multilabel(df: pd.DataFrame) -> np.ndarray:
    """
    Build a (N_RECORDS, N_CLASSES) int8 multi-label matrix from PTB-XL metadata.
    """
    subset = df.head(N_RECORDS)
    y = np.zeros((len(subset), N_CLASSES), dtype=np.int8)
    for i, (_, row) in enumerate(subset.iterrows()):
        codes = row["scp_codes"]   # dict {code: likelihood}
        for code in codes:
            idx = SCP_MAP.get(code.upper())
            if idx is not None:
                y[i, idx] = 1
    return y


def build_full_signals(df: pd.DataFrame) -> np.ndarray:
    """
    Load + clean all 12-lead signals. Returns (N, 12, 1000) float32.
    """
    subset = df.head(N_RECORDS)
    X = np.zeros((len(subset), 12, N_SAMPS), dtype=np.float32)

    for i, (_, row) in enumerate(subset.iterrows()):
        record_path = PTBXL_DIR / row["filename_lr"]
        try:
            rec = wfdb.rdrecord(str(record_path))
            raw = rec.p_signal  # (1000, 12)
            for lead in range(12):
                sig = raw[:N_SAMPS, lead].astype(np.float32)
                if len(sig) < N_SAMPS:
                    sig = np.pad(sig, (0, N_SAMPS - len(sig)))
                X[i, lead] = clean_signal(sig)
        except Exception as e:
            print(f"  [warn] record {i} ({record_path.name}): {e}")

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(subset)} records …")

    return X


def load_or_build_dataset():
    """Load cached arrays or build from scratch."""
    if X_CACHE.exists() and Y_CACHE.exists():
        print("Loading cached X_full.npy and y_multilabel.npy …")
        X = np.load(X_CACHE)
        y = np.load(Y_CACHE)
        return X, y

    # Need the metadata
    meta_path = PTBXL_DIR / "ptbxl_database.csv"
    if not meta_path.exists():
        print("Metadata not found — downloading PTB-XL subset …")
        sys.path.insert(0, str(SRC_DIR))
        from load_data import download_metadata, download_records
        df = download_metadata()
        download_records(df, n=N_RECORDS)
    else:
        df = pd.read_csv(meta_path, index_col="ecg_id")
        df.scp_codes = df.scp_codes.apply(eval)

    # Check that .dat files exist; download if missing
    first_path = PTBXL_DIR / df.head(1).iloc[0]["filename_lr"]
    if not Path(str(first_path) + ".dat").exists():
        print("ECG records not found — downloading …")
        from load_data import download_records
        download_records(df, n=N_RECORDS)

    print("Building full-signal dataset …")
    X = build_full_signals(df)
    y = build_multilabel(df)

    np.save(X_CACHE, X)
    np.save(Y_CACHE, y)
    print(f"Saved X_full.npy {X.shape}  y_multilabel.npy {y.shape}")
    return X, y


# ── main training script ──────────────────────────────────────────────────────

def main():
    pl.seed_everything(42, workers=True)

    # ── data ─────────────────────────────────────────────────────────────────
    print("\n=== Loading dataset ===")
    X, y = load_or_build_dataset()

    print(f"\nDataset shape : X={X.shape}  y={y.shape}")
    print("Label counts  :")
    for i, label in enumerate(LABELS):
        n_pos = int(y[:, i].sum())
        print(f"  {label:6s} — {n_pos:3d} positive / {len(y)} total")

    # ── tensors & split ───────────────────────────────────────────────────────
    X_t = torch.tensor(X)
    y_t = torch.tensor(y, dtype=torch.long)

    dataset   = TensorDataset(X_t, y_t)
    n_val     = max(10, int(0.2 * len(dataset)))
    n_train   = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0)

    print(f"\nTrain samples : {n_train}   Val samples : {n_val}")

    # ── compute positive-class weights for imbalanced labels ─────────────────
    # (handled internally by BCEWithLogitsLoss; kept simple here)

    # ── model ─────────────────────────────────────────────────────────────────
    model = ECGResNet(lr=1e-3, dropout=0.2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params  : {total_params:,}\n")

    # ── callbacks ─────────────────────────────────────────────────────────────
    ckpt_cb = ModelCheckpoint(
        dirpath=str(CKPT_DIR),
        filename="best",
        monitor="val_auc_mean",
        mode="max",
        save_top_k=1,
        verbose=True,
    )
    early_stop = EarlyStopping(
        monitor="val_auc_mean",
        patience=8,
        mode="max",
        verbose=True,
    )

    # ── trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=20,
        callbacks=[ckpt_cb, early_stop],
        enable_progress_bar=True,
        log_every_n_steps=1,
        deterministic=True,
    )

    print("=== Training ===")
    trainer.fit(model, train_dl, val_dl)

    # ── final AUC report ──────────────────────────────────────────────────────
    print("\n=== Validation AUC per condition ===")
    best_path = CKPT_DIR / "best.ckpt"
    best_model = ECGResNet.load_from_checkpoint(str(best_path))
    best_model.eval()

    with torch.no_grad():
        all_probs, all_labels = [], []
        for xb, yb in val_dl:
            all_probs.append(torch.sigmoid(best_model(xb)))
            all_labels.append(yb)

    probs  = torch.cat(all_probs)
    labels = torch.cat(all_labels)

    from torchmetrics.classification import MultilabelAUROC
    auc_metric = MultilabelAUROC(num_labels=N_CLASSES, average=None)
    aucs = auc_metric(probs, labels)

    for label, auc in zip(LABELS, aucs):
        val = auc.item()
        tag = "  (insufficient positives — random baseline)" if val != val else ""  # NaN check
        if auc.isnan():
            print(f"  {label:6s}  AUC = N/A  (no positive examples in val set)")
        else:
            print(f"  {label:6s}  AUC = {val:.4f}")

    print(f"\nBest checkpoint : {best_path}")
    print("Run explain.py to generate the SHAP explanation.")


if __name__ == "__main__":
    main()
