"""
explain.py — SHAP GradientExplainer on a trained ECGResNet.

Workflow
--------
1. Load best checkpoint from checkpoints/best.ckpt
2. Load X_full.npy  (N, 12, 1000)
3. Use 20 background samples for the SHAP explainer baseline
4. Compute SHAP values for one test sample and all 4 output classes
5. For each class, plot the 12-lead ECG with a red/blue SHAP heatmap overlay
6. Save composite figure to explanation.png

Interpreting the plot
---------------------
* Red regions  → high positive SHAP value → pushes the prediction higher
* Blue regions → negative SHAP value      → suppresses the prediction
* The intensity encodes magnitude.
"""

import sys
import numpy as np
import torch
import shap
import matplotlib
matplotlib.use("Agg")   # headless — safe on Windows without display
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker
from pathlib import Path

SRC_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from model import ECGResNet, LABELS, N_CLASSES

CKPT_PATH = PROJECT_ROOT / "checkpoints" / "best.ckpt"
DATA_DIR  = PROJECT_ROOT / "data"
OUT_PATH  = PROJECT_ROOT / "explanation.png"

LEAD_NAMES = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
N_BACKGROUND = 20   # number of background samples for SHAP
SAMPLE_IDX   = 0    # which test sample to explain


# ── SHAP computation ──────────────────────────────────────────────────────────

def compute_shap(
    model: ECGResNet,
    X: np.ndarray,
    sample_idx: int = SAMPLE_IDX,
    n_background: int = N_BACKGROUND,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute SHAP values for one sample using GradientExplainer.

    Returns
    -------
    shap_vals : (N_CLASSES, 12, 1000)  — one heatmap per class
    signal    : (12, 1000)             — the raw (cleaned) ECG signal
    """
    model.eval()

    # Background: first n_background records (excluding the test sample)
    bg_indices = [i for i in range(min(n_background + 1, len(X))) if i != sample_idx][:n_background]
    background = torch.tensor(X[bg_indices], dtype=torch.float32)
    test_input = torch.tensor(X[sample_idx : sample_idx + 1], dtype=torch.float32)

    print(f"  Background samples : {len(bg_indices)}")
    print(f"  Test sample index  : {sample_idx}")

    explainer = shap.GradientExplainer(model, background)

    # shap_values returns a list of (1, 12, 1000) arrays — one per output class
    raw_shap = explainer.shap_values(test_input)

    if isinstance(raw_shap, list):
        # Older shap: list of N_CLASSES arrays each (n_samples, 12, 1000)
        shap_vals = np.stack([sv[0] for sv in raw_shap], axis=0)  # (4, 12, 1000)
    else:
        raw_shap = np.array(raw_shap)
        if raw_shap.ndim == 4 and raw_shap.shape[-1] == N_CLASSES:
            # shap 0.46+: (n_samples, 12, 1000, 4) — classes in last axis
            shap_vals = np.transpose(raw_shap[0], (2, 0, 1))      # (4, 12, 1000)
        elif raw_shap.ndim == 4 and raw_shap.shape[0] == N_CLASSES:
            # (4, n_samples, 12, 1000)
            shap_vals = raw_shap[:, 0, :, :]                       # (4, 12, 1000)
        elif raw_shap.ndim == 3:
            # (n_samples, 12, 1000) — single-output model
            shap_vals = np.stack([raw_shap[0]] * N_CLASSES, axis=0)
        else:
            raise ValueError(f"Unexpected shap_values shape: {raw_shap.shape}")

    signal = X[sample_idx]  # (12, 1000)
    return shap_vals.astype(np.float32), signal.astype(np.float32)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_explanation(
    signal: np.ndarray,
    shap_vals: np.ndarray,
    out_path: Path = OUT_PATH,
):
    """
    4-column figure (one column per class) x 12-row (one row per lead).
    Each class column is paired with a dedicated narrow colorbar column
    so the colorbar never overlaps any waveform axes.

    signal    : (12, 1000)
    shap_vals : (4, 12, 1000)
    """
    from matplotlib.gridspec import GridSpec

    t = np.linspace(0, 10, signal.shape[1])   # time axis in seconds
    SEG = 20                                   # samples per coloured segment

    N_ROWS = 12
    # Layout: [wave | cbar | wave | cbar | wave | cbar | wave | cbar]
    # width_ratios: waveform columns are 14 units wide, colorbar columns 1 unit
    width_ratios = [14, 1] * N_CLASSES

    fig = plt.figure(figsize=(N_CLASSES * 5.2, N_ROWS * 1.55))
    fig.suptitle(
        "ECG-XAI: SHAP Explanation per Class\n"
        "(red = high importance  |  blue = suppresses prediction)",
        fontsize=13, fontweight="bold", y=1.002,
    )

    gs = GridSpec(
        N_ROWS, N_CLASSES * 2,
        figure=fig,
        width_ratios=width_ratios,
        hspace=0.10,   # vertical gap between lead rows
        wspace=0.08,   # horizontal gap between columns
    )

    cmap = plt.cm.RdBu_r

    for col, label in enumerate(LABELS):
        wave_col = col * 2        # waveform GridSpec column index
        cbar_col = col * 2 + 1   # colorbar GridSpec column index

        sv_class = shap_vals[col]                        # (12, 1000)
        abs_max  = float(np.abs(sv_class).max()) or 1.0
        norm     = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

        # ── waveform rows ────────────────────────────────────────────────────
        wave_axes = []
        for row in range(N_ROWS):
            ax = fig.add_subplot(
                gs[row, wave_col],
                sharex=wave_axes[0] if row > 0 else None,
            )
            wave_axes.append(ax)

            sv  = sv_class[row]
            sig = signal[row]

            # Coloured background bands
            n_segs = len(t) // SEG
            for s in range(n_segs):
                sl     = slice(s * SEG, (s + 1) * SEG)
                avg_sv = float(sv[sl].mean())
                color  = cmap(norm(avg_sv))
                ax.axvspan(
                    t[s * SEG],
                    t[min((s + 1) * SEG, len(t) - 1)],
                    alpha=0.55,
                    color=color,
                    linewidth=0,
                )

            ax.plot(t, sig, color="black", linewidth=0.6, zorder=2)
            ax.set_xlim(t[0], t[-1])
            ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

            # Lead name on leftmost column only
            if col == 0:
                ax.set_ylabel(
                    LEAD_NAMES[row], rotation=0,
                    labelpad=24, fontsize=8, va="center",
                )

        # Column title above first row
        wave_axes[0].set_title(label, fontsize=11, fontweight="bold", pad=5)

        # Time axis on bottom row only
        wave_axes[-1].tick_params(labelbottom=True, bottom=True, labelsize=7)
        wave_axes[-1].set_xlabel("Time (s)", fontsize=8)

        # ── dedicated colorbar axes (spans all 12 rows) ──────────────────────
        cbar_ax = fig.add_subplot(gs[:, cbar_col])
        cbar_ax.set_aspect("auto")   # fill the full GridSpec cell height

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax)

        # 3 ticks only: bottom value, zero, top value
        cb.set_ticks([-abs_max, 0.0, abs_max])
        cb.formatter = matplotlib.ticker.FuncFormatter(
            lambda x, _: "0" if abs(x) < 1e-12 else f"{x:.2g}"
        )
        cb.update_ticks()
        cb.set_label("SHAP", fontsize=8, labelpad=4)
        cb.ax.tick_params(labelsize=7)

    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not CKPT_PATH.exists():
        print(f"[error] Checkpoint not found at {CKPT_PATH}")
        print("Run train.py first.")
        sys.exit(1)

    X_path = DATA_DIR / "X_full.npy"
    if not X_path.exists():
        print(f"[error] X_full.npy not found at {X_path}")
        print("Run train.py first (it builds and caches the dataset).")
        sys.exit(1)

    print("=== ECG-XAI: SHAP Explanation ===")
    print(f"Checkpoint : {CKPT_PATH}")

    model = ECGResNet.load_from_checkpoint(str(CKPT_PATH))
    model.eval()

    X = np.load(X_path)
    print(f"Dataset    : {X.shape}")

    print("\nComputing SHAP values …")
    shap_vals, signal = compute_shap(model, X, sample_idx=SAMPLE_IDX)
    print(f"SHAP shape : {shap_vals.shape}  (classes × leads × samples)")

    print("Rendering explanation plot …")
    plot_explanation(signal, shap_vals, OUT_PATH)
    print("Done. Open explanation.png to view.")
