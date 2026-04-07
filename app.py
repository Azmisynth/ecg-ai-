"""
app.py — ECG-XAI Streamlit educational app.

Run with:
    streamlit run app.py
"""

import sys
import numpy as np
import pandas as pd
import torch
import shap as shap_lib
import neurokit2 as nk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker

import streamlit as st
from pathlib import Path
from matplotlib.gridspec import GridSpec

# ── project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
SRC_DIR      = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from model import ECGResNet, LABELS, N_CLASSES

CKPT_PATH = PROJECT_ROOT / "checkpoints" / "best.ckpt"
DATA_DIR  = PROJECT_ROOT / "data"
PTBXL_DIR = DATA_DIR / "ptbxl_small"

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

LEAD_DESC = {
    "I":   "lateral limb lead",
    "II":  "inferior limb lead",
    "III": "inferior limb lead",
    "aVR": "right-sided lead",
    "aVL": "high lateral lead",
    "aVF": "inferior lead",
    "V1":  "right precordial lead (best for atrial activity)",
    "V2":  "right precordial lead",
    "V3":  "anterior lead",
    "V4":  "anterior lead",
    "V5":  "lateral precordial lead",
    "V6":  "lateral precordial lead",
}

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECG-XAI",
    page_icon=":anatomical_heart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource
def load_model() -> ECGResNet:
    model = ECGResNet.load_from_checkpoint(str(CKPT_PATH))
    model.eval()
    return model


@st.cache_data
def load_dataset() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X = np.load(DATA_DIR / "X_full.npy")
    y = np.load(DATA_DIR / "y_multilabel.npy")
    df = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv", index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(eval)
    return X, y, df.head(len(X))


@st.cache_data
def get_predictions(record_idx: int) -> np.ndarray:
    """Return sigmoid probabilities (4,) for one record."""
    model = load_model()
    X, _, _ = load_dataset()
    x = torch.tensor(X[record_idx : record_idx + 1], dtype=torch.float32)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).numpy()[0]
    return probs


@st.cache_data(show_spinner=False)
def get_shap_values(record_idx: int) -> np.ndarray:
    """
    Compute SHAP values for one record.
    Returns shap_vals of shape (4, 12, 1000).
    """
    model = load_model()
    X, _, _ = load_dataset()

    bg_idx = [i for i in range(min(21, len(X))) if i != record_idx][:20]
    background  = torch.tensor(X[bg_idx], dtype=torch.float32)
    test_input  = torch.tensor(X[record_idx : record_idx + 1], dtype=torch.float32)

    explainer = shap_lib.GradientExplainer(model, background)
    raw = explainer.shap_values(test_input)

    if isinstance(raw, list):
        shap_vals = np.stack([sv[0] for sv in raw], axis=0)     # (4, 12, 1000)
    else:
        raw = np.array(raw)
        if raw.ndim == 4 and raw.shape[-1] == N_CLASSES:
            shap_vals = np.transpose(raw[0], (2, 0, 1))          # (4, 12, 1000)
        elif raw.ndim == 4 and raw.shape[0] == N_CLASSES:
            shap_vals = raw[:, 0, :, :]
        else:
            shap_vals = np.stack([raw[0]] * N_CLASSES, axis=0)

    return shap_vals.astype(np.float32)


# ── true-label helper ─────────────────────────────────────────────────────────

def get_true_label(y_row: np.ndarray) -> str:
    active = [LABELS[i] for i, v in enumerate(y_row) if v == 1]
    return "  |  ".join(active) if active else "Unknown"


# ── section 1: 12-lead ECG viewer ─────────────────────────────────────────────

def _annotate_lead_ii(ax: plt.Axes, signal_1d: np.ndarray, fs: int = 100) -> None:
    """
    Label P wave, QRS complex, and T wave on Lead II using R-peak detection.
    Positions are estimated from standard ECG timing relative to the R peak.
    Silently skipped if R-peak detection fails.
    """
    try:
        _, info = nk.ecg_peaks(signal_1d, sampling_rate=fs, method="neurokit")
        rpeaks = info["ECG_R_Peaks"]
        if len(rpeaks) < 2:
            return

        r = int(rpeaks[1])             # use 2nd beat — safely away from the edge
        t_axis = np.arange(len(signal_1d)) / fs
        y_range = signal_1d.max() - signal_1d.min()
        lift = y_range * 0.35          # vertical offset for annotation text

        # P wave: ~160 ms before R peak
        p_idx = max(0, r - int(0.16 * fs))
        ax.annotate(
            "P wave",
            xy=(t_axis[p_idx], signal_1d[p_idx]),
            xytext=(t_axis[p_idx] - 0.18, signal_1d[p_idx] + lift),
            fontsize=7, color="#1565C0", fontweight="bold",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#1565C0", lw=0.9),
        )

        # QRS complex: at the R peak
        ax.annotate(
            "QRS",
            xy=(t_axis[r], signal_1d[r]),
            xytext=(t_axis[r] + 0.06, signal_1d[r] + lift * 0.4),
            fontsize=7, color="#B71C1C", fontweight="bold",
            ha="left",
            arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=0.9),
        )

        # T wave: ~280 ms after R peak
        tw_idx = min(len(signal_1d) - 1, r + int(0.28 * fs))
        ax.annotate(
            "T wave",
            xy=(t_axis[tw_idx], signal_1d[tw_idx]),
            xytext=(t_axis[tw_idx] + 0.16, signal_1d[tw_idx] + lift),
            fontsize=7, color="#1B5E20", fontweight="bold",
            ha="left",
            arrowprops=dict(arrowstyle="->", color="#1B5E20", lw=0.9),
        )
    except Exception:
        pass


def plot_ecg(signal: np.ndarray) -> plt.Figure:
    """Plot all 12 leads; annotate P/QRS/T on Lead II."""
    fig, axes = plt.subplots(12, 1, figsize=(14, 11), sharex=True)
    fig.suptitle("12-Lead ECG", fontsize=13, fontweight="bold", y=1.002)
    t = np.linspace(0, 10, signal.shape[1])

    for i, ax in enumerate(axes):
        ax.plot(t, signal[i], color="#1a1a2e", linewidth=0.75)
        ax.axhline(0, color="#e0e0e0", linewidth=0.4, linestyle="--", zorder=0)
        ax.set_ylabel(LEAD_NAMES[i], rotation=0, labelpad=26, fontsize=8, va="center")
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#dddddd")
        ax.set_xlim(0, 10)

    # Wave annotations on Lead II (index 1)
    _annotate_lead_ii(axes[1], signal[1])

    axes[-1].tick_params(labelbottom=True, bottom=True, labelsize=7)
    axes[-1].set_xlabel("Time (s)", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 1])
    return fig


# ── section 2: prediction bars ────────────────────────────────────────────────

def plot_predictions(probs: np.ndarray) -> plt.Figure:
    """Horizontal probability bars, top prediction coloured green."""
    fig, ax = plt.subplots(figsize=(9, 2.8))

    top_idx = int(np.argmax(probs))
    colors  = ["#78909C"] * N_CLASSES
    colors[top_idx] = "#2E7D32"

    bars = ax.barh(
        LABELS, probs * 100,
        color=colors, edgecolor="white", height=0.52,
    )

    for bar, prob in zip(bars, probs):
        x_label = prob * 100 + 1.5
        ax.text(
            x_label, bar.get_y() + bar.get_height() / 2,
            f"{prob * 100:.1f}%",
            va="center", fontsize=10, fontweight="bold", color="#212121",
        )

    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)", fontsize=9)
    ax.set_title("Model Predictions", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)
    plt.tight_layout()
    return fig


# ── section 3: SHAP heatmap ───────────────────────────────────────────────────

def plot_shap_heatmap(signal: np.ndarray, shap_vals: np.ndarray) -> plt.Figure:
    """
    12-lead x 4-class SHAP heatmap with dedicated colourbar columns.
    Matches the style of explanation.png.
    """
    N_ROWS = 12
    width_ratios = [14, 1] * N_CLASSES
    cmap = plt.cm.RdBu_r
    t    = np.linspace(0, 10, signal.shape[1])
    SEG  = 20

    fig = plt.figure(figsize=(N_CLASSES * 5.2, N_ROWS * 1.55))
    gs  = GridSpec(
        N_ROWS, N_CLASSES * 2,
        figure=fig,
        width_ratios=width_ratios,
        hspace=0.10, wspace=0.08,
    )

    for col, label in enumerate(LABELS):
        wave_col = col * 2
        cbar_col = col * 2 + 1

        sv_class = shap_vals[col]
        abs_max  = float(np.abs(sv_class).max()) or 1.0
        norm     = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

        wave_axes: list[plt.Axes] = []
        for row in range(N_ROWS):
            ax = fig.add_subplot(
                gs[row, wave_col],
                sharex=wave_axes[0] if row > 0 else None,
            )
            wave_axes.append(ax)

            sv  = sv_class[row]
            sig = signal[row]
            n_segs = len(t) // SEG
            for s in range(n_segs):
                sl  = slice(s * SEG, (s + 1) * SEG)
                clr = cmap(norm(float(sv[sl].mean())))
                ax.axvspan(
                    t[s * SEG], t[min((s + 1) * SEG, len(t) - 1)],
                    alpha=0.55, color=clr, linewidth=0,
                )
            ax.plot(t, sig, color="black", linewidth=0.6, zorder=2)
            ax.set_xlim(t[0], t[-1])
            ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
            if col == 0:
                ax.set_ylabel(LEAD_NAMES[row], rotation=0, labelpad=24,
                              fontsize=8, va="center")

        wave_axes[0].set_title(label, fontsize=11, fontweight="bold", pad=5)
        wave_axes[-1].tick_params(labelbottom=True, bottom=True, labelsize=7)
        wave_axes[-1].set_xlabel("Time (s)", fontsize=8)

        cbar_ax = fig.add_subplot(gs[:, cbar_col])
        cbar_ax.set_aspect("auto")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax)
        cb.set_ticks([-abs_max, 0.0, abs_max])
        cb.formatter = matplotlib.ticker.FuncFormatter(
            lambda x, _: "0" if abs(x) < 1e-12 else f"{x:.2g}"
        )
        cb.update_ticks()
        cb.set_label("SHAP", fontsize=8, labelpad=4)
        cb.ax.tick_params(labelsize=7)

    return fig


# ── section 4: plain-English explanation ─────────────────────────────────────

# Per-condition text tiers
_COND_TEXT = {
    "NORM": {
        "high": "Rhythm and conduction appear normal. Regular P-QRS-T pattern detected with consistent PR intervals and narrow QRS complexes.",
        "mid":  "Mostly normal features, though minor variations are present. Clinical correlation advised.",
        "low":  "No strong evidence of normal sinus rhythm.",
    },
    "AFIB": {
        "high": "High suspicion for atrial fibrillation. Look for absent P waves and irregularly irregular R-R intervals.",
        "mid":  "Low-to-moderate suspicion. Some rhythm irregularity is present — monitor and correlate clinically.",
        "low":  "Low suspicion for atrial fibrillation. Rhythm appears regular with visible P waves.",
    },
    "STD": {
        "high": "Significant ST depression detected. This may indicate subendocardial ischaemia or demand ischaemia. Correlate with symptoms and troponin.",
        "mid":  "Borderline ST changes noted. Could reflect rate-related changes or early ischaemia.",
        "low":  "No significant ST depression detected.",
    },
    "STE": {
        "high": "ST elevation pattern detected. This raises concern for acute myocardial injury (STEMI pattern). Urgent assessment required.",
        "mid":  "Mild ST elevation noted. Consider early repolarisation, pericarditis, or Brugada pattern.",
        "low":  "No ST elevation pattern found.",
    },
}

_TEACHING_NOTES = {
    "NORM": (
        "**Teaching note:** Normal sinus rhythm requires: (1) a P wave before every QRS, "
        "(2) a constant PR interval of 120–200 ms, (3) a QRS duration < 120 ms, "
        "and (4) a heart rate of 60–100 bpm. "
        "The AI confirmed these features across both limb and precordial leads. "
        "Lead II is the best single lead for assessing P-wave axis and rhythm."
    ),
    "AFIB": (
        "**Teaching note:** In atrial fibrillation, chaotic atrial activity replaces discrete P waves. "
        "The key diagnostic features are: (1) absent P waves, (2) irregularly irregular R-R intervals, "
        "and (3) fibrillatory baseline most visible in **lead V1**. "
        "The AI correctly focused on V1 — the optimal window onto the right atrium — to assess atrial activity."
    ),
    "STD": (
        "**Teaching note:** ST depression \u2265 0.5 mm in \u2265 2 contiguous leads suggests subendocardial ischaemia. "
        "It is measured 80 ms after the J point. "
        "Common ischaemic territories: inferior (II, III, aVF), anterior (V3\u2013V4), lateral (I, aVL, V5\u2013V6). "
        "The AI focused on the ST segment — the isoelectric segment between the end of QRS and start of T wave."
    ),
    "STE": (
        "**Teaching note:** ST elevation \u2265 1 mm in \u2265 2 contiguous leads indicates transmural myocardial injury. "
        "Territory localisation: inferior STEMI (II, III, aVF), anterior STEMI (V1\u2013V4), "
        "lateral STEMI (I, aVL, V5\u2013V6). "
        "The AI correctly focused on the ST segment above the isoelectric line, "
        "identifying the leads most affected by the elevation."
    ),
}


def generate_explanation(probs: np.ndarray, shap_vals: np.ndarray) -> dict:
    top_idx   = int(np.argmax(probs))
    top_label = LABELS[top_idx]
    top_prob  = probs[top_idx]

    label_full = {
        "NORM": "NORMAL SINUS RHYTHM",
        "AFIB": "ATRIAL FIBRILLATION",
        "STD":  "ST DEPRESSION",
        "STE":  "ST ELEVATION",
    }

    # ── 1. Prediction summary ─────────────────────────────────────────────────
    summary = (
        f"This ECG is most likely **{label_full[top_label]}** "
        f"({top_prob * 100:.0f}% confidence)."
    )

    # ── 2. What the model focused on ─────────────────────────────────────────
    sv_top   = shap_vals[top_idx]                         # (12, 1000)
    lead_imp = np.abs(sv_top).mean(axis=1)                # (12,) mean importance per lead
    top_leads_idx = np.argsort(lead_imp)[::-1][:2]
    top_leads = [LEAD_NAMES[i] for i in top_leads_idx]

    # Dominant SHAP time window (which third of the signal has highest mean |SHAP|)
    sv_lead_ii  = np.abs(sv_top[1])                       # Lead II as representative
    thirds      = np.array_split(sv_lead_ii, 3)
    third_means = [t.mean() for t in thirds]
    dominant_third = int(np.argmax(third_means))
    region_desc = ["early P-wave / baseline", "QRS complex", "ST-T wave"][dominant_third]

    focus_template = {
        "NORM": (
            f"The model paid most attention to **{top_leads[0]}** and **{top_leads[1]}** "
            f"({LEAD_DESC.get(top_leads[0], '')} and {LEAD_DESC.get(top_leads[1], '')}), "
            f"particularly in the **{region_desc} region**. "
            f"The regular, narrow QRS morphology and consistent P-QRS-T pattern in these leads "
            f"strongly supported a normal classification."
        ),
        "AFIB": (
            f"The model paid most attention to **{top_leads[0]}** and **{top_leads[1]}** "
            f"({LEAD_DESC.get(top_leads[0], '')} and {LEAD_DESC.get(top_leads[1], '')}), "
            f"particularly in the **{region_desc} region**. "
            f"These leads are key for detecting absent P waves and irregular R-R intervals — "
            f"the hallmarks of atrial fibrillation."
        ),
        "STD": (
            f"The model paid most attention to **{top_leads[0]}** and **{top_leads[1]}** "
            f"({LEAD_DESC.get(top_leads[0], '')} and {LEAD_DESC.get(top_leads[1], '')}), "
            f"particularly in the **{region_desc} region**. "
            f"ST depression is most significant in leads facing the ischaemic territory."
        ),
        "STE": (
            f"The model paid most attention to **{top_leads[0]}** and **{top_leads[1]}** "
            f"({LEAD_DESC.get(top_leads[0], '')} and {LEAD_DESC.get(top_leads[1], '')}), "
            f"particularly in the **{region_desc} region**. "
            f"ST elevation is measured above the isoelectric line, most prominent in the leads "
            f"overlying the affected myocardial territory."
        ),
    }

    # ── 3. Per-condition bullets ──────────────────────────────────────────────
    bullets = []
    for i, label in enumerate(LABELS):
        p    = probs[i]
        tier = "high" if p > 0.50 else ("mid" if p > 0.25 else "low")
        bullets.append(f"- **{label} — {p * 100:.0f}%:** {_COND_TEXT[label][tier]}")

    return {
        "summary":  summary,
        "focus":    focus_template[top_label],
        "bullets":  "\n".join(bullets),
        "teaching": _TEACHING_NOTES[top_label],
    }


# ── app layout ─────────────────────────────────────────────────────────────────

st.title("ECG-XAI: Explainable ECG Analysis")
st.markdown(
    "*An educational tool for understanding how AI reads electrocardiograms*"
)
st.divider()

# ── preflight checks ──────────────────────────────────────────────────────────
if not CKPT_PATH.exists():
    st.error("Model checkpoint not found at `checkpoints/best.ckpt`. Run `train.py` first.")
    st.stop()
if not (DATA_DIR / "X_full.npy").exists():
    st.error("Dataset not found at `data/X_full.npy`. Run `train.py` first.")
    st.stop()

# ── load shared state ─────────────────────────────────────────────────────────
X, y, df = load_dataset()

# ── record selector ───────────────────────────────────────────────────────────
col_drop, col_label = st.columns([2, 3])
with col_drop:
    record_idx = st.selectbox(
        "Select ECG record (0 – 99)",
        options=list(range(len(X))),
        index=0,
        format_func=lambda i: f"Record {i}",
    )
with col_label:
    true_label_str = get_true_label(y[record_idx])
    st.metric("True label", true_label_str)

signal = X[record_idx]   # (12, 1000)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — RAW ECG
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("1. 12-Lead ECG")
st.caption(
    "Each row is one lead. Lead II is annotated with the key ECG waves "
    "(P wave = atrial depolarisation, QRS = ventricular depolarisation, "
    "T wave = ventricular repolarisation)."
)
fig_ecg = plot_ecg(signal)
st.pyplot(fig_ecg, use_container_width=True)
plt.close(fig_ecg)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — AI PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("2. Model Predictions")
st.caption(
    "Probability assigned by the AI to each condition. "
    "Green bar = highest prediction. This is a multi-label model — "
    "multiple conditions can score > 0% simultaneously."
)
probs   = get_predictions(record_idx)
fig_pred = plot_predictions(probs)
st.pyplot(fig_pred, use_container_width=True)
plt.close(fig_pred)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SHAP HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("3. What the AI is Looking At")
st.caption(
    "SHAP (SHapley Additive exPlanations) colours show which parts of the ECG "
    "drove each prediction. **Red** = pushed prediction up. **Blue** = suppressed it."
)
with st.spinner("Computing SHAP values (cached after first run)..."):
    shap_vals = get_shap_values(record_idx)

fig_shap = plot_shap_heatmap(signal, shap_vals)
st.pyplot(fig_shap, use_container_width=True)
plt.close(fig_shap)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PLAIN-ENGLISH EXPLANATION
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("4. Explanation")
st.caption("Written for medical students learning ECG interpretation with AI.")

explanation = generate_explanation(probs, shap_vals)

# ── prediction summary ────────────────────────────────────────────────────────
st.markdown("#### Prediction Summary")
st.markdown(explanation["summary"])

# ── what the model focused on ─────────────────────────────────────────────────
st.markdown("#### What the Model Focused On")
st.markdown(explanation["focus"])

# ── per-condition bullets ─────────────────────────────────────────────────────
st.markdown("#### What Each Finding Means")
st.markdown(explanation["bullets"])

# ── teaching note ─────────────────────────────────────────────────────────────
st.markdown("---")
with st.container(border=True):
    st.markdown(explanation["teaching"])

# ── disclaimer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "This is an AI-assisted educational tool only. "
    "Not validated for clinical use. "
    "Always consult a qualified cardiologist for medical decisions."
)
