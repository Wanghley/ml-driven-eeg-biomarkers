#!/usr/bin/env python3
"""
pipeline.py — Production EEG Cleaning Pipeline
================================================
Full pipeline from raw EDF to DBS-cleaned output with:

  Stage I  : Standard EEG Foundation (HPF 0.1 Hz, LPF 45 Hz, 60 Hz notch, CAR)
  Stage II : Surgical DBS Removal:
               1. Sinusoidal Regression  — exact-harmonic OLS subtraction
               2. Complex Spectral Interp — Hampel outlier → linear complex interp
               3. Median Phase Template  — continuous waveform subtraction
  Stage III: Automated ICA Cleaning (FastICA + spectral DBS reject + MI reject
             + EOG/muscle spatial heuristics)

Validation:
  • Scalp topomaps  : before / after cleaning (average amplitude)
  • PSD overlay     : Raw vs Stage-I vs Final (with DBS harmonic markers)
  • Time-domain     : 5-second sliding comparison for a representative channel
  • Biomarker table : Theta / Beta / Gamma power preservation %
  • Real-time dashboard (save as animation GIF)

Usage
-----
  python3 pipeline.py                                   # default: XUAWAKE7
  python3 pipeline.py --file data/raw/XU/XUAWAKE7_deidentified.edf --dbs-freq 7.0
  python3 pipeline.py --no-ica                          # skip ICA (faster)
  python3 pipeline.py --all-conditions                  # run all EDF files
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter

import mne
mne.set_log_level("ERROR")

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import kurtosis as sp_kurtosis

try:
    from sklearn.feature_selection import mutual_info_regression
    _HAVE_SKLEARN = True
except ImportError:
    _HAVE_SKLEARN = False

# ── Project imports ──────────────────────────────────────────────────────────
_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
from src.filters import ArtifactFilterFactory
from src.preprocessing import EEGPreprocessor

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
STANDARD_CH = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3",  "C3",  "Cz", "C4", "T4",
    "T5",  "P3",  "Pz", "P4", "T6",
    "O1",  "O2",
]
MAD_FACTOR      = 1.4826
TARGET_SFREQ    = 256.0
MAX_DUR_SEC     = 120.0
AA_CUTOFF_HZ    = 120.0          # Anti-alias before downsampling
HPF_HZ          = 0.1            # High-pass (DC removal)
LPF_HZ          = 100.0          # Low-pass
LINE_NOISE_HZ   = 60.0           # North American line noise
ICA_N_COMP      = 15
ANALYSIS_CH     = "Cz"

OUTPUT_DIR = pathlib.Path("output/validation_results")

# EEG band definitions  (name, lo, hi, colour)
BANDS = [
    ("Delta", 0.5,  4,  "#8c564b"),
    ("Theta",  4,   8,  "#9467bd"),
    ("Alpha",  8,  13,  "#1f77b4"),
    ("Beta",  13,  30,  "#2ca02c"),
    ("Gamma", 30, 100,  "#d62728"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Core Pipeline Execution
# ─────────────────────────────────────────────────────────────────────────────

def run_surgical_pipeline(path: pathlib.Path, f0: float, run_ica: bool = True):
    """
    Primary execution wrapper that uses the core library.
    """
    preprocessor = EEGPreprocessor()
    
    # Stage I: Foundation
    log.info(f"Loading {path.name}")
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
    
    # Crop to max duration
    raw.crop(tmin=0, tmax=min(MAX_DUR_SEC, raw.times[-1]))
    
    # Step 1: Standardize
    raw_std = preprocessor.standardize_channels(raw)
    
    # Step 2: Clinical Base (0.1 - 100 Hz)
    raw_stage1 = preprocessor.apply_clinical_filter(raw_std, l_freq=0.1, h_freq=100.0)
    
    # Step 3: Line Noise
    nyq = raw_stage1.info['sfreq'] / 2.0
    notch_freqs = sorted([f for f in np.arange(60.0, nyq, 60.0) if f < 105.0])
    if notch_freqs:
        raw_stage1.notch_filter(freqs=notch_freqs, method='fir', phase='zero', verbose=False)
    
    # Stage II & III: Surgical Removal
    raw_final = preprocessor.apply_surgical_pipeline(raw_stage1, f_dbs=f0, run_ica=run_ica)
    
    return raw, raw_stage1, raw_final


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def band_power(data_uv: np.ndarray, sfreq: float,
               lo: float, hi: float, n_fft: int = 2048) -> float:
    """Mean Welch band power (µV²) across all channels."""
    f, p = sp_signal.welch(data_uv, fs=sfreq, nperseg=n_fft,
                            noverlap=n_fft // 2, axis=1)
    m = (f >= lo) & (f <= hi)
    return float(np.trapz(p[:, m].mean(axis=0), f[m])) if m.any() else 0.0


def harmonic_power(data_uv: np.ndarray, sfreq: float, f_dbs: float,
                    bw: float = 0.5, n_fft: int = 2048) -> float:
    """Sum of Welch power within ±bw Hz of the first N DBS harmonics."""
    f, p = sp_signal.welch(data_uv, fs=sfreq, nperseg=n_fft,
                            noverlap=n_fft // 2, axis=1)
    pm = p.mean(axis=0)
    total = 0.0
    for k in range(1, 60):
        h = k * f_dbs
        if h + bw >= sfreq / 2:
            break
        m = (f >= h - bw) & (f <= h + bw)
        if m.any():
            total += float(np.trapz(pm[m], f[m]))
    return total


def compute_biomarker_table(raw_uv: np.ndarray, prep_uv: np.ndarray,
                             final_uv: np.ndarray, sfreq: float,
                             f_dbs: float) -> dict:
    out = {}
    for name, lo, hi, _ in BANDS:
        bp_raw  = band_power(raw_uv,   sfreq, lo, hi)
        bp_prep = band_power(prep_uv,  sfreq, lo, hi)
        bp_fin  = band_power(final_uv, sfreq, lo, hi)
        if bp_prep > 0:
            pres = round(100 * bp_fin / bp_prep, 2)
        else:
            pres = 0.0
        out[name] = {
            "baseline_uV2":    round(bp_prep, 3),
            "final_uV2":       round(bp_fin,  3),
            "preservation_%":  pres,
            "goal_met (>90%)": pres >= 90.0
        }

    h_raw   = harmonic_power(raw_uv,   sfreq, f_dbs)
    h_final = harmonic_power(final_uv, sfreq, f_dbs)
    dbs_red = round(100 * (1 - h_final / (h_raw + 1e-30)), 2) if h_raw > 0 else 0.0
    out["DBS_removal"] = {
        "harmonic_power_raw_uV2":   round(h_raw,   3),
        "harmonic_power_final_uV2": round(h_final, 3),
        "reduction_%":              dbs_red,
        "goal_met (>90%)":          dbs_red >= 90.0,
    }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION SUITE
# ─────────────────────────────────────────────────────────────────────────────

def _psd_mean(data_uv: np.ndarray, sfreq: float,
              fmin: float = 0.5, fmax: float = 55.0,
              n_fft: int = 2048, smooth_hz: float = 0.5) -> tuple:
    """
    Compute mean PSD (dB µV²/Hz) correctly:
      1. Welch per channel → average power spectra (NOT signals).
         Averaging signals first causes massive spatial cancellation.
      2. Light Hann-window smoothing in frequency to reduce hash.
    """
    f, p_all = sp_signal.welch(data_uv, fs=sfreq,
                                nperseg=n_fft, noverlap=n_fft // 2,
                                axis=1)          # (n_ch, n_freq)
    p_mean = p_all.mean(axis=0)                  # average power across channels

    # Optional light frequency-domain smoothing
    if smooth_hz > 0:
        df     = f[1] - f[0]
        k      = max(1, int(round(smooth_hz / df)))
        win    = np.hanning(2 * k + 1)
        win   /= win.sum()
        p_mean = np.convolve(p_mean, win, mode="same")

    m  = (f >= fmin) & (f <= fmax)
    db = 10 * np.log10(p_mean[m] + 1e-30)
    return f[m], db


def plot_triple_psd(raw_uv, pre_uv, stage1_uv, final_uv, sfreq, f_dbs, out_path):
    """
    2-panel dual-zoom PSD figure (dark theme, publication quality).

    Left  : 0.5 – 50 Hz overview — shows DBS harmonics, LPF rolloff, band structure.
    Right : 1 – 30 Hz clinical zoom — shows theta/alpha/beta preservation detail.

    Fixes applied vs previous version:
    • PSD computed per-channel then averaged (no spatial cancellation).
    • y-axis auto-scaled to data range + 6 dB pad per panel.
    • Hann-smoothed curves reduce frequency-bin hash.
    • LPF rolloff at 45 Hz clearly visible in left panel.
    """
    fig, axes = plt.subplots(1, 2, figsize=(17, 6),
                              gridspec_kw={"wspace": 0.30})
    fig.patch.set_facecolor("#0e1117")

    layers = [
        (stage1_uv, "Stage I — DBS (preprocessed)", "#e07b39", 1.2, "--"),
        (pre_uv,    "PRE baseline (no DBS)",         "#44cf6c", 1.6, "-."),
        (final_uv,  "Final — DBS removed + ICA",     "#4da6ff", 2.2, "-"),
    ]

    panels = [
        # (ax, fmin, fmax, title)
        (axes[0], 0.5, LPF_HZ + 5.0, f"Full spectrum  0.5 – {LPF_HZ + 5.0:.0f} Hz"),
        (axes[1], 1.0, 30.0, "Clinical zoom  1 – 30 Hz"),
    ]

    all_db_vals = []   # collect for global y-clip

    # Pre-compute all curves once
    curves = []
    for data, lbl, col, lw, ls in layers:
        fmax_calc = min(sfreq / 2.0, LPF_HZ + 10.0)
        f50, db50 = _psd_mean(data, sfreq, fmin=0.5, fmax=fmax_calc)
        curves.append((f50, db50, lbl, col, lw, ls))
        all_db_vals.append(db50)

    global_min = np.percentile(np.concatenate(all_db_vals), 2)
    global_max = np.percentile(np.concatenate(all_db_vals), 98)

    for ax, fmin_p, fmax_p, panel_title in panels:
        ax.set_facecolor("#0e1117")

        # ── Draw each curve, clipped to this panel's freq range ──────────
        panel_db_vals = []
        for f_all, db_all, lbl, col, lw, ls in curves:
            m   = (f_all >= fmin_p) & (f_all <= fmax_p)
            f_p = f_all[m]
            d_p = db_all[m]
            if len(f_p) < 2:
                continue
            ax.plot(f_p, d_p, color=col, lw=lw, ls=ls,
                    label=lbl, alpha=0.92, zorder=3)
            panel_db_vals.append(d_p)

        # ── Brain band spans ──────────────────────────────────────────────
        for name, lo, hi, col in BANDS:
            if hi < fmin_p or lo > fmax_p:
                continue
            lo_c = max(lo, fmin_p)
            hi_c = min(hi, fmax_p)
            ax.axvspan(lo_c, hi_c, color=col, alpha=0.08, zorder=1)
            # Band label
            mid = (lo_c + hi_c) / 2
            if global_min < global_max:
                ax.text(mid, global_max - 1, name,
                        ha="center", va="top", fontsize=6.5,
                        color=col, alpha=0.85, fontweight="bold")

        # ── DBS harmonic markers ──────────────────────────────────────────
        for k in range(1, 50):
            h = k * f_dbs
            if h < fmin_p:
                continue
            if h > fmax_p:
                break
            ax.axvline(h, color="#f9c74f", lw=0.8, ls=":",
                       alpha=0.55, zorder=2,
                       label="DBS harmonic" if k == 1 else "")
            ax.text(h, global_max + 0.5, f"{k}×",
                    ha="center", va="bottom", fontsize=5.5,
                    color="#f9c74f", alpha=0.7)

        # ── LPF rolloff marker (left panel only) ────────────────────────
        if fmax_p >= LPF_HZ and ax is axes[0]:
            ax.axvline(LPF_HZ, color="#ff6b6b", lw=1.2, ls="--",
                       alpha=0.7, label=f"LPF {LPF_HZ:.0f} Hz")
            ax.text(LPF_HZ - 0.4, global_max - 2, f"LPF\n{LPF_HZ:.0f}Hz",
                    ha="right", va="top", fontsize=7,
                    color="#ff6b6b", alpha=0.9)

        # ── Axis formatting ───────────────────────────────────────────────
        pad = max(3.0, (global_max - global_min) * 0.06)
        ax.set_ylim(global_min - pad, global_max + pad + 4)
        ax.set_xlim(fmin_p, fmax_p)

        ax.set_xlabel("Frequency (Hz)", color="white", fontsize=10)
        ax.set_ylabel("PSD  (dB µV²/Hz)", color="white", fontsize=10)
        ax.set_title(panel_title, color="white", fontsize=11, fontweight="bold")
        ax.tick_params(colors="white", labelsize=8)
        ax.grid(True, color="#2a2a3a", lw=0.5, ls="--", alpha=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444455")

        if ax is axes[0]:
            ax.legend(fontsize=8, facecolor="#1c1f26",
                      labelcolor="white", loc="lower left",
                      framealpha=0.85, edgecolor="#555566")

    fig.suptitle(
        "Triple-Overlay PSD  —  DBS Raw  ●  PRE Baseline (no DBS)  ●  Final Cleaned",
        color="white", fontsize=13, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"  PSD plot → {out_path.name}")


def plot_time_domain_comparison(raw_uv, prep_uv, final_uv,
                                 sfreq, ch_names, f_dbs, t_offset, dur, out_path):
    """3-panel time-domain comparison (channel Cz or first available)."""
    try:
        ci = ch_names.index(ANALYSIS_CH)
    except ValueError:
        ci = 0

    i0 = int(t_offset * sfreq)
    i1 = i0 + int(dur * sfreq)
    t  = np.linspace(t_offset, t_offset + dur, i1 - i0)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.patch.set_facecolor("#0e1117")
    triples = [
        (raw_uv,   "Raw (as-is)",              "#888888"),
        (prep_uv,  "Stage I Standard Prep",    "#ff7f0e"),
        (final_uv, "Final (DBS + ICA clean)", "#1f77b4"),
    ]
    for ax, (data, lbl, col) in zip(axes, triples):
        ax.set_facecolor("#0e1117")
        ax.plot(t, data[ci, i0:i1], color=col, lw=0.9)
        ax.set_ylabel(lbl, color="white", fontsize=8)
        ax.tick_params(colors="white")
        ax.spines[["top", "right"]].set_visible(False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")
        # DBS pulse markers
        for k in range(int(t_offset * f_dbs) - 1, int((t_offset + dur) * f_dbs) + 2):
            pt = k / f_dbs
            if t_offset <= pt <= t_offset + dur:
                ax.axvline(pt, color="#f9c74f", lw=0.5, alpha=0.4)

    axes[-1].set_xlabel("Time (s)", color="white")
    fig.suptitle(f"Time-Domain Comparison — Channel {ch_names[ci]}  ({dur:.0f}s segment)",
                 color="white", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"  Time-domain plot → {out_path.name}")


def plot_scalp_topomaps(prep_uv, final_uv, raw_mne, ch_names, out_path):
    """Scalp topomaps of mean absolute amplitude before and after cleaning."""
    try:
        info = raw_mne.info.copy()
        amp_before = np.abs(prep_uv).mean(axis=1)   # (n_ch,)
        amp_after  = np.abs(final_uv).mean(axis=1)

        ch_idx = mne.pick_types(info, eeg=True, exclude=[])
        if len(ch_idx) < 4:
            log.warning("Fewer than 4 EEG channels — skipping topomap.")
            return

        vmax = max(amp_before.max(), amp_after.max())
        vmin = 0.0
        diff = amp_before - amp_after

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.patch.set_facecolor("#0e1117")

        for ax, topo_data, title in [
            (axes[0], amp_before, "Before DBS Removal\n(Stage I output)"),
            (axes[1], amp_after,  "After DBS + ICA\n(Final)"),
            (axes[2], diff,       "Difference\n(Artifact Removed)"),
        ]:
            ax.set_facecolor("#0e1117")
            vmin_p = min(0, diff.min()) if ax is axes[2] else vmin
            vmax_p = max(diff.max(), 1e-3) if ax is axes[2] else vmax
            mne.viz.plot_topomap(
                topo_data, info, axes=ax, show=False,
                cmap="RdBu_r" if ax is axes[2] else "hot",
                vlim=(vmin_p, vmax_p),
                sphere="auto",
            )
            ax.set_title(title, color="white", fontsize=9)

        fig.suptitle("Scalp Topomaps — Mean Absolute Amplitude (µV)",
                     color="white", fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info(f"  Topomap plot → {out_path.name}")
    except Exception as e:
        log.warning(f"Topomap rendering failed: {e}")


def plot_biomarker_bars(table: dict, out_path: pathlib.Path):
    """Horizontal bar chart of band preservation percentages."""
    band_names = [b for b in table if b != "DBS_removal"]
    pct_vals   = [table[b]["preservation_%"] for b in band_names]
    colors     = ["#2ca02c" if v >= 90 else "#d62728" for v in pct_vals]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    bars = ax.barh(band_names, pct_vals, color=colors, height=0.5)
    ax.axvline(90, color="#f9c74f", lw=1.2, ls="--", label="90% target")
    ax.axvline(100, color="white", lw=0.7, ls=":", alpha=0.5)
    for bar, val in zip(bars, pct_vals):
        ax.text(min(val + 1, 98), bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color="white", fontsize=9)
    ax.set_xlabel("Band Power Preservation (%) vs PRE (No-DBS) Baseline", color="white")
    ax.set_title("Biomarker Integrity Check", color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="white")
    ax.set_xlim(0, 115)
    ax.legend(facecolor="#1c1f26", labelcolor="white", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ax.spines.values():
        sp.set_edgecolor("#555555")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"  Biomarker bar plot → {out_path.name}")


def save_realtime_dashboard(raw_uv, final_uv, sfreq, ch_names,
                             f_dbs, out_path: pathlib.Path,
                             window_sec: float = 5.0,
                             step_sec:   float = 0.5,
                             fps:        int   = 6):
    """
    Build the real-time sliding-window dashboard (Raw vs Final) and save as GIF.

    Layout
    ------
    Top    : sliding time-domain traces (Raw grey, Final blue)
    Bottom : live PSD updated every frame (Raw grey, Final blue)
    """
    try:
        ci = ch_names.index(ANALYSIS_CH)
    except ValueError:
        ci = 0

    n_s   = min(raw_uv.shape[1], final_uv.shape[1])
    w_s   = int(window_sec * sfreq)
    step  = int(step_sec   * sfreq)
    n_frames = max(1, (n_s - w_s) // step + 1)
    nyq   = sfreq / 2.0

    sig_raw = raw_uv[ci,   :n_s]
    sig_fin = final_uv[ci, :n_s]

    fig = plt.figure(figsize=(13, 7), facecolor="#0e1117")
    gs  = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.4)

    # ── Time panel ────────────────────────────────────────────────────────
    ax_t = fig.add_subplot(gs[0])
    ax_t.set_facecolor("#0e1117")
    ax_t.tick_params(colors="white")
    ax_t.set_ylabel("Amplitude (µV)", color="white")
    ax_t.set_xlabel("Time (s)", color="white")
    ax_t.spines[["top", "right"]].set_visible(False)
    for sp in ax_t.spines.values():
        sp.set_edgecolor("#555555")

    line_r, = ax_t.plot([], [], color="#888888", lw=0.8, label="Raw",  alpha=0.7)
    line_f, = ax_t.plot([], [], color="#4da6ff", lw=1.3, label="Final clean")
    ttl = ax_t.set_title("", color="white", fontsize=10)
    ax_t.legend(facecolor="#1c1f26", labelcolor="white", fontsize=8, loc="upper right")

    # DBS harmonic text annotation
    harm_lines = []
    for k in range(1, 15):
        h = k * f_dbs
        if h > 50:
            break
        vl = ax_t.axvline(h, color="#f9c74f", lw=0.0)   # invisible placeholder
        harm_lines.append(vl)

    # ── PSD panel ─────────────────────────────────────────────────────────
    ax_p = fig.add_subplot(gs[1])
    ax_p.set_facecolor("#0e1117")
    ax_p.tick_params(colors="white")
    ax_p.set_ylabel("PSD (dB µV²/Hz)", color="white")
    ax_p.set_xlabel("Frequency (Hz)", color="white")
    ax_p.set_xlim(0.5, 50)
    ax_p.spines[["top", "right"]].set_visible(False)
    for sp in ax_p.spines.values():
        sp.set_edgecolor("#555555")

    line_rp, = ax_p.plot([], [], color="#888888", lw=0.9, label="Raw")
    line_fp, = ax_p.plot([], [], color="#4da6ff", lw=1.2, label="Final")
    for k in range(1, 15):
        h = k * f_dbs
        if h > nyq:
            break
        ax_p.axvline(h, color="#f9c74f", lw=0.5, ls=":", alpha=0.5)
    for _, lo, hi, col in BANDS:
        ax_p.axvspan(lo, hi, color=col, alpha=0.07)
    ax_p.legend(facecolor="#1c1f26", labelcolor="white", fontsize=7, loc="upper right")

    # update frequencies up to LPF_HZ for the dashboard PSD
    dash_fmax = LPF_HZ + 5.0
    ax_p.set_xlim(0.5, dash_fmax)
    
    def update(frame):
        s = frame * step
        e = min(s + w_s, n_s)
        times = np.arange(s, e) / sfreq

        line_r.set_data(times, sig_raw[s:e])
        line_f.set_data(times, sig_fin[s:e])

        all_amp = np.concatenate([sig_raw[s:e], sig_fin[s:e]])
        pad = max(5.0, 0.1 * float(np.ptp(all_amp) or 10.0))
        ax_t.set_xlim(times[0], times[-1])
        ax_t.set_ylim(all_amp.min() - pad, all_amp.max() + pad)

        ttl.set_text(f"Real-Time Dashboard — ch {ch_names[ci]}  |  "
                     f"t = {times[0]:.1f}–{times[-1]:.1f} s")

        # Live PSD from current window
        nperseg = min(512, e - s)
        if nperseg > 4:
            fr, pr = sp_signal.welch(sig_raw[s:e], fs=sfreq, nperseg=nperseg)
            ff, pf = sp_signal.welch(sig_fin[s:e], fs=sfreq, nperseg=nperseg)
            mr = (fr >= 0.5) & (fr <= dash_fmax)
            mf = (ff >= 0.5) & (ff <= dash_fmax)
            dBr = 10 * np.log10(pr[mr] + 1e-30)
            dBf = 10 * np.log10(pf[mf] + 1e-30)
            line_rp.set_data(fr[mr], dBr)
            line_fp.set_data(ff[mf], dBf)
            all_db = np.concatenate([dBr, dBf])
            ax_p.set_ylim(all_db.min() - 3, all_db.max() + 3)

        return line_r, line_f, line_rp, line_fp, ttl

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)
    writer = PillowWriter(fps=fps)
    anim.save(str(out_path), writer=writer)
    plt.close(fig)
    log.info(f"  Real-time dashboard GIF → {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# MASTER PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

# Map DBS files to their corresponding PRE (no-DBS) recording
PRE_FILES = {
    "XUAWAKE7_deidentified.edf":    "XUAWAKEPRE_deidentified.edf",
    "XUSLEEP7_deidentified.edf":    "XUSLEEP_deidentified.edf",
    "XUAWAKE60_deidentified.edf":   "XUAWAKEPRE_deidentified.edf",
    "XUSLEEP60_deidentified.edf":   "XUSLEEP_deidentified.edf",
    "XUAWAKET100_deidentified.edf": "XUAWAKEPRE_deidentified.edf",
    "XUSLEEPT100_deidentified.edf": "XUSLEEP_deidentified.edf",
}


def run_pipeline(edf_path: pathlib.Path, f_dbs: float = 7.0,
                  run_ica: bool = True) -> dict:
    """
    Master pipeline entry point that coordinates the surgical removal
    and generates the full validation report.
    """
    stem   = edf_path.stem
    out_d  = OUTPUT_DIR / stem
    out_d.mkdir(parents=True, exist_ok=True)

    log.info(f"Starting master pipeline for {edf_path.name}")
    
    # ── Stage I-III: Process DBS recording via core library ──────────────
    raw_mne_full, stage1_raw, final_raw = run_surgical_pipeline(edf_path, f_dbs, run_ica=run_ica)
    
    sfreq    = final_raw.info["sfreq"]
    ch_names = final_raw.ch_names
    dbs_uv   = stage1_raw.get_data() * 1e6   # Stage I DBS baseline (µV)

    # ── Load PRE (no-DBS) recording as physiological baseline ─────────────
    pre_fname = PRE_FILES.get(edf_path.name)
    pre_path  = edf_path.parent / pre_fname if pre_fname else None
    
    if pre_path and pre_path.exists():
        log.info(f"Loading PRE (no-DBS) baseline: {pre_path.name}")
        # Process PRE exactly like DBS Stage I
        pre_preprocessor = EEGPreprocessor()
        pre_raw_orig = mne.io.read_raw_edf(str(pre_path), preload=True, verbose=False)
        pre_raw_orig.crop(tmin=0, tmax=min(MAX_DUR_SEC, pre_raw_orig.times[-1]))
        pre_raw_std = pre_preprocessor.standardize_channels(pre_raw_orig)
        pre_raw_stage1 = pre_preprocessor.apply_clinical_filter(pre_raw_std, l_freq=0.1, h_freq=100.0)
        
        # Align channels
        common    = [c for c in ch_names if c in pre_raw_stage1.ch_names]
        dbs_uv    = stage1_raw.get_data()[[ch_names.index(c) for c in common]] * 1e6
        final_uv  = final_raw.get_data()[[ch_names.index(c) for c in common]] * 1e6
        pre_uv_data = pre_raw_stage1.get_data()[[pre_raw_stage1.ch_names.index(c) for c in common]] * 1e6
        
        # Truncate to same length
        n_samp    = min(dbs_uv.shape[1], pre_uv_data.shape[1], final_uv.shape[1])
        baseline_uv = pre_uv_data[:, :n_samp]
        dbs_uv      = dbs_uv[:, :n_samp]
        final_uv    = final_uv[:, :n_samp]
        ch_names_use = common
    else:
        log.warning("No PRE file found — using Stage-I DBS output as baseline (less accurate).")
        baseline_uv  = dbs_uv
        final_uv     = final_raw.get_data() * 1e6
        ch_names_use = ch_names
        n_samp       = dbs_uv.shape[1]

    # ── Biomarker table (vs PRE/no-DBS baseline) ──────────────────────────
    table = compute_biomarker_table(dbs_uv, baseline_uv, final_uv, sfreq, f_dbs)

    # ── Print summary ─────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info(" BIOMARKER INTEGRITY REPORT  (vs PRE no-DBS baseline)")
    log.info("=" * 60)
    for band, vals in table.items():
        goal = "✓" if vals.get("goal_met (>90%)", False) else "✗"
        if band == "DBS_removal":
            log.info(f"  DBS reduction:  {vals['reduction_%']:.1f}%   {goal}")
        else:
            log.info(f"  {band:6s}  preservation: {vals['preservation_%']:6.1f}%  {goal}")
    log.info("=" * 60)

    with open(out_d / "biomarker_integrity.json", "w") as fh:
        json.dump(table, fh, indent=2)

    # ── Validation plots ──────────────────────────────────────────────────
    log.info("Generating validation plots …")

    plot_triple_psd(dbs_uv, baseline_uv, dbs_uv, final_uv, sfreq, f_dbs,
                    out_d / "01_triple_psd.png")

    plot_time_domain_comparison(dbs_uv, dbs_uv, final_uv, sfreq,
                                 ch_names_use, f_dbs,
                                 t_offset=10.0, dur=5.0,
                                 out_path=out_d / "02_time_domain.png")

    plot_scalp_topomaps(dbs_uv, final_uv, final_raw, ch_names_use,
                         out_d / "03_scalp_topomaps.png")

    plot_biomarker_bars(table, out_d / "04_biomarker_bars.png")

    log.info("Generating real-time dashboard GIF …")
    save_realtime_dashboard(dbs_uv, final_uv, sfreq, ch_names_use, f_dbs,
                             out_d / "05_realtime_dashboard.gif",
                             window_sec=5.0, step_sec=0.5, fps=6)

    log.info(f"\nAll outputs → {out_d}")
    return {"biomarkers": table, "output_dir": str(out_d)}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

RECORDINGS = {
    7.0:   {"awake": "XUAWAKE7_deidentified.edf",   "sleep": "XUSLEEP7_deidentified.edf"},
    60.0:  {"awake": "XUAWAKE60_deidentified.edf",  "sleep": "XUSLEEP60_deidentified.edf"},
    100.0: {"awake": "XUAWAKET100_deidentified.edf","sleep": "XUSLEEPT100_deidentified.edf"},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG DBS Cleaning Pipeline")
    parser.add_argument("--file",           default="data/raw/XU/XUAWAKE7_deidentified.edf")
    parser.add_argument("--dbs-freq",  type=float, default=7.0)
    parser.add_argument("--no-ica",    action="store_true")
    parser.add_argument("--all-conditions", action="store_true",
                        help="Run all recordings (7, 60, 100 Hz × awake/sleep)")
    args = parser.parse_args()

    data_dir = pathlib.Path("data/raw/XU")

    if args.all_conditions:
        for f_dbs, conds in RECORDINGS.items():
            for cond, fname in conds.items():
                p = data_dir / fname
                if p.exists():
                    log.info(f"\n{'='*60}\n  {cond.upper()} {f_dbs} Hz\n{'='*60}")
                    run_pipeline(p, f_dbs=f_dbs, run_ica=not args.no_ica)
                else:
                    log.warning(f"Not found: {p}")
    else:
        p = pathlib.Path(args.file)
        if not p.exists():
            log.error(f"File not found: {p}")
            sys.exit(1)
        run_pipeline(p, f_dbs=args.dbs_freq, run_ica=not args.no_ica)
