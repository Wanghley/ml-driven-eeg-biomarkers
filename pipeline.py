#!/usr/bin/env python3
"""pipeline.py — EEG biomarker pipeline orchestration.

Target data flow:
  Raw EDF → Ingestion & Channel Typing (src.ingestion)
         → Consensus Filtering + DBS Removal + ICA (src.artifact_removal)
         → Spike Morphology Extraction (src.spike_features)
         → Biomarker QC / Validation Export

Usage
-----
  python3 pipeline.py                                    # default: XUAWAKE7
  python3 pipeline.py --file data/raw/XU/XUAWAKE7_deidentified.edf --dbs-freq 7.0
  python3 pipeline.py --no-ica                           # skip ICA (faster)
  python3 pipeline.py --all-conditions                   # run all EDF files
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

# ── Project imports ──────────────────────────────────────────────────────────
_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.ingestion import IngestionConfig, load_edf
from src.artifact_removal import ArtifactRemovalConfig, run_artifact_removal
from src.spike_features import SpikeDetectionConfig, channel_summary, extract_spike_features


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TARGET_SFREQ  = 256.0
MAX_DUR_SEC   = 120.0
HPF_HZ        = 0.1        # high-pass passed to ArtifactRemovalConfig
LPF_HZ        = 100.0      # low-pass
LINE_NOISE_HZ = 60.0
ICA_N_COMP    = 15
ANALYSIS_CH   = "Cz"

OUTPUT_DIR = pathlib.Path("output/validation_results")

BANDS = [
    ("Delta", 0.5,  4,  "#8c564b"),
    ("Theta",  4,   8,  "#9467bd"),
    ("Alpha",  8,  13,  "#1f77b4"),
    ("Beta",  13,  30,  "#2ca02c"),
    ("Gamma", 30, 100,  "#d62728"),
]

# DBS filename → matching PRE (no-DBS) baseline recording
PRE_FILES = {
    "XUAWAKE7_deidentified.edf":    "XUAWAKEPRE_deidentified.edf",
    "XUSLEEP7_deidentified.edf":    "XUSLEEP_deidentified.edf",
    "XUAWAKE60_deidentified.edf":   "XUAWAKEPRE_deidentified.edf",
    "XUSLEEP60_deidentified.edf":   "XUSLEEP_deidentified.edf",
    "XUAWAKET100_deidentified.edf": "XUAWAKEPRE_deidentified.edf",
    "XUSLEEPT100_deidentified.edf": "XUSLEEP_deidentified.edf",
}

RECORDINGS = {
    7.0:   {"awake": "XUAWAKE7_deidentified.edf",    "sleep": "XUSLEEP7_deidentified.edf"},
    60.0:  {"awake": "XUAWAKE60_deidentified.edf",   "sleep": "XUSLEEP60_deidentified.edf"},
    100.0: {"awake": "XUAWAKET100_deidentified.edf", "sleep": "XUSLEEPT100_deidentified.edf"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _band_power(data_uv: np.ndarray, sfreq: float,
                lo: float, hi: float, n_fft: int = 2048) -> float:
    """Mean Welch band power (µV²) across all channels."""
    f, p = sp_signal.welch(data_uv, fs=sfreq, nperseg=n_fft,
                            noverlap=n_fft // 2, axis=1)
    m = (f >= lo) & (f <= hi)
    return float(np.trapz(p[:, m].mean(axis=0), f[m])) if m.any() else 0.0


def _harmonic_power(data_uv: np.ndarray, sfreq: float,
                    f_dbs: float, bw: float = 0.5, n_fft: int = 2048) -> float:
    """Sum of Welch power within ±bw Hz of all DBS harmonics below Nyquist."""
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


def _biomarker_table(baseline_uv: np.ndarray, final_uv: np.ndarray,
                     sfreq: float, f_dbs: float) -> dict:
    """Band-preservation and DBS-reduction metrics vs a clean baseline."""
    out = {}
    for name, lo, hi, _ in BANDS:
        bp_base = _band_power(baseline_uv, sfreq, lo, hi)
        bp_fin  = _band_power(final_uv,    sfreq, lo, hi)
        pres    = round(100 * bp_fin / bp_base, 2) if bp_base > 0 else 0.0
        out[name] = {
            "baseline_uV2":    round(bp_base, 3),
            "final_uV2":       round(bp_fin,  3),
            "preservation_%":  pres,
            "goal_met (>90%)": pres >= 90.0,
        }

    h_base  = _harmonic_power(baseline_uv, sfreq, f_dbs)
    h_final = _harmonic_power(final_uv,    sfreq, f_dbs)
    dbs_red = round(100 * (1 - h_final / (h_base + 1e-30)), 2) if h_base > 0 else 0.0
    out["DBS_removal"] = {
        "harmonic_power_baseline_uV2": round(h_base,  3),
        "harmonic_power_final_uV2":    round(h_final, 3),
        "reduction_%":                 dbs_red,
        "goal_met (>90%)":             dbs_red >= 90.0,
    }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Validation plots
# ─────────────────────────────────────────────────────────────────────────────

def _psd_mean(data_uv: np.ndarray, sfreq: float,
              fmin: float = 0.5, fmax: float = 55.0,
              n_fft: int = 2048, smooth_hz: float = 0.5) -> tuple:
    f, p_all = sp_signal.welch(data_uv, fs=sfreq,
                                nperseg=n_fft, noverlap=n_fft // 2, axis=1)
    p_mean = p_all.mean(axis=0)
    if smooth_hz > 0:
        df  = f[1] - f[0]
        k   = max(1, int(round(smooth_hz / df)))
        win = np.hanning(2 * k + 1)
        win /= win.sum()
        p_mean = np.convolve(p_mean, win, mode="same")
    m  = (f >= fmin) & (f <= fmax)
    db = 10 * np.log10(p_mean[m] + 1e-30)
    return f[m], db


def plot_triple_psd(baseline_uv, dbs_uv, final_uv, sfreq, f_dbs, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(17, 6),
                              gridspec_kw={"wspace": 0.30})
    fig.patch.set_facecolor("#0e1117")

    layers = [
        (dbs_uv,      "Stage I — DBS (preprocessed)",  "#e07b39", 1.2, "--"),
        (baseline_uv, "PRE baseline (no DBS)",          "#44cf6c", 1.6, "-."),
        (final_uv,    "Final — DBS removed + ICA",      "#4da6ff", 2.2, "-"),
    ]
    panels = [
        (axes[0], 0.5, LPF_HZ + 5.0, f"Full spectrum  0.5 – {LPF_HZ + 5.0:.0f} Hz"),
        (axes[1], 1.0, 30.0,          "Clinical zoom  1 – 30 Hz"),
    ]

    curves = []
    all_db_vals = []
    for data, lbl, col, lw, ls in layers:
        fmax_c = min(sfreq / 2.0, LPF_HZ + 10.0)
        f50, db50 = _psd_mean(data, sfreq, fmin=0.5, fmax=fmax_c)
        curves.append((f50, db50, lbl, col, lw, ls))
        all_db_vals.append(db50)

    global_min = np.percentile(np.concatenate(all_db_vals), 2)
    global_max = np.percentile(np.concatenate(all_db_vals), 98)

    for ax, fmin_p, fmax_p, title in panels:
        ax.set_facecolor("#0e1117")
        for f_all, db_all, lbl, col, lw, ls in curves:
            m = (f_all >= fmin_p) & (f_all <= fmax_p)
            if m.sum() >= 2:
                ax.plot(f_all[m], db_all[m], color=col, lw=lw, ls=ls,
                        label=lbl, alpha=0.92, zorder=3)

        for name, lo, hi, col in BANDS:
            if hi < fmin_p or lo > fmax_p:
                continue
            lo_c, hi_c = max(lo, fmin_p), min(hi, fmax_p)
            ax.axvspan(lo_c, hi_c, color=col, alpha=0.08, zorder=1)
            ax.text((lo_c + hi_c) / 2, global_max - 1, name,
                    ha="center", va="top", fontsize=6.5,
                    color=col, alpha=0.85, fontweight="bold")

        for k in range(1, 50):
            h = k * f_dbs
            if h < fmin_p:
                continue
            if h > fmax_p:
                break
            ax.axvline(h, color="#f9c74f", lw=0.8, ls=":", alpha=0.55, zorder=2,
                       label="DBS harmonic" if k == 1 else "")
            ax.text(h, global_max + 0.5, f"{k}×",
                    ha="center", va="bottom", fontsize=5.5, color="#f9c74f", alpha=0.7)

        if fmax_p >= LPF_HZ and ax is axes[0]:
            ax.axvline(LPF_HZ, color="#ff6b6b", lw=1.2, ls="--", alpha=0.7,
                       label=f"LPF {LPF_HZ:.0f} Hz")

        pad = max(3.0, (global_max - global_min) * 0.06)
        ax.set_ylim(global_min - pad, global_max + pad + 4)
        ax.set_xlim(fmin_p, fmax_p)
        ax.set_xlabel("Frequency (Hz)", color="white", fontsize=10)
        ax.set_ylabel("PSD  (dB µV²/Hz)", color="white", fontsize=10)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold")
        ax.tick_params(colors="white", labelsize=8)
        ax.grid(True, color="#2a2a3a", lw=0.5, ls="--", alpha=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444455")
        if ax is axes[0]:
            ax.legend(fontsize=8, facecolor="#1c1f26", labelcolor="white",
                      loc="lower left", framealpha=0.85, edgecolor="#555566")

    fig.suptitle("Triple-Overlay PSD  —  DBS Stage-I  ●  PRE Baseline  ●  Final Cleaned",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"  PSD plot → {out_path.name}")


def plot_time_domain_comparison(dbs_uv, baseline_uv, final_uv,
                                 sfreq, ch_names, f_dbs,
                                 t_offset, dur, out_path):
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
        (dbs_uv,      "Stage I — DBS (preprocessed)",  "#e07b39"),
        (baseline_uv, "PRE baseline (no DBS)",          "#44cf6c"),
        (final_uv,    "Final — DBS + ICA cleaned",      "#4da6ff"),
    ]
    for ax, (data, lbl, col) in zip(axes, triples):
        ax.set_facecolor("#0e1117")
        ax.plot(t, data[ci, i0:i1], color=col, lw=0.9)
        ax.set_ylabel(lbl, color="white", fontsize=8)
        ax.tick_params(colors="white")
        ax.spines[["top", "right"]].set_visible(False)
        for sp in ax.spines.values():
            sp.set_edgecolor("#555555")
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


def plot_scalp_topomaps(dbs_uv, final_uv, raw_mne, ch_names, out_path):
    try:
        info    = raw_mne.info.copy()
        ch_idx  = mne.pick_types(info, eeg=True, exclude=[])
        if len(ch_idx) < 4:
            log.warning("Fewer than 4 EEG channels — skipping topomap.")
            return

        amp_before = np.abs(dbs_uv).mean(axis=1)
        amp_after  = np.abs(final_uv).mean(axis=1)
        vmax       = max(amp_before.max(), amp_after.max())
        diff       = amp_before - amp_after

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.patch.set_facecolor("#0e1117")
        for ax, topo_data, title, cmap, vmin_p, vmax_p in [
            (axes[0], amp_before, "Before DBS Removal\n(Stage I)", "hot", 0.0, vmax),
            (axes[1], amp_after,  "After DBS + ICA\n(Final)",      "hot", 0.0, vmax),
            (axes[2], diff,       "Difference\n(Artifact Removed)", "RdBu_r",
             min(0, diff.min()), max(diff.max(), 1e-3)),
        ]:
            ax.set_facecolor("#0e1117")
            mne.viz.plot_topomap(topo_data, info, axes=ax, show=False,
                                  cmap=cmap, vlim=(vmin_p, vmax_p), sphere="auto")
            ax.set_title(title, color="white", fontsize=9)

        fig.suptitle("Scalp Topomaps — Mean Absolute Amplitude (µV)",
                     color="white", fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info(f"  Topomap plot → {out_path.name}")
    except Exception as exc:
        log.warning(f"Topomap rendering failed: {exc}")


def plot_biomarker_bars(table: dict, out_path: pathlib.Path):
    band_names = [b for b in table if b != "DBS_removal"]
    pct_vals   = [table[b]["preservation_%"] for b in band_names]
    colors     = ["#2ca02c" if v >= 90 else "#d62728" for v in pct_vals]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    bars = ax.barh(band_names, pct_vals, color=colors, height=0.5)
    ax.axvline(90,  color="#f9c74f", lw=1.2, ls="--", label="90% target")
    ax.axvline(100, color="white",   lw=0.7, ls=":",  alpha=0.5)
    for bar, val in zip(bars, pct_vals):
        ax.text(min(val + 1, 98), bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color="white", fontsize=9)
    ax.set_xlabel("Band Power Preservation (%) vs PRE (no-DBS) Baseline", color="white")
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


def save_realtime_dashboard(dbs_uv, final_uv, sfreq, ch_names, f_dbs,
                             out_path: pathlib.Path,
                             window_sec: float = 5.0,
                             step_sec: float = 0.5,
                             fps: int = 6):
    try:
        ci = ch_names.index(ANALYSIS_CH)
    except ValueError:
        ci = 0

    n_s     = min(dbs_uv.shape[1], final_uv.shape[1])
    w_s     = int(window_sec * sfreq)
    step    = int(step_sec   * sfreq)
    n_frames = max(1, (n_s - w_s) // step + 1)
    dash_fmax = LPF_HZ + 5.0

    sig_dbs = dbs_uv[ci,   :n_s]
    sig_fin = final_uv[ci, :n_s]

    fig = plt.figure(figsize=(13, 7), facecolor="#0e1117")
    gs  = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.4)

    ax_t = fig.add_subplot(gs[0])
    ax_t.set_facecolor("#0e1117")
    ax_t.tick_params(colors="white")
    ax_t.set_ylabel("Amplitude (µV)", color="white")
    ax_t.set_xlabel("Time (s)", color="white")
    ax_t.spines[["top", "right"]].set_visible(False)
    for sp in ax_t.spines.values():
        sp.set_edgecolor("#555555")
    line_r, = ax_t.plot([], [], color="#e07b39", lw=0.8, label="Stage I (DBS)", alpha=0.7)
    line_f, = ax_t.plot([], [], color="#4da6ff", lw=1.3, label="Final cleaned")
    ttl = ax_t.set_title("", color="white", fontsize=10)
    ax_t.legend(facecolor="#1c1f26", labelcolor="white", fontsize=8, loc="upper right")

    ax_p = fig.add_subplot(gs[1])
    ax_p.set_facecolor("#0e1117")
    ax_p.tick_params(colors="white")
    ax_p.set_ylabel("PSD (dB µV²/Hz)", color="white")
    ax_p.set_xlabel("Frequency (Hz)", color="white")
    ax_p.set_xlim(0.5, dash_fmax)
    ax_p.spines[["top", "right"]].set_visible(False)
    for sp in ax_p.spines.values():
        sp.set_edgecolor("#555555")
    line_rp, = ax_p.plot([], [], color="#e07b39", lw=0.9, label="Stage I")
    line_fp, = ax_p.plot([], [], color="#4da6ff", lw=1.2, label="Final")
    for k in range(1, 15):
        h = k * f_dbs
        if h > sfreq / 2:
            break
        ax_p.axvline(h, color="#f9c74f", lw=0.5, ls=":", alpha=0.5)
    for _, lo, hi, col in BANDS:
        ax_p.axvspan(lo, hi, color=col, alpha=0.07)
    ax_p.legend(facecolor="#1c1f26", labelcolor="white", fontsize=7, loc="upper right")

    def _update(frame):
        s = frame * step
        e = min(s + w_s, n_s)
        times = np.arange(s, e) / sfreq
        line_r.set_data(times, sig_dbs[s:e])
        line_f.set_data(times, sig_fin[s:e])
        all_amp = np.concatenate([sig_dbs[s:e], sig_fin[s:e]])
        pad = max(5.0, 0.1 * float(np.ptp(all_amp) or 10.0))
        ax_t.set_xlim(times[0], times[-1])
        ax_t.set_ylim(all_amp.min() - pad, all_amp.max() + pad)
        ttl.set_text(f"Real-Time Dashboard — ch {ch_names[ci]}  |  "
                     f"t = {times[0]:.1f}–{times[-1]:.1f} s")
        nperseg = min(512, e - s)
        if nperseg > 4:
            fr, pr = sp_signal.welch(sig_dbs[s:e], fs=sfreq, nperseg=nperseg)
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

    anim = FuncAnimation(fig, _update, frames=n_frames, interval=1000 // fps, blit=False)
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    log.info(f"  Real-time dashboard GIF → {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Master pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(edf_path: pathlib.Path, f_dbs: float = 7.0,
                  run_ica: bool = True) -> dict:
    """Run the full EEG biomarker pipeline for one EDF recording.

    Stages:
      I.  Ingestion — load_edf with strict channel typing (EEG/EOG/EMG preserved).
      II. Artifact removal — consensus filter → DBS removal → ICA cleanup.
      III.Spike features — morphology descriptors on clean EEG.
      IV. Validation — biomarker table, plots, JSON/CSV export.

    Returns:
        Dict with keys ``biomarkers`` (band integrity table) and ``output_dir``.
    """
    stem  = edf_path.stem
    out_d = OUTPUT_DIR / stem
    out_d.mkdir(parents=True, exist_ok=True)

    log.info(f"\n{'='*60}\n  {edf_path.name}  —  DBS {f_dbs} Hz\n{'='*60}")

    # ── Stage I: Ingestion ────────────────────────────────────────────────────
    log.info("Stage I — Ingestion & channel typing")
    ingest = load_edf(edf_path, IngestionConfig(max_duration_sec=MAX_DUR_SEC, dbs_freq=f_dbs))
    log.info(
        f"  {ingest.metadata['n_eeg']} EEG  "
        f"{ingest.metadata['n_eog']} EOG  "
        f"{ingest.metadata['n_emg']} EMG  "
        f"@ {ingest.metadata['sfreq']:.0f} Hz  "
        f"({ingest.metadata['duration_sec']:.1f}s)"
    )

    # ── Stage II: Artifact removal ────────────────────────────────────────────
    log.info("Stage II — Consensus filter + DBS removal + ICA")
    ar_config = ArtifactRemovalConfig(
        dbs_freq=f_dbs,
        l_freq=HPF_HZ,
        h_freq=LPF_HZ,
        line_freq=LINE_NOISE_HZ,
        dbs_method="spectrum_fit",
        run_ica=run_ica,
        ica_n_components=ICA_N_COMP,
    )
    ar_result = run_artifact_removal(ingest.raw, ar_config)
    log.info(
        f"  ICA excluded {len(ar_result.ica_excluded)} components: "
        f"{ar_result.ica_classification}"
    )

    # ── Stage III: Spike feature extraction ───────────────────────────────────
    log.info("Stage III — Spike morphology extraction")
    spike_cfg = SpikeDetectionConfig(
        threshold_mad_k=5.0,
        min_amplitude_uv=50.0,
        eeg_only=True,
    )
    spike_df = extract_spike_features(ar_result.raw, spike_cfg)
    spike_summary = channel_summary(spike_df, duration_sec=ar_result.metadata["duration_sec"])
    log.info(f"  {len(spike_df)} spikes detected across {spike_df['channel'].nunique()} channels")

    spike_df.to_csv(out_d / "spikes.csv", index=False)
    spike_summary.to_csv(out_d / "spike_summary.csv", index=False)
    log.info(f"  Spike tables → {out_d / 'spikes.csv'}")

    # ── Stage IV: Validation — biomarker table ────────────────────────────────
    log.info("Stage IV — Biomarker validation")
    sfreq      = float(ar_result.raw.info["sfreq"])
    ch_names   = ar_result.raw.ch_names
    final_uv   = ar_result.raw.get_data() * 1e6

    # Load PRE (no-DBS) recording as the physiological baseline for band comparison.
    # Pre-filtered during ingestion (no artifact_removal pass needed — baseline reference only).
    pre_fname = PRE_FILES.get(edf_path.name)
    pre_path  = edf_path.parent / pre_fname if pre_fname else None

    if pre_path and pre_path.exists():
        log.info(f"  Loading PRE baseline: {pre_path.name}")
        pre_result  = load_edf(pre_path, IngestionConfig(
            max_duration_sec=MAX_DUR_SEC, l_freq=HPF_HZ, h_freq=LPF_HZ,
        ))
        common   = [c for c in ch_names if c in pre_result.raw.ch_names]
        final_uv = ar_result.raw.get_data(
            picks=[ch_names.index(c) for c in common]
        ) * 1e6
        pre_uv = pre_result.raw.get_data(
            picks=[pre_result.raw.ch_names.index(c) for c in common]
        ) * 1e6
        n_samp    = min(final_uv.shape[1], pre_uv.shape[1])
        final_uv  = final_uv[:, :n_samp]
        baseline_uv = pre_uv[:, :n_samp]
        ch_names_use = common
    else:
        log.warning("No PRE file found — using Stage-II output as its own baseline (less accurate).")
        baseline_uv  = final_uv
        ch_names_use = ch_names

    table = _biomarker_table(baseline_uv, final_uv, sfreq, f_dbs)

    log.info("─" * 60)
    log.info("  BIOMARKER INTEGRITY  (vs PRE no-DBS baseline)")
    log.info("─" * 60)
    for band, vals in table.items():
        goal = "✓" if vals.get("goal_met (>90%)", False) else "✗"
        if band == "DBS_removal":
            log.info(f"  DBS reduction:  {vals['reduction_%']:.1f}%   {goal}")
        else:
            log.info(f"  {band:6s}  preservation: {vals['preservation_%']:6.1f}%  {goal}")

    with open(out_d / "biomarker_integrity.json", "w") as fh:
        json.dump(table, fh, indent=2)

    # ── Validation plots ──────────────────────────────────────────────────────
    log.info("Generating validation plots …")
    dbs_uv = ar_result.raw.get_data(
        picks=[ch_names.index(c) for c in ch_names_use]
    ) * 1e6 if ch_names_use != ch_names else final_uv

    plot_triple_psd(baseline_uv, dbs_uv, final_uv, sfreq, f_dbs,
                    out_d / "01_triple_psd.png")
    plot_time_domain_comparison(dbs_uv, baseline_uv, final_uv,
                                 sfreq, ch_names_use, f_dbs,
                                 t_offset=10.0, dur=5.0,
                                 out_path=out_d / "02_time_domain.png")
    plot_scalp_topomaps(dbs_uv, final_uv, ar_result.raw, ch_names_use,
                         out_d / "03_scalp_topomaps.png")
    plot_biomarker_bars(table, out_d / "04_biomarker_bars.png")
    save_realtime_dashboard(dbs_uv, final_uv, sfreq, ch_names_use, f_dbs,
                             out_d / "05_realtime_dashboard.gif",
                             window_sec=5.0, step_sec=0.5, fps=6)

    log.info(f"\nAll outputs → {out_d}")
    return {"biomarkers": table, "output_dir": str(out_d)}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG DBS Cleaning Pipeline")
    parser.add_argument("--file",             default="data/raw/XU/XUAWAKE7_deidentified.edf")
    parser.add_argument("--dbs-freq",   type=float, default=7.0)
    parser.add_argument("--no-ica",     action="store_true")
    parser.add_argument("--all-conditions", action="store_true",
                        help="Run all recordings (7, 60, 100 Hz × awake/sleep)")
    args = parser.parse_args()

    data_dir = pathlib.Path("data/raw/XU")

    if args.all_conditions:
        for f_dbs, conds in RECORDINGS.items():
            for cond, fname in conds.items():
                p = data_dir / fname
                if p.exists():
                    run_pipeline(p, f_dbs=f_dbs, run_ica=not args.no_ica)
                else:
                    log.warning(f"Not found: {p}")
    else:
        p = pathlib.Path(args.file)
        if not p.exists():
            log.error(f"File not found: {p}")
            sys.exit(1)
        run_pipeline(p, f_dbs=args.dbs_freq, run_ica=not args.no_ica)
