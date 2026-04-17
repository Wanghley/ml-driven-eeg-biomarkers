#!/usr/bin/env python3
"""
generate_dual_theme_plots.py
============================
Generates dark-background and light-background versions of every pipeline
output plot without modifying any existing source file.

Outputs
-------
  output/validation_results/XUAWAKE7_deidentified/dark/   (01–04 plots)
  output/validation_results/XUAWAKE7_deidentified/light/  (01–04 plots)
  results/experiments_7Hz_DBS/dark/   (psd, time-domain, beta-summary)
  results/experiments_7Hz_DBS/light/  (psd, time-domain, beta-summary)

Usage
-----
  conda run -n ml-eeg-biomarkers python generate_dual_theme_plots.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

import importlib.util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
import mne
mne.set_log_level("ERROR")
import numpy as np
from scipy import signal as sp_signal

# ── project root on path ──────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Load pipeline.py directly (avoids shadowing by the pipeline/ package) ────
def _load_module(name: str, fpath: pathlib.Path):
    spec   = importlib.util.spec_from_file_location(name, str(fpath))
    mod    = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_pl = _load_module("_pipeline_script", ROOT / "pipeline.py")
run_surgical_pipeline = _pl.run_surgical_pipeline
compute_biomarker_table = _pl.compute_biomarker_table
_psd_mean  = _pl._psd_mean
BANDS      = _pl.BANDS
LPF_HZ     = _pl.LPF_HZ
ANALYSIS_CH = _pl.ANALYSIS_CH
PRE_FILES  = _pl.PRE_FILES
MAX_DUR_SEC = _pl.MAX_DUR_SEC

from src.ingestion import load_edf, IngestionConfig
from src.preprocessing import EEGPreprocessor

# ── Stub out pipeline package to avoid dagster import in experimental_dbs_eval ─
import types as _types
_pipeline_stub = _types.ModuleType("pipeline")
_pipeline_constants = _types.ModuleType("pipeline.constants")

# Read the real constants directly (constants.py doesn't import dagster)
_constants_spec = importlib.util.spec_from_file_location(
    "pipeline.constants", str(ROOT / "pipeline" / "constants.py"))
_constants_mod = importlib.util.module_from_spec(_constants_spec)
_constants_spec.loader.exec_module(_constants_mod)

_pipeline_constants.STANDARD_CH = _constants_mod.STANDARD_CH
_pipeline_constants.BANDS       = _constants_mod.BANDS
_pipeline_constants.DBS_METHODS = _constants_mod.DBS_METHODS
_pipeline_stub.constants        = _pipeline_constants
sys.modules.setdefault("pipeline",          _pipeline_stub)
sys.modules.setdefault("pipeline.constants", _pipeline_constants)

# ── Experimental eval imports ─────────────────────────────────────────────────
_ev = _load_module("_eval_script", ROOT / "experimental_dbs_eval.py")
load_and_preprocess_edf  = _ev.load_and_preprocess_edf
raw_to_uv                = _ev.raw_to_uv
apply_svd_nlms           = _ev.apply_svd_nlms
apply_complex_hampel_fft = _ev.apply_complex_hampel_fft
apply_template_subtraction = _ev.apply_template_subtraction
apply_hybrid_abc         = _ev.apply_hybrid_abc
compute_psd              = _ev.compute_psd
METHOD_PALETTE           = _ev.METHOD_PALETTE
EVAL_BANDS               = _ev.BANDS
EVAL_SFREQ               = _ev.TARGET_SFREQ

# ─────────────────────────────────────────────────────────────────────────────
# Theme definitions
# ─────────────────────────────────────────────────────────────────────────────
THEMES: dict[str, dict] = {
    "dark": {
        "bg":          "#0e1117",
        "legend_bg":   "#1c1f26",
        "grid_color":  "#2a2a3a",
        "spine_color": "#444455",
        "text":        "white",
        "tick":        "white",
        "grid_alpha":  0.6,
        "ref_line":    "#f9c74f",
        "ref_zero":    "white",
        "dbs_marker":  "#f9c74f",
        "mpl_style":   "dark_background",
    },
    "light": {
        "bg":          "#ffffff",
        "legend_bg":   "#f8f8f8",
        "grid_color":  "#e0e0e0",
        "spine_color": "#aaaaaa",
        "text":        "#111111",
        "tick":        "#333333",
        "grid_alpha":  0.6,
        "ref_line":    "#c07c00",
        "ref_zero":    "#333333",
        "dbs_marker":  "#c07c00",
        "mpl_style":   "default",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Output paths
# ─────────────────────────────────────────────────────────────────────────────
VAL_DIR  = ROOT / "output" / "validation_results" / "XUAWAKE7_deidentified"
EVAL_DIR = ROOT / "results" / "experiments_7Hz_DBS"

for theme_name in THEMES:
    (VAL_DIR  / theme_name).mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / theme_name).mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# Shared axis styler
# ═════════════════════════════════════════════════════════════════════════════
def _style_ax(ax, t: dict, xlabel: str = "", ylabel: str = "",
              title: str = "", grid: bool = True):
    ax.set_facecolor(t["bg"])
    ax.tick_params(colors=t["tick"], labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ax.spines.values():
        sp.set_edgecolor(t["spine_color"])
    if grid:
        ax.grid(True, color=t["grid_color"], lw=0.5, ls="--",
                alpha=t["grid_alpha"])
    if xlabel:
        ax.set_xlabel(xlabel, color=t["text"], fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=t["text"], fontsize=10)
    if title:
        ax.set_title(title, color=t["text"], fontsize=11, fontweight="bold")


def _save(fig, path: pathlib.Path, t: dict):
    fig.savefig(str(path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"  Saved {path.relative_to(ROOT)}")


# ═════════════════════════════════════════════════════════════════════════════
# 1. TRIPLE PSD PLOT
# ═════════════════════════════════════════════════════════════════════════════
def plot_triple_psd_themed(raw_uv, pre_uv, stage1_uv, final_uv,
                            sfreq, f_dbs, out_path, t: dict):
    fig, axes = plt.subplots(1, 2, figsize=(17, 6),
                              gridspec_kw={"wspace": 0.30})
    fig.patch.set_facecolor(t["bg"])

    layers = [
        (stage1_uv, "Stage I — DBS (preprocessed)", "#e07b39", 1.2, "--"),
        (pre_uv,    "PRE baseline (no DBS)",         "#44cf6c", 1.6, "-."),
        (final_uv,  "Final — DBS removed + ICA",     "#4da6ff", 2.2, "-"),
    ]
    panels = [
        (axes[0], 0.5, LPF_HZ + 5.0, f"Full spectrum  0.5 – {LPF_HZ + 5.0:.0f} Hz"),
        (axes[1], 1.0, 30.0,          "Clinical zoom  1 – 30 Hz"),
    ]

    all_db_vals = []
    curves = []
    for data, lbl, col, lw, ls in layers:
        fmax_c = min(sfreq / 2.0, LPF_HZ + 10.0)
        f50, db50 = _psd_mean(data, sfreq, fmin=0.5, fmax=fmax_c)
        curves.append((f50, db50, lbl, col, lw, ls))
        all_db_vals.append(db50)

    global_min = np.percentile(np.concatenate(all_db_vals), 2)
    global_max = np.percentile(np.concatenate(all_db_vals), 98)

    for ax, fmin_p, fmax_p, panel_title in panels:
        _style_ax(ax, t, xlabel="Frequency (Hz)",
                  ylabel="PSD  (dB µV²/Hz)", title=panel_title)

        for f_all, db_all, lbl, col, lw, ls in curves:
            m   = (f_all >= fmin_p) & (f_all <= fmax_p)
            f_p = f_all[m]; d_p = db_all[m]
            if len(f_p) < 2:
                continue
            ax.plot(f_p, d_p, color=col, lw=lw, ls=ls,
                    label=lbl, alpha=0.92, zorder=3)

        for name, lo, hi, col in BANDS:
            if hi < fmin_p or lo > fmax_p:
                continue
            lo_c = max(lo, fmin_p); hi_c = min(hi, fmax_p)
            ax.axvspan(lo_c, hi_c, color=col, alpha=0.08, zorder=1)
            mid = (lo_c + hi_c) / 2
            if global_min < global_max:
                ax.text(mid, global_max - 1, name,
                        ha="center", va="top", fontsize=6.5,
                        color=col, alpha=0.85, fontweight="bold")

        for k in range(1, 50):
            h = k * f_dbs
            if h < fmin_p: continue
            if h > fmax_p: break
            ax.axvline(h, color=t["dbs_marker"], lw=0.8, ls=":",
                       alpha=0.55, zorder=2,
                       label="DBS harmonic" if k == 1 else "")
            ax.text(h, global_max + 0.5, f"{k}×",
                    ha="center", va="bottom", fontsize=5.5,
                    color=t["dbs_marker"], alpha=0.7)

        if fmax_p >= LPF_HZ and ax is axes[0]:
            ax.axvline(LPF_HZ, color="#ff6b6b", lw=1.2, ls="--",
                       alpha=0.7, label=f"LPF {LPF_HZ:.0f} Hz")
            ax.text(LPF_HZ - 0.4, global_max - 2, f"LPF\n{LPF_HZ:.0f}Hz",
                    ha="right", va="top", fontsize=7,
                    color="#ff6b6b", alpha=0.9)

        pad = max(3.0, (global_max - global_min) * 0.06)
        ax.set_ylim(global_min - pad, global_max + pad + 4)
        ax.set_xlim(fmin_p, fmax_p)

        if ax is axes[0]:
            ax.legend(fontsize=8, facecolor=t["legend_bg"],
                      labelcolor=t["text"], loc="lower left",
                      framealpha=0.85, edgecolor=t["spine_color"])

    fig.suptitle(
        "Triple-Overlay PSD  —  DBS Raw  ●  PRE Baseline (no DBS)  ●  Final Cleaned",
        color=t["text"], fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _save(fig, out_path, t)


# ═════════════════════════════════════════════════════════════════════════════
# 2. TIME-DOMAIN COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
def plot_time_domain_themed(raw_uv, prep_uv, final_uv,
                             sfreq, ch_names, f_dbs,
                             t_offset, dur, out_path, t: dict):
    try:
        ci = ch_names.index(ANALYSIS_CH)
    except ValueError:
        ci = 0

    i0 = int(t_offset * sfreq)
    i1 = i0 + int(dur * sfreq)
    tvec = np.linspace(t_offset, t_offset + dur, i1 - i0)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.patch.set_facecolor(t["bg"])

    triples = [
        (raw_uv,   "Raw (as-is)",             "#888888"),
        (prep_uv,  "Stage I Standard Prep",   "#ff7f0e"),
        (final_uv, "Final (DBS + ICA clean)", "#1f77b4"),
    ]
    for ax, (data, lbl, col) in zip(axes, triples):
        _style_ax(ax, t, ylabel=lbl, grid=False)
        ax.plot(tvec, data[ci, i0:i1], color=col, lw=0.9)
        for k in range(int(t_offset * f_dbs) - 1,
                        int((t_offset + dur) * f_dbs) + 2):
            pt = k / f_dbs
            if t_offset <= pt <= t_offset + dur:
                ax.axvline(pt, color=t["dbs_marker"], lw=0.5, alpha=0.4)

    axes[-1].set_xlabel("Time (s)", color=t["text"])
    fig.suptitle(
        f"Time-Domain Comparison — Channel {ch_names[ci]}  ({dur:.0f}s segment)",
        color=t["text"], fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, out_path, t)


# ═════════════════════════════════════════════════════════════════════════════
# 3. SCALP TOPOMAPS
# ═════════════════════════════════════════════════════════════════════════════
def plot_topomaps_themed(prep_uv, final_uv, raw_mne, ch_names,
                          out_path, t: dict):
    try:
        info = raw_mne.info.copy()
        amp_before = np.abs(prep_uv).mean(axis=1)
        amp_after  = np.abs(final_uv).mean(axis=1)

        ch_idx = mne.pick_types(info, eeg=True, exclude=[])
        if len(ch_idx) < 4:
            log.warning("Fewer than 4 EEG channels — skipping topomap.")
            return

        vmax = max(amp_before.max(), amp_after.max())
        diff = amp_before - amp_after

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.patch.set_facecolor(t["bg"])

        for ax, topo_data, title in [
            (axes[0], amp_before, "Before DBS Removal\n(Stage I output)"),
            (axes[1], amp_after,  "After DBS + ICA\n(Final)"),
            (axes[2], diff,       "Difference\n(Artifact Removed)"),
        ]:
            ax.set_facecolor(t["bg"])
            vmin_p = min(0, diff.min()) if ax is axes[2] else 0.0
            vmax_p = max(diff.max(), 1e-3) if ax is axes[2] else vmax
            mne.viz.plot_topomap(
                topo_data, info, axes=ax, show=False,
                cmap="RdBu_r" if ax is axes[2] else "hot",
                vlim=(vmin_p, vmax_p),
                sphere="auto",
            )
            ax.set_title(title, color=t["text"], fontsize=9)

        fig.suptitle("Scalp Topomaps — Mean Absolute Amplitude (µV)",
                     color=t["text"], fontsize=11, fontweight="bold")
        fig.tight_layout()
        _save(fig, out_path, t)
    except Exception as exc:
        log.warning(f"Topomap rendering failed ({t['bg']}): {exc}")


# ═════════════════════════════════════════════════════════════════════════════
# 4. BIOMARKER BAR CHART
# ═════════════════════════════════════════════════════════════════════════════
def plot_biomarker_bars_themed(table: dict, out_path: pathlib.Path, t: dict):
    band_names  = [b for b in table if b != "DBS_removal"]
    total_vals  = [table[b]["preservation_%"] for b in band_names]
    oh_vals     = [table[b].get("off_harmonic_preservation_%",
                                table[b]["preservation_%"]) for b in band_names]
    reliable    = [table[b].get("comparison_reliable", True) for b in band_names]
    DISPLAY_CAP = 200.0

    n = len(band_names)
    y = np.arange(n)
    height = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(t["bg"])
    ax.set_facecolor(t["bg"])

    for i, (name, tv, ohv, rel) in enumerate(
            zip(band_names, total_vals, oh_vals, reliable)):
        alpha_mod = 1.0 if rel else 0.45

        c_tot = "#2ca02c" if tv  >= 90 else "#d62728"
        ax.barh(y[i] + height / 2, min(tv, DISPLAY_CAP),
                height=height, color=c_tot, alpha=alpha_mod * 0.7,
                label="Total" if i == 0 else "")

        c_oh = "#4da6ff" if ohv >= 90 else "#ff7f50"
        ax.barh(y[i] - height / 2, min(ohv, DISPLAY_CAP),
                height=height, color=c_oh, alpha=alpha_mod,
                hatch="//", label="Off-harmonic (brain only)" if i == 0 else "")

        label_x = min(max(tv, ohv), DISPLAY_CAP) + 1
        tv_str  = f"T:{tv:.0f}%" if tv <= DISPLAY_CAP else f"T:>{DISPLAY_CAP:.0f}%"
        oh_str  = f"OH:{ohv:.0f}%"
        warn    = "  ⚠ unreliable" if not rel else ""
        ax.text(label_x, y[i],
                f"{tv_str}  {oh_str}{warn}",
                va="center",
                color=t["text"] if rel else t["spine_color"],
                fontsize=8.5)

    ax.set_yticks(y)
    ax.set_yticklabels(band_names, color=t["text"])
    ax.axvline(90, color=t["ref_line"], lw=1.2, ls="--",
               label="90% target")
    ax.axvline(100, color=t["ref_zero"], lw=0.7, ls=":", alpha=0.5)
    ax.axvline(DISPLAY_CAP, color=t["spine_color"], lw=0.6, ls=":",
               alpha=0.4, label=f">{DISPLAY_CAP:.0f}% capped")

    _style_ax(ax, t,
              xlabel="Band Power Preservation (%) vs PRE (No-DBS) Baseline",
              title="Biomarker Integrity Check\nTotal (solid) vs Off-Harmonic / Brain-Only (hatched)")
    ax.set_xlim(0, DISPLAY_CAP + 30)
    ax.legend(facecolor=t["legend_bg"], labelcolor=t["text"],
              fontsize=8, loc="lower right")

    fig.tight_layout()
    _save(fig, out_path, t)


# ═════════════════════════════════════════════════════════════════════════════
# 5. EXPERIMENTAL — PSD COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
def plot_psd_comparison_themed(results: dict, sfreq: float, f_dbs: float,
                                 title: str, out_path: pathlib.Path, t: dict):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(t["bg"])

    for ax, (fmin, fmax), zoom_label in zip(
        axes,
        [(0.5, 100.0), (0.5, 32.0)],
        ["Full 0.5–100 Hz", "Clinical zoom 0.5–32 Hz"],
    ):
        _style_ax(ax, t, xlabel="Frequency (Hz)",
                  ylabel="PSD (dB µV²/Hz)", title=zoom_label)
        for label, data_uv in results.items():
            col, ls, lw = METHOD_PALETTE.get(label, ("#888888", "-", 1.0))
            f, p = compute_psd(data_uv, sfreq, fmin=fmin, fmax=fmax)
            ax.plot(f, p, color=col, ls=ls, lw=lw, label=label, alpha=0.9)

        for k in range(1, 15):
            h = k * f_dbs
            if h > fmax: break
            ax.axvline(h, color=t["dbs_marker"], lw=0.5, ls=":", alpha=0.6,
                       label="DBS harmonic" if k == 1 else "")

        for band_name, lo, hi in EVAL_BANDS:
            ax.axvspan(lo, hi, color=t["grid_color"], alpha=0.08)

        ax.legend(fontsize=7, loc="upper right", framealpha=0.8,
                  facecolor=t["legend_bg"], labelcolor=t["text"])

    fig.suptitle(title, color=t["text"], fontsize=13,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, out_path, t)


# ═════════════════════════════════════════════════════════════════════════════
# 6. EXPERIMENTAL — TIME DOMAIN
# ═════════════════════════════════════════════════════════════════════════════
def plot_time_domain_exp_themed(results: dict, sfreq: float,
                                  ch_names: list[str], ch_label: str,
                                  t_offset: float, dur: float, f_dbs: float,
                                  title: str, out_path: pathlib.Path, t: dict):
    try:
        ch_idx = ch_names.index(ch_label)
    except ValueError:
        ch_idx = 0
        ch_label = ch_names[0]

    i0 = int(t_offset * sfreq)
    i1 = i0 + int(dur * sfreq)
    n_methods = len(results)
    fig, axes = plt.subplots(n_methods, 1,
                              figsize=(14, 2.0 * n_methods), sharex=True)
    if n_methods == 1:
        axes = [axes]
    fig.patch.set_facecolor(t["bg"])

    tvec = np.arange(i1 - i0) / sfreq + t_offset
    for ax, (label, data_uv) in zip(axes, results.items()):
        col, ls, lw = METHOD_PALETTE.get(label, ("#888888", "-", 1.0))
        _style_ax(ax, t, grid=False)
        ax.plot(tvec, data_uv[ch_idx, i0:i1], color=col, lw=lw, ls=ls)
        ax.set_ylabel(label, fontsize=7, rotation=0, labelpad=80,
                      va="center", color=t["text"])
        ax.set_yticks([])
        for k in range(int(t_offset * f_dbs),
                        int((t_offset + dur) * f_dbs) + 2):
            pt = k / f_dbs
            if t_offset <= pt <= t_offset + dur:
                ax.axvline(pt, color=t["dbs_marker"], lw=0.4, alpha=0.4)

    axes[-1].set_xlabel("Time (s)", fontsize=10, color=t["text"])
    fig.suptitle(f"{title}\nChannel {ch_label}",
                 color=t["text"], fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_path, t)


# ═════════════════════════════════════════════════════════════════════════════
# 7. EXPERIMENTAL — BETA / DBS SUMMARY BAR CHART
# ═════════════════════════════════════════════════════════════════════════════
def plot_beta_summary_themed(all_metrics: list[dict], out_path: pathlib.Path,
                               t: dict):
    conditions = sorted(set(m["condition"] for m in all_metrics))
    methods    = list(dict.fromkeys(m["method"] for m in all_metrics))

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.patch.set_facecolor(t["bg"])

    for ax, metric_key, ylabel, target, good_range in [
        (axes[0], "beta_preservation_%",
         "Beta preservation (%)",      100, (95, 105)),
        (axes[1], "dbs_reduction_%",
         "DBS harmonic reduction (%)",  90, (90, 100.5)),
    ]:
        _style_ax(ax, t, ylabel=ylabel, title=ylabel)
        x = np.arange(len(conditions))
        width = 0.8 / max(len(methods), 1)

        for mi, method in enumerate(methods):
            vals = []
            for cond in conditions:
                row = next((m for m in all_metrics
                            if m["method"] == method and m["condition"] == cond),
                           None)
                vals.append(row[metric_key] if row else np.nan)

            col = METHOD_PALETTE.get(method, ("#888888", "-", 1.0))[0]
            bars = ax.bar(x + mi * width, vals, width=width * 0.9,
                          color=col, alpha=0.85, label=method)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            min(v, good_range[1] - 1) + 0.5,
                            f"{v:.0f}", ha="center", va="bottom",
                            fontsize=5.5, fontweight="bold",
                            color=t["text"])

        ax.axhline(target, color=t["ref_line"], lw=1.2, ls="--",
                   label=f"Target={target}%")
        ax.axhspan(*good_range, color="#00aa00", alpha=0.08,
                   label="±5 % zone")
        ax.set_xticks(x + width * len(methods) / 2)
        ax.set_xticklabels(conditions, rotation=20, ha="right",
                           fontsize=9, color=t["text"])
        ax.legend(fontsize=6.5, loc="lower right", framealpha=0.8,
                  facecolor=t["legend_bg"], labelcolor=t["text"])

    fig.suptitle(
        "DBS Removal Performance Summary\n"
        "Beta preservation: 100 % = identical to pre-DBS baseline  "
        "| DBS reduction target > 90 %",
        color=t["text"], fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, out_path, t)


# ═════════════════════════════════════════════════════════════════════════════
# 5. REAL-TIME SLIDING-WINDOW DASHBOARD  (animated GIF)
# ═════════════════════════════════════════════════════════════════════════════
def save_realtime_dashboard_themed(raw_uv, final_uv, sfreq, ch_names,
                                    f_dbs, out_path: pathlib.Path, t: dict,
                                    window_sec: float = 5.0,
                                    step_sec:   float = 0.5,
                                    fps:        int   = 6):
    """Animated GIF: sliding Raw vs Final time panel + live PSD panel."""
    try:
        ci = ch_names.index(ANALYSIS_CH)
    except ValueError:
        ci = 0

    n_s      = min(raw_uv.shape[1], final_uv.shape[1])
    w_s      = int(window_sec * sfreq)
    step     = int(step_sec   * sfreq)
    n_frames = max(1, (n_s - w_s) // step + 1)
    nyq      = sfreq / 2.0
    dash_fmax = LPF_HZ + 5.0

    sig_raw = raw_uv[ci,   :n_s]
    sig_fin = final_uv[ci, :n_s]

    fig = plt.figure(figsize=(13, 7), facecolor=t["bg"])
    gs  = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.4)

    # ── Time panel ────────────────────────────────────────────────────────────
    ax_t = fig.add_subplot(gs[0])
    _style_ax(ax_t, t,
              xlabel="Time (s)", ylabel="Amplitude (µV)", grid=False)

    line_r, = ax_t.plot([], [], color="#888888", lw=0.8,
                         label="Raw", alpha=0.7)
    line_f, = ax_t.plot([], [], color="#4da6ff", lw=1.3,
                         label="Final clean")
    ttl = ax_t.set_title("", color=t["text"], fontsize=10)
    ax_t.legend(facecolor=t["legend_bg"], labelcolor=t["text"],
                fontsize=8, loc="upper right")

    # ── PSD panel ─────────────────────────────────────────────────────────────
    ax_p = fig.add_subplot(gs[1])
    _style_ax(ax_p, t,
              xlabel="Frequency (Hz)", ylabel="PSD (dB µV²/Hz)")
    ax_p.set_xlim(0.5, dash_fmax)

    line_rp, = ax_p.plot([], [], color="#888888", lw=0.9, label="Raw")
    line_fp, = ax_p.plot([], [], color="#4da6ff", lw=1.2, label="Final")

    for k in range(1, 15):
        h = k * f_dbs
        if h > nyq: break
        ax_p.axvline(h, color=t["dbs_marker"], lw=0.5, ls=":", alpha=0.5)

    for _, lo, hi, col in BANDS:
        ax_p.axvspan(lo, hi, color=col, alpha=0.07)

    ax_p.legend(facecolor=t["legend_bg"], labelcolor=t["text"],
                fontsize=7, loc="upper right")

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
        ttl.set_text(
            f"Real-Time Dashboard — ch {ch_names[ci]}  |  "
            f"t = {times[0]:.1f}–{times[-1]:.1f} s"
        )

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

    anim = FuncAnimation(fig, update, frames=n_frames,
                          interval=1000 // fps, blit=False)
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    log.info(f"  Saved {out_path.relative_to(ROOT)}")


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline runner — XUAWAKE7 validation plots
# ═════════════════════════════════════════════════════════════════════════════
def generate_validation_plots():
    log.info("=" * 60)
    log.info("Processing XUAWAKE7 — both themes")
    log.info("=" * 60)

    data_dir = ROOT / "data" / "raw" / "XU"
    dbs_path = data_dir / "XUAWAKE7_deidentified.edf"
    pre_fname = PRE_FILES.get(dbs_path.name)
    pre_path  = (data_dir / pre_fname) if pre_fname else None

    # ── Run pipeline once ────────────────────────────────────────────────────
    raw_mne, stage1_raw, final_raw = run_surgical_pipeline(
        dbs_path, f0=7.0, run_ica=True
    )
    sfreq     = final_raw.info["sfreq"]
    ch_names  = final_raw.ch_names
    raw_uv    = raw_mne.get_data()   * 1e6
    stage1_uv = stage1_raw.get_data() * 1e6
    final_uv  = final_raw.get_data()  * 1e6

    # Load PRE baseline
    if pre_path and pre_path.exists():
        pre_proc = EEGPreprocessor()
        pre_ing  = load_edf(pre_path, IngestionConfig(
            max_duration_sec=MAX_DUR_SEC, verbose=False))
        if pre_ing.eeg_channels:
            pre_ing.raw.pick(pre_ing.eeg_channels)
        pre_sfreq = float(pre_ing.raw.info["sfreq"])
        if pre_sfreq != sfreq:
            pre_ing.raw.resample(sfreq, verbose=False)
        pre_stage1 = pre_proc.apply_clinical_filter(
            pre_ing.raw, l_freq=0.1, h_freq=100.0)

        common   = [c for c in ch_names if c in pre_stage1.ch_names]
        s1_idx   = [ch_names.index(c)         for c in common]
        fin_idx  = s1_idx
        pre_idx  = [pre_stage1.ch_names.index(c) for c in common]
        s1_uv_c  = stage1_uv[s1_idx, :]
        fin_uv_c = final_uv[fin_idx, :]
        pre_uv_c = pre_stage1.get_data()[pre_idx, :] * 1e6
        n_samp   = min(s1_uv_c.shape[1], pre_uv_c.shape[1], fin_uv_c.shape[1])
        s1_uv_c  = s1_uv_c[:,  :n_samp]
        fin_uv_c = fin_uv_c[:, :n_samp]
        pre_uv_c = pre_uv_c[:, :n_samp]
        raw_uv_c = raw_uv[[ch_names.index(c) for c in common], :n_samp]
    else:
        log.warning("No PRE file — using Stage-I as baseline for plot.")
        common   = ch_names
        s1_uv_c  = stage1_uv
        fin_uv_c = final_uv
        pre_uv_c = stage1_uv
        raw_uv_c = raw_uv
        n_samp   = stage1_uv.shape[1]

    # Biomarker table (used for bar chart)
    table = compute_biomarker_table(s1_uv_c, pre_uv_c, fin_uv_c, sfreq, 7.0)

    # ── Plot in both themes ──────────────────────────────────────────────────
    for theme_name, t in THEMES.items():
        out = VAL_DIR / theme_name
        log.info(f"  Theme: {theme_name}")

        plot_triple_psd_themed(
            raw_uv_c, pre_uv_c, s1_uv_c, fin_uv_c, sfreq, 7.0,
            out / "01_triple_psd.png", t,
        )
        plot_time_domain_themed(
            raw_uv_c, s1_uv_c, fin_uv_c, sfreq, common, 7.0,
            t_offset=5.0, dur=5.0,
            out_path=out / "02_time_domain.png",
            t=t,
        )
        plot_topomaps_themed(
            s1_uv_c, fin_uv_c, raw_mne, common,
            out / "03_scalp_topomaps.png", t,
        )
        plot_biomarker_bars_themed(table, out / "04_biomarker_bars.png", t)
        save_realtime_dashboard_themed(
            raw_uv_c, fin_uv_c, sfreq, common, 7.0,
            out / "05_realtime_dashboard.gif", t,
        )

    log.info("Validation plots done.")


# ═════════════════════════════════════════════════════════════════════════════
# Experimental plots — 7 Hz AWAKE only (most important condition)
# ═════════════════════════════════════════════════════════════════════════════
def generate_experimental_plots():
    log.info("=" * 60)
    log.info("Processing Experimental 7 Hz AWAKE — both themes")
    log.info("=" * 60)

    # ── Beta summary from existing JSON (no reprocessing needed) ────────────
    metrics_json = EVAL_DIR / "metrics_summary.json"
    if metrics_json.exists():
        with open(metrics_json) as fh:
            all_metrics = json.load(fh)
        for theme_name, t in THEMES.items():
            plot_beta_summary_themed(
                all_metrics,
                EVAL_DIR / theme_name / "beta_preservation_summary.png",
                t,
            )
        log.info("  Beta summary done (from cached JSON).")
    else:
        log.warning(f"metrics_summary.json not found — skipping beta summary.")

    # ── PSD and time-domain: run 7 Hz AWAKE methods ─────────────────────────
    data_dir = ROOT / "data" / "raw" / "XU"
    dbs_path = data_dir / "XUAWAKE7_deidentified.edf"
    pre_path = data_dir / "XUAWAKEPRE_deidentified.edf"

    if not (dbs_path.exists() and pre_path.exists()):
        log.warning("EDF files not found — skipping experimental PSD/time plots.")
        return

    log.info("  Loading DBS + PRE EDF …")
    raw_dbs = load_and_preprocess_edf(dbs_path, target_sfreq=EVAL_SFREQ)
    raw_pre = load_and_preprocess_edf(pre_path,  target_sfreq=EVAL_SFREQ)

    sfreq    = raw_dbs.info["sfreq"]
    ch_names = raw_dbs.ch_names
    dbs_uv   = raw_to_uv(raw_dbs)
    pre_uv   = raw_to_uv(raw_pre)

    # Align channels and lengths
    common   = [c for c in ch_names if c in raw_pre.ch_names]
    di       = [ch_names.index(c)         for c in common]
    pi       = [raw_pre.ch_names.index(c) for c in common]
    min_samp = min(dbs_uv.shape[1], pre_uv.shape[1])
    dbs_uv   = dbs_uv[di, :min_samp]

    # Apply all removal methods
    log.info("  Applying removal methods …")
    results: dict[str, np.ndarray] = {
        "Raw DBS":              dbs_uv.copy(),
        "FFT Spectral Interp":  _apply("fft_spectral_interp", dbs_uv, sfreq, 7.0),
        "Comb Notch Q=200":     _apply("comb_notch",          dbs_uv, sfreq, 7.0),
        "Sinusoidal Regression":_apply("sinusoidal_regression",dbs_uv, sfreq, 7.0),
        "Stage A (SVD+NLMS)":   apply_svd_nlms(              dbs_uv.copy(), sfreq, 7.0),
        "Stage B (Cplx Hampel)":apply_complex_hampel_fft(    dbs_uv.copy(), sfreq, 7.0),
        "Stage C (Template)":   apply_template_subtraction(  dbs_uv.copy(), sfreq, 7.0),
        "Hybrid A→B→C":         apply_hybrid_abc(            dbs_uv.copy(), sfreq, 7.0),
    }

    cond_tag = "AWAKE_7Hz"
    ch_label = "Cz" if "Cz" in common else common[0]

    for theme_name, t in THEMES.items():
        out = EVAL_DIR / theme_name
        log.info(f"  Theme: {theme_name}")

        plot_psd_comparison_themed(
            results, sfreq, 7.0,
            f"PSD Comparison — {cond_tag}",
            out / f"psd_comparison_{cond_tag}.png", t,
        )
        plot_time_domain_exp_themed(
            results, sfreq, common, ch_label,
            t_offset=5.0, dur=2.0, f_dbs=7.0,
            title=f"Time-Domain — {cond_tag}",
            out_path=out / f"time_domain_{cond_tag}.png",
            t=t,
        )

    log.info("Experimental plots done.")


def _apply(method_name: str, data: np.ndarray, sfreq: float,
           f_dbs: float) -> np.ndarray:
    from src.filters import ArtifactFilterFactory
    kw: dict = {}
    if method_name == "fft_spectral_interp":
        kw = {"f_target": f_dbs}
    elif method_name == "comb_notch":
        kw = {"f0": f_dbs, "q_factor": 200}
    elif method_name == "sinusoidal_regression":
        kw = {"f_target": f_dbs}
    return ArtifactFilterFactory.process(method_name, data.copy(), sfreq, **kw)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    generate_validation_plots()
    generate_experimental_plots()
    log.info("All dual-theme plots generated.")
