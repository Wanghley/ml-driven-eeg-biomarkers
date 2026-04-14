"""
Dagster software-defined assets for the *real-data* EEG pipeline.

Asset graph
───────────
  real_raw_loaded
        │
  real_dbs_filtered        ← all 7 methods × AWAKE7 + SLEEP7
        │
  real_ica_pipeline        ← best method → ICA → final figures + summary table

All heavy MNE Raw objects are saved to .fif on disk; assets exchange
lightweight metadata dicts so Dagster's default IOManager can serialize them.
"""


import sys
import pathlib
import warnings

_ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mne

mne.set_log_level("WARNING")

import base64
from dagster import asset, AssetExecutionContext, MaterializeResult, MetadataValue, Output


def _png_md(path) -> MetadataValue:
    with open(str(path), "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    name = pathlib.Path(str(path)).name
    return MetadataValue.md(f"![{name}](data:image/png;base64,{b64})")


from pipeline.resources import EEGPipelineConfig
from pipeline.constants import BANDS, DBS_METHODS, METHOD_PALETTE, STANDARD_CH
from src.filters import ArtifactFilterFactory


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers (mirror real_data_pipeline.ipynb helpers)
# ──────────────────────────────────────────────────────────────────────────────

def _load_eeg(
    path: pathlib.Path,
    target_sfreq: float = 256.0,
    l_freq: float = 1.0,
    h_freq: float = 119.0,
) -> mne.io.Raw:
    """Load an EDF, pick 19 standard channels, resample, bandpass, avg-ref."""
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
    upper_map = {s.upper(): s for s in STANDARD_CH}
    rename = {ch: upper_map[ch.upper()] for ch in raw.ch_names if ch.upper() in upper_map}
    raw.rename_channels(rename)
    raw.pick(STANDARD_CH)
    raw.set_channel_types({ch: "eeg" for ch in STANDARD_CH})
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="ignore", verbose=False)
    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq, verbose=False)
    h_safe = min(h_freq, target_sfreq / 2 - 1)
    raw.filter(l_freq=l_freq, h_freq=h_safe, fir_design="firwin", phase="zero", verbose=False)
    raw.set_eeg_reference("average", projection=True, verbose=False)
    raw.apply_proj()
    return raw


def _raw_from_fif(path: str) -> mne.io.Raw:
    return mne.io.read_raw_fif(path, preload=True, verbose=False)


def _apply_method(raw: mne.io.Raw, filter_key: str, kwargs: dict) -> mne.io.Raw:
    r = raw.copy()
    r.load_data()
    d = r.get_data() * 1e6
    d_clean = ArtifactFilterFactory.process(filter_key, d, raw.info["sfreq"], **kwargs)
    r._data = d_clean * 1e-6
    return r


def _psd_db(raw: mne.io.Raw, fmax: float = 80.0, n_fft: int = 2048):
    obj = raw.compute_psd(method="welch", fmax=fmax, n_fft=n_fft, verbose=False)
    return obj.freqs, 10 * np.log10(obj.get_data().mean(axis=0) + 1e-30)


def _band_power(raw: mne.io.Raw, lo: float, hi: float, n_fft: int = 2048) -> float:
    obj = raw.compute_psd(method="welch", fmin=lo, fmax=hi, n_fft=n_fft, verbose=False)
    return float(obj.get_data().mean())


def _harmonic_atten_db(
    raw_before: mne.io.Raw, raw_after: mne.io.Raw, h: float, bw: float = 0.10
) -> float:
    def bp(r):
        obj = r.compute_psd(method="welch", fmin=h - bw, fmax=h + bw, n_fft=2048, verbose=False)
        return obj.get_data().mean()
    return float(10 * np.log10(bp(raw_after) / (bp(raw_before) + 1e-30)))


def _compute_metrics(
    cleaned_dict: dict[str, mne.io.Raw],
    raw_dbs: mne.io.Raw,
    raw_ref: mne.io.Raw,
    dbs_freq: float,
) -> list[dict]:
    harmonics = [k * dbs_freq for k in range(1, 5)]
    rows = []
    for label, _, _ in DBS_METHODS:
        rc = cleaned_dict[label]
        bp = {b: 100 * _band_power(rc, lo, hi) / (_band_power(raw_ref, lo, hi) + 1e-30)
              for b, lo, hi in BANDS}
        att = float(np.mean([_harmonic_atten_db(raw_dbs, rc, h) for h in harmonics]))
        rows.append({"label": label, **bp, "atten_db": att})
    return rows


def _classify_ica(ica, raw: mne.io.Raw) -> tuple[list, list]:
    from scipy import signal as sp_signal
    sfreq = raw.info["sfreq"]
    mixing  = ica.get_components()
    sources = ica.get_sources(raw).get_data()
    ch_lower = [c.lower() for c in raw.ch_names]
    fp_idx     = [i for i, c in enumerate(ch_lower) if c in ("fp1", "fp2")]
    muscle_idx = [i for i, c in enumerate(ch_lower) if c in ("t3", "t4", "t5", "t6", "f7", "f8")]
    eye_comps, muscle_comps = [], []
    for ic in range(ica.n_components_):
        col    = np.abs(mixing[:, ic])
        top_ch = int(np.argmax(col))
        f_s, psd_s = sp_signal.welch(sources[ic], fs=sfreq, nperseg=512)
        p_all = np.trapz(psd_s[(f_s >= 1) & (f_s <= 80)], f_s[(f_s >= 1) & (f_s <= 80)]) + 1e-30
        lf_r  = np.trapz(psd_s[(f_s >= 1) & (f_s <= 15)], f_s[(f_s >= 1) & (f_s <= 15)]) / p_all
        hf_r  = np.trapz(psd_s[(f_s >= 30) & (f_s <= 80)], f_s[(f_s >= 30) & (f_s <= 80)]) / p_all
        if top_ch in fp_idx and lf_r > 0.60:
            eye_comps.append(ic)
        elif top_ch in muscle_idx and hf_r > 0.55:
            muscle_comps.append(ic)
    return eye_comps, muscle_comps


def _topomap_band(raw: mne.io.Raw, lo: float, hi: float, ax, title: str):
    obj   = raw.compute_psd(method="welch", fmin=lo, fmax=hi, n_fft=2048, verbose=False)
    bp    = obj.get_data().mean(axis=1)
    bp_db = 10 * np.log10(bp + 1e-30)
    mne.viz.plot_topomap(
        bp_db, raw.info, axes=ax, show=False, cmap="RdBu_r",
        vlim=(np.percentile(bp_db, 5), np.percentile(bp_db, 95)),
    )
    ax.set_title(title, fontsize=8)


# ──────────────────────────────────────────────────────────────────────────────
# Asset 1 — Load and standardise all four EDF recordings
# ──────────────────────────────────────────────────────────────────────────────

@asset(
    group_name="real_data",
    description=(
        "Load all four XU patient EDF recordings (AWAKE7, SLEEP7, PRE-awake, PRE-sleep). "
        "Apply channel standardisation, resample to target_sfreq if needed, "
        "bandpass 1–119 Hz, average reference. Save each as .fif."
    ),
)
def real_raw_loaded(
    context: AssetExecutionContext,
    eeg_config: EEGPipelineConfig,
) -> Output:
    """
    Returns
    -------
    dict
        ``awake7_path``, ``sleep7_path``, ``pre_awake_path``, ``pre_sleep_path``
        — absolute paths to the saved .fif files.
        ``recording_info`` — duration / sfreq metadata per recording.
    """
    data_dir = eeg_config.data_path()
    proc     = eeg_config.processed_path()

    recordings = {
        "awake7":    (eeg_config.awake7_file,    "real_awake7.fif"),
        "sleep7":    (eeg_config.sleep7_file,    "real_sleep7.fif"),
        "pre_awake": (eeg_config.pre_awake_file, "real_pre_awake.fif"),
        "pre_sleep": (eeg_config.pre_sleep_file, "real_pre_sleep.fif"),
    }

    paths: dict[str, str] = {}
    recording_info: dict[str, dict] = {}

    for key, (edf_name, fif_name) in recordings.items():
        edf_path = data_dir / edf_name
        context.log.info(f"Loading {edf_path.name} …")
        raw = _load_eeg(edf_path, target_sfreq=eeg_config.target_sfreq)
        out = proc / fif_name
        raw.save(str(out), overwrite=True)
        paths[f"{key}_path"] = str(out)
        recording_info[key] = {
            "duration_min": float(round(raw.times[-1] / 60, 2)),
            "sfreq":        float(raw.info["sfreq"]),
            "n_channels":   len(raw.ch_names),
        }
        context.log.info(
            f"  → {recording_info[key]['duration_min']:.1f} min  "
            f"{recording_info[key]['sfreq']:.0f} Hz  "
            f"{recording_info[key]['n_channels']} ch"
        )

    value = {**paths, "recording_info": recording_info}
    return Output(
        value=value,
        metadata={
            f"{k}_duration_min": MetadataValue.float(float(v["duration_min"]))
            for k, v in recording_info.items()
        } | {
            f"{k}_sfreq": MetadataValue.float(float(v["sfreq"]))
            for k, v in recording_info.items()
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Asset 2 — Apply all 7 DBS removal methods to AWAKE7 and SLEEP7
# ──────────────────────────────────────────────────────────────────────────────

@asset(
    group_name="real_data",
    description=(
        "Apply all 7 DBS removal methods to both AWAKE7 and SLEEP7 recordings. "
        "Compute brain-band preservation % vs PRE baseline and harmonic attenuation dB. "
        "Save PSD comparison figures and master comparison bar chart."
    ),
)
def real_dbs_filtered(
    context: AssetExecutionContext,
    real_raw_loaded: dict,
    eeg_config: EEGPipelineConfig,
) -> Output:
    """
    Returns
    -------
    dict
        ``awake_fif_paths``  : {method_label: str path}
        ``sleep_fif_paths``  : {method_label: str path}
        ``metrics_awake``    : list of per-method metric dicts
        ``metrics_sleep``    : list of per-method metric dicts
        ``figure_psd_awake`` : path to AWAKE PSD comparison figure
        ``figure_psd_sleep`` : path to SLEEP PSD comparison figure
        ``figure_master``    : path to master bar-chart comparison figure
    """
    proc     = eeg_config.processed_path()
    figs     = eeg_config.real_figures_path()
    dbs_freq = eeg_config.dbs_freq

    raw_awake7   = _raw_from_fif(real_raw_loaded["awake7_path"])
    raw_sleep7   = _raw_from_fif(real_raw_loaded["sleep7_path"])
    raw_pre_aw   = _raw_from_fif(real_raw_loaded["pre_awake_path"])
    raw_pre_sl   = _raw_from_fif(real_raw_loaded["pre_sleep_path"])

    # ── Run methods ──────────────────────────────────────────────────────
    awake_fif: dict[str, str] = {}
    sleep_fif: dict[str, str] = {}

    for label, filter_key, kwargs_factory in DBS_METHODS:
        kw = kwargs_factory(dbs_freq)
        safe = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")

        context.log.info(f"AWAKE7 → {label}")
        raw_aw_cl = _apply_method(raw_awake7, filter_key, kw)
        p = proc / f"real_awake_{safe}.fif"
        raw_aw_cl.save(str(p), overwrite=True)
        awake_fif[label] = str(p)

        context.log.info(f"SLEEP7 → {label}")
        raw_sl_cl = _apply_method(raw_sleep7, filter_key, kw)
        p = proc / f"real_sleep_{safe}.fif"
        raw_sl_cl.save(str(p), overwrite=True)
        sleep_fif[label] = str(p)

    # ── Reload cleaned dicts ─────────────────────────────────────────────
    cleaned_awake = {lbl: _raw_from_fif(path) for lbl, path in awake_fif.items()}
    cleaned_sleep = {lbl: _raw_from_fif(path) for lbl, path in sleep_fif.items()}

    metrics_awake = _compute_metrics(cleaned_awake, raw_awake7, raw_pre_aw, dbs_freq)
    metrics_sleep = _compute_metrics(cleaned_sleep, raw_sleep7, raw_pre_sl, dbs_freq)

    for cond, rows in [("AWAKE7", metrics_awake), ("SLEEP7", metrics_sleep)]:
        context.log.info(f"\n=== {cond} vs PRE ===")
        for r in rows:
            context.log.info(
                f"  {r['label']:<26} "
                f"Theta={r['Theta']:5.1f}%  Beta={r['Beta']:5.1f}%  Atten={r['atten_db']:+.1f} dB"
            )

    # ── PSD comparison figures ───────────────────────────────────────────
    def _psd_comparison_fig(
        raw_dbs, cleaned, raw_ref, cond_label, ref_color, out_path
    ):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        for ax, (fmin, fmax), title in zip(
            axes, [(1, 80), (1, 30)], ["Full 1–80 Hz", "Zoom 1–30 Hz"]
        ):
            fb, pb = _psd_db(raw_ref, fmax=fmax)
            fc, pc = _psd_db(raw_dbs, fmax=fmax)
            m = (fb >= fmin) & (fb <= fmax)
            ax.fill_between(fb[m], pb[m] - 2, pb[m] + 2, color=ref_color, alpha=0.10)
            ax.plot(fb[m], pb[m], color=ref_color, lw=1.2, ls="--", label="PRE baseline")
            ax.plot(fc[m], pc[m], color="gray",    lw=0.7, alpha=0.5, label="DBS contaminated")
            for label, _, _ in DBS_METHODS:
                col, ls, lw = METHOD_PALETTE[label]
                fn, pn = _psd_db(cleaned[label], fmax=fmax)
                mn = (fn >= fmin) & (fn <= fmax)
                ax.plot(fn[mn], pn[mn], color=col, ls=ls, lw=lw, label=label)
            for k in range(1, int(fmax // dbs_freq) + 1):
                ax.axvline(k * dbs_freq, color="orange", lw=0.6, ls=":", alpha=0.6,
                           label="DBS harmonic" if k == 1 else "")
            ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (dB)")
            ax.set_title(title); ax.legend(fontsize=7, loc="upper right")
        fig.suptitle(f"XU {cond_label} — 7-method DBS removal vs PRE baseline",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)

    fig_psd_aw = figs / "real_01_awake7_method_comparison.png"
    fig_psd_sl = figs / "real_02_sleep7_method_comparison.png"
    _psd_comparison_fig(raw_awake7, cleaned_awake, raw_pre_aw, "AWAKE7", "limegreen", fig_psd_aw)
    _psd_comparison_fig(raw_sleep7, cleaned_sleep, raw_pre_sl, "SLEEP7", "navy",      fig_psd_sl)

    # ── Master comparison bar chart ──────────────────────────────────────
    band_names   = [b for b, *_ in BANDS]
    method_lbls  = [r["label"] for r in metrics_awake]
    n_methods    = len(method_lbls)
    bar_colors   = [METHOD_PALETTE[l][0] for l in method_lbls]
    x            = np.arange(n_methods)
    w            = 0.38

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    for col_i, bname in enumerate(band_names):
        for row_i, (rows, cond) in enumerate([(metrics_awake, "AWAKE"), (metrics_sleep, "SLEEP")]):
            ax = fig.add_subplot(gs[row_i, col_i])
            vals = [r[bname] for r in rows]
            bars = ax.bar(range(n_methods), vals, color=bar_colors, alpha=0.85)
            ax.axhline(100, color="k", lw=1.0, ls="--")
            ax.axhline(90,  color="orange", lw=0.8, ls=":")
            ax.set_title(f"{cond} — {bname}", fontsize=9, fontweight="bold")
            ax.set_xticks(range(n_methods))
            ax.set_xticklabels([m.split("(")[0].strip() for m in method_lbls],
                                rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Preservation %", fontsize=8); ax.set_ylim(20, 400)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, min(val, 395) + 2,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=6, fontweight="bold")

    # Attenuation & beta summary (bottom row)
    att_aw = [abs(r["atten_db"]) for r in metrics_awake]
    att_sl = [abs(r["atten_db"]) for r in metrics_sleep]
    ax_att = fig.add_subplot(gs[2, :2])
    ax_att.bar(x - w / 2, att_aw, w, label="Awake", color=bar_colors, alpha=0.8)
    ax_att.bar(x + w / 2, att_sl, w, label="Sleep", color=bar_colors, alpha=0.5,
               edgecolor="k", linewidth=0.4)
    ax_att.set_xticks(x)
    ax_att.set_xticklabels([m.split("(")[0].strip() for m in method_lbls],
                             rotation=30, ha="right", fontsize=8)
    ax_att.set_ylabel("|DBS attenuation| dB", fontsize=8)
    ax_att.set_title("DBS harmonic attenuation (7–28 Hz avg)\n"
                     "⚠ Wider methods inflate this by removing brain signal", fontsize=9)
    ax_att.legend(fontsize=8)

    beta_aw = [r["Beta"] for r in metrics_awake]
    beta_sl = [r["Beta"] for r in metrics_sleep]
    ax_beta = fig.add_subplot(gs[2, 2:])
    ax_beta.bar(x - w / 2, beta_aw, w, label="Awake", color=bar_colors, alpha=0.8)
    ax_beta.bar(x + w / 2, beta_sl, w, label="Sleep", color=bar_colors, alpha=0.5,
                edgecolor="k", linewidth=0.4)
    ax_beta.axhline(100, color="k",      lw=1.2, ls="--", label="100% = perfect")
    ax_beta.axhline(90,  color="orange", lw=0.9, ls=":",  label="90% threshold")
    ax_beta.set_xticks(x)
    ax_beta.set_xticklabels([m.split("(")[0].strip() for m in method_lbls],
                              rotation=30, ha="right", fontsize=8)
    ax_beta.set_ylabel("Beta preservation %", fontsize=8)
    ax_beta.set_title("Brain beta preservation (13–30 Hz)", fontsize=9)
    ax_beta.set_ylim(20, 400); ax_beta.legend(fontsize=8)

    fig.suptitle(
        "Real Data (Patient XU, 7 Hz DBS) — Full 7-Method Comparison\n"
        "Preservation % vs PRE baseline  |  100% = no brain lost  |  "
        "<100% = over-filtered  |  >100% = DBS residual",
        fontsize=12, fontweight="bold",
    )
    fig_master = figs / "real_03_master_comparison.png"
    fig.savefig(str(fig_master), dpi=150, bbox_inches="tight")
    plt.close(fig)

    value = {
        "awake_fif_paths":  awake_fif,
        "sleep_fif_paths":  sleep_fif,
        "metrics_awake":    metrics_awake,
        "metrics_sleep":    metrics_sleep,
        "figure_psd_awake": str(fig_psd_aw),
        "figure_psd_sleep": str(fig_psd_sl),
        "figure_master":    str(fig_master),
    }
    return Output(
        value=value,
        metadata={
            "AWAKE7 — PSD comparison":  _png_md(fig_psd_aw),
            "SLEEP7 — PSD comparison":  _png_md(fig_psd_sl),
            "Master comparison":        _png_md(fig_master),
            "awake_best_theta_pct": MetadataValue.float(float(
                max(m["Theta"] for m in metrics_awake))
            ),
            "sleep_best_theta_pct": MetadataValue.float(float(
                max(m["Theta"] for m in metrics_sleep))
            ),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Asset 3 — ICA + final evaluation figures
# ──────────────────────────────────────────────────────────────────────────────

@asset(
    group_name="real_data",
    description=(
        "Apply FastICA to the best-method DBS-filtered AWAKE7 and SLEEP7 signals. "
        "Classify and remove eye/muscle components. "
        "Save ICA topomap figure, 4-panel final PSD, band-power topomap grid, "
        "and print quantitative summary table."
    ),
)
def real_ica_pipeline(
    context: AssetExecutionContext,
    real_raw_loaded: dict,
    real_dbs_filtered: dict,
    eeg_config: EEGPipelineConfig,
) -> MaterializeResult:
    """
    Returns MaterializeResult with per-condition preservation % and ICA component counts
    shown in the Dagster UI asset detail pane.
    """
    proc     = eeg_config.processed_path()
    figs     = eeg_config.real_figures_path()
    best     = eeg_config.best_method
    dbs_freq = eeg_config.dbs_freq

    awake_fif_paths = real_dbs_filtered["awake_fif_paths"]
    sleep_fif_paths = real_dbs_filtered["sleep_fif_paths"]

    if best not in awake_fif_paths:
        raise ValueError(
            f"best_method='{best}' not found in filtered outputs. "
            f"Available: {list(awake_fif_paths.keys())}"
        )

    raw_awake7  = _raw_from_fif(real_raw_loaded["awake7_path"])
    raw_sleep7  = _raw_from_fif(real_raw_loaded["sleep7_path"])
    raw_pre_aw  = _raw_from_fif(real_raw_loaded["pre_awake_path"])
    raw_pre_sl  = _raw_from_fif(real_raw_loaded["pre_sleep_path"])
    raw_aw_dbs  = _raw_from_fif(awake_fif_paths[best])
    raw_sl_dbs  = _raw_from_fif(sleep_fif_paths[best])

    # ── Fit ICA for each condition ───────────────────────────────────────
    results: dict[str, dict] = {}
    for label, raw_dbs in [("AWAKE7", raw_aw_dbs), ("SLEEP7", raw_sl_dbs)]:
        ica = mne.preprocessing.ICA(
            n_components=eeg_config.n_ica_components,
            method="fastica",
            max_iter=eeg_config.ica_max_iter,
            random_state=eeg_config.ica_random_state,
        )
        ica.fit(raw_dbs, verbose=False)
        eye_comps, muscle_comps = _classify_ica(ica, raw_dbs)
        all_bad = sorted(set(eye_comps + muscle_comps))
        context.log.info(f"{label}: eye={eye_comps}  muscle={muscle_comps}  → exclude {all_bad}")
        ica.exclude = all_bad
        raw_clean = raw_dbs.copy()
        ica.apply(raw_clean, verbose=False)
        results[label] = {
            "ica": ica, "bad": all_bad, "eye": eye_comps,
            "muscle": muscle_comps, "clean": raw_clean,
        }

    raw_aw_clean = results["AWAKE7"]["clean"]
    raw_sl_clean = results["SLEEP7"]["clean"]

    # Save cleaned signals
    aw_clean_path = proc / "real_awake_final_clean.fif"
    sl_clean_path = proc / "real_sleep_final_clean.fif"
    raw_aw_clean.save(str(aw_clean_path), overwrite=True)
    raw_sl_clean.save(str(sl_clean_path), overwrite=True)

    # ── ICA topomap figure (excluded components) ─────────────────────────
    fig = plt.figure(figsize=(16, 8))
    gs_ica = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    for row_i, (cond, raw_dbs, raw_ref, raw_final) in enumerate([
        ("AWAKE7", raw_aw_dbs, raw_pre_aw, raw_aw_clean),
        ("SLEEP7", raw_sl_dbs, raw_pre_sl, raw_sl_clean),
    ]):
        ica_obj = results[cond]["ica"]
        bad     = results[cond]["bad"]
        mixing  = ica_obj.get_components()
        n_show  = min(len(bad), 3)

        for col_i in range(3):
            ax = fig.add_subplot(gs_ica[row_i, col_i])
            if col_i < n_show:
                ic  = bad[col_i]
                mix = mixing[:, ic]
                mne.viz.plot_topomap(
                    mix, raw_dbs.info, axes=ax, show=False, cmap="RdBu_r",
                    vlim=(-np.abs(mix).max(), np.abs(mix).max()),
                )
                tag = "eye" if ic in results[cond]["eye"] else "muscle"
                ax.set_title(f"{cond}\nIC{ic} [{tag}]", fontsize=8)
            else:
                ax.axis("off")

        ax_psd = fig.add_subplot(gs_ica[row_i, 3])
        raw_cont = raw_awake7 if "AWAKE" in cond else raw_sleep7
        for raw_s, col, lw, ls, lbl in [
            (raw_cont,  "gray",      1.0, "-",  "DBS contaminated"),
            (raw_dbs,   "steelblue", 1.2, "-",  "DBS removed"),
            (raw_final, "darkblue",  2.0, "-",  "Final (+ ICA)"),
            (raw_ref,   "limegreen" if "AWAKE" in cond else "navy", 1.0, "--", "PRE baseline"),
        ]:
            f_, p_ = _psd_db(raw_s, fmax=30.0)
            m_ = (f_ >= 1) & (f_ <= 30)
            ax_psd.plot(f_[m_], p_[m_], color=col, lw=lw, ls=ls, label=lbl)
        for k in range(1, 5):
            ax_psd.axvline(k * dbs_freq, color="orange", lw=0.7, ls=":", alpha=0.6)
        ax_psd.set_xlabel("Hz"); ax_psd.set_ylabel("PSD (dB)")
        ax_psd.set_title(f"{cond} — pipeline stages", fontsize=9)
        ax_psd.legend(fontsize=7)

    fig.suptitle("ICA artifact removal — AWAKE7 and SLEEP7", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig_ica = figs / "real_04_ica_results.png"
    fig.savefig(str(fig_ica), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 4-panel final PSD ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for row_i, (raw_dbs_cond, raw_dbs_rem, raw_final, raw_ref, cond_lbl, ref_col) in enumerate([
        (raw_awake7, raw_aw_dbs, raw_aw_clean, raw_pre_aw, "AWAKE7", "limegreen"),
        (raw_sleep7, raw_sl_dbs, raw_sl_clean, raw_pre_sl, "SLEEP7", "navy"),
    ]):
        for col_i, (fmin, fmax) in enumerate([(1, 80), (1, 30)]):
            ax = axes[row_i, col_i]
            for raw_s, col, lw, ls, lbl in [
                (raw_dbs_cond, "gray",     0.7, "-",  "DBS contaminated"),
                (raw_dbs_rem,  "steelblue",1.2, "-",  f"DBS removed ({best})"),
                (raw_final,    "darkblue", 2.0, "-",  "Final clean (+ ICA)"),
                (raw_ref,      ref_col,    1.0, "--", "PRE baseline"),
            ]:
                f_, p_ = _psd_db(raw_s, fmax=fmax)
                m_ = (f_ >= fmin) & (f_ <= fmax)
                ax.plot(f_[m_], p_[m_], color=col, lw=lw, ls=ls, alpha=0.85, label=lbl)
            for k in range(1, 5):
                h = k * dbs_freq
                if h <= fmax:
                    ax.axvline(h, color="orange", lw=0.9, ls=":", alpha=0.7,
                               label="DBS harmonic" if k == 1 else "")
            ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (dB)")
            suffix = "Full 1–80 Hz" if fmax == 80 else "Zoom 1–30 Hz"
            ax.set_title(f"{cond_lbl} — {suffix}", fontsize=10)
            ax.legend(fontsize=7)
    fig.suptitle(
        f"XU Patient — Final Pipeline: DBS Removal + ICA\nMethod: {best}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fig_psd = figs / "real_05_final_pipeline_psd.png"
    fig.savefig(str(fig_psd), dpi=150)
    plt.close(fig)

    # ── Band-power topomap grid (4 stages × 2 conditions × theta+alpha) ──
    awake_stages = [
        (raw_awake7,    "AWAKE\nContaminated"),
        (raw_aw_dbs,    "AWAKE\nDBS removed"),
        (raw_aw_clean,  "AWAKE\nFinal clean"),
        (raw_pre_aw,    "AWAKE\nPRE baseline"),
    ]
    sleep_stages = [
        (raw_sleep7,    "SLEEP\nContaminated"),
        (raw_sl_dbs,    "SLEEP\nDBS removed"),
        (raw_sl_clean,  "SLEEP\nFinal clean"),
        (raw_pre_sl,    "SLEEP\nPRE baseline"),
    ]
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    for col_i, (raw_s, lbl) in enumerate(awake_stages):
        _topomap_band(raw_s, 4,  8, axes[0, col_i], f"{lbl}\nTheta 4–8 Hz")
        _topomap_band(raw_s, 8, 13, axes[1, col_i], f"{lbl}\nAlpha 8–13 Hz")
    for col_i, (raw_s, lbl) in enumerate(sleep_stages):
        _topomap_band(raw_s, 4,  8, axes[2, col_i], f"{lbl}\nTheta 4–8 Hz")
        _topomap_band(raw_s, 8, 13, axes[3, col_i], f"{lbl}\nAlpha 8–13 Hz")
    fig.suptitle("Band-power topomaps — pipeline stages (AWAKE + SLEEP)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig_topo = figs / "real_06_topomaps.png"
    fig.savefig(str(fig_topo), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Quantitative summary ─────────────────────────────────────────────
    metadata: dict = {
        "best_method":     MetadataValue.text(best),
        "figure_ica":      MetadataValue.path(str(fig_ica)),
        "figure_final_psd":MetadataValue.path(str(fig_psd)),
        "figure_topomaps": MetadataValue.path(str(fig_topo)),
        "awake_clean_fif": MetadataValue.path(str(aw_clean_path)),
        "sleep_clean_fif": MetadataValue.path(str(sl_clean_path)),
    }
    for cond, raw_dbs_c, raw_clean_c, raw_ref_c in [
        ("awake", raw_awake7, raw_aw_clean, raw_pre_aw),
        ("sleep", raw_sleep7, raw_sl_clean, raw_pre_sl),
    ]:
        for bname, lo, hi in BANDS:
            pct = 100 * _band_power(raw_clean_c, lo, hi) / (_band_power(raw_ref_c, lo, hi) + 1e-30)
            metadata[f"{cond}_{bname.lower()}_pct"] = MetadataValue.float(float(round(pct, 1)))
        for res_label, res_dict in results.items():
            if cond.upper() in res_label:
                metadata[f"{cond}_ica_eye"]    = MetadataValue.text(str(res_dict["eye"]))
                metadata[f"{cond}_ica_muscle"] = MetadataValue.text(str(res_dict["muscle"]))
                metadata[f"{cond}_ica_excluded"] = MetadataValue.int(int(len(res_dict["bad"])))

    # Embed figures inline for Dagster UI
    metadata["ICA results"]      = _png_md(fig_ica)
    metadata["Final PSD"]        = _png_md(fig_psd)
    metadata["Band topomaps"]    = _png_md(fig_topo)
    metadata["Master comparison"]= _png_md(
        pathlib.Path(real_dbs_filtered["figure_master"])
    )

    context.log.info("real_ica_pipeline complete — all figures saved.")
    return MaterializeResult(metadata=metadata)


# ──────────────────────────────────────────────────────────────────────────────
# Asset 4 — Export final cleaned signals as EDF files
# ──────────────────────────────────────────────────────────────────────────────

@asset(
    group_name="real_data",
    description=(
        "Export the final pipeline-cleaned AWAKE7 and SLEEP7 signals as EDF files. "
        "Also exports DBS-only-removed (pre-ICA) variants for comparison. "
        "EDF files are written to data/processed/edf/."
    ),
)
def real_edf_export(
    context: AssetExecutionContext,
    real_raw_loaded: dict,
    real_dbs_filtered: dict,
    real_ica_pipeline: None,
    eeg_config: EEGPipelineConfig,
) -> Output:
    """
    Reads the saved .fif files from upstream assets and writes EDF files.

    Returns
    -------
    dict
        Mapping of label → absolute EDF path for all exported files.
    """
    import mne.export  # noqa: F401  (ensures export API is available)

    proc = eeg_config.processed_path()
    edf_dir = proc / "edf"
    edf_dir.mkdir(parents=True, exist_ok=True)

    best = eeg_config.best_method

    to_export = {
        "raw_awake7":       real_raw_loaded["awake7_path"],
        "raw_sleep7":       real_raw_loaded["sleep7_path"],
        "raw_pre_awake":    real_raw_loaded["pre_awake_path"],
        "raw_pre_sleep":    real_raw_loaded["pre_sleep_path"],
        f"dbs_removed_awake7": real_dbs_filtered["awake_fif_paths"][best],
        f"dbs_removed_sleep7": real_dbs_filtered["sleep_fif_paths"][best],
        "final_clean_awake7":  str(proc / "real_awake_final_clean.fif"),
        "final_clean_sleep7":  str(proc / "real_sleep_final_clean.fif"),
    }

    exported: dict[str, str] = {}
    for label, fif_path in to_export.items():
        p = pathlib.Path(fif_path)
        if not p.exists():
            context.log.warning(f"Skipping {label}: {fif_path} not found")
            continue
        raw = _raw_from_fif(fif_path)
        out_edf = edf_dir / f"{label}.edf"
        # MNE export requires physical_range to be set; use data min/max
        raw.export(str(out_edf), fmt="edf", physical_range="auto", overwrite=True, verbose=False)
        exported[label] = str(out_edf)
        size_mb = out_edf.stat().st_size / 1_048_576
        context.log.info(f"Exported {label} → {out_edf.name}  ({size_mb:.1f} MB)")

    return Output(
        value=exported,
        metadata={
            label: MetadataValue.path(path)
            for label, path in exported.items()
        } | {
            "n_files_exported": MetadataValue.int(int(len(exported))),
            "edf_directory": MetadataValue.path(str(edf_dir)),
        },
    )
