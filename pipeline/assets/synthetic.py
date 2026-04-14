"""
Dagster software-defined assets for the *synthetic* EEG pipeline.

Asset graph
───────────
  synthetic_signal
        │
  synthetic_dbs_filtered          ← runs all 7 DBS removal methods
        │
  synthetic_ica_pipeline          ← ICA on best method; final evaluation figures

All heavy MNE Raw objects are saved to .fif on disk; assets exchange
lightweight metadata dicts so Dagster's default IOManager can serialize them.
"""


import sys
import pathlib
import warnings

# Make src/ importable from the pipeline package
_ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

mne.set_log_level("WARNING")

import base64
from dagster import asset, AssetExecutionContext, MaterializeResult, MetadataValue, Output


# ── Helper: encode a saved PNG as base64 markdown for inline Dagster UI display ──
def _png_md(path) -> MetadataValue:
    with open(str(path), "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    name = pathlib.Path(str(path)).name
    return MetadataValue.md(f"![{name}](data:image/png;base64,{b64})")


from pipeline.resources import EEGPipelineConfig
from pipeline.constants import BANDS, DBS_METHODS, METHOD_PALETTE, STANDARD_CH
from src.synthetic_eeg import SyntheticEEG
from src.filters import ArtifactFilterFactory


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _raw_from_fif(path: str) -> mne.io.Raw:
    return mne.io.read_raw_fif(path, preload=True, verbose=False)


def _apply_method(raw: mne.io.Raw, filter_key: str, kwargs: dict) -> mne.io.Raw:
    r = raw.copy()
    r.load_data()
    d = r.get_data() * 1e6  # V → µV
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
        obj = r.compute_psd(
            method="welch", fmin=h - bw, fmax=h + bw, n_fft=2048, verbose=False
        )
        return obj.get_data().mean()

    return float(10 * np.log10(bp(raw_after) / (bp(raw_before) + 1e-30)))


def _classify_ica(ica: mne.preprocessing.ICA, raw: mne.io.Raw) -> tuple[list, list]:
    """Conservative two-condition ICA classifier (eye + muscle)."""
    from scipy import signal as sp_signal

    sfreq = raw.info["sfreq"]
    mixing = ica.get_components()
    sources = ica.get_sources(raw).get_data()
    ch_lower = [c.lower() for c in raw.ch_names]
    fp_idx = [i for i, c in enumerate(ch_lower) if c in ("fp1", "fp2")]
    muscle_idx = [
        i for i, c in enumerate(ch_lower) if c in ("t3", "t4", "t5", "t6", "f7", "f8")
    ]
    eye_comps, muscle_comps = [], []
    for ic in range(ica.n_components_):
        col = np.abs(mixing[:, ic])
        top_ch = int(np.argmax(col))
        f_s, psd_s = sp_signal.welch(sources[ic], fs=sfreq, nperseg=512)
        p_all = np.trapz(psd_s[(f_s >= 1) & (f_s <= 80)], f_s[(f_s >= 1) & (f_s <= 80)]) + 1e-30
        lf_r = np.trapz(psd_s[(f_s >= 1) & (f_s <= 15)], f_s[(f_s >= 1) & (f_s <= 15)]) / p_all
        hf_r = np.trapz(psd_s[(f_s >= 30) & (f_s <= 80)], f_s[(f_s >= 30) & (f_s <= 80)]) / p_all
        if top_ch in fp_idx and lf_r > 0.60:
            eye_comps.append(ic)
        elif top_ch in muscle_idx and hf_r > 0.55:
            muscle_comps.append(ic)
    return eye_comps, muscle_comps


# ──────────────────────────────────────────────────────────────────────────────
# Asset 1 — Generate synthetic signal
# ──────────────────────────────────────────────────────────────────────────────

@asset(
    group_name="synthetic",
    description=(
        "Generate a 19-channel synthetic EEG signal with 7 Hz DBS harmonics, "
        "eye blink/saccade, and EMG burst artifacts. "
        "Saves the mixed (contaminated) signal to a .fif file."
    ),
)
def synthetic_signal(
    context: AssetExecutionContext,
    eeg_config: EEGPipelineConfig,
) -> Output:
    """
    Returns
    -------
    dict
        ``fif_path``   : absolute path to saved MNE Raw (.fif)
        ``brain_path`` : path to brain-only MNE Raw (.fif) used as reference
        ``sfreq``      : sampling rate
        ``duration``   : recording duration in seconds
        ``dbs_freq``   : DBS frequency used
        ``n_channels`` : number of EEG channels
        ``rms``        : per-component RMS in µV
    """
    proc = eeg_config.processed_path()

    gen = SyntheticEEG(
        sfreq=eeg_config.synthetic_sfreq,
        duration=eeg_config.synthetic_duration,
        seed=eeg_config.synthetic_seed,
    )
    sig = gen.generate(dbs_freq=eeg_config.dbs_freq)

    # ── Save mixed (contaminated) signal ────────────────────────────────
    raw_mixed = gen.to_mne_raw(sig, "mixed")
    fif_path = proc / "synth_mixed.fif"
    raw_mixed.save(str(fif_path), overwrite=True)

    # ── Save brain-only reference ────────────────────────────────────────
    raw_brain = gen.to_mne_raw(sig, "brain")
    brain_path = proc / "synth_brain.fif"
    raw_brain.save(str(brain_path), overwrite=True)

    rms = {
        k: float(np.sqrt(np.mean(sig[k] ** 2)))
        for k in ("brain", "dbs", "eyes", "muscle", "mixed")
    }
    context.log.info(
        "Synthetic signal RMS (µV): "
        + ", ".join(f"{k}={v:.2f}" for k, v in rms.items())
    )

    value = {
        "fif_path":   str(fif_path),
        "brain_path": str(brain_path),
        "sfreq":      float(sig["sfreq"]),
        "duration":   float(eeg_config.synthetic_duration),
        "dbs_freq":   float(eeg_config.dbs_freq),
        "n_channels": int(len(sig["ch_names"])),
        "rms":        rms,
    }
    return Output(
        value=value,
        metadata={
            "n_channels":  MetadataValue.int(int(value["n_channels"])),
            "duration_s":  MetadataValue.float(float(value["duration"])),
            "dbs_freq_hz": MetadataValue.float(float(value["dbs_freq"])),
            "brain_rms_uv":MetadataValue.float(float(round(rms["brain"], 2))),
            "dbs_rms_uv":  MetadataValue.float(float(round(rms["dbs"],   2))),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Asset 2 — Run all 7 DBS removal methods on the synthetic signal
# ──────────────────────────────────────────────────────────────────────────────

@asset(
    group_name="synthetic",
    description=(
        "Apply all 7 DBS removal methods to the synthetic signal. "
        "Saves one .fif per method. "
        "Computes brain-band preservation % and harmonic attenuation dB. "
        "Saves PSD comparison figure and bar chart."
    ),
)
def synthetic_dbs_filtered(
    context: AssetExecutionContext,
    synthetic_signal: dict,
    eeg_config: EEGPipelineConfig,
) -> Output:
    """
    Returns
    -------
    dict
        ``fif_paths``  : {method_label: str path to cleaned .fif}
        ``metrics``    : list of per-method metric dicts
        ``figure_psd`` : path to PSD comparison figure
        ``figure_bars``: path to bar chart figure
    """
    proc = eeg_config.processed_path()
    figs = eeg_config.synth_figures_path()
    dbs_freq = eeg_config.dbs_freq

    raw_cont = _raw_from_fif(synthetic_signal["fif_path"])
    raw_brain = _raw_from_fif(synthetic_signal["brain_path"])

    fif_paths: dict[str, str] = {}
    metrics: list[dict] = []

    for label, filter_key, kwargs_factory in DBS_METHODS:
        context.log.info(f"Applying: {label}")
        kw = kwargs_factory(dbs_freq)
        raw_clean = _apply_method(raw_cont, filter_key, kw)

        # Save to .fif
        safe_name = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        out_path = proc / f"synth_{safe_name}.fif"
        raw_clean.save(str(out_path), overwrite=True)
        fif_paths[label] = str(out_path)

        # Metrics
        bp = {b: 100 * _band_power(raw_clean, lo, hi) / (_band_power(raw_brain, lo, hi) + 1e-30)
              for b, lo, hi in BANDS}
        harmonics = [k * dbs_freq for k in range(1, 5)]
        atten = float(np.mean([_harmonic_atten_db(raw_cont, raw_clean, h) for h in harmonics]))
        metrics.append({"label": label, **bp, "atten_db": atten})
        context.log.info(
            f"  Theta={bp['Theta']:.1f}%  Beta={bp['Beta']:.1f}%  Atten={atten:.1f} dB"
        )

    # ── PSD comparison figure ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (fmin, fmax), title in zip(
        axes,
        [(1, 80), (1, 30)],
        ["Full spectrum 1–80 Hz", "Zoom 1–30 Hz"],
    ):
        fb, pb = _psd_db(raw_brain, fmax=fmax)
        fc, pc = _psd_db(raw_cont,  fmax=fmax)
        m = (fb >= fmin) & (fb <= fmax)
        ax.fill_between(fb[m], pb[m] - 2, pb[m] + 2, color="limegreen", alpha=0.12)
        ax.plot(fb[m], pb[m], color="limegreen", lw=1.2, ls="--", label="Brain only")
        ax.plot(fc[m], pc[m], color="gray",      lw=0.7, alpha=0.5, label="DBS contaminated")
        for label, _, _ in DBS_METHODS:
            col, ls, lw = METHOD_PALETTE[label]
            raw_cl = _raw_from_fif(fif_paths[label])
            fn, pn = _psd_db(raw_cl, fmax=fmax)
            mn = (fn >= fmin) & (fn <= fmax)
            ax.plot(fn[mn], pn[mn], color=col, ls=ls, lw=lw, label=label)
        for k in range(1, int(fmax // dbs_freq) + 1):
            ax.axvline(k * dbs_freq, color="orange", lw=0.6, ls=":", alpha=0.6,
                       label="DBS harmonic" if k == 1 else "")
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (dB)")
        ax.set_title(title); ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Synthetic EEG — 7-method DBS removal comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    psd_fig = figs / "synth_method_comparison_psd.png"
    fig.savefig(psd_fig, dpi=150)
    plt.close(fig)

    # ── Bar chart: band preservation ────────────────────────────────────
    band_names = [b for b, *_ in BANDS]
    labels = [m["label"] for m in metrics]
    n = len(labels)
    colors = [METHOD_PALETTE[l][0] for l in labels]

    fig, axes = plt.subplots(1, len(band_names), figsize=(16, 5))
    for ax, bname in zip(axes, band_names):
        vals = [m[bname] for m in metrics]
        bars = ax.bar(range(n), vals, color=colors, alpha=0.85)
        ax.axhline(100, color="k", lw=1.0, ls="--")
        ax.axhline(90,  color="orange", lw=0.8, ls=":")
        ax.set_xticks(range(n))
        ax.set_xticklabels([l.split("(")[0].strip() for l in labels],
                            rotation=40, ha="right", fontsize=7)
        ax.set_ylabel("Preservation %"); ax.set_title(f"{bname} band"); ax.set_ylim(40, 250)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, min(val, 245) + 1,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=6, fontweight="bold")
    fig.suptitle("Synthetic — Brain-band preservation % vs brain-only reference", fontsize=12)
    plt.tight_layout()
    bar_fig = figs / "synth_method_comparison_bars.png"
    fig.savefig(bar_fig, dpi=150)
    plt.close(fig)

    context.log.info(f"Saved figures: {psd_fig}, {bar_fig}")

    value = {
        "fif_paths":   fif_paths,
        "metrics":     metrics,
        "figure_psd":  str(psd_fig),
        "figure_bars": str(bar_fig),
    }
    return Output(
        value=value,
        metadata={
            "PSD comparison":  _png_md(psd_fig),
            "Band preservation bars": _png_md(bar_fig),
            "n_methods": MetadataValue.int(int(len(metrics))),
            "best_theta_pct": MetadataValue.float(float(
                max(m["Theta"] for m in metrics))
            ),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Asset 3 — ICA artifact removal on the best DBS filter output
# ──────────────────────────────────────────────────────────────────────────────

@asset(
    group_name="synthetic",
    description=(
        "Apply FastICA to the best-method DBS-filtered synthetic signal. "
        "Classify eye (Fp1/Fp2 dominant + low-freq) and muscle components. "
        "Saves ICA topomaps, final PSD comparison, and band-power topomaps."
    ),
)
def synthetic_ica_pipeline(
    context: AssetExecutionContext,
    synthetic_signal: dict,
    synthetic_dbs_filtered: dict,
    eeg_config: EEGPipelineConfig,
) -> MaterializeResult:
    """
    Returns MaterializeResult with quantitative metadata shown in the Dagster UI.
    Saves the final clean signal to ``data/processed/synth_final_clean.fif``.
    """
    proc = eeg_config.processed_path()
    figs = eeg_config.synth_figures_path()
    best = eeg_config.best_method

    if best not in synthetic_dbs_filtered["fif_paths"]:
        available = list(synthetic_dbs_filtered["fif_paths"].keys())
        raise ValueError(
            f"best_method='{best}' not found. Available: {available}"
        )

    raw_dbs_removed = _raw_from_fif(synthetic_dbs_filtered["fif_paths"][best])
    raw_brain = _raw_from_fif(synthetic_signal["brain_path"])
    raw_cont  = _raw_from_fif(synthetic_signal["fif_path"])

    # ── Fit ICA ─────────────────────────────────────────────────────────
    ica = mne.preprocessing.ICA(
        n_components=eeg_config.n_ica_components,
        method="fastica",
        max_iter=eeg_config.ica_max_iter,
        random_state=eeg_config.ica_random_state,
    )
    ica.fit(raw_dbs_removed, verbose=False)
    eye_comps, muscle_comps = _classify_ica(ica, raw_dbs_removed)
    all_bad = sorted(set(eye_comps + muscle_comps))
    context.log.info(f"ICA: eye={eye_comps}  muscle={muscle_comps}  → exclude {all_bad}")

    ica.exclude = all_bad
    raw_clean = raw_dbs_removed.copy()
    ica.apply(raw_clean, verbose=False)

    # Save final clean signal
    clean_path = proc / "synth_final_clean.fif"
    raw_clean.save(str(clean_path), overwrite=True)

    # ── ICA topomap figure ───────────────────────────────────────────────
    if all_bad:
        mixing = ica.get_components()
        n_bad = len(all_bad)
        fig, axes = plt.subplots(1, n_bad, figsize=(4 * n_bad, 4))
        axes = [axes] if n_bad == 1 else list(axes)
        for ax, ic in zip(axes, all_bad):
            mix = mixing[:, ic]
            mne.viz.plot_topomap(
                mix, raw_dbs_removed.info, axes=ax, show=False, cmap="RdBu_r",
                vlim=(-np.abs(mix).max(), np.abs(mix).max()),
            )
            tag = "eye" if ic in eye_comps else "muscle"
            ax.set_title(f"IC{ic} [{tag}]", fontsize=9)
        fig.suptitle("ICA excluded components — synthetic pipeline", fontsize=11)
        plt.tight_layout()
        topo_fig = figs / "synth_ica_topomaps.png"
        fig.savefig(topo_fig, dpi=150)
        plt.close(fig)
    else:
        context.log.info("No ICA components excluded.")
        topo_fig = None

    # ── Final PSD comparison ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (fmin, fmax), title in zip(
        axes,
        [(1, 80), (1, 30)],
        ["Full 1–80 Hz", "Zoom 1–30 Hz"],
    ):
        for raw_s, col, lw, ls, lbl in [
            (raw_cont,        "gray",       0.7, "-",  "DBS contaminated"),
            (raw_dbs_removed, "steelblue",  1.2, "-",  f"DBS removed ({best})"),
            (raw_clean,       "darkblue",   2.0, "-",  "Final (+ ICA)"),
            (raw_brain,       "limegreen",  1.0, "--", "Brain reference"),
        ]:
            f_, p_ = _psd_db(raw_s, fmax=fmax)
            m_ = (f_ >= fmin) & (f_ <= fmax)
            ax.plot(f_[m_], p_[m_], color=col, lw=lw, ls=ls, alpha=0.85, label=lbl)
        for k in range(1, int(fmax // eeg_config.dbs_freq) + 1):
            ax.axvline(k * eeg_config.dbs_freq, color="orange", lw=0.7, ls=":", alpha=0.6,
                       label="DBS harmonic" if k == 1 else "")
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (dB)")
        ax.set_title(title); ax.legend(fontsize=7)
    fig.suptitle("Synthetic — Final pipeline PSD", fontsize=13, fontweight="bold")
    plt.tight_layout()
    psd_fig = figs / "synth_final_psd.png"
    fig.savefig(psd_fig, dpi=150)
    plt.close(fig)

    # ── Compute final metrics ────────────────────────────────────────────
    final_metrics = {}
    for bname, lo, hi in BANDS:
        pct = 100 * _band_power(raw_clean, lo, hi) / (_band_power(raw_brain, lo, hi) + 1e-30)
        final_metrics[bname] = round(pct, 1)

    context.log.info(
        "Final preservation: "
        + "  ".join(f"{b}={v}%" for b, v in final_metrics.items())
    )

    return MaterializeResult(
        metadata={
            "best_method":            MetadataValue.text(best),
            "n_components_excluded":  MetadataValue.int(int(len(all_bad))),
            "eye_components":         MetadataValue.text(str(eye_comps)),
            "muscle_components":      MetadataValue.text(str(muscle_comps)),
            "delta_preservation_pct": MetadataValue.float(float(final_metrics["Delta"])),
            "theta_preservation_pct": MetadataValue.float(float(final_metrics["Theta"])),
            "alpha_preservation_pct": MetadataValue.float(float(final_metrics["Alpha"])),
            "beta_preservation_pct":  MetadataValue.float(float(final_metrics["Beta"])),
            "ICA topomaps":           _png_md(topo_fig) if topo_fig else MetadataValue.text("no components excluded"),
            "Final PSD":              _png_md(psd_fig),
        }
    )


# ──────────────────────────────────────────────────────────────────────────────
# Asset 4 — Export synthetic pipeline signals as EDF files
# ──────────────────────────────────────────────────────────────────────────────

@asset(
    group_name="synthetic",
    description=(
        "Export synthetic pipeline signals (contaminated, best DBS-removed, final clean, "
        "and brain reference) as EDF files for external analysis or viewer."
    ),
)
def synthetic_edf_export(
    context: AssetExecutionContext,
    synthetic_signal: dict,
    synthetic_dbs_filtered: dict,
    synthetic_ica_pipeline: None,
    eeg_config: EEGPipelineConfig,
) -> Output:
    """
    Returns
    -------
    dict
        Mapping of label → absolute EDF path for all exported files.
    """
    import mne.export  # noqa: F401

    proc = eeg_config.processed_path()
    edf_dir = proc / "edf"
    edf_dir.mkdir(parents=True, exist_ok=True)

    best = eeg_config.best_method

    to_export = {
        "synth_contaminated":   synthetic_signal["fif_path"],
        "synth_brain_only":     synthetic_signal["brain_path"],
        f"synth_dbs_removed":   synthetic_dbs_filtered["fif_paths"][best],
        "synth_final_clean":    str(proc / "synth_final_clean.fif"),
    }

    exported: dict[str, str] = {}
    for label, fif_path in to_export.items():
        p = pathlib.Path(fif_path)
        if not p.exists():
            context.log.warning(f"Skipping {label}: {fif_path} not found")
            continue
        raw = _raw_from_fif(fif_path)
        out_edf = edf_dir / f"{label}.edf"
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
            "edf_directory":    MetadataValue.path(str(edf_dir)),
        },
    )
