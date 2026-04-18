"""pipeline/assets/eeg_assets.py — Generic single-file EEG asset chain.

Asset graph (group: eeg_pipeline)
──────────────────────────────────
  ingested_edf          ← per-run Config: file_path + dbs_freq
        │
  cleaned_edf           ← SVD → Hampel FFT → (NLMS) → ICA → Spectral Interp
        │
  spike_features        ← extract_spike_features()
        │
  eeg_ml_features       ← channel_summary() + ml_feature_matrix()

All heavy MNE Raw objects are written to .fif on disk.  Assets exchange
lightweight metadata dicts so Dagster's default IOManager can serialise them.

This asset chain is intentionally decoupled from the legacy XU-specific
real_data / synthetic groups — it operates on *any* EDF file dropped into
data/raw/ and is the target of ``edf_ingest_sensor``.
"""

import pathlib
import sys
import warnings
from typing import Optional

_ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import mne

mne.set_log_level("WARNING")

from dagster import (
    asset,
    AssetExecutionContext,
    Config,
    MaterializeResult,
    MetadataValue,
    Output,
)

from src.ingestion import load_edf, IngestionConfig
from src.preprocessing import EEGPreprocessor
from src.spike_features import (
    SpikeDetectionConfig,
    extract_spike_features,
    channel_summary,
    ml_feature_matrix,
)


# ──────────────────────────────────────────────────────────────────────────────
# Per-run configuration (supplied via RunRequest.run_config or CLI --dg-config)
# ──────────────────────────────────────────────────────────────────────────────

class EdfIngestConfig(Config):
    """Per-run parameters for the eeg_pipeline asset chain.

    Passed in run_config as::

        {"ops": {"ingested_edf": {"config": {"file_path": "...", "dbs_freq": 7.0}}}}
    """
    file_path: str
    """Absolute or project-relative path to the source EDF file."""
    dbs_freq: float = 7.0
    """Primary DBS stimulation frequency in Hz."""
    target_sfreq: float = 256.0
    """Target sample rate after resampling."""
    max_duration_sec: float = 0.0
    """Crop recording to this many seconds (0 = keep full recording)."""
    run_svd: bool = True
    """Enable SVD spatial pre-filter stage."""
    run_nlms: bool = False
    """Enable harmonic-aware NLMS stage (off by default — validate first)."""
    run_ica: bool = True
    """Enable conservative ICA stage."""
    run_spectral_interp: bool = True
    """Enable targeted spectral interpolation stage."""


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _processed_dir() -> pathlib.Path:
    p = _ROOT / "data" / "processed" / "eeg_pipeline"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _stem(file_path: str) -> str:
    """Derive a safe filename stem from the source EDF path."""
    return pathlib.Path(file_path).stem


# ──────────────────────────────────────────────────────────────────────────────
# Asset 1 — Ingest EDF
# ──────────────────────────────────────────────────────────────────────────────

@asset(
    group_name="eeg_pipeline",
    description=(
        "Load a single EDF file via src.ingestion.load_edf(). "
        "Renames channels to standard 10-20 labels, classifies EEG/EOG/EMG, "
        "resamples if needed, bandpass-filters 1–119 Hz, applies average reference. "
        "Saves the typed MNE Raw object as a .fif file and returns provenance metadata."
    ),
)
def ingested_edf(
    context: AssetExecutionContext,
    config: EdfIngestConfig,
) -> Output:
    """
    Returns
    -------
    dict
        ``fif_path``        — absolute path to the saved .fif file
        ``eeg_channels``    — list of standard 10-20 channel names found
        ``eog_channels``    — list of EOG channel names
        ``emg_channels``    — list of EMG channel names
        ``other_channels``  — remaining channels
        ``metadata``        — provenance dict from IngestionResult
        ``stem``            — filename stem for downstream naming
    """
    src_path = pathlib.Path(config.file_path)
    if not src_path.is_absolute():
        src_path = _ROOT / src_path
    context.log.info(f"Ingesting {src_path.name} …")

    ing_config = IngestionConfig(
        max_duration_sec=config.max_duration_sec if config.max_duration_sec > 0 else None,
        target_sfreq=config.target_sfreq,
        l_freq=1.0,
        h_freq=119.0,
        apply_avg_ref=True,
        dbs_freq=config.dbs_freq,
        verbose=False,
    )
    result = load_edf(src_path, ing_config)
    raw = result.raw

    stem = _stem(config.file_path)
    fif_path = _processed_dir() / f"{stem}_ingested.fif"
    raw.save(str(fif_path), overwrite=True)
    context.log.info(
        f"  → {result.metadata['duration_sec'] / 60:.1f} min  "
        f"{result.metadata['sfreq']:.0f} Hz  "
        f"{result.metadata['n_eeg']} EEG channels  "
        f"saved → {fif_path.name}"
    )

    value = {
        "fif_path":       str(fif_path),
        "eeg_channels":   result.eeg_channels,
        "eog_channels":   result.eog_channels,
        "emg_channels":   result.emg_channels,
        "other_channels": result.other_channels,
        "metadata":       result.metadata,
        "stem":           stem,
        "dbs_freq":       config.dbs_freq,
        "target_sfreq":   config.target_sfreq,
        "run_svd":        config.run_svd,
        "run_nlms":       config.run_nlms,
        "run_ica":        config.run_ica,
        "run_spectral_interp": config.run_spectral_interp,
    }
    return Output(
        value=value,
        metadata={
            "source_file":    MetadataValue.path(str(src_path)),
            "fif_path":       MetadataValue.path(str(fif_path)),
            "duration_min":   MetadataValue.float(float(round(result.metadata["duration_sec"] / 60, 2))),
            "sfreq_hz":       MetadataValue.float(float(result.metadata["sfreq"])),
            "n_eeg_channels": MetadataValue.int(int(result.metadata["n_eeg"])),
            "n_eog_channels": MetadataValue.int(int(result.metadata["n_eog"])),
            "dbs_freq_hz":    MetadataValue.float(float(config.dbs_freq)),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Asset 2 — DBS Artifact Removal
# ──────────────────────────────────────────────────────────────────────────────

@asset(
    group_name="eeg_pipeline",
    description=(
        "Apply the surgical DBS removal pipeline to the ingested EEG. "
        "Stages (all configurable): SVD spatial pre-filter → Allen/Hampel FFT → "
        "(NLMS adaptive filter) → conservative ICA → targeted spectral interpolation. "
        "Saves cleaned .fif and returns provenance + band-power preservation metrics."
    ),
)
def cleaned_edf(
    context: AssetExecutionContext,
    ingested_edf: dict,
) -> Output:
    """
    Returns
    -------
    dict
        ``fif_path``     — absolute path to the cleaned .fif file
        ``stem``         — filename stem
        ``dbs_freq``     — DBS frequency used (Hz)
        ``band_power``   — {band_name: mean_power_uv2} for Delta/Theta/Alpha/Beta/Gamma
        ``n_eeg``        — number of EEG channels in cleaned signal
    """
    fif_in   = ingested_edf["fif_path"]
    stem     = ingested_edf["stem"]
    dbs_freq = float(ingested_edf["dbs_freq"])
    sfreq    = float(ingested_edf["target_sfreq"])

    context.log.info(f"Loading {pathlib.Path(fif_in).name} for DBS cleaning …")
    raw = mne.io.read_raw_fif(fif_in, preload=True, verbose=False)

    preprocessor = EEGPreprocessor(
        l_freq=1.0,
        h_freq=119.0,
    )

    # apply_clinical_filter is a no-op here because ingestion already filtered;
    # calling it is harmless but redundant — we skip it to avoid double-filtering.
    context.log.info(
        f"Running surgical pipeline: "
        f"SVD={ingested_edf['run_svd']}  NLMS={ingested_edf['run_nlms']}  "
        f"ICA={ingested_edf['run_ica']}  SpectralInterp={ingested_edf['run_spectral_interp']}"
    )
    raw_clean = preprocessor.apply_surgical_pipeline(
        raw,
        f_dbs=dbs_freq,
        target_sfreq=sfreq,
        run_ica=ingested_edf["run_ica"],
        run_svd=ingested_edf["run_svd"],
        run_nlms=ingested_edf["run_nlms"],
        run_spectral_interp=ingested_edf["run_spectral_interp"],
    )

    fif_out = _processed_dir() / f"{stem}_cleaned.fif"
    raw_clean.save(str(fif_out), overwrite=True)
    context.log.info(f"Cleaned signal saved → {fif_out.name}")

    # Compute band-power preservation metrics
    bands = [("Delta", 0.5, 4), ("Theta", 4, 8), ("Alpha", 8, 13),
             ("Beta", 13, 30), ("Gamma", 30, 80)]
    band_power: dict[str, float] = {}
    for bname, lo, hi in bands:
        psd_obj = raw_clean.compute_psd(method="welch", fmin=lo, fmax=hi,
                                         n_fft=2048, verbose=False)
        band_power[bname] = float(psd_obj.get_data().mean())

    value = {
        "fif_path":   str(fif_out),
        "stem":       stem,
        "dbs_freq":   dbs_freq,
        "band_power": band_power,
        "n_eeg":      len(mne.pick_types(raw_clean.info, eeg=True)),
        "sfreq":      float(raw_clean.info["sfreq"]),
        "duration_sec": float(raw_clean.times[-1]),
    }
    return Output(
        value=value,
        metadata={
            "cleaned_fif":    MetadataValue.path(str(fif_out)),
            "dbs_freq_hz":    MetadataValue.float(dbs_freq),
            **{
                f"{b}_power_uv2": MetadataValue.float(float(v))
                for b, v in band_power.items()
            },
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Asset 3 — Spike Feature Extraction
# ──────────────────────────────────────────────────────────────────────────────

@asset(
    group_name="eeg_pipeline",
    description=(
        "Detect epileptiform / DBS-related transient events on the cleaned EEG "
        "and compute the full ML-grade spike morphology + spectral feature set "
        "via src.spike_features.extract_spike_features(). "
        "Saves the per-spike DataFrame as a Parquet file and returns the path."
    ),
)
def spike_features(
    context: AssetExecutionContext,
    cleaned_edf: dict,
) -> Output:
    """
    Returns
    -------
    dict
        ``parquet_path``  — path to the per-spike Parquet file
        ``csv_path``      — path to the per-spike CSV file (human-readable)
        ``stem``          — filename stem
        ``n_spikes``      — total number of detected spikes
        ``n_channels``    — number of channels with at least one spike
        ``duration_sec``  — recording duration fed to the detector
    """
    fif_path     = cleaned_edf["fif_path"]
    stem         = cleaned_edf["stem"]
    duration_sec = cleaned_edf["duration_sec"]

    context.log.info(f"Extracting spike features from {pathlib.Path(fif_path).name} …")
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)

    config = SpikeDetectionConfig()
    df = extract_spike_features(raw, config=config, recording_id=stem)

    context.log.info(f"  → {len(df)} spikes detected across {df['channel'].nunique() if not df.empty else 0} channels")

    proc = _processed_dir()
    csv_path = proc / f"{stem}_spikes.csv"
    df.to_csv(str(csv_path), index=False)

    # Save as parquet if pyarrow is available, otherwise CSV only
    parquet_path = None  # type: Optional[str]
    try:
        import pyarrow  # noqa: F401
        _pq = proc / f"{stem}_spikes.parquet"
        df.to_parquet(str(_pq), index=False)
        parquet_path = str(_pq)
    except ImportError:
        pass

    value = {
        "parquet_path": parquet_path,
        "csv_path":     str(csv_path),
        "stem":         stem,
        "n_spikes":     int(len(df)),
        "n_channels":   int(df["channel"].nunique()) if not df.empty else 0,
        "duration_sec": duration_sec,
    }
    meta: dict = {
        "csv_path":   MetadataValue.path(str(csv_path)),
        "n_spikes":   MetadataValue.int(int(len(df))),
        "n_channels": MetadataValue.int(int(df["channel"].nunique()) if not df.empty else 0),
    }
    if parquet_path:
        meta["parquet_path"] = MetadataValue.path(parquet_path)
    return Output(value=value, metadata=meta)


# ──────────────────────────────────────────────────────────────────────────────
# Asset 4 — ML Feature Matrix
# ──────────────────────────────────────────────────────────────────────────────

@asset(
    group_name="eeg_pipeline",
    description=(
        "Build the final ML-ready feature matrix from the per-spike DataFrame. "
        "Calls src.spike_features.channel_summary() for per-channel aggregates and "
        "src.spike_features.ml_feature_matrix() for the imputed spike-level matrix. "
        "Saves channel summary CSV + spike feature numpy array (.npy)."
    ),
)
def eeg_ml_features(
    context: AssetExecutionContext,
    spike_features: dict,
) -> MaterializeResult:
    """
    Returns MaterializeResult with feature shape and channel summary counts
    shown in the Dagster UI.
    """
    import pandas as pd

    stem         = spike_features["stem"]
    duration_sec = spike_features["duration_sec"]

    context.log.info(f"Building ML feature matrix for {stem} …")
    # Prefer parquet if available (faster), fall back to CSV
    if spike_features.get("parquet_path"):
        df = pd.read_parquet(spike_features["parquet_path"])
    else:
        df = pd.read_csv(spike_features["csv_path"])

    proc = _processed_dir()

    if df.empty:
        context.log.warning("No spikes detected — feature matrix will be empty.")
        ch_summary_df = pd.DataFrame()
        X             = np.empty((0, 0), dtype=np.float64)
        feature_names: list[str] = []
    else:
        ch_summary_df = channel_summary(df, duration_sec=duration_sec)
        X, feature_names = ml_feature_matrix(df, impute_strategy="median")

    # Save outputs
    ch_summary_path = proc / f"{stem}_channel_summary.csv"
    features_path   = proc / f"{stem}_features.npy"
    fname_path      = proc / f"{stem}_feature_names.txt"

    ch_summary_df.to_csv(str(ch_summary_path), index=False)
    np.save(str(features_path), X)
    fname_path.write_text("\n".join(feature_names))

    context.log.info(
        f"  → Feature matrix shape: {X.shape}  "
        f"  Channel summary: {len(ch_summary_df)} rows"
    )

    return MaterializeResult(
        metadata={
            "channel_summary_csv": MetadataValue.path(str(ch_summary_path)),
            "features_npy":        MetadataValue.path(str(features_path)),
            "n_spikes":            MetadataValue.int(int(X.shape[0])),
            "n_features":          MetadataValue.int(int(X.shape[1]) if X.ndim == 2 else 0),
            "n_channels_with_spikes": MetadataValue.int(int(len(ch_summary_df))),
            "feature_names_preview": MetadataValue.text(
                ", ".join(feature_names[:10]) + (" …" if len(feature_names) > 10 else "")
            ),
        }
    )
