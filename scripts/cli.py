#!/usr/bin/env python
"""scripts/cli.py — Unified EEG pipeline command-line interface.

Calls the exact same src/ functions as the Dagster asset chain in
pipeline/assets/eeg_assets.py — zero divergence between the two entry points.

Usage examples
--------------
# Single file, auto-detect DBS freq from filename (7 Hz for XUAWAKE7_...)
python scripts/cli.py --file data/raw/XU/XUAWAKE7_deidentified.edf

# Single file, explicit DBS freq
python scripts/cli.py \\
    --file data/raw/XU/XUAWAKE7_deidentified.edf \\
    --dbs-freq 7.0

# Batch: every EDF under data/raw/XU/
python scripts/cli.py --dir data/raw/XU --dbs-freq 7.0

# Disable ICA, enable NLMS (advanced)
python scripts/cli.py \\
    --file data/raw/XU/XUAWAKE7_deidentified.edf \\
    --dbs-freq 7.0 \\
    --no-ica \\
    --enable-nlms

# Crop to first 120 s for a quick smoke test
python scripts/cli.py \\
    --file data/raw/XU/XUAWAKE7_deidentified.edf \\
    --dbs-freq 7.0 \\
    --max-duration 120
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import warnings

warnings.filterwarnings("ignore")

# Ensure project root is on the path regardless of where the script is invoked
_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import mne
import numpy as np

mne.set_log_level("WARNING")

from src.ingestion import load_edf, IngestionConfig, find_edf_files
from src.preprocessing import EEGPreprocessor
from src.spike_features import (
    SpikeDetectionConfig,
    extract_spike_features,
    channel_summary,
    ml_feature_matrix,
)


# ──────────────────────────────────────────────────────────────────────────────
# DBS frequency inference (mirrors pipeline/sensors.py logic)
# ──────────────────────────────────────────────────────────────────────────────

_DBS_FREQ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:hz|Hz)", re.IGNORECASE)
_DBS_STEM_RE = re.compile(r"(?<!\d)(\d+)(?!\d)")


def _infer_dbs_freq(stem: str) -> float:
    """Infer DBS frequency from filename stem."""
    m = _DBS_FREQ_RE.search(stem)
    if m:
        return float(m.group(1))
    m = _DBS_STEM_RE.search(stem)
    if m:
        return float(m.group(1))
    return 7.0


# ──────────────────────────────────────────────────────────────────────────────
# Core pipeline logic (shared with Dagster assets)
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    edf_path: pathlib.Path,
    dbs_freq: float,
    output_dir: pathlib.Path,
    target_sfreq: float = 256.0,
    max_duration_sec: float | None = None,
    run_svd: bool = True,
    run_nlms: bool = False,
    run_ica: bool = True,
    run_spectral_interp: bool = True,
    verbose: bool = True,
) -> dict:
    """Run the full EEG pipeline on a single EDF file.

    This function calls the exact same src/ functions that the Dagster asset
    chain uses, ensuring CLI and automated sensor outputs are identical.

    Parameters
    ----------
    edf_path         : Path to the source EDF file.
    dbs_freq         : Primary DBS stimulation frequency (Hz).
    output_dir       : Directory where all output files are written.
    target_sfreq     : Target sample rate after resampling (Hz).
    max_duration_sec : Crop recording to this duration (None = full).
    run_svd          : Enable SVD spatial pre-filter.
    run_nlms         : Enable harmonic-aware NLMS adaptive filter.
    run_ica          : Enable conservative ICA component removal.
    run_spectral_interp : Enable targeted spectral interpolation.
    verbose          : Print progress to stdout.

    Returns
    -------
    dict with keys:
        fif_ingested, fif_cleaned, parquet_spikes, csv_spikes,
        csv_channel_summary, npy_features, feature_names, n_spikes
    """
    def _log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = edf_path.stem

    # ── Stage 1: Ingest ──────────────────────────────────────────────────────
    _log(f"\n[1/4] Ingesting {edf_path.name}  (DBS = {dbs_freq} Hz) …")
    ing_config = IngestionConfig(
        max_duration_sec=max_duration_sec,
        target_sfreq=target_sfreq,
        l_freq=1.0,
        h_freq=119.0,
        apply_avg_ref=True,
        dbs_freq=dbs_freq,
        verbose=False,
    )
    result = load_edf(edf_path, ing_config)
    raw = result.raw

    fif_ingested = output_dir / f"{stem}_ingested.fif"
    raw.save(str(fif_ingested), overwrite=True)
    _log(
        f"    {result.metadata['duration_sec'] / 60:.1f} min  "
        f"{result.metadata['sfreq']:.0f} Hz  "
        f"{result.metadata['n_eeg']} EEG channels  "
        f"→ {fif_ingested.name}"
    )

    # ── Stage 2: DBS Artifact Removal ────────────────────────────────────────
    _log(
        f"\n[2/4] Surgical DBS removal "
        f"(SVD={run_svd}  NLMS={run_nlms}  ICA={run_ica}  "
        f"SpectralInterp={run_spectral_interp}) …"
    )
    preprocessor = EEGPreprocessor(
        l_freq=1.0,
        h_freq=119.0,
    )
    raw_clean = preprocessor.apply_surgical_pipeline(
        raw,
        f_dbs=dbs_freq,
        target_sfreq=target_sfreq,
        run_ica=run_ica,
        run_svd=run_svd,
        run_nlms=run_nlms,
        run_spectral_interp=run_spectral_interp,
    )

    fif_cleaned = output_dir / f"{stem}_cleaned.fif"
    raw_clean.save(str(fif_cleaned), overwrite=True)
    _log(f"    Cleaned signal → {fif_cleaned.name}")

    # Band-power summary
    bands = [("Delta", 0.5, 4), ("Theta", 4, 8), ("Alpha", 8, 13),
             ("Beta", 13, 30), ("Gamma", 30, 80)]
    if verbose:
        print("    Band power (µV²):")
        for bname, lo, hi in bands:
            psd_obj = raw_clean.compute_psd(
                method="welch", fmin=lo, fmax=hi, n_fft=2048, verbose=False
            )
            power_uv2 = float(psd_obj.get_data().mean())
            print(f"      {bname:<8} {power_uv2:.4e}")

    # ── Stage 3: Spike Feature Extraction ────────────────────────────────────
    _log(f"\n[3/4] Extracting spike features …")
    spike_config = SpikeDetectionConfig()
    spike_df = extract_spike_features(raw_clean, config=spike_config, recording_id=stem)
    n_spikes   = len(spike_df)
    n_channels = spike_df["channel"].nunique() if not spike_df.empty else 0
    _log(f"    {n_spikes} spikes detected across {n_channels} channels")

    csv_spikes = output_dir / f"{stem}_spikes.csv"
    spike_df.to_csv(str(csv_spikes), index=False)
    parquet_spikes: pathlib.Path | None = None
    try:
        import pyarrow  # noqa: F401
        parquet_spikes = output_dir / f"{stem}_spikes.parquet"
        spike_df.to_parquet(str(parquet_spikes), index=False)
    except ImportError:
        pass
    saved = f"{csv_spikes.name}" + (f"  {parquet_spikes.name}" if parquet_spikes else "")
    _log(f"    Saved → {saved}")

    # ── Stage 4: ML Feature Matrix ────────────────────────────────────────────
    _log(f"\n[4/4] Building ML feature matrix …")
    duration_sec = float(raw_clean.times[-1])

    if spike_df.empty:
        _log("    No spikes — feature matrix is empty.")
        ch_summary_df  = __import__("pandas").DataFrame()
        X              = np.empty((0, 0), dtype=np.float64)
        feature_names: list[str] = []
    else:
        ch_summary_df = channel_summary(spike_df, duration_sec=duration_sec)
        X, feature_names = ml_feature_matrix(spike_df, impute_strategy="median")

    csv_ch_summary  = output_dir / f"{stem}_channel_summary.csv"
    npy_features    = output_dir / f"{stem}_features.npy"
    fname_txt       = output_dir / f"{stem}_feature_names.txt"

    ch_summary_df.to_csv(str(csv_ch_summary), index=False)
    np.save(str(npy_features), X)
    fname_txt.write_text("\n".join(feature_names))

    _log(f"    Feature matrix: {X.shape}  → {npy_features.name}")
    _log(f"    Channel summary: {len(ch_summary_df)} rows  → {csv_ch_summary.name}")

    return {
        "fif_ingested":       str(fif_ingested),
        "fif_cleaned":        str(fif_cleaned),
        "parquet_spikes":     str(parquet_spikes),
        "csv_spikes":         str(csv_spikes),
        "csv_channel_summary":str(csv_ch_summary),
        "npy_features":       str(npy_features),
        "feature_names":      feature_names,
        "n_spikes":           n_spikes,
        "n_channels":         n_channels,
        "feature_shape":      X.shape,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cli.py",
        description="EEG DBS artifact removal + ML feature extraction pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input (mutually exclusive: single file or directory batch)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--file", "-f",
        metavar="PATH",
        help="Path to a single EDF file.",
    )
    src.add_argument(
        "--dir", "-d",
        metavar="DIR",
        help="Directory to search recursively for EDF files.",
    )

    # DBS parameters
    p.add_argument(
        "--dbs-freq",
        type=float,
        default=None,
        metavar="HZ",
        help=(
            "Primary DBS stimulation frequency in Hz.  "
            "If omitted, inferred from the filename "
            "(e.g. 'AWAKE7' → 7.0 Hz, 'SLEEP60' → 60.0 Hz; default 7.0)."
        ),
    )

    # Output
    p.add_argument(
        "--output-dir", "-o",
        metavar="DIR",
        default=str(_ROOT / "data" / "processed" / "eeg_pipeline"),
        help="Directory for all output files (default: data/processed/eeg_pipeline/).",
    )

    # Pipeline stage flags
    p.add_argument(
        "--no-svd",
        action="store_true",
        help="Disable SVD spatial pre-filter stage.",
    )
    p.add_argument(
        "--enable-nlms",
        action="store_true",
        help="Enable harmonic-aware NLMS adaptive filter (off by default — validate first).",
    )
    p.add_argument(
        "--no-ica",
        action="store_true",
        help="Disable conservative ICA component removal.",
    )
    p.add_argument(
        "--no-spectral-interp",
        action="store_true",
        help="Disable targeted spectral interpolation post-ICA.",
    )

    # Signal parameters
    p.add_argument(
        "--sfreq",
        type=float,
        default=256.0,
        metavar="HZ",
        help="Target sample rate in Hz (default: 256).",
    )
    p.add_argument(
        "--max-duration",
        type=float,
        default=None,
        metavar="SEC",
        help="Crop each recording to this many seconds (default: keep full).",
    )

    # Misc
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )

    return p


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    output_dir = pathlib.Path(args.output_dir)
    verbose    = not args.quiet

    # Resolve file list
    if args.file:
        files = [pathlib.Path(args.file)]
    else:
        files = find_edf_files(pathlib.Path(args.dir))
        if not files:
            print(f"ERROR: no EDF files found under {args.dir}", file=sys.stderr)
            return 1
        if verbose:
            print(f"Found {len(files)} EDF file(s) under {args.dir}")

    errors: list[tuple[pathlib.Path, str]] = []

    for edf_path in files:
        # Infer DBS freq per file if not explicitly provided
        dbs_freq = args.dbs_freq if args.dbs_freq is not None else _infer_dbs_freq(edf_path.stem)

        if verbose and len(files) > 1:
            print(f"\n{'─' * 60}")
            print(f"File: {edf_path.name}  (DBS = {dbs_freq} Hz)")

        try:
            result = run_pipeline(
                edf_path=edf_path,
                dbs_freq=dbs_freq,
                output_dir=output_dir,
                target_sfreq=args.sfreq,
                max_duration_sec=args.max_duration,
                run_svd=not args.no_svd,
                run_nlms=args.enable_nlms,
                run_ica=not args.no_ica,
                run_spectral_interp=not args.no_spectral_interp,
                verbose=verbose,
            )
            if verbose:
                print(
                    f"\n  ✓ {edf_path.name}  "
                    f"spikes={result['n_spikes']}  "
                    f"features={result['feature_shape']}  "
                    f"→ {output_dir}"
                )
        except Exception as exc:  # noqa: BLE001
            msg = f"ERROR processing {edf_path.name}: {exc}"
            print(msg, file=sys.stderr)
            errors.append((edf_path, str(exc)))

    if errors:
        print(f"\n{len(errors)} file(s) failed:", file=sys.stderr)
        for p, e in errors:
            print(f"  {p.name}: {e}", file=sys.stderr)
        return 1

    if verbose:
        print(f"\nDone. Outputs in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
