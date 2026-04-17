"""src/spike_features.py — spike morphology feature extraction.

Detects epileptiform spikes on clean EEG data and computes morphology
descriptors: amplitude, half-width, rise time, decay time, and sharpness.

INPUT CONTRACT
==============
Pass data AFTER ``src.artifact_removal.run_artifact_removal`` — i.e., consensus-
filtered, DBS-cleaned, ICA-corrected. Spike morphology is highly sensitive to
filtering; measuring on pre-cleaned data produces invalid descriptors.

UNITS
=====
All amplitude values are in µV (MNE stores data in V; conversion is applied
internally). Time values are in milliseconds.

THRESHOLD
=========
Detection uses a per-channel adaptive threshold: median + N × MAD, where MAD
is the median absolute deviation. This is robust to non-Gaussian noise and does
not require a normality assumption.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import mne
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SpikeDetectionConfig:
    """Parameters controlling detection and morphology measurement.

    Attributes:
        threshold_mad_k:    Multiplier on MAD for adaptive threshold.
        min_peak_distance_ms: Minimum inter-spike interval (refractory period).
        peak_width_ms:      Maximum accepted spike duration at half prominence
                            (filters out slow waves).
        halfwidth_rel:      Amplitude fraction for half-width measurement (0.5 = FWHM).
        rise_lo:            Lower fraction for rise-time measurement (10 %).
        rise_hi:            Upper fraction (90 %).
        decay_hi:           Upper fraction for decay-time measurement (90 %).
        decay_lo:           Lower fraction (10 %).
        min_amplitude_uv:   Hard minimum peak amplitude to accept (µV).
        max_amplitude_uv:   Hard maximum (rejects saturated artefacts) (µV).
        search_window_ms:   Half-window around peak for morphology search (ms).
        eeg_only:           If True, restrict detection to EEG-typed channels.
        verbose:            Passed to MNE functions where applicable.
    """
    threshold_mad_k: float = 5.0
    min_peak_distance_ms: float = 70.0    # ms — ~14 Hz refractory period
    peak_width_ms: float = 200.0          # ms — ignore slow waves wider than this
    halfwidth_rel: float = 0.5            # FWHM
    rise_lo: float = 0.10
    rise_hi: float = 0.90
    decay_hi: float = 0.90
    decay_lo: float = 0.10
    min_amplitude_uv: float = 20.0        # µV
    max_amplitude_uv: float = 2000.0      # µV
    search_window_ms: float = 150.0       # half-window for morphology (ms)
    eeg_only: bool = True
    verbose: Union[bool, str] = False


# ──────────────────────────────────────────────────────────────────────────────
# Per-spike morphology helpers
# ──────────────────────────────────────────────────────────────────────────────

def _mad(x: np.ndarray) -> float:
    """Median absolute deviation, a robust scale estimator."""
    return float(np.median(np.abs(x - np.median(x))))


def _find_threshold_crossings(
    signal: np.ndarray,
    peak_idx: int,
    level: float,
    sfreq: float,
    half_window_samples: int,
    ascending: bool,
) -> Optional[int]:
    """Find the sample index where signal crosses ``level`` near a peak.

    Args:
        signal:             1-D signal array (µV).
        peak_idx:           Sample index of the spike peak.
        level:              Voltage threshold to locate (µV).
        sfreq:              Sampling frequency (Hz).
        half_window_samples: Search radius in samples.
        ascending:          If True, search on the rising (pre-peak) flank;
                            if False, on the falling (post-peak) flank.

    Returns:
        Sample index of the crossing, or None if not found.
    """
    if ascending:
        segment = signal[max(0, peak_idx - half_window_samples): peak_idx + 1]
        if segment.size < 2:
            return None
        # Search backward from peak for the first sample below level
        for i in range(len(segment) - 2, -1, -1):
            if segment[i] < level <= segment[i + 1]:
                # Linear interpolation
                frac = (level - segment[i]) / (segment[i + 1] - segment[i] + 1e-12)
                return max(0, peak_idx - (len(segment) - 1 - i)) + int(frac)
        return None
    else:
        start = peak_idx
        end = min(len(signal), peak_idx + half_window_samples + 1)
        segment = signal[start:end]
        if segment.size < 2:
            return None
        for i in range(len(segment) - 1):
            if segment[i] >= level > segment[i + 1]:
                frac = (segment[i] - level) / (segment[i] - segment[i + 1] + 1e-12)
                return start + i + int(frac)
        return None


def measure_spike_morphology(
    signal_uv: np.ndarray,
    peak_idx: int,
    sfreq: float,
    config: SpikeDetectionConfig,
) -> dict:
    """Measure morphology descriptors for a single detected spike.

    Args:
        signal_uv:  1-D EEG channel data in µV.
        peak_idx:   Sample index of the spike peak.
        sfreq:      Sampling frequency (Hz).
        config:     SpikeDetectionConfig.

    Returns:
        Dict with keys: amplitude_uv, half_width_ms, rise_time_ms,
        decay_time_ms, sharpness.  Values are NaN where measurement failed.
    """
    hw = int(config.search_window_ms * sfreq / 1000)
    peak_amp = float(signal_uv[peak_idx])

    result = dict(
        amplitude_uv=peak_amp,
        half_width_ms=np.nan,
        rise_time_ms=np.nan,
        decay_time_ms=np.nan,
        sharpness=np.nan,
    )

    # ── Half-width (FWHM at halfwidth_rel of peak amplitude) ─────────────────
    hw_level = peak_amp * config.halfwidth_rel
    left_idx = _find_threshold_crossings(
        signal_uv, peak_idx, hw_level, sfreq, hw, ascending=True
    )
    right_idx = _find_threshold_crossings(
        signal_uv, peak_idx, hw_level, sfreq, hw, ascending=False
    )
    if left_idx is not None and right_idx is not None and right_idx > left_idx:
        result["half_width_ms"] = (right_idx - left_idx) / sfreq * 1000.0

    # ── Rise time (rise_lo → rise_hi fraction of peak amplitude) ─────────────
    lo_level = peak_amp * config.rise_lo
    hi_level = peak_amp * config.rise_hi
    lo_idx = _find_threshold_crossings(
        signal_uv, peak_idx, lo_level, sfreq, hw, ascending=True
    )
    hi_idx = _find_threshold_crossings(
        signal_uv, peak_idx, hi_level, sfreq, hw, ascending=True
    )
    if lo_idx is not None and hi_idx is not None and hi_idx > lo_idx:
        result["rise_time_ms"] = (hi_idx - lo_idx) / sfreq * 1000.0

    # ── Decay time (decay_hi → decay_lo fraction of peak amplitude) ──────────
    dhi_idx = _find_threshold_crossings(
        signal_uv, peak_idx, peak_amp * config.decay_hi, sfreq, hw, ascending=False
    )
    dlo_idx = _find_threshold_crossings(
        signal_uv, peak_idx, peak_amp * config.decay_lo, sfreq, hw, ascending=False
    )
    if dhi_idx is not None and dlo_idx is not None and dlo_idx > dhi_idx:
        result["decay_time_ms"] = (dlo_idx - dhi_idx) / sfreq * 1000.0

    # ── Sharpness — second derivative at peak (curvature), normalised ────────
    half_w_ms = result["half_width_ms"]
    if np.isfinite(half_w_ms) and half_w_ms > 0:
        # Amplitude / half-width ratio: large value = sharp spike
        result["sharpness"] = peak_amp / half_w_ms
    else:
        # Fallback: second derivative at peak sample
        if 1 <= peak_idx < len(signal_uv) - 1:
            d2 = (
                signal_uv[peak_idx + 1]
                - 2 * signal_uv[peak_idx]
                + signal_uv[peak_idx - 1]
            ) * (sfreq / 1000.0) ** 2  # units: µV/ms²
            result["sharpness"] = float(-d2)  # negative because peak is positive

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Per-channel spike detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_spikes(
    signal_uv: np.ndarray,
    sfreq: float,
    config: SpikeDetectionConfig,
) -> np.ndarray:
    """Detect spike peak indices on a single channel.

    Threshold: median + ``threshold_mad_k`` × MAD (computed per channel).
    Candidate peaks are further filtered by minimum distance and width.

    Args:
        signal_uv:  1-D channel data in µV.
        sfreq:      Sampling frequency (Hz).
        config:     SpikeDetectionConfig.

    Returns:
        Array of sample indices of accepted spike peaks.
    """
    med = np.median(signal_uv)
    mad = _mad(signal_uv)
    threshold = med + config.threshold_mad_k * mad

    min_dist_samples = max(1, int(config.min_peak_distance_ms * sfreq / 1000))
    max_width_samples = int(config.peak_width_ms * sfreq / 1000)

    peaks, props = find_peaks(
        signal_uv,
        height=threshold,
        distance=min_dist_samples,
        width=(1, max_width_samples),
    )

    # Amplitude gates in µV
    if peaks.size == 0:
        return peaks

    heights = signal_uv[peaks]
    mask = (heights >= config.min_amplitude_uv) & (heights <= config.max_amplitude_uv)
    return peaks[mask]


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def extract_spike_features(
    data: Union[mne.io.BaseRaw, mne.Epochs],
    config: Optional[SpikeDetectionConfig] = None,
) -> pd.DataFrame:
    """Detect spikes and extract morphology features from clean EEG data.

    Accepts either a continuous ``mne.io.BaseRaw`` or an ``mne.Epochs`` object.
    For Epochs, each epoch is processed independently and the epoch index is
    included in the output.

    Args:
        data:   MNE Raw or Epochs object **after** artifact removal.
        config: SpikeDetectionConfig (defaults applied if None).

    Returns:
        DataFrame with one row per detected spike.  Columns:

        =========== ===================================================
        channel     Channel name (str)
        epoch       Epoch index (int; 0 for Raw)
        time_sec    Absolute time of peak within recording (s)
        amplitude_uv  Peak amplitude (µV)
        half_width_ms  FWHM of spike (ms)
        rise_time_ms   10–90 % rise time (ms)
        decay_time_ms  90–10 % decay time (ms)
        sharpness    amplitude_uv / half_width_ms (or curvature estimate)
        =========== ===================================================
    """
    if config is None:
        config = SpikeDetectionConfig()

    rows: list[dict] = []

    if isinstance(data, mne.io.BaseRaw):
        _process_raw(data, config, rows)
    elif isinstance(data, mne.Epochs):
        _process_epochs(data, config, rows)
    else:
        raise TypeError(
            f"data must be mne.io.BaseRaw or mne.Epochs, got {type(data).__name__}"
        )

    cols = [
        "channel", "epoch", "time_sec",
        "amplitude_uv", "half_width_ms", "rise_time_ms", "decay_time_ms", "sharpness",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)


def _process_raw(
    raw: mne.io.BaseRaw,
    config: SpikeDetectionConfig,
    rows: list[dict],
) -> None:
    raw.load_data()
    picks = (
        mne.pick_types(raw.info, eeg=True, exclude=[])
        if config.eeg_only
        else np.arange(len(raw.ch_names))
    )
    sfreq = float(raw.info["sfreq"])
    data_v = raw.get_data(picks=picks)  # (n_ch, n_times) in V
    times = raw.times  # seconds

    for ch_i, pick in enumerate(picks):
        ch_name = raw.ch_names[pick]
        signal_uv = data_v[ch_i] * 1e6  # V → µV
        peak_idxs = detect_spikes(signal_uv, sfreq, config)
        for pidx in peak_idxs:
            morph = measure_spike_morphology(signal_uv, int(pidx), sfreq, config)
            rows.append(
                {
                    "channel": ch_name,
                    "epoch": 0,
                    "time_sec": float(times[pidx]),
                    **morph,
                }
            )


def _process_epochs(
    epochs: mne.Epochs,
    config: SpikeDetectionConfig,
    rows: list[dict],
) -> None:
    picks = (
        mne.pick_types(epochs.info, eeg=True, exclude=[])
        if config.eeg_only
        else np.arange(len(epochs.ch_names))
    )
    sfreq = float(epochs.info["sfreq"])
    times = epochs.times  # relative to epoch onset (s)
    data_v = epochs.get_data(picks=picks)  # (n_epochs, n_ch, n_times) in V

    for ep_i in range(data_v.shape[0]):
        for ch_i, pick in enumerate(picks):
            ch_name = epochs.ch_names[pick]
            signal_uv = data_v[ep_i, ch_i] * 1e6
            peak_idxs = detect_spikes(signal_uv, sfreq, config)
            for pidx in peak_idxs:
                morph = measure_spike_morphology(signal_uv, int(pidx), sfreq, config)
                rows.append(
                    {
                        "channel": ch_name,
                        "epoch": ep_i,
                        "time_sec": float(times[pidx]),
                        **morph,
                    }
                )


# ──────────────────────────────────────────────────────────────────────────────
# Channel-level summary statistics
# ──────────────────────────────────────────────────────────────────────────────

def channel_summary(
    spike_df: pd.DataFrame,
    duration_sec: Optional[float] = None,
) -> pd.DataFrame:
    """Compute per-channel summary statistics from a spike DataFrame.

    Args:
        spike_df:      Output of ``extract_spike_features``.
        duration_sec:  Total recording length in seconds, used to compute
                       ``spike_rate_per_min``.  If None, inferred from
                       ``max(time_sec) – min(time_sec)`` per channel.

    Returns:
        DataFrame with one row per channel.  For each morphology feature
        (amplitude_uv, half_width_ms, rise_time_ms, decay_time_ms, sharpness)
        the columns ``<feature>_mean``, ``<feature>_median``, ``<feature>_std``
        are included, plus ``spike_count`` and ``spike_rate_per_min``.
    """
    if spike_df.empty:
        return pd.DataFrame(
            columns=[
                "channel", "spike_count", "spike_rate_per_min",
                "amplitude_uv_mean", "amplitude_uv_median", "amplitude_uv_std",
                "half_width_ms_mean", "half_width_ms_median", "half_width_ms_std",
                "rise_time_ms_mean", "rise_time_ms_median", "rise_time_ms_std",
                "decay_time_ms_mean", "decay_time_ms_median", "decay_time_ms_std",
                "sharpness_mean", "sharpness_median", "sharpness_std",
            ]
        )

    morph_cols = [
        "amplitude_uv", "half_width_ms", "rise_time_ms", "decay_time_ms", "sharpness"
    ]

    agg_funcs = {col: ["mean", "median", "std"] for col in morph_cols}
    agg_funcs["time_sec"] = ["count", "min", "max"]

    summary = spike_df.groupby("channel").agg(agg_funcs)
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary = summary.reset_index()

    # ── Spike count and rate ──────────────────────────────────────────────────
    summary.rename(columns={"time_sec_count": "spike_count"}, inplace=True)

    if duration_sec is not None:
        dur = float(duration_sec)
        summary["spike_rate_per_min"] = summary["spike_count"] / (dur / 60.0)
    else:
        dur_per_ch = summary["time_sec_max"] - summary["time_sec_min"]
        dur_per_ch = dur_per_ch.clip(lower=1e-3)
        summary["spike_rate_per_min"] = summary["spike_count"] / (dur_per_ch / 60.0)

    summary.drop(columns=["time_sec_min", "time_sec_max"], inplace=True)
    return summary
