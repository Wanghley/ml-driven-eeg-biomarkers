"""src/spike_features.py — ML-grade spike morphology and spectral feature extraction.

Detects epileptiform / DBS-related transient events on clean EEG and computes
a rich, ML-ready feature vector per spike.  The output DataFrame can be passed
directly to scikit-learn after minimal preprocessing (NaN imputation, optional
scaling) via the helper :func:`ml_feature_matrix`.

INPUT CONTRACT
==============
Pass data **after** ``src.artifact_removal.run_artifact_removal`` — i.e., consensus-
filtered, DBS-cleaned, ICA-corrected.  Spike morphology is highly sensitive to
filtering; measuring on pre-cleaned data produces invalid descriptors.

UNITS
=====
* Amplitude  — µV
* Time       — milliseconds (ms) unless noted as seconds (s)
* Frequency  — Hz
* Ratios     — dimensionless [0, 1] unless noted

FEATURE GROUPS
==============
1. **Detection metadata**    : channel, epoch, time_sec, recording_id
2. **Amplitude morphology**  : amplitude_uv, relative_amplitude, local_snr,
                                peak_to_trough_uv, pre_spike_baseline_uv
3. **Temporal morphology**   : half_width_ms, rise_time_ms, decay_time_ms,
                                spike_duration_ms, symmetry_ratio,
                                slope_rising_uv_ms
4. **Sharpness**             : sharpness, curvature_uv_ms2
5. **Area / energy**         : area_uv_ms, post_spike_suppression_ratio
6. **Spectral (spike window)**: dominant_freq_hz, spectral_centroid_hz,
                                spectral_entropy, theta_power_ratio,
                                alpha_power_ratio, beta_power_ratio,
                                high_freq_ratio
7. **Inter-spike**           : isi_prev_ms, isi_next_ms
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import mne
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import entropy as sp_entropy


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SpikeDetectionConfig:
    """Parameters controlling detection sensitivity and morphology measurement.

    Attributes
    ----------
    threshold_mad_k : float
        Adaptive threshold = channel_median + k × MAD.
        Typical range: 4–8.  Higher → fewer but more certain detections.
    min_peak_distance_ms : float
        Minimum inter-spike interval enforcing a refractory period (ms).
        Prevents double-counting of broad spikes.
    peak_width_ms : float
        Maximum accepted width at half prominence (ms).
        Rejects slow waves broader than this.
    halfwidth_rel : float
        Amplitude fraction at which half-width is measured (FWHM = 0.5).
    rise_lo / rise_hi : float
        Lower / upper amplitude fractions for rise-time measurement (10 %–90 %).
    decay_hi / decay_lo : float
        Upper / lower amplitude fractions for decay-time measurement (90 %–10 %).
    min_amplitude_uv : float
        Hard minimum peak amplitude (µV).  Filters sub-threshold fluctuations.
    max_amplitude_uv : float
        Hard maximum (µV).  Rejects saturated recording artifacts.
    search_window_ms : float
        Half-window around each spike peak used for morphology measurements (ms).
    spectral_window_ms : float
        Half-window used for per-spike spectral analysis (ms).
        Should be larger than search_window_ms to capture the waveform context.
    eeg_only : bool
        Restrict detection to MNE-typed EEG channels.
    verbose : bool or str
        MNE verbosity flag.
    """
    threshold_mad_k: float       = 5.0
    min_peak_distance_ms: float  = 70.0
    peak_width_ms: float         = 200.0
    halfwidth_rel: float         = 0.5
    rise_lo: float               = 0.10
    rise_hi: float               = 0.90
    decay_hi: float              = 0.90
    decay_lo: float              = 0.10
    min_amplitude_uv: float      = 20.0
    max_amplitude_uv: float      = 2000.0
    search_window_ms: float      = 150.0
    spectral_window_ms: float    = 250.0
    eeg_only: bool               = True
    verbose: Union[bool, str]    = False


# ──────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────────────────────────────────────────

def _mad(x: np.ndarray) -> float:
    """Median absolute deviation — robust scale estimator."""
    return float(np.median(np.abs(x - np.median(x))))


def _find_threshold_crossing(
    signal: np.ndarray,
    peak_idx: int,
    level: float,
    half_window: int,
    ascending: bool,
) -> Optional[float]:
    """Return fractional sample index of a threshold crossing near a spike peak.

    Uses linear interpolation between adjacent samples for sub-sample accuracy.

    Parameters
    ----------
    signal      : 1-D µV array
    peak_idx    : sample index of spike peak
    level       : voltage threshold (µV)
    half_window : search radius in samples
    ascending   : True → search rising flank (pre-peak);
                  False → search falling flank (post-peak)

    Returns
    -------
    Fractional sample index, or None if not found.
    """
    if ascending:
        lo = max(0, peak_idx - half_window)
        seg = signal[lo: peak_idx + 1]
        for i in range(len(seg) - 2, -1, -1):
            if seg[i] < level <= seg[i + 1]:
                frac = (level - seg[i]) / (seg[i + 1] - seg[i] + 1e-12)
                return (lo + i) + frac
    else:
        hi = min(len(signal), peak_idx + half_window + 1)
        seg = signal[peak_idx: hi]
        for i in range(len(seg) - 1):
            if seg[i] >= level > seg[i + 1]:
                frac = (seg[i] - level) / (seg[i] - seg[i + 1] + 1e-12)
                return peak_idx + i + frac
    return None


def _spike_spectrum(
    signal_uv: np.ndarray,
    peak_idx: int,
    sfreq: float,
    half_win_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (freqs_hz, power) for the Hann-windowed spike waveform."""
    lo = max(0, peak_idx - half_win_samples)
    hi = min(len(signal_uv), peak_idx + half_win_samples + 1)
    waveform = signal_uv[lo:hi].copy()
    n = len(waveform)
    if n < 4:
        return np.array([0.0]), np.array([0.0])
    window = np.hanning(n)
    waveform *= window
    # Zero-pad to next power of 2 for efficiency
    n_fft = max(64, int(2 ** np.ceil(np.log2(n))))
    spec = np.abs(np.fft.rfft(waveform, n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sfreq)
    return freqs, spec


# ──────────────────────────────────────────────────────────────────────────────
# Per-spike feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def measure_spike_morphology(
    signal_uv: np.ndarray,
    peak_idx: int,
    sfreq: float,
    config: SpikeDetectionConfig,
    channel_rms: float = 1.0,
) -> dict:
    """Compute the full ML feature vector for a single detected spike.

    Returns a flat dict with all features.  Values are ``np.nan`` where
    measurement failed (e.g., truncated window, no threshold crossing found).

    Parameters
    ----------
    signal_uv   : full 1-D channel array in µV
    peak_idx    : sample index of the spike peak
    sfreq       : sampling frequency (Hz)
    config      : SpikeDetectionConfig
    channel_rms : RMS of the full channel (µV), used for relative amplitude

    Returns
    -------
    dict with keys documented in the module docstring.
    """
    hw = int(config.search_window_ms * sfreq / 1000)
    sw = int(config.spectral_window_ms * sfreq / 1000)
    pre_win = int(100 * sfreq / 1000)   # 100 ms context window
    n = len(signal_uv)

    peak_amp = float(signal_uv[peak_idx])

    # ── Baseline: mean of 100 ms pre-spike window ─────────────────────────
    bl_lo = max(0, peak_idx - pre_win)
    pre_signal = signal_uv[bl_lo:peak_idx]
    pre_baseline = float(np.mean(pre_signal)) if pre_signal.size else 0.0
    pre_rms = float(np.sqrt(np.mean(pre_signal ** 2))) if pre_signal.size else 1.0
    peak_corrected = peak_amp - pre_baseline

    # ── Post-spike RMS (100 ms after peak) ───────────────────────────────
    post_signal = signal_uv[peak_idx: min(n, peak_idx + pre_win)]
    post_rms = float(np.sqrt(np.mean(post_signal ** 2))) if post_signal.size else pre_rms

    # ── Peak-to-trough (preceding 50 ms minimum) ─────────────────────────
    trough_lo = max(0, peak_idx - int(50 * sfreq / 1000))
    pre_trough_amp = float(np.min(signal_uv[trough_lo:peak_idx])) if peak_idx > trough_lo else peak_amp
    peak_to_trough = peak_amp - pre_trough_amp

    # ── FWHM ──────────────────────────────────────────────────────────────
    hw_level = pre_baseline + peak_corrected * config.halfwidth_rel
    left_frac = _find_threshold_crossing(signal_uv, peak_idx, hw_level, hw, ascending=True)
    right_frac = _find_threshold_crossing(signal_uv, peak_idx, hw_level, hw, ascending=False)
    half_width_ms = (
        (right_frac - left_frac) / sfreq * 1000.0
        if left_frac is not None and right_frac is not None and right_frac > left_frac
        else np.nan
    )

    # ── Rise time (rise_lo → rise_hi) ────────────────────────────────────
    lo_frac = _find_threshold_crossing(
        signal_uv, peak_idx, pre_baseline + peak_corrected * config.rise_lo, hw, ascending=True)
    hi_frac = _find_threshold_crossing(
        signal_uv, peak_idx, pre_baseline + peak_corrected * config.rise_hi, hw, ascending=True)
    rise_time_ms = (
        (hi_frac - lo_frac) / sfreq * 1000.0
        if lo_frac is not None and hi_frac is not None and hi_frac > lo_frac
        else np.nan
    )

    # ── Decay time (decay_hi → decay_lo) ─────────────────────────────────
    dhi_frac = _find_threshold_crossing(
        signal_uv, peak_idx, pre_baseline + peak_corrected * config.decay_hi, hw, ascending=False)
    dlo_frac = _find_threshold_crossing(
        signal_uv, peak_idx, pre_baseline + peak_corrected * config.decay_lo, hw, ascending=False)
    decay_time_ms = (
        (dlo_frac - dhi_frac) / sfreq * 1000.0
        if dhi_frac is not None and dlo_frac is not None and dlo_frac > dhi_frac
        else np.nan
    )

    # ── Spike total duration (baseline crossings) ─────────────────────────
    baseline_cross_left = _find_threshold_crossing(
        signal_uv, peak_idx, pre_baseline, hw, ascending=True)
    baseline_cross_right = _find_threshold_crossing(
        signal_uv, peak_idx, pre_baseline, hw, ascending=False)
    spike_duration_ms = (
        (baseline_cross_right - baseline_cross_left) / sfreq * 1000.0
        if baseline_cross_left is not None and baseline_cross_right is not None
           and baseline_cross_right > baseline_cross_left
        else np.nan
    )

    # ── Sharpness = amplitude / half-width ───────────────────────────────
    sharpness = (
        peak_corrected / half_width_ms
        if np.isfinite(half_width_ms) and half_width_ms > 0
        else np.nan
    )

    # ── Second-derivative curvature (µV/ms²) ─────────────────────────────
    if 1 <= peak_idx < n - 1:
        dt_ms = 1000.0 / sfreq
        curvature = -(
            signal_uv[peak_idx + 1] - 2.0 * signal_uv[peak_idx] + signal_uv[peak_idx - 1]
        ) / (dt_ms ** 2)
    else:
        curvature = np.nan

    # ── Symmetry ratio (rise / decay — 1.0 = symmetric) ──────────────────
    symmetry_ratio = (
        rise_time_ms / decay_time_ms
        if np.isfinite(rise_time_ms) and np.isfinite(decay_time_ms) and decay_time_ms > 0
        else np.nan
    )

    # ── Slope of rising phase (µV/ms) ────────────────────────────────────
    slope_rising = (
        peak_corrected / rise_time_ms
        if np.isfinite(rise_time_ms) and rise_time_ms > 0
        else np.nan
    )

    # ── Area under spike above baseline ──────────────────────────────────
    if baseline_cross_left is not None and baseline_cross_right is not None:
        i0 = int(np.floor(baseline_cross_left))
        i1 = int(np.ceil(baseline_cross_right)) + 1
        i0, i1 = max(0, i0), min(n, i1)
        seg = signal_uv[i0:i1] - pre_baseline
        area_uv_ms = float(np.trapz(np.maximum(seg, 0.0))) * 1000.0 / sfreq
    else:
        area_uv_ms = np.nan

    # ── Relative amplitude (z-score vs channel RMS) ───────────────────────
    relative_amplitude = peak_corrected / (channel_rms + 1e-12)

    # ── Local SNR ─────────────────────────────────────────────────────────
    local_snr = peak_corrected / (pre_rms + 1e-12)

    # ── Post-spike suppression ratio ──────────────────────────────────────
    post_suppression = (post_rms / pre_rms) if pre_rms > 0 else np.nan

    # ── Spectral features ─────────────────────────────────────────────────
    freqs, power = _spike_spectrum(signal_uv, peak_idx, sfreq, sw)
    total_power = power.sum()

    if total_power > 0 and len(freqs) > 2:
        norm_power = power / total_power

        # Dominant frequency
        dominant_freq_hz = float(freqs[np.argmax(power)])

        # Spectral centroid
        spectral_centroid_hz = float(np.dot(freqs, norm_power))

        # Shannon spectral entropy (nats → bits via log2)
        # Guard against zero bins with small epsilon
        safe_p = np.maximum(norm_power, 1e-12)
        safe_p /= safe_p.sum()
        spectral_entropy = float(sp_entropy(safe_p, base=2))

        def _band_ratio(lo, hi):
            mask = (freqs >= lo) & (freqs <= hi)
            return float(norm_power[mask].sum()) if mask.any() else 0.0

        theta_power_ratio  = _band_ratio(4.0,  8.0)
        alpha_power_ratio  = _band_ratio(8.0,  13.0)
        beta_power_ratio   = _band_ratio(13.0, 30.0)
        high_freq_ratio    = _band_ratio(80.0, 200.0)
    else:
        dominant_freq_hz = spectral_centroid_hz = spectral_entropy = np.nan
        theta_power_ratio = alpha_power_ratio = beta_power_ratio = high_freq_ratio = np.nan

    return {
        # ── amplitude morphology ──────────────────────────────────────────
        "amplitude_uv":              peak_amp,
        "amplitude_corrected_uv":    peak_corrected,
        "peak_to_trough_uv":         peak_to_trough,
        "pre_spike_baseline_uv":     pre_baseline,
        "relative_amplitude":        relative_amplitude,
        "local_snr":                 local_snr,
        # ── temporal morphology ──────────────────────────────────────────
        "half_width_ms":             half_width_ms,
        "rise_time_ms":              rise_time_ms,
        "decay_time_ms":             decay_time_ms,
        "spike_duration_ms":         spike_duration_ms,
        "symmetry_ratio":            symmetry_ratio,
        "slope_rising_uv_ms":        slope_rising,
        # ── sharpness / energy ───────────────────────────────────────────
        "sharpness":                 sharpness,
        "curvature_uv_ms2":          curvature,
        "area_uv_ms":                area_uv_ms,
        "post_spike_suppression":    post_suppression,
        # ── spectral ──────────────────────────────────────────────────────
        "dominant_freq_hz":          dominant_freq_hz,
        "spectral_centroid_hz":      spectral_centroid_hz,
        "spectral_entropy":          spectral_entropy,
        "theta_power_ratio":         theta_power_ratio,
        "alpha_power_ratio":         alpha_power_ratio,
        "beta_power_ratio":          beta_power_ratio,
        "high_freq_ratio":           high_freq_ratio,
        # ── inter-spike filled in later (placeholder) ────────────────────
        "isi_prev_ms":               np.nan,
        "isi_next_ms":               np.nan,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-channel spike detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_spikes(
    signal_uv: np.ndarray,
    sfreq: float,
    config: SpikeDetectionConfig,
) -> np.ndarray:
    """Return sample indices of detected spike peaks on one channel.

    Threshold: per-channel ``median + threshold_mad_k × MAD``.

    Parameters
    ----------
    signal_uv : 1-D array in µV
    sfreq     : sampling frequency (Hz)
    config    : SpikeDetectionConfig

    Returns
    -------
    np.ndarray of sample indices (dtype int64), possibly empty.
    """
    med = np.median(signal_uv)
    mad = _mad(signal_uv)
    threshold = med + config.threshold_mad_k * mad

    min_dist = max(1, int(config.min_peak_distance_ms * sfreq / 1000))
    max_w    = int(config.peak_width_ms * sfreq / 1000)

    peaks, _ = find_peaks(
        signal_uv,
        height=threshold,
        distance=min_dist,
        width=(1, max_w),
    )

    if peaks.size == 0:
        return peaks

    heights = signal_uv[peaks]
    mask = (heights >= config.min_amplitude_uv) & (heights <= config.max_amplitude_uv)
    return peaks[mask]


# ──────────────────────────────────────────────────────────────────────────────
# Inter-spike interval (ISI) annotation
# ──────────────────────────────────────────────────────────────────────────────

def _annotate_isi(df: pd.DataFrame) -> pd.DataFrame:
    """Fill isi_prev_ms and isi_next_ms per channel group (in place)."""
    if df.empty or "time_sec" not in df.columns:
        return df

    rows = df.copy()
    for ch, grp in rows.groupby("channel"):
        t = grp["time_sec"].values * 1000.0   # → ms
        idx = grp.index
        prev_isi = np.full(len(t), np.nan)
        next_isi = np.full(len(t), np.nan)
        if len(t) > 1:
            prev_isi[1:]  = np.diff(t)
            next_isi[:-1] = np.diff(t)
        rows.loc[idx, "isi_prev_ms"] = prev_isi
        rows.loc[idx, "isi_next_ms"] = next_isi

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Main entry points
# ──────────────────────────────────────────────────────────────────────────────

#: All scalar feature columns produced by this module (excludes metadata cols).
FEATURE_COLS: list[str] = [
    "amplitude_uv", "amplitude_corrected_uv", "peak_to_trough_uv",
    "pre_spike_baseline_uv", "relative_amplitude", "local_snr",
    "half_width_ms", "rise_time_ms", "decay_time_ms", "spike_duration_ms",
    "symmetry_ratio", "slope_rising_uv_ms",
    "sharpness", "curvature_uv_ms2", "area_uv_ms", "post_spike_suppression",
    "dominant_freq_hz", "spectral_centroid_hz", "spectral_entropy",
    "theta_power_ratio", "alpha_power_ratio", "beta_power_ratio",
    "high_freq_ratio",
    "isi_prev_ms", "isi_next_ms",
]

META_COLS: list[str] = ["channel", "epoch", "time_sec", "recording_id"]


def extract_spike_features(
    data: Union[mne.io.BaseRaw, mne.Epochs],
    config: Optional[SpikeDetectionConfig] = None,
    recording_id: str = "",
) -> pd.DataFrame:
    """Detect spikes and extract the full ML feature set from clean EEG data.

    Parameters
    ----------
    data         : MNE Raw or Epochs **after** artifact removal.
    config       : SpikeDetectionConfig (defaults applied if None).
    recording_id : Optional label attached to every row (useful for pooling
                   data from multiple recordings before ML training).

    Returns
    -------
    pd.DataFrame  — one row per spike, columns = META_COLS + FEATURE_COLS.
    All features are float64.  Rows where every feature is NaN are dropped.
    """
    if config is None:
        config = SpikeDetectionConfig()

    rows: list[dict] = []

    if isinstance(data, mne.io.BaseRaw):
        _process_raw(data, config, rows, recording_id)
    elif isinstance(data, mne.Epochs):
        _process_epochs(data, config, rows, recording_id)
    else:
        raise TypeError(
            f"data must be mne.io.BaseRaw or mne.Epochs, got {type(data).__name__}"
        )

    all_cols = META_COLS + FEATURE_COLS
    if not rows:
        return pd.DataFrame(columns=all_cols).astype({c: "float64" for c in FEATURE_COLS})

    df = pd.DataFrame(rows, columns=all_cols)

    # Drop rows where every feature column is NaN
    df = df.dropna(subset=FEATURE_COLS, how="all").reset_index(drop=True)

    # Annotate inter-spike intervals per channel
    df = _annotate_isi(df)

    return df


def _process_raw(
    raw: mne.io.BaseRaw,
    config: SpikeDetectionConfig,
    rows: list[dict],
    recording_id: str,
) -> None:
    raw.load_data()
    picks = (
        mne.pick_types(raw.info, eeg=True, exclude=[])
        if config.eeg_only
        else np.arange(len(raw.ch_names))
    )
    sfreq = float(raw.info["sfreq"])
    data_v = raw.get_data(picks=picks)
    times  = raw.times

    for ch_i, pick in enumerate(picks):
        ch_name   = raw.ch_names[pick]
        signal_uv = data_v[ch_i] * 1e6
        ch_rms    = float(np.sqrt(np.mean(signal_uv ** 2)))
        peak_idxs = detect_spikes(signal_uv, sfreq, config)

        for pidx in peak_idxs:
            feats = measure_spike_morphology(
                signal_uv, int(pidx), sfreq, config, channel_rms=ch_rms
            )
            rows.append({
                "channel":      ch_name,
                "epoch":        0,
                "time_sec":     float(times[pidx]),
                "recording_id": recording_id,
                **feats,
            })


def _process_epochs(
    epochs: mne.Epochs,
    config: SpikeDetectionConfig,
    rows: list[dict],
    recording_id: str,
) -> None:
    picks = (
        mne.pick_types(epochs.info, eeg=True, exclude=[])
        if config.eeg_only
        else np.arange(len(epochs.ch_names))
    )
    sfreq  = float(epochs.info["sfreq"])
    times  = epochs.times
    data_v = epochs.get_data(picks=picks)   # (n_epochs, n_ch, n_times)

    for ep_i in range(data_v.shape[0]):
        for ch_i, pick in enumerate(picks):
            ch_name   = epochs.ch_names[pick]
            signal_uv = data_v[ep_i, ch_i] * 1e6
            ch_rms    = float(np.sqrt(np.mean(signal_uv ** 2)))
            peak_idxs = detect_spikes(signal_uv, sfreq, config)

            for pidx in peak_idxs:
                feats = measure_spike_morphology(
                    signal_uv, int(pidx), sfreq, config, channel_rms=ch_rms
                )
                rows.append({
                    "channel":      ch_name,
                    "epoch":        ep_i,
                    "time_sec":     float(times[pidx]),
                    "recording_id": recording_id,
                    **feats,
                })


# ──────────────────────────────────────────────────────────────────────────────
# Channel-level summary (for per-recording ML feature vectors)
# ──────────────────────────────────────────────────────────────────────────────

# Topographic groupings for spatial context features
_HEMISPHERE = {
    "Fp1": "L", "F7": "L", "F3": "L", "T3": "L", "C3": "L",
    "T5": "L", "P3": "L", "O1": "L",
    "Fp2": "R", "F8": "R", "F4": "R", "T4": "R", "C4": "R",
    "T6": "R", "P4": "R", "O2": "R",
    "Fz": "M", "Cz": "M", "Pz": "M",
}
_REGION = {
    "Fp1": "frontal",  "Fp2": "frontal",
    "F7":  "frontal",  "F3":  "frontal", "Fz": "frontal", "F4": "frontal",  "F8":  "frontal",
    "T3":  "temporal", "T4":  "temporal", "T5": "temporal", "T6": "temporal",
    "C3":  "central",  "Cz":  "central",  "C4": "central",
    "P3":  "parietal", "Pz":  "parietal", "P4": "parietal",
    "O1":  "occipital","O2":  "occipital",
}


def channel_summary(
    spike_df: pd.DataFrame,
    duration_sec: Optional[float] = None,
) -> pd.DataFrame:
    """Per-channel summary statistics ready for ML feature engineering.

    For each scalar feature column the following statistics are computed:
    mean, median, std, p25, p75, skewness, kurtosis (excess).

    Additional columns added per channel:
    * spike_count, spike_rate_per_min
    * first_spike_sec, last_spike_sec, activity_span_sec
    * burst_count  — number of spikes with isi_next_ms < 200 ms
    * hemisphere, brain_region

    Parameters
    ----------
    spike_df     : Output of :func:`extract_spike_features`.
    duration_sec : Full recording duration (s).  Used to compute rate.
                   Inferred from spike times if None.

    Returns
    -------
    pd.DataFrame — one row per channel.
    """
    if spike_df.empty:
        return pd.DataFrame()

    from scipy.stats import skew as sp_skew, kurtosis as sp_kurt

    stat_cols = [c for c in FEATURE_COLS if c in spike_df.columns]
    records = []

    for ch, grp in spike_df.groupby("channel"):
        row: dict = {"channel": ch}

        # ── Spike count and rate ─────────────────────────────────────────
        row["spike_count"] = len(grp)
        t_vals = grp["time_sec"].values
        t_span = float(np.ptp(t_vals)) if len(t_vals) > 1 else 1e-3
        dur = float(duration_sec) if duration_sec is not None else t_span
        row["spike_rate_per_min"] = round(len(grp) / (dur / 60.0), 4)
        row["first_spike_sec"]    = float(t_vals.min())
        row["last_spike_sec"]     = float(t_vals.max())
        row["activity_span_sec"]  = t_span

        # ── Burst count (ISI < 200 ms) ────────────────────────────────
        if "isi_next_ms" in grp.columns:
            row["burst_count"] = int((grp["isi_next_ms"].dropna() < 200).sum())
        else:
            row["burst_count"] = 0

        # ── Per-feature statistics ───────────────────────────────────
        for col in stat_cols:
            vals = grp[col].dropna().values
            if vals.size == 0:
                for stat in ("mean", "median", "std", "p25", "p75", "skew", "kurt"):
                    row[f"{col}_{stat}"] = np.nan
                continue
            row[f"{col}_mean"]   = float(np.mean(vals))
            row[f"{col}_median"] = float(np.median(vals))
            row[f"{col}_std"]    = float(np.std(vals, ddof=min(1, len(vals) - 1)))
            row[f"{col}_p25"]    = float(np.percentile(vals, 25))
            row[f"{col}_p75"]    = float(np.percentile(vals, 75))
            row[f"{col}_skew"]   = float(sp_skew(vals))   if len(vals) >= 3 else np.nan
            row[f"{col}_kurt"]   = float(sp_kurt(vals, fisher=True)) if len(vals) >= 4 else np.nan

        # ── Spatial metadata ─────────────────────────────────────────
        row["hemisphere"]   = _HEMISPHERE.get(ch, "unknown")
        row["brain_region"] = _REGION.get(ch, "unknown")

        records.append(row)

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# ML utility: clean feature matrix for direct scikit-learn use
# ──────────────────────────────────────────────────────────────────────────────

def ml_feature_matrix(
    spike_df: pd.DataFrame,
    impute_strategy: str = "median",
    include_channel_dummies: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Convert a spike DataFrame into a clean numeric feature matrix.

    Parameters
    ----------
    spike_df                : Output of :func:`extract_spike_features`.
    impute_strategy         : "median" (default) or "mean" — strategy for
                              filling NaN values column-wise.
    include_channel_dummies : If True, one-hot encode the ``channel`` column
                              and append as additional features.

    Returns
    -------
    X             : np.ndarray of shape (n_spikes, n_features), float64
    feature_names : list[str] — column names matching X's columns

    Notes
    -----
    * Constant-variance columns (std == 0) are kept — the caller can apply
      ``sklearn.feature_selection.VarianceThreshold`` if needed.
    * No scaling is applied here.  Use ``sklearn.preprocessing.StandardScaler``
      or ``RobustScaler`` downstream.
    """
    feat_cols = [c for c in FEATURE_COLS if c in spike_df.columns]
    X_df = spike_df[feat_cols].copy().astype(np.float64)

    # Column-wise imputation
    for col in X_df.columns:
        if X_df[col].isna().any():
            fill = X_df[col].median() if impute_strategy == "median" else X_df[col].mean()
            X_df[col] = X_df[col].fillna(fill if np.isfinite(fill) else 0.0)

    feature_names = list(X_df.columns)

    if include_channel_dummies and "channel" in spike_df.columns:
        dummies = pd.get_dummies(spike_df["channel"], prefix="ch").astype(np.float64)
        X_df = pd.concat([X_df, dummies], axis=1)
        feature_names = list(X_df.columns)

    return X_df.values, feature_names
