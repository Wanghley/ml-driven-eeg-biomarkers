#!/usr/bin/env python3
"""
experimental_dbs_eval.py
========================
Standalone experimental evaluation suite for multi-stage DBS artifact removal
from EEG recordings (Patient XU dataset — 7 Hz, 60 Hz, 100 Hz conditions).

Three novel hybrid removal stages are benchmarked against existing
ArtifactFilterFactory methods.

Stages
------
  A: Epoch-locked SVD artefact-subspace projection + NLMS adaptive residual filter
  B: Complex-valued FFT Hampel with phase-preserving spectral interpolation
  C: Median pulse-template subtraction (handles temporal jitter)
  Hybrid: A → B → C pipeline

Post-stage ICA
--------------
  auto_reject_dbs_components — spectral + spatial + mutual-information heuristics

Performance targets
-------------------
  DBS attenuation  : > 90 % power reduction at 7 Hz ± harmonics
  Beta preservation: within ± 5 % of pre-DBS baseline (13–30 Hz)

Results saved to: results/experiments_7Hz_DBS/

Usage
-----
  python experimental_dbs_eval.py [--dbs-freq 7] [--mode awake|sleep|both]
"""

from __future__ import annotations

# ── Standard library ────────────────────────────────────────────────────────
import argparse
import json
import pathlib
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Numeric / scientific ────────────────────────────────────────────────────
import numpy as np
from scipy import signal as sp_signal
from scipy.interpolate import CubicSpline
from scipy.stats import kurtosis as spkurtosis

# ── Visualisation ───────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── MNE ─────────────────────────────────────────────────────────────────────
import mne
mne.set_log_level("ERROR")

# ── Project utilities ────────────────────────────────────────────────────────
_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.filters import ArtifactFilterFactory
from pipeline.constants import STANDARD_CH, BANDS, DBS_METHODS

# ── sklearn (mutual information) ─────────────────────────────────────────────
try:
    from sklearn.feature_selection import mutual_info_regression
    _HAVE_SKLEARN = True
except ImportError:
    _HAVE_SKLEARN = False

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR      = _ROOT / "data" / "raw" / "XU"
RESULTS_DIR   = _ROOT / "results" / "experiments_7Hz_DBS"
TARGET_SFREQ  = 256.0          # Hz — common resampling target
AA_LPF_HZ    = 100.0          # strict anti-alias cutoff (well below Nyquist/2)
MAX_DUR_SEC   = 120.0          # analyse first 2 min per recording
ANALYSIS_CH   = "Cz"          # representative channel for time-domain plots
ICA_N_COMP    = 15
ICA_MAX_ITER  = 1000
MAD_FACTOR    = 1.4826         # consistent estimator σ from MAD

# Recording file registry  {dbs_freq: {condition: filename}}
RECORDINGS = {
    7.0: {
        "awake": "XUAWAKE7_deidentified.edf",
        "sleep": "XUSLEEP7_deidentified.edf",
    },
    60.0: {
        "awake": "XUAWAKE60_deidentified.edf",
        "sleep": "XUSLEEP60_deidentified.edf",
    },
    100.0: {
        "awake": "XUAWAKET100_deidentified.edf",
        "sleep": "XUSLEEPT100_deidentified.edf",
    },
}
PRE_FILES = {
    "awake": "XUAWAKEPRE_deidentified.edf",
    "sleep": "XUSLEEP_deidentified.edf",
}

# Palette for consistent plotting across methods
METHOD_PALETTE = {
    "Raw DBS":              ("#888888", "-",  0.8),
    "PRE baseline":         ("#2ca02c", "--", 1.2),
    "FFT Spectral Interp":  ("#1f77b4", "-",  1.4),
    "Comb Notch Q=200":     ("#d62728", "-.", 1.2),
    "Sinusoidal Regression":("#ff7f0e", "-",  1.2),
    "Stage A (SVD+NLMS)":   ("#9467bd", "-",  1.6),
    "Stage B (Cplx Hampel)":("#8c564b", "-",  1.6),
    "Stage C (Template)":   ("#e377c2", "-",  1.6),
    "Hybrid A→B→C":         ("#17becf", "-",  2.2),
    "Hybrid + ICA":         ("#bcbd22", "-",  2.4),
}


# ═════════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ═════════════════════════════════════════════════════════════════════════════

def load_and_preprocess_edf(
    path: pathlib.Path,
    target_sfreq: float = TARGET_SFREQ,
    max_dur_sec: float  = MAX_DUR_SEC,
    aa_lpf_hz: float    = AA_LPF_HZ,
    l_freq: float       = 0.5,
    h_freq: float       = 45.0,
) -> mne.io.RawArray:
    """
    Load an EDF file and apply the standard EEG preprocessing chain.

    Steps
    -----
    1. Load EDF, pick 19 standard 10-20 channels.
    2. Crop to ``max_dur_sec``.
    3. ANTI-ALIASING: apply strict FIR low-pass at ``aa_lpf_hz`` *before*
       any resampling step so that DBS harmonics cannot alias into baseband.
    4. Resample to ``target_sfreq`` if needed.
    5. High-pass filter (default 0.5 Hz) to remove DC drift.
    6. Average reference projection.

    Returns
    -------
    mne.io.RawArray  (preloaded, 19 ch, target_sfreq Hz)
    """
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)

    # ── Channel standardisation ───────────────────────────────────────────
    upper_map = {s.upper(): s for s in STANDARD_CH}
    rename = {ch: upper_map[ch.upper()]
              for ch in raw.ch_names if ch.upper() in upper_map}
    raw.rename_channels(rename)
    available = [ch for ch in STANDARD_CH if ch in raw.ch_names]
    raw.pick(available)
    raw.set_channel_types({ch: "eeg" for ch in available})
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="ignore", verbose=False)

    # ── Crop ─────────────────────────────────────────────────────────────
    raw.crop(tmin=0.0, tmax=min(max_dur_sec, raw.times[-1]))

    # ── Anti-aliasing LPF (MUST precede any downsampling) ─────────────────
    # Safe cutoff: min(aa_lpf_hz, target_sfreq/2 - 1)
    aa_safe = min(aa_lpf_hz, target_sfreq / 2.0 - 1.0, raw.info["sfreq"] / 2.0 - 1.0)
    raw.filter(l_freq=None, h_freq=aa_safe, method="fir",
               fir_design="firwin", phase="zero", verbose=False)

    # ── Resample ─────────────────────────────────────────────────────────
    if not np.isclose(raw.info["sfreq"], target_sfreq, rtol=0.01):
        raw.resample(target_sfreq, verbose=False)

    # ── Band-pass to remove slow drift and high-frequency noise ───────────
    raw.filter(l_freq=l_freq, h_freq=h_freq, method="fir",
               fir_design="firwin", phase="zero", verbose=False)

    # ── Line Noise Removal ────────────────────────────────────────────────
    # Metadata for XU defaults to 60 Hz line noise.
    raw.notch_filter(freqs=60.0, method="fir", phase="zero", verbose=False)

    # ── Bad Channel Detection & Interpolation ─────────────────────────────
    # Heuristic: flat channels or extreme variance (e.g. saturated)
    data = raw.get_data()
    vars_ch = np.var(data, axis=1)
    med_var = np.median(vars_ch)
    mad_var = np.median(np.abs(vars_ch - med_var)) * MAD_FACTOR + 1e-15

    bad_chs = []
    for i, ch in enumerate(raw.ch_names):
        if vars_ch[i] < 1e-16 or vars_ch[i] > med_var + 4 * mad_var:
            bad_chs.append(ch)

    if bad_chs:
        raw.info["bads"] = bad_chs
        try:
            # Requires spherical spline for MNE eeg interpolation
            raw.interpolate_bads(reset_bads=True, method="spline", verbose=False)
        except Exception:
            pass # Skip if montage/spherical geometry issues

    # ── Average reference ─────────────────────────────────────────────────
    raw.set_eeg_reference("average", projection=True, verbose=False)
    raw.apply_proj()

    return raw


def raw_to_uv(raw: mne.io.Raw) -> np.ndarray:
    """Return data array in µV, shape (n_ch, n_samples)."""
    return raw.get_data() * 1e6


def uv_to_raw(data_uv: np.ndarray, template_raw: mne.io.Raw) -> mne.io.RawArray:
    """Wrap a µV data array back into an MNE RawArray cloned from template_raw."""
    info = template_raw.info.copy()
    raw_new = mne.io.RawArray(data_uv * 1e-6, info, verbose=False)
    return raw_new


def compute_psd(data_uv: np.ndarray, sfreq: float,
                fmin: float = 0.5, fmax: float = 120.0,
                n_fft: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Mean-channel Welch PSD in dB µV²/Hz."""
    f, p = sp_signal.welch(data_uv.mean(axis=0), fs=sfreq,
                            nperseg=n_fft, noverlap=n_fft // 2)
    mask = (f >= fmin) & (f <= fmax)
    return f[mask], 10 * np.log10(p[mask] + 1e-30)


def band_power_uv2(data_uv: np.ndarray, sfreq: float,
                   lo: float, hi: float, n_fft: int = 2048) -> float:
    """Mean absolute band power (µV²) across channels using Welch."""
    f, p = sp_signal.welch(data_uv, fs=sfreq, nperseg=n_fft,
                            noverlap=n_fft // 2, axis=1)
    mask = (f >= lo) & (f <= hi)
    return float(np.trapz(p[:, mask].mean(axis=0), f[mask]))


def harmonic_power_uv2(data_uv: np.ndarray, sfreq: float,
                        f_dbs: float, n_harm: int = 6,
                        bw: float = 0.5, n_fft: int = 2048) -> float:
    """Sum of power within ±bw Hz of the first n_harm DBS harmonics."""
    f, p = sp_signal.welch(data_uv, fs=sfreq, nperseg=n_fft,
                            noverlap=n_fft // 2, axis=1)
    p_mean = p.mean(axis=0)
    total = 0.0
    for k in range(1, n_harm + 1):
        h = k * f_dbs
        if h + bw > sfreq / 2:
            break
        mask = (f >= h - bw) & (f <= h + bw)
        if np.any(mask):
            total += float(np.trapz(p_mean[mask], f[mask]))
    return total


# ═════════════════════════════════════════════════════════════════════════════
# STAGE A — SVD epoch-subspace projection + NLMS adaptive residual filter
# ═════════════════════════════════════════════════════════════════════════════

def find_dbs_pulses(data_uv: np.ndarray, sfreq: float,
                    f_dbs: float) -> np.ndarray:
    """
    Detect DBS stimulation pulse times using the mean-channel amplitude
    envelope of a band-pass filtered signal.

    Strategy
    --------
    1. Bandpass around the fundamental (f_dbs ± 1 Hz) to extract dominant
       DBS oscillation.
    2. Detect positive peaks at the expected inter-pulse interval.
    3. Fall back to a regular grid if peaks are not reliably detected.

    Returns
    -------
    peaks : (n_pulses,) array of sample indices
    """
    mean_ch = data_uv.mean(axis=0)
    # Bandpass around DBS fundamental
    sos_bp = sp_signal.butter(
        4, [max(0.5, f_dbs - 1.0), min(sfreq / 2 - 1, f_dbs + 1.0)],
        btype="band", fs=sfreq, output="sos"
    )
    bp = sp_signal.sosfiltfilt(sos_bp, mean_ch)

    expected_period_samp = sfreq / f_dbs
    min_distance = int(0.70 * expected_period_samp)

    peaks, _ = sp_signal.find_peaks(
        np.abs(bp),
        distance=min_distance,
        height=np.percentile(np.abs(bp), 70),
    )

    # Require at least 20 detected pulses; otherwise use regular grid
    n_expected = int(len(mean_ch) / expected_period_samp)
    if len(peaks) < max(20, n_expected // 4):
        peaks = np.round(
            np.arange(expected_period_samp / 2,
                      len(mean_ch) - expected_period_samp / 2,
                      expected_period_samp)
        ).astype(int)

    return peaks


def svd_project_dbs(data_uv: np.ndarray, sfreq: float,
                     f_dbs: float, n_components: int = 3) -> np.ndarray:
    """
    Epoch-locked SVD artefact-subspace projection.

    Mathematical derivation
    -----------------------
    Let the data matrix be X ∈ ℝ^{C×N} (C channels, N samples).
    We epoch X at the DBS period T_s = ⌊sfreq/f_dbs⌋ samples:

        E ∈ ℝ^{K × (C·T_s)}   (K complete epochs stacked)

    SVD of the mean-centred epoch matrix:
        E_c = U Σ V^T

    The first k left singular vectors (columns of U) span the DBS artefact
    subspace in epoch space.  The corresponding right singular vectors (rows
    of V^T) express the DBS template in channel-time space.

    Reconstruction of the DBS artefact contribution:
        A_k = U[:, :k] (U[:, :k]^T E_c)    (projection onto artefact subspace)

    Cleaned epoch matrix:
        E_clean = E_c − A_k

    The cleaned epochs are then reassembled back into a continuous signal.

    Parameters
    ----------
    n_components : number of SVD components identified as DBS artefact
                   (typically 1–3; first component dominates).
    """
    n_ch, n_samp = data_uv.shape
    T = int(round(sfreq / f_dbs))            # samples per DBS period
    n_epochs = n_samp // T
    if n_epochs < 10:
        return data_uv.copy()

    # Build epoch tensor  (n_epochs, n_ch, T)
    usable = n_epochs * T
    ep = data_uv[:, :usable].reshape(n_ch, n_epochs, T)
    ep = ep.transpose(1, 0, 2)               # → (n_epochs, n_ch, T)
    E  = ep.reshape(n_epochs, n_ch * T)      # flatten channel × time

    # Mean-centre across epochs (removes stationary brain signal)
    E_mean = E.mean(axis=0)
    E_c    = E - E_mean

    # Economy SVD
    U, S, Vt = np.linalg.svd(E_c, full_matrices=False)

    # Project out top k components (DBS artefact subspace)
    U_k  = U[:, :n_components]              # (n_epochs, k)
    proj = U_k @ (U_k.T @ E_c)             # (n_epochs, n_ch*T)
    E_clean = E_c - proj + E_mean          # restore mean (brain baseline)

    # Reassemble continuous signal
    ep_clean = E_clean.reshape(n_epochs, n_ch, T).transpose(1, 0, 2)
    data_clean = data_uv.copy()
    data_clean[:, :usable] = ep_clean.reshape(n_ch, usable)
    return data_clean


def nlms_filter_channel(x: np.ndarray, ref: np.ndarray,
                         N: int = 64, mu: float = 0.008,
                         eps: float = 1e-8,
                         n_conv_samp: int = 2048) -> np.ndarray:
    """
    Normalized Least Mean Square (NLMS) adaptive filter — single channel.

    Reference model
    ---------------
    The adaptive filter h(n) ∈ ℝ^N models the DBS artefact component
    linearly predictable from the reference signal r(n):

        ŷ(n) = h(n)^T · r_vec(n)          where r_vec = [r(n), …, r(n-N+1)]
        e(n)  = x(n) − ŷ(n)              (cleaned estimate)

    NLMS weight update rule:
        h(n+1) = h(n) + μ / (||r_vec||² + ε) · e(n) · r_vec(n)

    Normalisation by ||r_vec||² ensures convergence for any bounded input
    power (step size is effectively self-tuning).

    Implementation
    --------------
    Phase 1 (adaptation): run n_conv_samp steps to learn h.
    Phase 2 (application): apply converged FIR h to full reference signal
                           and subtract from input (O(N·n) via scipy.lfilter).
    """
    n = len(x)
    n_conv = min(n_conv_samp, n // 4)

    h = np.zeros(N)
    ref_buf = np.zeros(N)

    # ── Phase 1: weight convergence ───────────────────────────────────────
    for i in range(n_conv):
        # Update circular buffer (newest sample at index 0)
        ref_buf = np.roll(ref_buf, 1)
        ref_buf[0] = ref[i]

        y_i = np.dot(h, ref_buf)
        e_i = x[i] - y_i
        norm = np.dot(ref_buf, ref_buf) + eps
        h += (mu / norm) * e_i * ref_buf

    # ── Phase 2: apply converged FIR (vectorised) ─────────────────────────
    # h models  DBS → x  transfer; artifact estimate = conv(ref, h)
    artifact = sp_signal.lfilter(h, [1.0], ref)
    return x - artifact


def make_dbs_reference(n_samples: int, sfreq: float,
                        f_dbs: float, n_harmonics: int = 12) -> np.ndarray:
    """Normalised multi-harmonic DBS reference signal (sine sum with 1/k taper)."""
    t   = np.arange(n_samples) / sfreq
    nyq = sfreq / 2.0
    ref = np.zeros(n_samples)
    for k in range(1, n_harmonics + 1):
        if k * f_dbs >= nyq:
            break
        ref += np.sin(2.0 * np.pi * k * f_dbs * t) / k
    peak = np.abs(ref).max()
    return ref / (peak + 1e-10)


def apply_svd_nlms(data_uv: np.ndarray, sfreq: float,
                    f_dbs: float,
                    n_svd_components: int = 3,
                    nlms_order: int = 64,
                    nlms_mu: float = 0.008) -> np.ndarray:
    """
    Full Stage-A pipeline: SVD subspace projection followed by
    per-channel NLMS adaptive residual cancellation.
    """
    # SVD epoch decomposition
    data_svd = svd_project_dbs(data_uv, sfreq, f_dbs, n_svd_components)

    # NLMS residual cancellation
    ref = make_dbs_reference(data_uv.shape[1], sfreq, f_dbs)
    data_clean = np.zeros_like(data_svd)
    for ch in range(data_svd.shape[0]):
        data_clean[ch] = nlms_filter_channel(
            data_svd[ch], ref, N=nlms_order, mu=nlms_mu
        )
    return data_clean


# ═════════════════════════════════════════════════════════════════════════════
# STAGE B — Complex-valued FFT Hampel with phase-preserving interpolation
# ═════════════════════════════════════════════════════════════════════════════

def apply_complex_hampel_fft(data_uv: np.ndarray, sfreq: float,
                               f_dbs: float,
                               n_sigmas: float = 3.0,
                               bw_hz: float = 0.5,
                               flank_hz: float = 3.0) -> np.ndarray:
    """
    Phase-preserving frequency-domain Hampel filter.

    Improvement over existing hampel_freq
    --------------------------------------
    The existing implementation operates on *magnitudes* then reconstructs
    with the original phase — this introduces phase discontinuities at
    harmonic frequencies that corrupt Phase-Amplitude Coupling (PAC) measures.

    This implementation interpolates the **complex** spectrum directly
    using **linear** interpolation (not cubic spline):
        Xf_real[outlier_bins] ← np.interp(target, ref_bins, Xf_real[ref_bins])
        Xf_imag[outlier_bins] ← np.interp(target, ref_bins, Xf_imag[ref_bins])

    Linear interpolation is preferred over cubic splines for spectral data
    because the complex spectrum is **not** smooth at harmonic edges — cubic
    splines exhibit Runge's phenomenon and can severely overshoot, amplifying
    the very harmonics we want to remove.  Linear interpolation is
    monotone-bounded between reference points, guaranteeing no amplification.

    Hampel criterion (applied per harmonic, per channel)
    -------------------------------------------------------
    In a local neighbourhood of M bins around harmonic h:
        M_local = median(|Xf|)
        σ_local = 1.4826 × MAD(|Xf|)
        outlier if |Xf[h_bin]| > M_local + n_sigmas × σ_local

    Parameters
    ----------
    bw_hz   : total width of the interpolated zone around each harmonic (Hz)
              (kept narrow — 0.5 Hz default — to minimise brain-signal loss)
    flank_hz: width of each reference flank on each side of the zone (Hz)
              (uses bins clearly outside the harmonic zone; avoids adjacent
              harmonics by capping at half the DBS inter-harmonic spacing)
    """
    n_ch, n_samp = data_uv.shape
    Xf    = np.fft.rfft(data_uv, axis=1)          # (n_ch, n_freq) complex
    freqs = np.fft.rfftfreq(n_samp, 1.0 / sfreq)
    freq_res   = freqs[1] - freqs[0]
    n_freq     = len(freqs)

    half_bw    = max(1, int(round(bw_hz  / 2.0 / freq_res)))
    # Cap flank to half the inter-harmonic spacing to avoid neighbouring
    # harmonics contaminating the reference bins
    max_flank  = max(5, int(round(min(flank_hz, f_dbs * 0.4) / freq_res)))
    local_win  = max(20, int(round(2.5 / freq_res)))  # Hampel local window

    Xf_clean = Xf.copy()
    nyq = sfreq / 2.0

    for k in range(1, 60):
        h = k * f_dbs
        if h >= nyq:
            break
        h_bin = int(round(h / freq_res))
        if h_bin >= n_freq - max_flank - 1:
            break

        zone_s = max(1, h_bin - half_bw)
        zone_e = min(n_freq - 2, h_bin + half_bw + 1)
        left_s = max(1, zone_s - max_flank)
        right_e = min(n_freq - 1, zone_e + max_flank)

        ref_bins    = np.concatenate([
            np.arange(left_s, zone_s),
            np.arange(zone_e, right_e),
        ])
        target_bins = np.arange(zone_s, zone_e)

        if len(ref_bins) < 4 or len(target_bins) == 0:
            continue

        # Hampel outlier check (using magnitude)
        local_s = max(1, h_bin - local_win)
        local_e = min(n_freq - 1, h_bin + local_win + 1)

        for ch in range(n_ch):
            mag = np.abs(Xf_clean[ch])
            local_mag = mag[local_s:local_e]
            med   = np.median(local_mag)
            mad   = np.median(np.abs(local_mag - med))
            sigma = MAD_FACTOR * mad
            threshold = med + n_sigmas * sigma

            if mag[h_bin] > threshold:
                # Linear interpolation of complex spectrum (real + imag separately)
                # Linear interp is monotone-bounded → no Runge overshoot
                rb = ref_bins.astype(float)
                tb = target_bins.astype(float)
                interp_r = np.interp(tb, rb, Xf_clean[ch, ref_bins].real)
                interp_i = np.interp(tb, rb, Xf_clean[ch, ref_bins].imag)
                Xf_clean[ch, target_bins] = interp_r + 1j * interp_i

    return np.fft.irfft(Xf_clean, n=n_samp, axis=1)


# ═════════════════════════════════════════════════════════════════════════════
# STAGE C — Median pulse-template subtraction
# ═════════════════════════════════════════════════════════════════════════════

def apply_template_subtraction(data_uv: np.ndarray, sfreq: float,
                                 f_dbs: float,
                                 n_phase_bins: int = 128) -> np.ndarray:
    """
    Phase-sorted median template subtraction for continuous periodic DBS.

    Rationale
    ---------
    7 Hz DBS produces a *continuous* periodic waveform.  A naive period-folded
    approach (reshape into integer-length blocks) accumulates a phase drift of
    (T_integer − T_exact) = (37 − 36.571) = 0.429 samples per period, which
    over 830 periods equals ≈ 356 samples ≈ 10 full DBS periods.  The
    resulting blurred template explains why subtraction removes < 1% of power.

    The phase-sorted approach eliminates this drift entirely by using the
    **exact instantaneous phase** φ(t) = 2π f_dbs t  (mod 2π) at each sample.
    Samples with the same DBS phase bin are pooled and their median computed —
    yielding an accurate template irrespective of DBS frequency or recording
    length.

    Mathematical derivation
    -----------------------
    Let B = n_phase_bins (default 128), phase bin width δ = 2π / B.

    Instantaneous phase at sample n:
        φ[n] = (2π f_dbs n / sfreq) mod (2π)

    Bin index:
        b[n] = floor(φ[n] / δ)   ∈ {0, 1, …, B-1}

    DBS template for channel c:
        τ_c[b] = median { x_c[n] : b[n] = b }   b = 0 … B-1

    With N = 30720 samples and B = 128, each bin receives N/B ≈ 240 samples
    on average — sufficient for a robust median estimate.

    Reconstruction of artifact at each sample:
        artifact[n] = τ_c[b[n]]   (lookup + optional cubic-spline upsampling)

    Cleaned signal:
        x_clean[n] = x_c[n] − artifact[n]

    Parameters
    ----------
    n_phase_bins : number of phase bins for the template (higher = finer
                   time resolution of DBS waveform; 128 gives < 3° per bin).
    """
    n_ch, n_samp = data_uv.shape

    # ── Exact instantaneous phase (no integer rounding) ───────────────────
    t     = np.arange(n_samp, dtype=np.float64) / sfreq
    phase = (2.0 * np.pi * f_dbs * t) % (2.0 * np.pi)   # 0 … 2π

    # ── Bin each sample into a phase bin ─────────────────────────────────
    bin_idx = (phase / (2.0 * np.pi) * n_phase_bins).astype(int) % n_phase_bins

    # ── Build phase-sorted template and subtract ──────────────────────────
    data_clean = data_uv.copy()
    for ch in range(n_ch):
        template = np.zeros(n_phase_bins)
        for b in range(n_phase_bins):
            mask = bin_idx == b
            if mask.sum() > 0:
                template[b] = np.median(data_uv[ch, mask])

        # Reconstruct artifact time series by lookup
        artifact = template[bin_idx]
        data_clean[ch] -= artifact

    return data_clean


# ═════════════════════════════════════════════════════════════════════════════
# HYBRID PIPELINE  A → B → C
# ═════════════════════════════════════════════════════════════════════════════

def apply_hybrid_abc(data_uv: np.ndarray, sfreq: float,
                      f_dbs: float) -> np.ndarray:
    """Sequential A → B → C pipeline."""
    out = apply_svd_nlms(data_uv, sfreq, f_dbs)
    out = apply_complex_hampel_fft(out, sfreq, f_dbs)
    out = apply_template_subtraction(out, sfreq, f_dbs)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# AUTO ICA — Spectral + Spatial + Mutual Information heuristics
# ═════════════════════════════════════════════════════════════════════════════

def auto_reject_dbs_components(
    ica: mne.preprocessing.ICA,
    raw: mne.io.Raw,
    f_dbs: float = 7.0,
    dbs_n_sigmas: float = 3.0,
    mi_n_sigmas: float  = 2.0,
) -> dict[str, list[int]]:
    """
    Automated ICA component rejection for DBS + ocular + muscle artefacts.

    Three independent heuristics are applied:

    1. Spectral (DBS)
       For each IC time series, compute Welch PSD.  Flag the component if
       power within ±0.5 Hz of the DBS fundamental exceeds the broad-band
       (1–50 Hz) median by more than `dbs_n_sigmas` × σ_broad.

    2. Spatial (ocular / muscle)
       · Eye: maximal mixing-matrix weight on Fp1/Fp2 AND low-frequency
         ratio (1–15 Hz / 1–80 Hz) > 0.55.
       · Muscle: high-frequency ratio (30–80 Hz) > 0.55 AND topomap
         kurtosis > 3 (localised peripheral activation).

    3. Mutual Information (MI)
       MI between each IC and a 7 Hz sine reference is estimated via
       sklearn's mutual_info_regression (k-NN entropy estimator).  ICs
       with MI > mean + mi_n_sigmas × std (across unflagged ICs) are
       rejected as DBS-correlated.  This catches non-sinusoidal residuals
       that the spectral heuristic misses.

    Returns
    -------
    dict with keys 'dbs', 'eye', 'muscle', 'mi' → list of IC indices
    """
    sfreq    = raw.info["sfreq"]
    sources  = ica.get_sources(raw).get_data()   # (n_comp, n_samples)
    mixing   = ica.get_components()              # (n_ch, n_comp)
    n_comp   = sources.shape[0]

    ch_lower    = [c.lower() for c in raw.ch_names]
    fp_idx      = [i for i, c in enumerate(ch_lower) if c in ("fp1", "fp2")]
    frontal_idx = [i for i, c in enumerate(ch_lower)
                   if c in ("fp1", "fp2", "f7", "f8", "f3", "f4", "fz")]

    dbs_comps, eye_comps, muscle_comps = [], [], []

    for ic in range(n_comp):
        f_w, psd = sp_signal.welch(sources[ic], fs=sfreq, nperseg=512)

        # ── Spectral DBS heuristic ────────────────────────────────────────
        dbs_mask    = (f_w >= f_dbs - 0.5) & (f_w <= f_dbs + 0.5)
        broad_mask  = (f_w >= 1.0)         & (f_w <= 50.0)
        if np.any(dbs_mask) and np.any(broad_mask):
            p_dbs  = psd[dbs_mask].mean()
            p_b    = psd[broad_mask]
            med_b  = np.median(p_b)
            std_b  = MAD_FACTOR * np.median(np.abs(p_b - med_b))
            if p_dbs > med_b + dbs_n_sigmas * std_b:
                dbs_comps.append(ic)
                continue

        # ── Spatial / spectral ocular + muscle heuristic ──────────────────
        col    = np.abs(mixing[:, ic])
        top_ch = int(np.argmax(col))

        p_all = np.trapz(psd[(f_w >= 1) & (f_w <= 80)],
                          f_w[(f_w >= 1) & (f_w <= 80)]) + 1e-30
        lf_r  = (np.trapz(psd[(f_w >= 1)  & (f_w <= 15)],
                            f_w[(f_w >= 1)  & (f_w <= 15)]) / p_all)
        hf_r  = (np.trapz(psd[(f_w >= 30) & (f_w <= 80)],
                            f_w[(f_w >= 30) & (f_w <= 80)]) / p_all)
        topo_kurt = float(spkurtosis(col))

        if top_ch in fp_idx and lf_r > 0.55:
            eye_comps.append(ic)
        elif top_ch in frontal_idx and lf_r > 0.65 and topo_kurt > 2.0:
            eye_comps.append(ic)
        elif hf_r > 0.55 and topo_kurt > 3.0:
            muscle_comps.append(ic)

    # ── Mutual information (unflagged ICs only) ───────────────────────────
    mi_comps = []
    if _HAVE_SKLEARN:
        already = set(dbs_comps + eye_comps + muscle_comps)
        t  = np.arange(sources.shape[1]) / sfreq
        ref_signal = np.sin(2.0 * np.pi * f_dbs * t)

        mi_scores = np.zeros(n_comp)
        unflagged = [i for i in range(n_comp) if i not in already]
        for ic in unflagged:
            mi_scores[ic] = mutual_info_regression(
                sources[ic].reshape(-1, 1),
                ref_signal,
                n_neighbors=5,
                random_state=42,
            )[0]

        if len(unflagged) > 3:
            mi_vals  = mi_scores[unflagged]
            mi_thresh = np.mean(mi_vals) + mi_n_sigmas * np.std(mi_vals)
            mi_comps  = [i for i in unflagged if mi_scores[i] > mi_thresh]

    return {"dbs": dbs_comps, "eye": eye_comps,
            "muscle": muscle_comps, "mi": mi_comps}


def apply_ica_pipeline(data_uv: np.ndarray,
                        template_raw: mne.io.Raw,
                        f_dbs: float,
                        n_components: int = ICA_N_COMP,
                        random_state: int = 42) -> tuple[np.ndarray, dict]:
    """
    Fit FastICA, auto-reject components, return cleaned data and report dict.
    """
    raw_in = uv_to_raw(data_uv, template_raw)
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method="fastica",
        max_iter=ICA_MAX_ITER,
        random_state=random_state,
    )
    ica.fit(raw_in, verbose=False)

    rejected = auto_reject_dbs_components(ica, raw_in, f_dbs=f_dbs)
    all_bad  = sorted(set(sum(rejected.values(), [])))

    ica.exclude = all_bad
    raw_clean = raw_in.copy()
    ica.apply(raw_clean, verbose=False)

    report = {k: v for k, v in rejected.items()}
    report["all_excluded"] = all_bad
    return raw_clean.get_data() * 1e6, report


# ═════════════════════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    cleaned_uv:  np.ndarray,
    dbs_uv:      np.ndarray,
    baseline_uv: np.ndarray,
    sfreq: float,
    f_dbs: float,
    label: str = "",
) -> dict:
    """
    Compute the full set of performance metrics for one method.

    Metrics
    -------
    * Beta preservation %  : band power 13–30 Hz relative to pre-DBS baseline
    * Alpha preservation % : band power 8–13 Hz
    * Theta preservation % : band power 4–8 Hz
    * DBS attenuation dB   : power reduction at first 6 harmonics (< 0 = good)
    * DBS reduction %      : 1 − (post/pre) harmonic power × 100
    * Goals met            : attenuation > 90 % AND |beta − 100 %| < 5 %
    """
    def bp(arr, lo, hi):
        return band_power_uv2(arr, sfreq, lo, hi)

    beta_clean = bp(cleaned_uv, 13, 30)
    beta_base  = bp(baseline_uv, 13, 30)
    alpha_clean = bp(cleaned_uv, 8, 13)
    alpha_base  = bp(baseline_uv, 8, 13)
    theta_clean = bp(cleaned_uv, 4, 8)
    theta_base  = bp(baseline_uv, 4, 8)

    h_before = harmonic_power_uv2(dbs_uv,     sfreq, f_dbs)
    h_after  = harmonic_power_uv2(cleaned_uv, sfreq, f_dbs)

    atten_db  = float(10 * np.log10((h_after + 1e-30) / (h_before + 1e-30)))
    reduc_pct = float(100 * (1 - h_after / (h_before + 1e-30)))

    beta_pct  = float(100 * beta_clean  / (beta_base  + 1e-30))
    alpha_pct = float(100 * alpha_clean / (alpha_base + 1e-30))
    theta_pct = float(100 * theta_clean / (theta_base + 1e-30))

    return {
        "method":              label,
        "beta_preservation_%": round(beta_pct,  2),
        "alpha_preservation_%":round(alpha_pct, 2),
        "theta_preservation_%":round(theta_pct, 2),
        "dbs_attenuation_dB":  round(atten_db,  2),
        "dbs_reduction_%":     round(reduc_pct, 2),
        "beta_goal_met":       bool(abs(beta_pct - 100) < 5.0),
        "attenuation_goal_met":bool(reduc_pct > 90.0),
        "both_goals_met":      bool(abs(beta_pct - 100) < 5.0 and reduc_pct > 90.0),
    }


# ═════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═════════════════════════════════════════════════════════════════════════════

def plot_psd_comparison(
    results: dict[str, np.ndarray],
    sfreq: float,
    f_dbs: float,
    title: str,
    out_path: pathlib.Path,
):
    """
    Save a 2-panel PSD comparison figure:
    left = full 0.5–100 Hz, right = zoom 0.5–30 Hz (clinical DBS band).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (fmin, fmax), zoom_label in zip(
        axes,
        [(0.5, 100.0), (0.5, 32.0)],
        ["Full 0.5–100 Hz", "Clinical zoom 0.5–32 Hz"],
    ):
        for label, data_uv in results.items():
            col, ls, lw = METHOD_PALETTE.get(label, ("#333333", "-", 1.0))
            f, p = compute_psd(data_uv, sfreq, fmin=fmin, fmax=fmax)
            ax.plot(f, p, color=col, ls=ls, lw=lw, label=label, alpha=0.9)

        # DBS harmonic markers
        for k in range(1, 15):
            h = k * f_dbs
            if h > fmax:
                break
            ax.axvline(h, color="orange", lw=0.5, ls=":", alpha=0.6,
                       label="DBS harmonic" if k == 1 else "")

        # Brain band spans
        for band_name, lo, hi in BANDS:
            ax.axvspan(lo, hi, color="gray", alpha=0.04)

        ax.set_xlabel("Frequency (Hz)", fontsize=10)
        ax.set_ylabel("PSD (dB µV²/Hz)", fontsize=10)
        ax.set_title(zoom_label, fontsize=11)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_time_domain(
    results: dict[str, np.ndarray],
    sfreq: float,
    ch_names: list[str],
    ch_label: str,
    t_offset: float,
    dur: float,
    f_dbs: float,
    title: str,
    out_path: pathlib.Path,
):
    """
    Save a 2-second time-domain comparison for a single representative channel.

    Requirements met
    ----------------
    The output MUST NOT show the repetitive "comb" or "ringing" artefacts
    associated with narrow notch filters.  Ringing produces periodic dips/spikes
    at exactly 1/f_dbs s intervals; template subtraction and SVD-based methods
    avoid this because they operate on epoch structure, not IIR pole placement.
    """
    try:
        ch_idx = ch_names.index(ch_label)
    except ValueError:
        ch_idx = 0
        ch_label = ch_names[0]

    i0 = int(t_offset * sfreq)
    i1 = i0 + int(dur * sfreq)

    n_methods = len(results)
    fig, axes = plt.subplots(n_methods, 1,
                              figsize=(14, 2.0 * n_methods),
                              sharex=True)
    if n_methods == 1:
        axes = [axes]

    t = np.arange(i1 - i0) / sfreq + t_offset

    for ax, (label, data_uv) in zip(axes, results.items()):
        col, ls, lw = METHOD_PALETTE.get(label, ("#333333", "-", 1.0))
        sig = data_uv[ch_idx, i0:i1]
        ax.plot(t, sig, color=col, lw=lw, ls=ls)
        ax.set_ylabel(label, fontsize=7, rotation=0, labelpad=80, va="center")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_yticks([])

        # DBS pulse markers
        for k in range(int(t_offset * f_dbs), int((t_offset + dur) * f_dbs) + 2):
            pulse_t = k / f_dbs
            if t_offset <= pulse_t <= t_offset + dur:
                ax.axvline(pulse_t, color="orange", lw=0.4, alpha=0.4)

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle(f"{title}\nChannel {ch_label}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_beta_summary(all_metrics: list[dict], out_path: pathlib.Path):
    """
    Bar chart of Beta preservation % and DBS reduction % across all
    methods and recording conditions.
    """
    from itertools import groupby

    conditions = sorted(set(m["condition"] for m in all_metrics))
    methods    = list(dict.fromkeys(m["method"] for m in all_metrics))

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for ax, metric_key, ylabel, target, good_range in [
        (axes[0], "beta_preservation_%",  "Beta preservation (%)",   100, (95, 105)),
        (axes[1], "dbs_reduction_%",      "DBS harmonic reduction (%)", 90, (90, 100.5)),
    ]:
        x = np.arange(len(conditions))
        width = 0.8 / max(len(methods), 1)

        for mi, method in enumerate(methods):
            vals = []
            for cond in conditions:
                row = next((m for m in all_metrics
                            if m["method"] == method and m["condition"] == cond), None)
                vals.append(row[metric_key] if row else np.nan)

            col = METHOD_PALETTE.get(method, ("#888888", "-", 1.0))[0]
            bars = ax.bar(x + mi * width, vals, width=width * 0.9,
                          color=col, alpha=0.85, label=method)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            min(v, good_range[1] - 1) + 0.5,
                            f"{v:.0f}", ha="center", va="bottom",
                            fontsize=5.5, fontweight="bold")

        ax.axhline(target, color="k", lw=1.2, ls="--", label=f"Target={target}%")
        ax.axhspan(*good_range, color="green", alpha=0.08, label="±5 % zone")
        ax.set_xticks(x + width * len(methods) / 2)
        ax.set_xticklabels(conditions, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11, fontweight="bold")
        ax.legend(fontsize=6.5, loc="lower right", framealpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "DBS Removal Performance Summary\n"
        "Beta preservation: 100 % = identical to pre-DBS baseline  "
        "| DBS reduction target > 90 %",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_single_experiment(
    dbs_path:      pathlib.Path,
    pre_path:      pathlib.Path,
    f_dbs:         float,
    condition_tag: str,
    results_dir:   pathlib.Path,
    run_ica:       bool = True,
) -> list[dict]:
    """
    Run the full experiment for one (DBS recording, PRE baseline) pair.

    Returns list of metric dicts (one per method).
    """
    t0 = time.perf_counter()
    print(f"\n{'='*70}")
    print(f"  {condition_tag}  |  DBS = {f_dbs} Hz")
    print(f"  DBS file : {dbs_path.name}")
    print(f"  PRE file : {pre_path.name}")
    print(f"{'='*70}")

    # ── Load data ────────────────────────────────────────────────────────
    print("  Loading DBS recording …")
    raw_dbs = load_and_preprocess_edf(dbs_path, target_sfreq=TARGET_SFREQ)
    print("  Loading PRE baseline …")
    raw_pre = load_and_preprocess_edf(pre_path,  target_sfreq=TARGET_SFREQ)

    sfreq    = raw_dbs.info["sfreq"]
    ch_names = raw_dbs.ch_names
    dbs_uv   = raw_to_uv(raw_dbs)
    pre_uv   = raw_to_uv(raw_pre)

    # Crop PRE to same length if needed
    min_samp = min(dbs_uv.shape[1], pre_uv.shape[1])
    dbs_uv   = dbs_uv[:, :min_samp]
    pre_uv   = pre_uv[:min_samp].T[:dbs_uv.shape[0]].T if pre_uv.shape[0] != dbs_uv.shape[0] else pre_uv[:, :min_samp]

    # Align channels (use intersection)
    common = [c for c in ch_names if c in raw_pre.ch_names]
    dbs_idx = [ch_names.index(c)            for c in common]
    pre_idx = [raw_pre.ch_names.index(c)    for c in common]
    dbs_uv  = dbs_uv[dbs_idx, :min_samp]
    pre_uv  = raw_to_uv(raw_pre)[pre_idx, :min_samp]
    ch_names = common

    print(f"  Channels: {len(ch_names)}  |  Duration: {dbs_uv.shape[1]/sfreq:.1f} s  |  Fs: {sfreq:.0f} Hz")

    # ── Existing baseline methods (from ArtifactFilterFactory) ───────────
    print("  Running baseline methods …")
    fft_uv = ArtifactFilterFactory.process(
        "fft_spectral_interp", dbs_uv.copy(), sfreq, f_target=f_dbs
    )
    notch_uv = ArtifactFilterFactory.process(
        "comb_notch", dbs_uv.copy(), sfreq, f0=f_dbs, q_factor=200
    )
    sinreg_uv = ArtifactFilterFactory.process(
        "sinusoidal_regression", dbs_uv.copy(), sfreq, f_target=f_dbs
    )

    # ── Novel stages ─────────────────────────────────────────────────────
    print("  Stage A: SVD + NLMS …")
    stageA_uv = apply_svd_nlms(dbs_uv.copy(), sfreq, f_dbs)

    print("  Stage B: Complex Hampel FFT …")
    stageB_uv = apply_complex_hampel_fft(dbs_uv.copy(), sfreq, f_dbs)

    print("  Stage C: Template subtraction …")
    stageC_uv = apply_template_subtraction(dbs_uv.copy(), sfreq, f_dbs)

    print("  Hybrid A→B→C …")
    hybrid_uv = apply_hybrid_abc(dbs_uv.copy(), sfreq, f_dbs)

    # ── ICA on Hybrid output ─────────────────────────────────────────────
    hybrid_ica_uv = hybrid_uv.copy()
    ica_report = {}
    if run_ica:
        print("  Fitting ICA (this may take ~60 s) …")
        try:
            hybrid_ica_uv, ica_report = apply_ica_pipeline(
                hybrid_uv.copy(), raw_dbs, f_dbs
            )
            print(f"    Excluded: {ica_report['all_excluded']}")
        except Exception as exc:
            print(f"    ICA failed: {exc}")

    # ── Collect named results ─────────────────────────────────────────────
    named = {
        "Raw DBS":               dbs_uv,
        "PRE baseline":          pre_uv,
        "FFT Spectral Interp":   fft_uv,
        "Comb Notch Q=200":      notch_uv,
        "Sinusoidal Regression": sinreg_uv,
        "Stage A (SVD+NLMS)":    stageA_uv,
        "Stage B (Cplx Hampel)": stageB_uv,
        "Stage C (Template)":    stageC_uv,
        "Hybrid A→B→C":          hybrid_uv,
        "Hybrid + ICA":          hybrid_ica_uv,
    }

    # ── PSD comparison figure ─────────────────────────────────────────────
    tag = condition_tag.replace(" ", "_")
    psd_fig = results_dir / f"psd_comparison_{tag}.png"
    print(f"  Saving PSD figure → {psd_fig.name}")
    plot_psd_comparison(
        named, sfreq, f_dbs,
        title=f"{condition_tag} ({f_dbs} Hz DBS) — method comparison",
        out_path=psd_fig,
    )

    # ── 2-second time-domain figure ───────────────────────────────────────
    td_fig = results_dir / f"time_domain_{tag}.png"
    print(f"  Saving time-domain figure → {td_fig.name}")
    plot_time_domain(
        named, sfreq, ch_names, ANALYSIS_CH,
        t_offset=10.0, dur=2.0, f_dbs=f_dbs,
        title=f"{condition_tag} ({f_dbs} Hz DBS)",
        out_path=td_fig,
    )

    # ── Compute metrics for all cleaned methods ───────────────────────────
    print("  Computing metrics …")
    metrics_list = []
    skip = {"Raw DBS", "PRE baseline"}
    for label, data_uv in named.items():
        if label in skip:
            continue
        m = compute_metrics(data_uv, dbs_uv, pre_uv, sfreq, f_dbs, label=label)
        m["condition"] = condition_tag
        m["f_dbs_hz"]  = f_dbs
        metrics_list.append(m)

        goal = "✓" if m["both_goals_met"] else "✗"
        print(f"    {goal} {label:<28} "
              f"β={m['beta_preservation_%']:6.1f}%  "
              f"DBS↓={m['dbs_reduction_%']:5.1f}%  "
              f"atten={m['dbs_attenuation_dB']:+.1f} dB")

    elapsed = time.perf_counter() - t0
    print(f"  Done ({elapsed:.1f} s)")
    return metrics_list


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experimental DBS artifact removal evaluation suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dbs-freq", type=float, default=7.0,
                   help="Primary DBS frequency to evaluate (default: 7). "
                        "Use 'all' flag to include 60/100 Hz conditions.")
    p.add_argument("--all-freqs", action="store_true",
                   help="Evaluate all DBS conditions (7, 60, 100 Hz)")
    p.add_argument("--mode", choices=["awake", "sleep", "both"], default="both",
                   help="Which recording state to process (default: both)")
    p.add_argument("--no-ica", action="store_true",
                   help="Skip ICA stage (faster for quick testing)")
    p.add_argument("--data-dir", default=str(DATA_DIR),
                   help=f"EDF data directory (default: {DATA_DIR})")
    p.add_argument("--results-dir", default=str(RESULTS_DIR),
                   help=f"Output directory (default: {RESULTS_DIR})")
    return p.parse_args()


def main():
    args    = parse_args()
    data_d  = pathlib.Path(args.data_dir)
    res_d   = pathlib.Path(args.results_dir)
    res_d.mkdir(parents=True, exist_ok=True)

    # ── Choose which DBS frequencies to run ──────────────────────────────
    if args.all_freqs:
        eval_freqs = [7.0, 60.0, 100.0]
    else:
        eval_freqs = [float(args.dbs_freq)]

    # ── Choose conditions ─────────────────────────────────────────────────
    conditions = {"awake": ["awake"], "sleep": ["sleep"],
                  "both": ["awake", "sleep"]}[args.mode]

    print(f"\nExperimental DBS Evaluation Suite")
    print(f"  Data dir   : {data_d}")
    print(f"  Results dir: {res_d}")
    print(f"  DBS freqs  : {eval_freqs} Hz")
    print(f"  Conditions : {conditions}")
    print(f"  Max duration: {MAX_DUR_SEC:.0f} s  |  Target Fs: {TARGET_SFREQ:.0f} Hz")
    print(f"  ICA        : {'disabled' if args.no_ica else 'enabled'}")
    print(f"  Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_metrics: list[dict] = []

    for f_dbs in eval_freqs:
        if f_dbs not in RECORDINGS:
            print(f"\nWarning: no file registry for {f_dbs} Hz DBS — skipping.")
            continue

        for cond in conditions:
            dbs_fname = RECORDINGS[f_dbs].get(cond)
            pre_fname = PRE_FILES.get(cond)
            if not dbs_fname or not pre_fname:
                continue

            dbs_path = data_d / dbs_fname
            pre_path = data_d / pre_fname

            if not dbs_path.exists():
                print(f"\nSkipping: {dbs_path} not found")
                continue
            if not pre_path.exists():
                print(f"\nSkipping: {pre_path} not found")
                continue

            tag = f"{cond.upper()}_{int(f_dbs)}Hz"
            metrics = run_single_experiment(
                dbs_path, pre_path, f_dbs,
                condition_tag=tag,
                results_dir=res_d,
                run_ica=not args.no_ica,
            )
            all_metrics.extend(metrics)

    if not all_metrics:
        print("\nNo experiments completed. Check data directory.")
        return

    # ── Summary bar chart ─────────────────────────────────────────────────
    print("\nGenerating beta preservation summary …")
    plot_beta_summary(all_metrics, res_d / "beta_preservation_summary.png")

    # ── Save metrics to JSON and CSV ─────────────────────────────────────
    metrics_json = res_d / "metrics_summary.json"
    with open(metrics_json, "w") as fh:
        json.dump(all_metrics, fh, indent=2, default=str)
    print(f"Metrics JSON → {metrics_json}")

    try:
        import pandas as pd
        df = pd.DataFrame(all_metrics)
        csv_path = res_d / "metrics_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"Metrics CSV  → {csv_path}")

        # ── Console summary table ─────────────────────────────────────────
        print("\n" + "─" * 100)
        print(f"{'Condition':<22} {'Method':<28} {'Beta%':>8} "
              f"{'DBS↓%':>8} {'Atten(dB)':>10} {'Both✓?':>8}")
        print("─" * 100)
        for _, row in df.sort_values(["condition", "dbs_reduction_%"],
                                      ascending=[True, False]).iterrows():
            flag = "✓" if row["both_goals_met"] else " "
            print(f"  {flag} {row['condition']:<20} {row['method']:<28} "
                  f"{row['beta_preservation_%']:>7.1f}% "
                  f"{row['dbs_reduction_%']:>7.1f}% "
                  f"{row['dbs_attenuation_dB']:>10.1f} dB "
                  f"{'yes' if row['both_goals_met'] else 'no':>8}")
        print("─" * 100)
    except ImportError:
        pass

    print(f"\nAll results saved to: {res_d}")
    print("Done.")


if __name__ == "__main__":
    main()
