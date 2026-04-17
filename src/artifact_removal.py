"""
Surgical DBS Artifact Removal — Innovation 1 + 2 Hybrid.

Standard frequency-carving (notch/comb/band-nulling) is deliberately
avoided because a 7 Hz DBS fundamental sits inside the physiological
theta band (4–8 Hz).  We instead exploit two orthogonal mathematical
properties of the artifact:

Innovation 1 — Allen Complex-Domain Hampel FFT
----------------------------------------------
For a T-second recording at fs Hz the full-length FFT has resolution
δf = fs / N = 1/T Hz.  The DBS artifact is a perfectly-periodic
machine oscillation, so it concentrates all its energy in ≤ 1 FFT bin.
Physiological theta is a stochastic process and distributes energy
across ~(8–4)/δf = 4·T bins.

A sliding-window MAD scan over the *complex* spectrum (real and
imaginary parts independently) detects the narrow spike as a
statistical outlier and replaces it with the local window median —
an estimate of the brain's expected spectral amplitude at that exact
frequency — NOT with zero.  Bins outside the spike are untouched.

Reference: Allen et al. (2010) "Suppression of stimulation artefacts
from local field potential recordings", J Neurosci Methods.

Innovation 2 — EMD Phase-Locked Template Subtraction
------------------------------------------------------
Empirical Mode Decomposition decomposes each channel adaptively into
Intrinsic Mode Functions (IMFs).  The DBS artifact is deterministic
and perfectly phase-coherent, yielding one IMF whose Hilbert-derived
instantaneous frequency (IF) is nearly constant (σ_IF ≈ 0).
Physiological theta is stochastic: σ_IF >> 0.  We identify the
minimally-variant IF component near f_dbs and subtract it, leaving
stochastic brain theta completely untouched.

Reference: Huang et al. (1998) "The empirical mode decomposition and
the Hilbert spectrum for nonlinear and non-stationary time series",
Proc Royal Soc A.
"""

from __future__ import annotations

import warnings
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import ndimage, signal
from scipy.interpolate import CubicSpline
from scipy.signal import argrelmax, argrelmin

MAD_SCALE = 1.4826  # converts MAD → σ for Gaussian distributions


# ---------------------------------------------------------------------------
# Innovation 1: Allen Complex-Domain Hampel FFT
# ---------------------------------------------------------------------------

def allen_hampel_fft(
    raw: mne.io.BaseRaw,
    dbs_freq: float = 7.0,
    window_hz: float = 2.0,
    n_sigmas: float = 3.5,
    target_half_hz: float = 0.15,
    n_harmonics_max: int = 20,
    report: bool = True,
) -> mne.io.BaseRaw:
    """Remove DBS harmonics via sliding-window MAD scan of the complex spectrum.

    Full-length FFT frequency resolution is δf = fs / N = 1/T Hz.  For a
    60-second recording at 256 Hz, δf ≈ 17 mHz — each DBS harmonic occupies
    only 1–3 bins while physiological theta spans ~(8–4)/δf ≈ 240 bins.

    Algorithm (per channel, per DBS harmonic):
      1. Compute the one-sided rFFT of the full-length signal.
      2. Around each harmonic h = k·f_dbs, take a *reference window* of
         ±window_hz/2 Hz.  Compute the rolling median and MAD of the real
         and imaginary parts within that window.
      3. Flag any bin whose deviation exceeds n_sigmas × 1.4826 × MAD as an
         outlier — but **only replace bins within ±target_half_hz** of the
         harmonic centre (the actual spike zone).
      4. Replace each outlier spike bin with the local window median — a
         statistically consistent estimate of the brain signal at that
         frequency — NOT with zero.
      5. Reconstruct via IFFT.

    Bins outside ±target_half_hz of every harmonic are **never touched**,
    preserving all physiological theta/alpha power regardless of amplitude.

    Args:
        raw: MNE Raw object (copied internally).
        dbs_freq: Fundamental DBS frequency in Hz.
        window_hz: Reference window width (Hz) for MAD estimation.
                   Default 2.0 Hz gives enough background samples while
                   staying within a single inter-harmonic gap for 7 Hz DBS.
        n_sigmas: Detection threshold in MAD units.  DBS spikes are typically
                  20–100 σ; 3.5 is conservative against broad brain peaks.
        target_half_hz: Half-width (Hz) of the **replacement zone** around
                        each harmonic.  Only bins this close to the harmonic
                        centre are ever replaced.  Default 0.15 Hz.
        n_harmonics_max: Maximum harmonic index to process.
        report: Print per-harmonic attenuation estimates.

    Returns:
        Cleaned MNE Raw object.
    """
    raw_clean = raw.copy().load_data()
    data = raw_clean.get_data()          # (n_ch, n_samples)  [V]
    n_ch, n_s = data.shape
    sfreq = float(raw_clean.info["sfreq"])
    nyquist = sfreq / 2.0

    df = sfreq / n_s                     # Hz per FFT bin
    win_bins = max(3, int(round(window_hz / df)))
    if win_bins % 2 == 0:
        win_bins += 1
    tgt_bins = max(1, int(round(target_half_hz / df)))

    if report:
        print(
            f"[AllenHampelFFT] f₀={dbs_freq} Hz | "
            f"ref-window={window_hz} Hz ({win_bins} bins) | "
            f"target-zone=±{target_half_hz} Hz (±{tgt_bins} bins) | "
            f"δf={df * 1000:.2f} mHz | σ={n_sigmas}"
        )

    # Full one-sided FFT: shape (n_ch, n_fft_bins)
    S = np.fft.rfft(data, axis=1).copy()
    n_fft = S.shape[1]

    harmonics = [k * dbs_freq for k in range(1, n_harmonics_max + 1)
                 if k * dbs_freq < nyquist]

    total_replaced = 0

    for h in harmonics:
        h_bin = int(round(h / df))
        if h_bin >= n_fft:
            break

        # Reference window around harmonic for MAD estimation
        ref_lo = max(0, h_bin - win_bins // 2)
        ref_hi = min(n_fft, h_bin + win_bins // 2 + 1)

        # Replacement zone (narrower) — only these bins may be replaced
        tgt_lo = max(1, h_bin - tgt_bins)
        tgt_hi = min(n_fft - 1, h_bin + tgt_bins + 1)
        tgt_slice = slice(tgt_lo, tgt_hi)

        replaced_this = 0
        for part_attr in ("real", "imag"):
            part_window = getattr(S[:, ref_lo:ref_hi], part_attr)  # (n_ch, win)

            # Median and MAD of the reference window per channel
            med_win = np.median(part_window, axis=1, keepdims=True)  # (n_ch, 1)
            mad_win = np.median(
                np.abs(part_window - med_win), axis=1, keepdims=True
            )
            thr = n_sigmas * MAD_SCALE * np.maximum(mad_win, 1e-30)

            part_tgt = getattr(S[:, tgt_slice], part_attr).copy()   # (n_ch, tgt)
            abs_dev = np.abs(part_tgt - med_win)
            outlier = abs_dev > thr                                   # (n_ch, tgt)

            if outlier.any():
                # Replace each channel's outlier bins with the window median
                if part_attr == "real":
                    S.real[:, tgt_slice][outlier] = np.broadcast_to(
                        med_win, part_tgt.shape
                    )[outlier]
                else:
                    S.imag[:, tgt_slice][outlier] = np.broadcast_to(
                        med_win, part_tgt.shape
                    )[outlier]
                replaced_this += int(outlier.sum())

        total_replaced += replaced_this

        if report:
            # Estimate attenuation: compare pre/post magnitude at harmonic bin
            orig_mag = np.abs(
                np.fft.rfft(data, axis=1)[:, h_bin]
            ).mean()
            clean_mag = np.abs(S[:, h_bin]).mean()
            atten_db = 20.0 * np.log10(
                (orig_mag + 1e-30) / (clean_mag + 1e-30)
            )
            print(
                f"  {h:6.1f} Hz (bin {h_bin:5d}): "
                f"replaced {replaced_this} bins across {n_ch} ch | "
                f"attenuation ≈ {atten_db:+.1f} dB"
            )

    if report:
        print(f"  Total bins replaced: {total_replaced}")

    # Reconstruct time-domain signal via IFFT
    cleaned = np.fft.irfft(S, n=n_s, axis=1)
    raw_clean._data[:] = cleaned
    return raw_clean


# ---------------------------------------------------------------------------
# Innovation 2: EMD Phase-Locked Template Subtraction
# ---------------------------------------------------------------------------

def _sift_one_imf(
    x: np.ndarray,
    max_sifts: int = 50,
    sd_tol: float = 0.05,
    min_extrema: int = 4,
) -> np.ndarray:
    """Extract the first IMF from signal x via the EMD sifting process."""
    h = x.copy()
    n = len(h)
    t = np.arange(n, dtype=float)

    for _ in range(max_sifts):
        max_idx = argrelmax(h, order=1)[0]
        min_idx = argrelmin(h, order=1)[0]

        if len(max_idx) < min_extrema or len(min_idx) < min_extrema:
            break

        # Boundary mirroring: prepend/append one reflected extremum
        max_idx_ext = np.concatenate([[2 * max_idx[0] - max_idx[1]], max_idx,
                                       [2 * max_idx[-1] - max_idx[-2]]])
        min_idx_ext = np.concatenate([[2 * min_idx[0] - min_idx[1]], min_idx,
                                       [2 * min_idx[-1] - min_idx[-2]]])
        max_idx_ext = np.clip(max_idx_ext, 0, n - 1)
        min_idx_ext = np.clip(min_idx_ext, 0, n - 1)

        try:
            upper_env = CubicSpline(max_idx_ext, h[max_idx_ext])(t)
            lower_env = CubicSpline(min_idx_ext, h[min_idx_ext])(t)
        except Exception:
            break

        mean_env = (upper_env + lower_env) / 2.0
        h_prev = h.copy()
        h = h - mean_env

        # Cauchy-type stopping criterion
        sd = float(np.sum((h - h_prev) ** 2) / (np.sum(h_prev ** 2) + 1e-30))
        if sd < sd_tol:
            break

    return h


def _emd_decompose(
    x: np.ndarray,
    max_imfs: int = 8,
    min_extrema: int = 4,
    sd_tol: float = 0.05,
) -> np.ndarray:
    """Decompose x into IMFs via EMD sifting.

    Returns array of shape (n_imfs+1, n_samples) where the last row is the
    final monotone residue.
    """
    imfs = []
    residue = x.copy()

    for _ in range(max_imfs):
        max_idx = argrelmax(residue, order=1)[0]
        if len(max_idx) < min_extrema:
            break
        imf = _sift_one_imf(residue, sd_tol=sd_tol)
        imfs.append(imf)
        residue = residue - imf

    imfs.append(residue)
    return np.array(imfs)                # (n_imfs, n_samples)


def _instantaneous_frequency(imf: np.ndarray, sfreq: float) -> np.ndarray:
    """Compute per-sample instantaneous frequency via Hilbert transform (Hz)."""
    analytic = signal.hilbert(imf)
    phase = np.unwrap(np.angle(analytic))
    # Central difference for derivative stability
    if_hz = np.gradient(phase, 1.0 / sfreq) / (2.0 * np.pi)
    return if_hz


def _find_dbs_imf(
    imfs: np.ndarray,
    sfreq: float,
    f_dbs: float,
    tol_hz: float = 1.5,
    max_if_std: float = 1.2,
) -> int | None:
    """Return index of the IMF most phase-locked to f_dbs.

    Criterion: mean(|IF|) within tol_hz of f_dbs AND std(IF) < max_if_std.
    Among candidates, choose the one with the smallest std(IF) — i.e., the
    most deterministic (least stochastic) component.
    """
    best_idx: int | None = None
    best_std = np.inf

    for i, imf in enumerate(imfs):
        if_hz = _instantaneous_frequency(imf, sfreq)
        # Trim 5% from each end to reduce Hilbert edge effects
        trim = max(1, len(if_hz) // 20)
        if_hz_inner = if_hz[trim:-trim]

        mean_if = float(np.median(np.abs(if_hz_inner)))
        std_if = float(np.std(if_hz_inner))

        if abs(mean_if - f_dbs) <= tol_hz and std_if < max_if_std:
            if std_if < best_std:
                best_std = std_if
                best_idx = i

    return best_idx


def emd_phase_locked_subtraction(
    raw: mne.io.BaseRaw,
    dbs_freq: float = 7.0,
    max_imfs: int = 8,
    tol_hz: float = 1.5,
    max_if_std: float = 1.2,
    report: bool = True,
) -> mne.io.BaseRaw:
    """Remove the phase-locked DBS IMF via EMD decomposition.

    Each channel is independently decomposed into IMFs.  The component
    whose Hilbert instantaneous frequency is closest to f_dbs with minimal
    variance (the deterministic DBS oscillation) is identified and
    subtracted.  All other IMFs — including stochastic theta — are
    retained verbatim.

    Args:
        raw: MNE Raw object.
        dbs_freq: DBS fundamental frequency in Hz.
        max_imfs: Maximum IMFs to extract per channel.
        tol_hz: Acceptable deviation of mean IF from f_dbs.
        max_if_std: Maximum allowed std of IF to qualify as 'phase-locked'.
        report: Print per-channel status.

    Returns:
        Cleaned MNE Raw object.
    """
    raw_clean = raw.copy().load_data()
    data = raw_clean.get_data()          # (n_ch, n_samples) [V]
    n_ch, n_s = data.shape
    sfreq = float(raw_clean.info["sfreq"])

    if report:
        print(
            f"[EMD Phase-Lock Subtraction] f₀={dbs_freq} Hz | "
            f"max_imfs={max_imfs} | tol=±{tol_hz} Hz | max_std={max_if_std} Hz"
        )

    n_subtracted = 0
    for ch in range(n_ch):
        x = data[ch]
        imfs = _emd_decompose(x, max_imfs=max_imfs)
        dbs_idx = _find_dbs_imf(imfs, sfreq, dbs_freq, tol_hz, max_if_std)

        if dbs_idx is not None:
            data[ch] = x - imfs[dbs_idx]
            n_subtracted += 1
            if report:
                if_hz = _instantaneous_frequency(imfs[dbs_idx], sfreq)
                trim = max(1, len(if_hz) // 20)
                if_inner = if_hz[trim:-trim]
                print(
                    f"  ch {ch:2d} ({raw_clean.ch_names[ch]}): "
                    f"IMF[{dbs_idx}] subtracted — "
                    f"mean_IF={np.median(np.abs(if_inner)):.2f} Hz, "
                    f"std_IF={np.std(if_inner):.3f} Hz"
                )
        elif report:
            print(
                f"  ch {ch:2d} ({raw_clean.ch_names[ch]}): "
                f"no phase-locked IMF found — channel unchanged"
            )

    if report:
        print(f"  {n_subtracted}/{n_ch} channels had a DBS IMF subtracted.")

    raw_clean._data[:] = data
    return raw_clean


# ---------------------------------------------------------------------------
# Hybrid Remover (MNE-compatible class)
# ---------------------------------------------------------------------------

class SurgicalDBSRemover:
    """MNE-compatible surgical DBS artifact remover.

    Applies Innovation 1 (Allen Complex Hampel FFT) first, then optionally
    Innovation 2 (EMD phase-locked subtraction) as a residual-cleanup pass.

    Usage::

        remover = SurgicalDBSRemover(dbs_freq=7.0)
        raw_clean = remover.remove(raw_contaminated)
        fig = remover.validate(raw_contaminated, raw_clean)
        fig.savefig("psd_validation.png")
    """

    def __init__(
        self,
        dbs_freq: float = 7.0,
        method: Literal["hampel", "emd", "hybrid"] = "hampel",
        # Hampel params
        window_hz: float = 2.0,
        n_sigmas: float = 3.5,
        # EMD params
        max_imfs: int = 8,
        tol_hz: float = 1.5,
        max_if_std: float = 1.2,
    ) -> None:
        self.dbs_freq = dbs_freq
        self.method = method
        self.window_hz = window_hz
        self.n_sigmas = n_sigmas
        self.max_imfs = max_imfs
        self.tol_hz = tol_hz
        self.max_if_std = max_if_std

    def remove(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Apply the configured method(s) and return cleaned Raw."""
        if self.method in ("hampel", "hybrid"):
            raw_out = allen_hampel_fft(
                raw,
                dbs_freq=self.dbs_freq,
                window_hz=self.window_hz,
                n_sigmas=self.n_sigmas,
            )
        else:
            raw_out = raw.copy().load_data()

        if self.method in ("emd", "hybrid"):
            raw_out = emd_phase_locked_subtraction(
                raw_out,
                dbs_freq=self.dbs_freq,
                max_imfs=self.max_imfs,
                tol_hz=self.tol_hz,
                max_if_std=self.max_if_std,
            )

        return raw_out

    def validate(
        self,
        raw_before: mne.io.BaseRaw,
        raw_after: mne.io.BaseRaw,
        fmin: float = 1.0,
        fmax: float = 15.0,
        n_fft: int = 8192,
        save_path: str | None = None,
    ) -> plt.Figure:
        """Plot and optionally save the PSD validation figure."""
        return validate_removal(
            raw_before,
            raw_after,
            dbs_freq=self.dbs_freq,
            fmin=fmin,
            fmax=fmax,
            n_fft=n_fft,
            save_path=save_path,
        )


# ---------------------------------------------------------------------------
# Validation: PSD comparison zoomed on 1–15 Hz
# ---------------------------------------------------------------------------

def _welch_psd_mean(
    raw: mne.io.BaseRaw, fmin: float, fmax: float, n_fft: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (freqs, mean_power_across_eeg_channels) via Welch."""
    eeg = raw.copy().pick_types(eeg=True, exclude=[])
    data = eeg.get_data()
    sfreq = float(eeg.info["sfreq"])
    nperseg = min(n_fft, data.shape[1])
    freqs, psd = signal.welch(
        data, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2,
        window="hann", axis=1,
    )
    mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[mask], psd[:, mask].mean(axis=0)


def validate_removal(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    dbs_freq: float = 7.0,
    fmin: float = 1.0,
    fmax: float = 15.0,
    n_fft: int = 8192,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Generate the mandatory PSD validation figure.

    Produces three panels:
      1. PSD overlay (before / after) on 1–15 Hz, log-power scale.
      2. Attenuation curve: before_dB − after_dB, highlighting DBS harmonics.
      3. Theta-band (4–8 Hz) power preservation bar chart.

    Mathematical proof of correctness:
      - 7 Hz spike: ΔdB >> 0 (artifact removed).
      - 4–8 Hz integrated power: ratio ≈ 1.0 (theta preserved).

    Args:
        raw_before: Original contaminated Raw.
        raw_after: Cleaned Raw.
        dbs_freq: DBS fundamental in Hz.
        fmin/fmax: Frequency axis range for PSD panels.
        n_fft: Welch window length in samples.
        save_path: If given, save figure to this path (PNG).

    Returns:
        matplotlib Figure.
    """
    freqs, psd_before = _welch_psd_mean(raw_before, fmin, fmax, n_fft)
    _, psd_after = _welch_psd_mean(raw_after, fmin, fmax, n_fft)

    psd_before_db = 10.0 * np.log10(psd_before + 1e-30)
    psd_after_db = 10.0 * np.log10(psd_after + 1e-30)
    attenuation_db = psd_before_db - psd_after_db   # positive = removed

    # Exclude ±0.5 Hz around each DBS harmonic when computing band preservation.
    # This measures how much *off-harmonic* (i.e., brain) signal is retained,
    # which is the correct proof that theta/alpha are not harmed.
    excl_hz = 0.5
    harm_excl = np.zeros(len(freqs), dtype=bool)
    for k in range(1, 30):
        h = k * dbs_freq
        harm_excl |= np.abs(freqs - h) <= excl_hz

    # Theta (4–8 Hz) off-harmonic integrated power ratio
    theta_mask = (freqs >= 4.0) & (freqs <= 8.0) & ~harm_excl
    theta_before = float(np.trapz(psd_before[theta_mask], freqs[theta_mask]))
    theta_after = float(np.trapz(psd_after[theta_mask], freqs[theta_mask]))
    theta_preservation_pct = 100.0 * theta_after / (theta_before + 1e-30)

    # Alpha (8–13 Hz) off-harmonic integrated power ratio
    alpha_mask = (freqs >= 8.0) & (freqs <= 13.0) & ~harm_excl
    alpha_before = float(np.trapz(psd_before[alpha_mask], freqs[alpha_mask]))
    alpha_after = float(np.trapz(psd_after[alpha_mask], freqs[alpha_mask]))
    alpha_preservation_pct = 100.0 * alpha_after / (alpha_before + 1e-30)

    # DBS harmonics within [fmin, fmax]
    nyquist = float(raw_before.info["sfreq"]) / 2.0
    harmonics = [k * dbs_freq for k in range(1, 20) if fmin <= k * dbs_freq <= fmax]

    # Attenuation at each harmonic (peak dB in a ±0.5 Hz window)
    harm_attens = []
    for h in harmonics:
        window = np.abs(freqs - h) <= 0.5
        if window.any():
            harm_attens.append(float(attenuation_db[window].max()))
        else:
            harm_attens.append(0.0)

    # ------- Build figure -------
    fig, axes = plt.subplots(3, 1, figsize=(10, 11))
    fig.suptitle(
        f"Surgical DBS Artifact Removal Validation\n"
        f"DBS f₀ = {dbs_freq} Hz  |  "
        f"Off-harmonic θ preserved: {theta_preservation_pct:.1f}%  |  "
        f"Off-harmonic α preserved: {alpha_preservation_pct:.1f}%",
        fontsize=13, fontweight="bold",
    )

    # Panel 1 — PSD overlay
    ax1 = axes[0]
    ax1.plot(freqs, psd_before_db, color="#D62728", lw=1.6,
             label="Before (contaminated)", alpha=0.85)
    ax1.plot(freqs, psd_after_db, color="#1F77B4", lw=1.8,
             label="After (cleaned)", alpha=0.95)
    for h in harmonics:
        ax1.axvline(h, color="orange", lw=0.8, ls="--", alpha=0.7)
    ax1.axvspan(4.0, 8.0, color="cyan", alpha=0.08, label="Theta band 4–8 Hz")
    ax1.axvspan(8.0, 13.0, color="green", alpha=0.06, label="Alpha band 8–13 Hz")
    ax1.set_xlim(fmin, fmax)
    ax1.set_ylabel("Power (dB rel. V²/Hz)")
    ax1.set_title("PSD Comparison (Welch, mean across EEG channels)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2 — Attenuation curve
    ax2 = axes[1]
    ax2.fill_between(freqs, 0, attenuation_db,
                     where=attenuation_db > 0,
                     color="#D62728", alpha=0.4, label="Attenuated regions")
    ax2.fill_between(freqs, attenuation_db, 0,
                     where=attenuation_db < 0,
                     color="#1F77B4", alpha=0.3, label="Added energy (unexpected)")
    ax2.plot(freqs, attenuation_db, color="black", lw=1.0)
    ax2.axhline(0, color="gray", lw=0.8)
    for h, a in zip(harmonics, harm_attens):
        ax2.annotate(
            f"{h:.0f}Hz\n{a:+.1f}dB",
            xy=(h, a), fontsize=7.5, ha="center",
            xytext=(0, 12), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", lw=0.8),
        )
    ax2.axvspan(4.0, 8.0, color="cyan", alpha=0.08)
    ax2.set_xlim(fmin, fmax)
    ax2.set_ylabel("Δ Power (dB)")
    ax2.set_title("Attenuation Curve — positive = removed, near-zero outside spikes = preserved")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3 — Band preservation bar chart
    ax3 = axes[2]
    bands = {
        "Delta\n1–4 Hz": ((1.0, 4.0), "#7F7F7F"),
        "Theta\n4–8 Hz": ((4.0, 8.0), "#17BECF"),
        "Alpha\n8–13 Hz": ((8.0, 13.0), "#2CA02C"),
        "Beta\n13–15 Hz": ((13.0, 15.0), "#9467BD"),
    }
    bar_labels, bar_vals, bar_colors = [], [], []
    for band_label, (band_range, color) in bands.items():
        lo, hi = band_range
        if hi > fmax:
            continue
        bm = (freqs >= lo) & (freqs <= hi) & ~harm_excl   # off-harmonic only
        if bm.sum() < 2:
            continue
        p_b = float(np.trapz(psd_before[bm], freqs[bm]))
        p_a = float(np.trapz(psd_after[bm], freqs[bm]))
        bar_labels.append(band_label)
        bar_vals.append(100.0 * p_a / (p_b + 1e-30))
        bar_colors.append(color)

    bars = ax3.bar(bar_labels, bar_vals, color=bar_colors, alpha=0.8, edgecolor="black")
    ax3.axhline(100.0, color="gray", lw=1.0, ls="--", label="100% = perfect preservation")
    ax3.axhline(95.0, color="orange", lw=0.8, ls=":", label="95% threshold")
    for bar, val in zip(bars, bar_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax3.set_ylim(0, max(130, max(bar_vals) + 15) if bar_vals else 130)
    ax3.set_ylabel("Preserved power (%)")
    ax3.set_title("Band-Integrated Power Preservation (100% = no signal loss)")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[validate_removal] Figure saved → {save_path}")

    # Print mathematical summary
    print("\n=== Validation Summary ===")
    print(f"  Off-harmonic θ (4–8 Hz,  excl ±{excl_hz} Hz of harmonics): {theta_preservation_pct:.1f}%")
    print(f"  Off-harmonic α (8–13 Hz, excl ±{excl_hz} Hz of harmonics): {alpha_preservation_pct:.1f}%")
    for h, a in zip(harmonics, harm_attens):
        print(f"  {h:5.1f} Hz harmonic attenuation:    {a:+.1f} dB")
    print("=========================\n")

    return fig
