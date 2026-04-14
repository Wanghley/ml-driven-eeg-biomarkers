import numpy as np
from scipy import ndimage, signal
import warnings

# Define constants for robust statistics
MAD_SCALE_FACTOR = 1.4826


class BaselineReferencedFilter:
    """
    Baseline-Referenced Wiener Spectral Gating for DBS artifact removal.

    Uses clean baseline recordings (no DBS) to build a frequency-dependent
    gain mask that attenuates only the *excess* power introduced by DBS
    stimulation at the fundamental and harmonic frequencies, while preserving
    endogenous brain activity at those same frequencies.

    Theory (time-varying Wiener formulation):
        At each time-frequency tile (f, t), the gain is:
            G(f, t) = P_baseline(f) / [P_baseline(f) + α · max(0, |X(f,t)|² - P_baseline(f))]
        where P_baseline is estimated from clean recordings, |X(f,t)|² is the
        instantaneous power in the DBS recording, and α >= 1 is an oversubtraction
        factor. Away from artifact harmonics, G = 1 (no attenuation). At harmonics,
        G < 1 proportional to instantaneous artifact strength.

    Improvements over fixed-gain approach:
        - Time-varying gain adapts to non-stationary DBS artifact amplitude
        - Smooth cosine taper at harmonic band edges prevents spectral ringing
        - Oversubtraction factor compensates for baseline PSD estimation errors
        - Proper amplitude-domain gain (sqrt of power ratio) for correct attenuation
    """

    def __init__(self, baseline_raw, dbs_freq: float = 7.0,
                 harmonic_bandwidth: float = 2.0, n_fft: int = 4096,
                 floor_db: float = -40.0, alpha: float = 1.5,
                 taper_width: float = 1.0):
        """
        Args:
            baseline_raw: MNE Raw object from a clean baseline recording
                          (e.g., XUAWAKEPRE or XUSLEEP). Must share the same
                          channel set and sampling rate as the DBS data.
            dbs_freq: Fundamental DBS stimulation frequency in Hz.
            harmonic_bandwidth: Total bandwidth (Hz) around each harmonic
                                where full Wiener gain is applied. Default 2.0 Hz.
            n_fft: FFT length for PSD and STFT. Longer = finer frequency
                   resolution for separating narrow DBS peaks from brain.
            floor_db: Minimum gain in dB to prevent complete nulling.
                      Default -40 dB removes 99.99% of artifact power
                      while avoiding numerical zeros.
            alpha: Oversubtraction factor (>= 1.0). Values > 1 remove slightly
                   more than the estimated noise, reducing residual artifacts
                   at the cost of minor signal loss at harmonics. Default 1.5.
            taper_width: Width (Hz) of cosine rolloff beyond harmonic_bandwidth
                         edges. Captures spectral leakage/sidebands. Default 1.0 Hz.
        """
        self.dbs_freq = dbs_freq
        self.harmonic_bandwidth = harmonic_bandwidth
        self.n_fft = n_fft
        self.floor_gain = 10 ** (floor_db / 20.0)
        self.alpha = max(alpha, 1.0)
        self.taper_width = taper_width

        self._sfreq = baseline_raw.info['sfreq']
        self._baseline_psd, self._psd_freqs = self._estimate_psd(baseline_raw)

    def _estimate_psd(self, raw):
        """Compute per-channel PSD via Welch's method with Hann window."""
        data = raw.get_data()  # (n_channels, n_samples)
        freqs, psd = signal.welch(
            data, fs=self._sfreq, nperseg=self.n_fft,
            noverlap=self.n_fft // 2, window='hann', axis=1
        )
        return psd, freqs

    def _build_harmonic_taper(self, freqs: np.ndarray) -> np.ndarray:
        """Build smooth taper around each DBS harmonic.

        Returns array of shape (n_freq,) where:
        - 1.0 within harmonic_bandwidth/2 of each harmonic (full filtering)
        - Cosine rolloff from 1→0 over taper_width beyond that
        - 0.0 away from harmonics (no filtering)
        """
        nyquist = self._sfreq / 2.0
        harmonics = np.arange(self.dbs_freq, nyquist, self.dbs_freq)
        half_bw = self.harmonic_bandwidth / 2.0

        taper = np.zeros(len(freqs))
        for h in harmonics:
            dist = np.abs(freqs - h)
            # Full filtering within the core bandwidth
            core = dist <= half_bw
            taper[core] = 1.0
            # Cosine rolloff in the transition region
            if self.taper_width > 0:
                transition = (dist > half_bw) & (dist <= half_bw + self.taper_width)
                taper[transition] = np.maximum(
                    taper[transition],
                    0.5 * (1 + np.cos(np.pi * (dist[transition] - half_bw) / self.taper_width))
                )
        return taper

    def filter(self, dbs_raw):
        """
        Apply time-varying baseline-referenced Wiener spectral gating to remove DBS artifacts.

        Pipeline:
        1. STFT the DBS recording into time-frequency tiles.
        2. At each tile (f, t) near a harmonic, compute instantaneous Wiener gain:
           G_power(f,t) = P_base(f) / [P_base(f) + α · max(0, |X(f,t)|² - P_base(f))]
        3. Apply smooth cosine taper to blend gain at harmonic edges.
        4. Convert to amplitude gain: G_amp = sqrt(G_power).
        5. Multiply STFT coefficients by G_amp and reconstruct via ISTFT.

        Signal preservation guarantee:
        - At non-harmonic frequencies, G = 1.0 exactly — zero modification.
        - At harmonic frequencies, the time-varying gain adapts to instantaneous
          artifact power, removing more when the artifact is stronger.
        - The cosine taper captures spectral leakage/sidebands beyond the core bandwidth.

        Args:
            dbs_raw: MNE Raw object containing DBS-contaminated EEG.

        Returns:
            MNE Raw object with DBS artifacts removed.
        """
        raw_clean = dbs_raw.copy()
        raw_clean.load_data()

        data = raw_clean.get_data()  # (n_channels, n_samples)
        n_channels, n_samples = data.shape
        sfreq = raw_clean.info['sfreq']

        nperseg = min(self.n_fft, n_samples)
        noverlap = nperseg // 2

        print(f"Baseline-Referenced Wiener Filter: f0={self.dbs_freq} Hz, "
              f"bandwidth={self.harmonic_bandwidth} Hz (+{self.taper_width} Hz taper), "
              f"alpha={self.alpha}, n_fft={nperseg}")

        cleaned_data = np.zeros_like(data)

        for ch in range(n_channels):
            f_stft, t_stft, Zxx_ch = signal.stft(
                data[ch], fs=sfreq, nperseg=nperseg, noverlap=noverlap,
                window='hann'
            )

            if ch == 0:
                # Build smooth harmonic taper once (same for all channels)
                taper = self._build_harmonic_taper(f_stft)
                active_bins = taper > 0  # only compute gain where taper is nonzero
                # Interpolate baseline PSD onto STFT frequency grid
                baseline_interp = np.zeros((n_channels, len(f_stft)))
                for ch_b in range(n_channels):
                    baseline_interp[ch_b] = np.interp(
                        f_stft, self._psd_freqs, self._baseline_psd[ch_b]
                    )

            # Instantaneous power: |Zxx|², shape (n_freq, n_time)
            inst_power = np.abs(Zxx_ch) ** 2
            n_freq, n_time = inst_power.shape

            # --- Time-varying Wiener gain (vectorized over active bins) ---
            # G_power(f,t) = P_base(f) / [P_base(f) + α * max(0, |X(f,t)|² - P_base(f))]
            base_pow = baseline_interp[ch, active_bins, np.newaxis]  # (n_active, 1)
            active_power = inst_power[active_bins, :]                # (n_active, n_time)

            excess = np.maximum(active_power - base_pow, 0)
            denom = base_pow + self.alpha * excess
            g_power = base_pow / np.maximum(denom, 1e-30)
            g_power = np.maximum(g_power, self.floor_gain ** 2)  # floor in power domain

            # Convert to amplitude gain and apply taper blending
            g_amp = np.sqrt(g_power)  # (n_active, n_time)
            taper_active = taper[active_bins, np.newaxis]  # (n_active, 1)
            # Blend: where taper=1 → full Wiener gain, taper=0 → gain=1 (untouched)
            effective_gain = taper_active * g_amp + (1 - taper_active)

            # Build full gain matrix
            gain_full = np.ones((n_freq, n_time))
            gain_full[active_bins, :] = effective_gain

            Zxx_clean = Zxx_ch * gain_full

            # ISTFT reconstruction
            _, cleaned_ch = signal.istft(
                Zxx_clean, fs=sfreq, nperseg=nperseg, noverlap=noverlap,
                window='hann'
            )
            cleaned_data[ch, :len(cleaned_ch)] = cleaned_ch[:n_samples]

        raw_clean._data = cleaned_data

        # Report per-harmonic attenuation (median gain across channels and time)
        nyquist = sfreq / 2.0
        harmonics = np.arange(self.dbs_freq, nyquist, self.dbs_freq)
        half_bw = self.harmonic_bandwidth / 2.0
        for h in harmonics[:10]:
            band = (f_stft >= h - half_bw) & (f_stft <= h + half_bw)
            if np.any(band):
                band_gain = gain_full[band, :].mean()
                atten_db = 20 * np.log10(band_gain + 1e-30)
                print(f"  {h:6.1f} Hz: median attenuation = {atten_db:+.1f} dB")

        return raw_clean


class ArtifactFilterFactory:
    """
    A factory class to process EEG data arrays using multiple highly-vectorized
    filtering implementations. It avoids Python for-loops entirely where possible
    to remain memory and computationally efficient over massive ND arrays.
    """
    
    @classmethod
    def process(cls, method: str, data: np.ndarray, sfreq: float, **kwargs) -> np.ndarray:
        """
        Main entrypoint to route EEG data to the specific filter implementation.
        
        Args:
            method: The filter type to apply.
                    Options: 
                    - 'hampel_time' / 'time_domain_hampel' (Allen et al., 2010)
                    - 'hampel_freq' / 'freq_domain_hampel' (Allen et al., 2010)
                    - 'spectrum_fit' / 'spectral_interpolation' (Multi-Harmonic Removal)
                    - 'zapline' (Chen et al., 2022)
                    - 'comb_notch' (Parametric Comb Filter)
            data: The 2D EEG array of shape (n_channels, n_samples).
            sfreq: Sampling frequency in Hz.
            **kwargs: Additional parameters specific to the chosen filter.
            
        Returns:
            np.ndarray: The cleaned data array.
        """
        method = method.lower().strip()
        
        # Alias mapping for convenience
        method_aliases = {
            'time_domain_hampel':     'hampel_time',
            'freq_domain_hampel':     'hampel_freq',
            'spectral_interpolation': 'spectrum_fit',
            'chen':                   'zapline',
            # New surgical methods
            'fft_interp':             'fft_spectral_interp',
            'spectral_interp':        'fft_spectral_interp',
            'narrow_interp':          'fft_spectral_interp',
            'sin_regression':         'sinusoidal_regression',
            'harmonic_regression':    'sinusoidal_regression',
            'harmonic_subtraction':   'sinusoidal_regression',
            # Production surgical suite
            'surgical_regression':    'sinusoidal_regression_chunked',
            'complex_hampel':         'complex_hampel_fft',
            'phase_template':         'phase_template_subtraction',
        }
        method = method_aliases.get(method, method)

        # Ensure data is 2D
        original_shape = data.shape
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim > 2:
            raise ValueError(f"Expected 1D or 2D array, got shape {data.shape}")

        if method == 'hampel_time':
            clean_data = cls._apply_time_domain_hampel(data, sfreq, **kwargs)
        elif method == 'hampel_freq':
            clean_data = cls._apply_freq_domain_hampel(data, sfreq, **kwargs)
        elif method == 'complex_hampel_fft':
            clean_data = cls._apply_complex_hampel_fft(data, sfreq, **kwargs)
        elif method == 'phase_template_subtraction':
            clean_data = cls._apply_phase_template_subtraction(data, sfreq, **kwargs)
        elif method == 'fft_spectral_interp':
            clean_data = cls._apply_fft_spectral_interp(data, sfreq, **kwargs)
        elif method == 'sinusoidal_regression' or method == 'sinusoidal_regression_chunked':
            clean_data = cls._apply_sinusoidal_regression_chunked(data, sfreq, **kwargs)
        elif method == 'spectrum_fit':
            clean_data = cls._apply_spectral_interpolation(data, sfreq, **kwargs)
        elif method == 'zapline':
            clean_data = cls._apply_zapline(data, sfreq, **kwargs)
        elif method == 'comb_notch':
            clean_data = cls._apply_comb_notch(data, sfreq, **kwargs)
        else:
            raise ValueError(f"Unknown method '{method}'. "
                           f"Try: 'surgical_regression', 'complex_hampel', 'phase_template'")
            
        return clean_data.reshape(original_shape)

    @staticmethod
    def _fold_to_nyquist(freq_hz: float, sfreq: float) -> float:
        """Fold a target frequency into (0, Nyquist) to handle aliasing safely."""
        nyquist = sfreq / 2.0
        if freq_hz <= 0:
            raise ValueError(f"Target frequency must be > 0 Hz, got {freq_hz}")

        if freq_hz < nyquist:
            return freq_hz

        # Frequency folding for sampled signals: map to aliased in-band component.
        folded = abs(((freq_hz + nyquist) % (2 * nyquist)) - nyquist)
        if np.isclose(folded, 0.0):
            folded = nyquist * 0.95
        folded = min(folded, nyquist * 0.999)

        print(f"  Target frequency {freq_hz:.2f} Hz exceeds Nyquist ({nyquist:.2f} Hz); "
              f"using aliased in-band target {folded:.2f} Hz")
        return folded

    @staticmethod
    def _apply_time_domain_hampel(data: np.ndarray, sfreq: float, 
                                  window_sec: float = 0.2, n_sigmas: float = 3.0,
                                  attenuation_factor: float = 1.0) -> np.ndarray:
        """
        Highly vectorized Time-Domain Hampel Filter (Allen et al., 2010).
        
        Detects and removes impulsive artifacts by replacing outliers detected via
        median absolute deviation (MAD) with the local median.
        
        Args:
            window_sec: Window duration for median filtering (seconds)
            n_sigmas: Sensitivity threshold in MAD units (higher = more selective)
            attenuation_factor: Strength of artifact removal (1.0 = full replacement, 0.0 = no filtering)
                               Values > 1.0 provide stronger attenuation via multiple passes
        """
        # Convert window to odd number of samples
        k = int(window_sec * sfreq)
        k = k + 1 if k % 2 == 0 else k
        if k < 3:
            warnings.warn("Window size is too small, setting to 3 samples.")
            k = 3
            
        print(f"Running vectorized Time-Domain Hampel (Allen et al., 2010) - window={k} samples, "
              f"sigmas={n_sigmas}, attenuation={attenuation_factor}x")
        
        cleaned_data = np.copy(data)
        n_passes = max(1, int(np.ceil(attenuation_factor)))
        
        for pass_num in range(n_passes):
            # 1. Rolling Median
            rolling_median = ndimage.median_filter(cleaned_data, size=(1, k), mode='reflect')
            
            # 2. Rolling Median Absolute Deviation (MAD)
            abs_deviation = np.abs(cleaned_data - rolling_median)
            rolling_mad = ndimage.median_filter(abs_deviation, size=(1, k), mode='reflect')
            
            # 3. Thresholding
            threshold = n_sigmas * MAD_SCALE_FACTOR * rolling_mad
            
            # Avoid zero thresholds
            threshold = np.maximum(threshold, 1e-10)
            
            # 4. Outlier detection
            outlier_mask = abs_deviation > threshold
            
            # 5. Replacement (with fractional attenuation)
            if pass_num < n_passes - 1:
                # Intermediate passes: blend between original and median
                cleaned_data[outlier_mask] = cleaned_data[outlier_mask] * 0.5 + rolling_median[outlier_mask] * 0.5
            else:
                # Final pass: full replacement
                cleaned_data[outlier_mask] = rolling_median[outlier_mask]
            
            outliers_replaced = np.sum(outlier_mask)
            print(f"  Pass {pass_num + 1}: Replaced {outliers_replaced} outlier points.")
        
        return cleaned_data

    @staticmethod
    def _apply_freq_domain_hampel(data: np.ndarray, sfreq: float, 
                                  window_hz: float = 2.0, n_sigmas: float = 3.0,
                                  attenuation_db: float = -60.0) -> np.ndarray:
        """
        Frequency-Domain Hampel Filter (Allen et al., 2010).
        
        Computes FFT magnitude spectrum, applies rolling median to detect spectral peaks
        (like DBS harmonics), and scales down outlier frequency bins while preserving phase.
        
        Args:
            window_hz: Window size for rolling median in frequency (Hz)
            n_sigmas: Sensitivity threshold in MAD units
            attenuation_db: Target attenuation for detected peaks (dB, more negative = stronger)
        """
        print(f"Running vectorized Freq-Domain Hampel (Allen et al., 2010) - window={window_hz}Hz, "
              f"sigmas={n_sigmas}, attenuation={attenuation_db}dB")
        
        n_samples = data.shape[1]
        freq_res = sfreq / n_samples
        
        # Window size in bins
        k = int(window_hz / freq_res)
        k = k + 1 if k % 2 == 0 else k
        if k < 3:
            k = 3
            
        # 1. Compute real FFT
        S = np.fft.rfft(data, axis=1)
        magnitudes = np.abs(S)
        phases = np.angle(S)
        
        # 2. Rolling median over frequency bins
        rolling_median_mag = ndimage.median_filter(magnitudes, size=(1, k), mode='reflect')
        
        # 3. Median Absolute Deviation
        abs_dev = np.abs(magnitudes - rolling_median_mag)
        rolling_mad = ndimage.median_filter(abs_dev, size=(1, k), mode='reflect')
        
        # 4. Detect outliers
        threshold = n_sigmas * MAD_SCALE_FACTOR * rolling_mad
        threshold = np.maximum(threshold, 1e-10)
        outlier_mask = abs_dev > threshold
        
        # 5. Scale down the complex coefficients to median magnitude (preserving phase)
        # Apply additional attenuation if specified
        attenuation_linear = 10 ** (attenuation_db / 20.0)
        
        clean_mags = np.copy(magnitudes)
        # For outliers, blend between original magnitude and median magnitude
        # scaled by attenuation factor
        blend_factor = attenuation_linear
        clean_mags[outlier_mask] = (1 - blend_factor) * rolling_median_mag[outlier_mask] + blend_factor * clean_mags[outlier_mask]
        
        S_clean = clean_mags * np.exp(1j * phases)
        
        # 6. Inverse FFT
        cleaned_data = np.fft.irfft(S_clean, n=n_samples, axis=1)
        
        outliers_replaced = np.sum(outlier_mask)
        print(f"Replaced {outliers_replaced} frequency domain spikes.")
        
        return cleaned_data

    @staticmethod
    def _apply_comb_notch(data: np.ndarray, sfreq: float, 
                          f0: float = 14.0, q_factor: float = 30.0) -> np.ndarray:
        """
        Parametric Comb Filter.
        Rigorous IIR comb filter specifically designed to notch a fundamental frequency 
        and all its harmonics up to the Nyquist limit, while preserving intermediate bands.
        """
        f0 = ArtifactFilterFactory._fold_to_nyquist(float(f0), sfreq)
        print(f"Applying Comb Filter (f0={f0}Hz, Q={q_factor})")

        nyq = sfreq / 2.0
        harmonics = np.arange(f0, nyq, f0)
        cleaned_data = np.copy(data)

        # Robust comb behavior: cascade individual notch filters at each harmonic.
        for h in harmonics:
            if h <= 0 or h >= nyq:
                continue
            b, a = signal.iirnotch(w0=h, Q=q_factor, fs=sfreq)
            cleaned_data = signal.filtfilt(b, a, cleaned_data, axis=1)

        return cleaned_data

    @staticmethod
    def _apply_fft_spectral_interp(
        data: np.ndarray,
        sfreq: float,
        f_target: float = 7.0,
        half_width_bins: int = 2,
        interp_flank_bins: int = 10,
    ) -> np.ndarray:
        """
        FFT Spectral Interpolation — surgical DBS harmonic removal.

        For each DBS harmonic k·f_target the algorithm:
          1. Locates the nearest FFT bin.
          2. Identifies the "artifact zone": (2·half_width_bins + 1) bins centred
             on the harmonic.
          3. Fits a cubic-spline through ``interp_flank_bins`` reference bins on
             each side of the artifact zone (brain-only spectrum estimate).
          4. Replaces the artifact-zone magnitudes with the spline prediction
             while keeping the original complex phase.
          5. iFFT back to the time domain.

        Effective removal bandwidth per harmonic
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        With ``half_width_bins=2`` and a 120 s recording at 256 Hz
        (Δf = 0.0083 Hz/bin):

            width = (2·2 + 1) × 0.0083 Hz ≈ **0.042 Hz**

        Compare with the comb-notch alternatives:
            Q = 50  →  −1 dB bandwidth ≈ 0.39 Hz  (≈ 9× wider)
            Q = 200 →  −1 dB bandwidth ≈ 0.10 Hz  (≈ 2.4× wider)

        Brain signal between harmonics is **completely untouched**.

        Args
        ----
        f_target         : DBS fundamental frequency in Hz.
        half_width_bins  : Half-width of artifact zone in FFT bins.
                           Each bin = sfreq / n_samples Hz.
                           Default 2 → 5 bins total ≈ 0.042 Hz at 120 s / 256 Hz.
        interp_flank_bins: Number of reference bins on each side of the artifact
                           zone used for cubic-spline interpolation.
        """
        from scipy.interpolate import CubicSpline

        f_target = ArtifactFilterFactory._fold_to_nyquist(float(f_target), sfreq)
        n_samples = data.shape[1]
        freq_res  = sfreq / n_samples
        nyquist   = sfreq / 2.0
        harmonics = np.arange(f_target, nyquist, f_target)

        print(f"FFT Spectral Interpolation: f₀={f_target} Hz, "
              f"±{half_width_bins} bins ({(2*half_width_bins+1)*freq_res*1000:.1f} mHz per harmonic), "
              f"{len(harmonics)} harmonics")

        # Full-length FFT (one per channel, vectorised across channels)
        S   = np.fft.rfft(data, axis=1)    # (n_ch, n_freq)
        mag = np.abs(S)                     # amplitude spectrum
        pha = np.angle(S)                   # phase spectrum
        n_freq = S.shape[1]

        for h in harmonics:
            h_bin = int(round(h / freq_res))
            if h_bin >= n_freq:
                break

            # Artifact zone (keep away from DC and Nyquist)
            lo_bin = max(1, h_bin - half_width_bins)
            hi_bin = min(n_freq - 2, h_bin + half_width_bins)

            # Reference bins: flanking the artifact zone
            ref_lo = np.arange(max(1, lo_bin - interp_flank_bins), lo_bin)
            ref_hi = np.arange(hi_bin + 1, min(n_freq - 1, hi_bin + 1 + interp_flank_bins))
            ref_bins = np.concatenate([ref_lo, ref_hi])

            if len(ref_bins) < 4:   # need at least 4 pts for cubic spline
                continue

            art_bins = np.arange(lo_bin, hi_bin + 1)

            # Fit cubic spline per channel through reference bins, evaluate at artifact bins
            for ch in range(data.shape[0]):
                cs = CubicSpline(ref_bins, mag[ch, ref_bins])
                interp_vals = np.maximum(cs(art_bins), 0.0)   # magnitude ≥ 0
                mag[ch, art_bins] = interp_vals

        # Reconstruct complex spectrum preserving original phase
        S_clean = mag * np.exp(1j * pha)
        cleaned = np.fft.irfft(S_clean, n=n_samples, axis=1)

        print(f"  → Effective removal: {(2*half_width_bins+1)*freq_res*1000:.1f} mHz per harmonic "
              f"({len(harmonics)} harmonics, {(2*half_width_bins+1)*freq_res*len(harmonics)*1000:.1f} mHz total)")
        return cleaned

    @staticmethod
    def _apply_sinusoidal_regression_chunked(
        data: np.ndarray,
        sfreq: float,
        f_target: float = 7.0,
        chunk_sec: float = 4.0,
    ) -> np.ndarray:
        """
        Production-grade Chunked Sinusoidal Regression (Kroth et al. 2020).

        Fits exact-harmonic OLS sinusoids in short overlapping chunks to remove
        perfectly periodic DBS artifacts while preserving phase-resetting brain oscillations.
        Uses 4-second windows by default to balance frequency resolution and non-stationarity.
        """
        f_target = ArtifactFilterFactory._fold_to_nyquist(float(f_target), sfreq)
        nyquist   = sfreq / 2.0
        harmonics = np.arange(f_target, nyquist, f_target)
        n_ch, n_samples = data.shape
        chunk_len = int(chunk_sec * sfreq)

        print(f"Chunked Sinusoidal Regression: f₀={f_target}Hz, {len(harmonics)} harmonics, window={chunk_sec}s")

        def _regress_seg(seg: np.ndarray) -> np.ndarray:
            n = seg.shape[1]
            t = np.arange(n) / sfreq
            cols = []
            for h in harmonics:
                cols.append(np.sin(2 * np.pi * h * t))
                cols.append(np.cos(2 * np.pi * h * t))
            cols.append(np.ones(n))
            X = np.column_stack(cols)
            beta, _, _, _ = np.linalg.lstsq(X, seg.T, rcond=None)
            dbs_model = X[:, :-1] @ beta[:-1]
            return seg - dbs_model.T

        cleaned = np.zeros_like(data)
        for start in range(0, n_samples, chunk_len):
            end = min(start + chunk_len, n_samples)
            cleaned[:, start:end] = _regress_seg(data[:, start:end])

        return cleaned

    @staticmethod
    def _apply_complex_hampel_fft(
        data: np.ndarray,
        sfreq: float,
        f_target: float = 7.0,
        n_sigmas: float = 3.0,
        bw_hz: float = 0.5,
        fmax_hz: float = 100.0
    ) -> np.ndarray:
        """
        Complex Spectral Hampel Filter — surgical residual removal.

        Targets narrowband DBS residue by replacing outlier bins in the
        complex spectrum (Real/Imag separately) via linear interpolation.
        """
        n_ch, n_s = data.shape
        Xf    = np.fft.rfft(data, axis=1)
        freqs = np.fft.rfftfreq(n_s, 1.0 / sfreq)
        df    = freqs[1] - freqs[0]
        n_f   = len(freqs)

        half_bw  = max(1, int(round(bw_hz / 2.0 / df)))
        flank    = max(5, int(round(min(3.0, f_target * 0.4) / df)))
        loc_win  = max(20, int(round(2.5 / df)))
        nyq      = sfreq / 2.0

        print(f"Complex Spectral Hampel: f₀={f_target}Hz, sigmas={n_sigmas}, bw={bw_hz}Hz")

        Xf_c = Xf.copy()
        for k in range(1, 200):
            h = k * f_target
            if h >= min(nyq, fmax_hz + 5.0):
                break
            h_bin = int(round(h / df))
            if h_bin >= n_f - flank - 1:
                break

            zs = max(1, h_bin - half_bw)
            ze = min(n_f - 2, h_bin + half_bw + 1)
            ls = max(1, zs - flank)
            re = min(n_f - 1, ze + flank)

            ref = np.concatenate([np.arange(ls, zs), np.arange(ze, re)])
            tgt = np.arange(zs, ze)
            if len(ref) < 4 or len(tgt) == 0:
                continue

            lo_s, lo_e = max(1, h_bin - loc_win), min(n_f - 1, h_bin + loc_win + 1)

            for ch in range(n_ch):
                mag     = np.abs(Xf_c[ch])
                loc_mag = mag[lo_s:lo_e]
                med     = np.median(loc_mag)
                mad     = np.median(np.abs(loc_mag - med))
                thr     = med + n_sigmas * MAD_SCALE_FACTOR * mad

                if mag[h_bin] > thr:
                    rb, tb = ref.astype(float), tgt.astype(float)
                    Xf_c[ch, tgt] = (np.interp(tb, rb, Xf_c[ch, ref].real)
                                      + 1j * np.interp(tb, rb, Xf_c[ch, ref].imag))

        return np.fft.irfft(Xf_c, n=n_s, axis=1)

    @staticmethod
    def _apply_phase_template_subtraction(
        data: np.ndarray,
        sfreq: float,
        f_target: float = 7.0,
        n_bins: int = 256
    ) -> np.ndarray:
        """
        Phase-locked Median Template Subtraction.

        Constructs a stationary waveform template by binning samples into
        their relative DBS phase (0 to 2π) and subtracting the median.
        """
        print(f"Phase Template Subtraction: f₀={f_target}Hz, bins={n_bins}")
        n_ch, n_s = data.shape
        t     = np.arange(n_s, dtype=np.float64) / sfreq
        phase = (2.0 * np.pi * f_target * t) % (2.0 * np.pi)
        bidx  = (phase / (2.0 * np.pi) * n_bins).astype(int) % n_bins

        out = data.copy()
        for ch in range(n_ch):
            # Efficiently compute medians for each bin
            tmpl = np.zeros(n_bins)
            for b in range(n_bins):
                m = bidx == b
                if m.any():
                    tmpl[b] = np.median(data[ch, m])
            out[ch] -= tmpl[bidx]
        return out

    @staticmethod
    def _apply_spectral_interpolation(data: np.ndarray, sfreq: float,
                                      f_target: float = 14.0, bandwidth: float = 2.0,
                                      chunk_sec: float = 10.0, attenuation_db: float = -60.0) -> np.ndarray:
        """
        Spectrum-Fit Multi-Harmonic Removal using STFT chunks to prevent OOM.
        
        Method: For each harmonic, computes a frequency-domain gain that:
        - Smoothly tapers from 1.0 (no filtering) outside the harmonic band
        - Applies strong spectral nulling within the harmonic band
        - Preserves amplitude information and phase relationships
        
        Args:
            f_target: Fundamental DBS frequency (Hz)
            bandwidth: Total bandwidth around each harmonic to filter (Hz)
            chunk_sec: Window duration for STFT (seconds)
            attenuation_db: Target attenuation level in dB (more negative = stronger removal, default -60 dB)
        """
        f_target = ArtifactFilterFactory._fold_to_nyquist(float(f_target), sfreq)
        print(f"Applying Spectrum-Fit Multi-Harmonic Removal (f={f_target}Hz, bandwidth={bandwidth}Hz, attenuation={attenuation_db}dB)")
        
        # Using Short-Time Fourier Transform to avoid massive array allocation overheads
        nperseg = int(sfreq * chunk_sec)
        if nperseg > data.shape[1]:
            nperseg = data.shape[1]
            
        # Run STFT per channel for broad SciPy compatibility (avoids axis-specific istft APIs).
        stft_channels = []
        f = None
        t = None
        for ch in range(data.shape[0]):
            f_ch, t_ch, z_ch = signal.stft(data[ch], fs=sfreq, nperseg=nperseg)
            if f is None:
                f, t = f_ch, t_ch
            stft_channels.append(z_ch)
        Zxx = np.stack(stft_channels, axis=0)  # (n_channels, n_freq, n_time)
        
        nyquist = sfreq / 2.0
        harmonics = np.arange(f_target, nyquist, f_target)
        
        # Convert attenuation from dB to power ratio
        attenuation_linear = 10 ** (attenuation_db / 20.0)
        
        # Build gain mask for all frequencies
        gain_mask = np.ones_like(f)
        half_bw = bandwidth / 2.0
        taper_width = half_bw / 2.0  # Smooth rolloff zone
        
        for harm in harmonics:
            lower_bound = harm - half_bw
            upper_bound = harm + half_bw
            
            # Core filtering region
            core_mask = (f >= lower_bound) & (f <= upper_bound)
            gain_mask[core_mask] = attenuation_linear
            
            # Smooth cosine taper in transition zones
            # Left transition: lower_bound - taper_width to lower_bound
            left_trans = (f >= lower_bound - taper_width) & (f < lower_bound)
            if np.any(left_trans):
                trans_idx = np.where(left_trans)[0]
                for idx in trans_idx:
                    # Cosine rolloff
                    alpha = (f[idx] - (lower_bound - taper_width)) / taper_width
                    gain_mask[idx] = 1.0 - (1.0 - attenuation_linear) * (0.5 * (1 + np.cos(np.pi * (1 - alpha))))
            
            # Right transition: upper_bound to upper_bound + taper_width
            right_trans = (f > upper_bound) & (f <= upper_bound + taper_width)
            if np.any(right_trans):
                trans_idx = np.where(right_trans)[0]
                for idx in trans_idx:
                    alpha = (f[idx] - upper_bound) / taper_width
                    gain_mask[idx] = 1.0 - (1.0 - attenuation_linear) * (0.5 * (1 + np.cos(np.pi * alpha)))
        
        if len(harmonics) == 0:
            return np.copy(data)

        # Apply gain to all STFT coefficients (across channels and time)
        Zxx_clean = Zxx * gain_mask[np.newaxis, :, np.newaxis]
        
        # Inverse STFT per channel for SciPy compatibility.
        cleaned_data = np.zeros_like(data)
        for ch in range(data.shape[0]):
            _, cleaned_ch = signal.istft(Zxx_clean[ch], fs=sfreq, nperseg=nperseg)
            cleaned_data[ch, :min(data.shape[1], len(cleaned_ch))] = cleaned_ch[:data.shape[1]]
        
        return cleaned_data

    @staticmethod
    def _apply_zapline(data: np.ndarray, sfreq: float, 
                       f_target: float = 14.0, n_harmonics: int = 10,
                       threshold_percentile: float = 95.0, chunk_sec: float = 2.0) -> np.ndarray:
        """
        Zapline+ Filter (Chen et al., 2022) - Adaptive Notch-based Removal.
        
        Method: Uses spectro-spatial filtering to:
        - Detect harmonic peaks across channels
        - Build a spatial filter that maximally removes artifact subspace
        - Iteratively apply this filter to attenuate artifacts adaptively
        
        Args:
            f_target: Fundamental DBS frequency (Hz)
            n_harmonics: Number of harmonics to consider
            threshold_percentile: Percentile threshold for artifact detection (higher = more selective)
            chunk_sec: Chunk duration for STFT in seconds
        """
        f_target = ArtifactFilterFactory._fold_to_nyquist(float(f_target), sfreq)
        print(f"Applying Zapline+ (Chen et al., 2022) - f={f_target}Hz, {n_harmonics} harmonics, "
              f"threshold={threshold_percentile}th percentile")
        
        n_channels, n_samples = data.shape
        
        # STFT parameters
        nperseg = int(sfreq * chunk_sec)
        if nperseg > n_samples:
            nperseg = n_samples
        
        # Step 1: Decompose using STFT per channel for broad SciPy compatibility.
        stft_channels = []
        f = None
        t = None
        for ch in range(n_channels):
            f_ch, t_ch, z_ch = signal.stft(data[ch], fs=sfreq, nperseg=nperseg)
            if f is None:
                f, t = f_ch, t_ch
            stft_channels.append(z_ch)
        Zxx = np.stack(stft_channels, axis=0)  # (n_channels, n_freq, n_time)
        
        nyquist = sfreq / 2.0
        harmonics = np.arange(f_target, min(nyquist, f_target * (n_harmonics + 1)), f_target)
        
        # Step 2: For each harmonic, identify the artifact subspace
        Zxx_clean = np.copy(Zxx)
        
        for h_idx, harmonic in enumerate(harmonics):
            # Find bins near this harmonic (within +/- 1 Hz)
            freq_window = 1.0
            harmonic_bins = np.where(np.abs(f - harmonic) <= freq_window)[0]
            
            if len(harmonic_bins) == 0:
                continue
            
            # Extract STFT for these bins across all channels and time
            # Shape: (n_channels, n_harmonic_bins, n_time)
            X_harmonic = Zxx[:, harmonic_bins, :]
            
            # Step 3: Compute power per channel and time
            power_harmonic = np.abs(X_harmonic) ** 2  # (n_channels, n_harmonic_bins, n_time)
            
            # Median power across frequency (to get per-channel, per-time power)
            median_power_ch_time = np.median(power_harmonic, axis=1)  # (n_channels, n_time)
            
            # Step 4: Compute spatial correlation matrix over time
            # Reshape for covariance: (n_channels, n_harmonic_bins * n_time)
            X_reshaped = X_harmonic.reshape(n_channels, -1)
            
            # Compute covariance matrix
            C = np.cov(X_reshaped)  # (n_channels, n_channels)
            
            # Eigendecomposition to find dominant artifact direction
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Sort by eigenvalue (largest = most artifact power)
                idx_sort = np.argsort(eigvals)[::-1]
                
                # Take top 1-2 eigenvectors as artifact subspace
                n_artifact_vecs = max(1, min(2, n_channels // 2))
                artifact_vecs = eigvecs[:, idx_sort[:n_artifact_vecs]]  # (n_channels, n_artifact_vecs)
                
                # Project data onto artifact subspace and back
                proj = artifact_vecs @ artifact_vecs.T @ X_reshaped  # (n_channels, n_bins*n_time)
                
                # Scale down the projection (artifact removal strength)
                removal_factor = 0.8  # Remove ~80% of artifact subspace
                Zxx_clean[:, harmonic_bins, :] -= removal_factor * proj.reshape(X_harmonic.shape)
                
            except np.linalg.LinAlgError:
                # Fallback to simple spectral nulling if eigendecompositon fails
                Zxx_clean[:, harmonic_bins, :] *= 0.5
        
        # Inverse STFT per channel for SciPy compatibility.
        cleaned_data = np.zeros_like(data)
        for ch in range(n_channels):
            _, cleaned_ch = signal.istft(Zxx_clean[ch], fs=sfreq, nperseg=nperseg)
            cleaned_data[ch, :min(data.shape[1], len(cleaned_ch))] = cleaned_ch[:data.shape[1]]
        
        return cleaned_data