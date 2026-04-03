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

    Theory:
        At each frequency bin f, the Wiener-optimal gain is:
            G(f) = P_baseline(f) / [P_baseline(f) + P_noise(f)]
        where P_baseline is estimated from clean recordings, and P_noise is the
        excess power in the DBS recording above baseline. Away from artifact
        harmonics, P_noise ~ 0 so G ~ 1 (no attenuation). At harmonics,
        G < 1 proportional to artifact strength.
    """

    def __init__(self, baseline_raw, dbs_freq: float = 7.0,
                 harmonic_bandwidth: float = 1.5, n_fft: int = 4096,
                 floor_db: float = -40.0):
        """
        Args:
            baseline_raw: MNE Raw object from a clean baseline recording
                          (e.g., XUAWAKEPRE or XUSLEEP). Must share the same
                          channel set and sampling rate as the DBS data.
            dbs_freq: Fundamental DBS stimulation frequency in Hz.
            harmonic_bandwidth: Total bandwidth (Hz) around each harmonic
                                where the Wiener gain is applied. Outside this
                                band, G(f) = 1 (untouched). Default 1.5 Hz
                                is narrow enough to preserve theta/alpha.
            n_fft: FFT length for PSD and STFT. Longer = finer frequency
                   resolution for separating narrow DBS peaks from brain.
            floor_db: Minimum gain in dB to prevent complete nulling.
                      Default -40 dB removes 99.99% of artifact power
                      while avoiding numerical zeros.
        """
        self.dbs_freq = dbs_freq
        self.harmonic_bandwidth = harmonic_bandwidth
        self.n_fft = n_fft
        self.floor_gain = 10 ** (floor_db / 20.0)

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

    def _build_harmonic_mask(self, freqs: np.ndarray) -> np.ndarray:
        """Boolean mask: True at frequency bins within harmonic_bandwidth of any DBS harmonic."""
        nyquist = self._sfreq / 2.0
        harmonics = np.arange(self.dbs_freq, nyquist, self.dbs_freq)
        half_bw = self.harmonic_bandwidth / 2.0

        mask = np.zeros(len(freqs), dtype=bool)
        for h in harmonics:
            mask |= (freqs >= h - half_bw) & (freqs <= h + half_bw)
        return mask

    def filter(self, dbs_raw):
        """
        Apply baseline-referenced Wiener spectral gating to remove DBS artifacts.

        Pipeline:
        1. STFT the DBS recording into time-frequency tiles.
        2. Compute median instantaneous power per frequency bin (robust to transients).
        3. At harmonic bins only, compute Wiener gain:
           G(f) = P_baseline(f) / max(P_baseline(f), P_dbs(f))
        4. Multiply STFT coefficients by G(f) and reconstruct via ISTFT.

        Signal preservation guarantee:
        - At non-harmonic frequencies, G(f) = 1.0 exactly — zero modification.
        - At harmonic frequencies, the gain retains the baseline-level power
          (i.e., the brain's natural contribution) and removes only the excess.
        - The harmonic_bandwidth (default 1.5 Hz) is narrower than the 7 Hz
          spacing between harmonics, so inter-harmonic brain activity is untouched.

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
              f"bandwidth={self.harmonic_bandwidth} Hz, n_fft={nperseg}")

        # STFT: (n_channels, n_freq, n_time)
        f_stft, t_stft, Zxx = signal.stft(
            data, fs=sfreq, nperseg=nperseg, noverlap=noverlap,
            window='hann', axis=1
        )

        harmonic_mask = self._build_harmonic_mask(f_stft)

        # Interpolate baseline PSD onto STFT frequency grid
        # shape: (n_channels, n_stft_freqs)
        baseline_interp = np.zeros((n_channels, len(f_stft)))
        for ch in range(n_channels):
            baseline_interp[ch] = np.interp(f_stft, self._psd_freqs, self._baseline_psd[ch])

        # Instantaneous power from STFT: |Zxx|^2
        inst_power = np.abs(Zxx) ** 2

        # Median power over time for robust gain estimation: (n_channels, n_freq)
        median_power = np.median(inst_power, axis=2)

        # Build gain: G(f) = 1.0 everywhere, then attenuate at harmonic bins
        gain = np.ones((n_channels, len(f_stft)))
        harm_idx = np.where(harmonic_mask)[0]

        for idx in harm_idx:
            dbs_pow = median_power[:, idx]       # (n_channels,)
            base_pow = baseline_interp[:, idx]   # (n_channels,)

            # Only attenuate where DBS power exceeds baseline
            excess = dbs_pow > base_pow
            ratio = np.ones(n_channels)
            ratio[excess] = base_pow[excess] / np.maximum(dbs_pow[excess], 1e-30)
            ratio = np.maximum(ratio, self.floor_gain)
            gain[:, idx] = ratio

        # Apply gain: broadcast (n_ch, n_freq, 1) over (n_ch, n_freq, n_time)
        Zxx_clean = Zxx * gain[:, :, np.newaxis]

        # ISTFT reconstruction
        _, cleaned_data = signal.istft(
            Zxx_clean, fs=sfreq, nperseg=nperseg, noverlap=noverlap,
            window='hann', axis=1
        )
        cleaned_data = cleaned_data[:, :n_samples]

        raw_clean._data = cleaned_data

        # Report per-harmonic attenuation for first 10 harmonics
        nyquist = sfreq / 2.0
        harmonics = np.arange(self.dbs_freq, nyquist, self.dbs_freq)
        half_bw = self.harmonic_bandwidth / 2.0
        for h in harmonics[:10]:
            band = (f_stft >= h - half_bw) & (f_stft <= h + half_bw)
            if np.any(band):
                avg_gain_db = 20 * np.log10(np.mean(gain[:, band]) + 1e-30)
                print(f"  {h:6.1f} Hz: avg gain = {avg_gain_db:+.1f} dB")

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
                    Options: 'hampel_time', 'hampel_freq', 'comb_notch', 'spectral_interpolation'
            data: The 2D EEG array of shape (n_channels, n_samples).
            sfreq: Sampling frequency in Hz.
            **kwargs: Additional parameters specific to the chosen filter.
            
        Returns:
            np.ndarray: The cleaned data array.
        """
        method = method.lower().strip()
        
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
        elif method == 'comb_notch':
            clean_data = cls._apply_comb_notch(data, sfreq, **kwargs)
        elif method == 'spectral_interpolation':
            clean_data = cls._apply_spectral_interpolation(data, sfreq, **kwargs)
        else:
            raise ValueError(f"Unknown method '{method}'.")
            
        return clean_data.reshape(original_shape)

    @staticmethod
    def _apply_time_domain_hampel(data: np.ndarray, sfreq: float, 
                                  window_sec: float = 0.2, n_sigmas: float = 3.0) -> np.ndarray:
        """
        Highly vectorized Time-Domain Hampel Filter.
        Instead of sliding loops, uses `scipy.ndimage.median_filter` which runs
        efficiently in C.
        """
        # Convert window to odd number of samples
        k = int(window_sec * sfreq)
        k = k + 1 if k % 2 == 0 else k
        if k < 3:
            warnings.warn("Window size is too small, setting to 3 samples.")
            k = 3
            
        print(f"Running vectorized Time-Domain Hampel (window={k} samples, sigmas={n_sigmas})")
        
        # 1. Rolling Median
        rolling_median = ndimage.median_filter(data, size=(1, k), mode='reflect')
        
        # 2. Rolling Median Absolute Deviation (MAD)
        abs_deviation = np.abs(data - rolling_median)
        rolling_mad = ndimage.median_filter(abs_deviation, size=(1, k), mode='reflect')
        
        # 3. Thresholding
        threshold = n_sigmas * MAD_SCALE_FACTOR * rolling_mad
        
        # Avoid zero thresholds
        threshold = np.maximum(threshold, 1e-10)
        
        # 4. Outlier detection
        outlier_mask = abs_deviation > threshold
        
        # 5. Replacement
        cleaned_data = np.copy(data)
        cleaned_data[outlier_mask] = rolling_median[outlier_mask]
        
        outliers_replaced = np.sum(outlier_mask)
        print(f"Replaced {outliers_replaced} outlier points overall.")
        
        return cleaned_data

    @staticmethod
    def _apply_freq_domain_hampel(data: np.ndarray, sfreq: float, 
                                  window_hz: float = 2.0, n_sigmas: float = 3.0) -> np.ndarray:
        """
        Frequency-Domain Hampel Filter.
        Computes FFT, applies rolling median to magnitudes to find spikes (like DBS artifacts),
        scales down the outlier complex bins to the rolling median magnitude, computing IFFT.
        """
        print(f"Running vectorized Freq-Domain Hampel (window={window_hz}Hz, sigmas={n_sigmas})")
        
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
        # S_clean = median_mag * e^(j * phase)
        clean_mags = np.copy(magnitudes)
        clean_mags[outlier_mask] = rolling_median_mag[outlier_mask]
        
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
        print(f"Applying Comb Filter (f0={f0}Hz, Q={q_factor})")
        
        # Design IIR comb filter
        # w0 is the normalized frequency (w0 = f0 / (fs/2))
        nyq = sfreq / 2.0
        w0 = f0 / nyq
        
        # Create numerator and denominator polynomials
        b, a = signal.iircomb(w0, q_factor, ftype='notch')
        
        # Zero-phase forward and reverse filtering
        cleaned_data = signal.filtfilt(b, a, data, axis=1)
        return cleaned_data

    @staticmethod
    def _apply_spectral_interpolation(data: np.ndarray, sfreq: float,
                                      f_target: float = 14.0, bandwidth: float = 2.0,
                                      chunk_sec: float = 10.0) -> np.ndarray:
        """
        Multi-Taper inspired Spectral Interpolation using STFT chunks to prevent OOM.
        Zeros out the fundamental and all harmonics, interpolating the signal from
        adjacent frequency bins. Excellent for preserving brain signals with stationary noise.
        """
        print(f"Applying Spectral Interpolation (f={f_target}Hz +/- {bandwidth/2}Hz)")
        
        # Using Short-Time Fourier Transform to avoid massive array allocation overheads
        nperseg = int(sfreq * chunk_sec)
        if nperseg > data.shape[1]:
            nperseg = data.shape[1]
            
        f, t, Zxx = signal.stft(data, fs=sfreq, nperseg=nperseg, axis=1)
        
        nyquist = sfreq / 2.0
        harmonics = np.arange(f_target, nyquist, f_target)
        
        db_half = bandwidth / 2.0
        
        Zxx_clean = np.copy(Zxx)
        
        # Over each target harmonic, linear interpolate across the gap in freq bins
        for harm in harmonics:
            # Mask bins within the bandwidth of the harmonic
            lower_bound = harm - db_half
            upper_bound = harm + db_half
            
            mask = (f >= lower_bound) & (f <= upper_bound)
            idx_mask = np.where(mask)[0]
            
            if len(idx_mask) == 0:
                continue
                
            # Nearest neighbor bins for interpolation
            left_idx = max(0, idx_mask[0] - 1)
            right_idx = min(len(f) - 1, idx_mask[-1] + 1)
            
            # For each time column, and each channel, perform linear interpolation across the freq gap
            left_vals = Zxx[:, left_idx, :]   # shape: (channels, time)
            right_vals = Zxx[:, right_idx, :] # shape: (channels, time)
            
            # Number of points to interpolate
            n_points = len(idx_mask)
            
            # Simple linear interpolation between left_vals and right_vals across the frequency bins
            # Using vectorized calculation
            step = (right_vals - left_vals) / (n_points + 1)
            for i, idx in enumerate(idx_mask):
                Zxx_clean[:, idx, :] = left_vals + step * (i + 1)
                
        # Inverse STFT
        _, cleaned_data = signal.istft(Zxx_clean, fs=sfreq, nperseg=nperseg, axis=1)
        
        # Edge case: Istft might return a slightly larger array or smaller array due to padding
        # Slice to the original exact size
        cleaned_data = cleaned_data[:, :data.shape[1]]
        
        return cleaned_data
