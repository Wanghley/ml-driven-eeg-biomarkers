"""
Synthetic EEG Signal Generator — 10-20 Configuration

Generates realistic multi-channel EEG with three independent artifact sources:

  1. DBS artifact  — 7 Hz fundamental + harmonics (large amplitude, global)
  2. Eye artifacts — blinks (sharp Gaussian transients) + saccades (slow sinusoidal)
  3. Muscle artifacts — bandlimited EMG bursts on temporal/frontal channels

All signals are returned in µV.

Typical usage
-------------
>>> from src.synthetic_eeg import SyntheticEEG
>>> gen = SyntheticEEG(sfreq=256.0, duration=120.0, seed=42)
>>> eeg = gen.generate(dbs_freq=7.0)
>>> raw = gen.to_mne_raw(eeg, which='mixed')
"""

import numpy as np
from scipy import signal
import mne

# Standard 19-channel 10-20 layout
CHANNELS_1020 = [
    'Fp1', 'Fp2',
    'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'O2',
]


class SyntheticEEG:
    """
    Synthetic 19-channel EEG generator with controllable artifact sources.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz (default 256).
    duration : float
        Signal duration in seconds (default 120).
    seed : int
        Random seed for reproducibility.
    """

    N_CHANNELS = 19
    CHANNELS = CHANNELS_1020

    # --- topographic channel groups (indices into CHANNELS_1020) ---
    # Fp1=0, Fp2=1, F7=2, F3=3, Fz=4, F4=5, F8=6
    FRONTAL_IDX  = [0, 1, 2, 3, 4, 5, 6]
    # T3=7, T4=11, T5=12, T6=16
    TEMPORAL_IDX = [7, 11, 12, 16]
    # C3=8, Cz=9, C4=10
    CENTRAL_IDX  = [8, 9, 10]
    # P3=13, Pz=14, P4=15
    PARIETAL_IDX = [13, 14, 15]
    # O1=17, O2=18
    OCCIPITAL_IDX = [17, 18]

    def __init__(self, sfreq: float = 256.0, duration: float = 120.0, seed: int = 42):
        self.sfreq = float(sfreq)
        self.duration = float(duration)
        self.n_samples = int(self.sfreq * self.duration)
        self.times = np.arange(self.n_samples) / self.sfreq
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Internal signal primitives
    # ------------------------------------------------------------------

    def _pink_noise(self, n: int, amplitude: float = 1.0) -> np.ndarray:
        """1/f (pink) noise via FFT shaping."""
        white = self.rng.standard_normal(n)
        freqs = np.fft.rfftfreq(n)
        freqs[0] = 1e-9
        fft_in = np.fft.rfft(white)
        fft_in *= 1.0 / np.sqrt(freqs)
        fft_in[0] = 0.0  # remove DC
        return amplitude * np.fft.irfft(fft_in, n=n)

    def _oscillation(self, freq: float, amplitude: float, phase: float | None = None) -> np.ndarray:
        """Pure sinusoidal oscillation at a given frequency."""
        if phase is None:
            phase = self.rng.uniform(0, 2 * np.pi)
        return amplitude * np.sin(2 * np.pi * freq * self.times + phase)

    # ------------------------------------------------------------------
    # Brain signal
    # ------------------------------------------------------------------

    def generate_brain(self) -> np.ndarray:
        """
        Generate realistic 19-channel brain EEG.

        Each channel is a superposition of:
          - 1/f background (pink noise)
          - Delta   0.5–4  Hz  — global slow waves
          - Theta   4–8    Hz  — frontal/central emphasis
          - Alpha   8–13   Hz  — occipital dominant (~10 Hz peak)
          - Beta   13–30   Hz  — frontal/central
          - Gamma  30–80   Hz  — weak, global

        Returns
        -------
        np.ndarray, shape (19, n_samples), units µV
        """
        data = np.zeros((self.N_CHANNELS, self.n_samples))

        for ch in range(self.N_CHANNELS):
            s = self._pink_noise(self.n_samples, amplitude=2.5)

            # Delta
            s += self._oscillation(1.5 + self.rng.uniform(-0.5, 0.5), 2.0)

            # Theta — stronger on frontal/central
            theta_amp = 2.0 if ch in self.FRONTAL_IDX + self.CENTRAL_IDX else 0.8
            s += self._oscillation(6.0 + self.rng.uniform(-0.5, 0.5), theta_amp)

            # Alpha — occipital dominant
            alpha_amp = 5.0 if ch in self.OCCIPITAL_IDX else 1.2
            s += self._oscillation(10.0 + self.rng.uniform(-0.5, 0.5), alpha_amp)

            # Beta
            beta_amp = 1.0 if ch in self.FRONTAL_IDX + self.CENTRAL_IDX else 0.4
            s += self._oscillation(20.0 + self.rng.uniform(-2, 2), beta_amp)

            # Gamma
            s += self._oscillation(40.0 + self.rng.uniform(-3, 3), 0.25)

            data[ch] = s

        return data  # µV

    # ------------------------------------------------------------------
    # DBS artifact
    # ------------------------------------------------------------------

    def generate_dbs(self, f_dbs: float = 7.0, n_harmonics: int = 20,
                     amplitude: float = 80.0) -> np.ndarray:
        """
        Generate DBS stimulation artifact as a harmonic series.

        In scalp EEG, DBS pulses (60–90 µs) are capacitively coupled and
        band-limited by electrode impedances, appearing as a strong oscillation
        at f_dbs with progressively weaker harmonics.  We model this as:

            x_dbs(t) = Σ_{k=1}^{N} (A / k) · sin(2π · k · f_dbs · t + φ_k)

        giving a rich harmonic spectrum with 1/k amplitude decay, similar to
        what is observed in clinical EEG recordings from DBS patients.

        Parameters
        ----------
        f_dbs : float
            DBS fundamental frequency in Hz.
        n_harmonics : int
            Number of harmonics to include (up to Nyquist).
        amplitude : float
            Amplitude of the fundamental in µV.

        Returns
        -------
        np.ndarray, shape (19, n_samples), units µV
        """
        nyquist = self.sfreq / 2.0
        single_channel = np.zeros(self.n_samples)

        for k in range(1, n_harmonics + 1):
            freq = k * f_dbs
            if freq >= nyquist:
                break
            phase = self.rng.uniform(0, 2 * np.pi)
            single_channel += (amplitude / k) * np.sin(2 * np.pi * freq * self.times + phase)

        # DBS artifact is global; slightly stronger at central/parietal channels
        weights = np.ones(self.N_CHANNELS)
        for i in self.CENTRAL_IDX + self.PARIETAL_IDX:
            weights[i] = 1.15
        for i in self.OCCIPITAL_IDX:
            weights[i] = 0.85

        return single_channel[np.newaxis, :] * weights[:, np.newaxis]

    # ------------------------------------------------------------------
    # Eye movement artifacts
    # ------------------------------------------------------------------

    def generate_eyes(self) -> np.ndarray:
        """
        Generate eye movement artifacts: blinks and horizontal saccades.

        Blinks
          - Gaussian transients, ~250 ms, ~15/min
          - Strongest on Fp1/Fp2; fall off with distance from eyes

        Saccades
          - Slow sinusoidal shifts at ~0.3 Hz on Fp1 vs Fp2 (opposite polarity)
          - Propagate weakly to lateral frontal channels

        Returns
        -------
        np.ndarray, shape (19, n_samples), units µV
        """
        n = self.n_samples
        data = np.zeros((self.N_CHANNELS, n))

        # --- Blinks ---
        blink_rate = 15.0 / 60.0     # blinks per second
        blink_interval_mean = int(self.sfreq / blink_rate)
        blink_width = int(0.25 * self.sfreq)  # 250 ms FWHM
        sigma = blink_width / 6.0

        t0 = 0
        while t0 < n - blink_width:
            jitter = self.rng.integers(-blink_interval_mean // 3, blink_interval_mean // 3)
            t0 = max(0, t0 + blink_interval_mean + jitter)
            if t0 + blink_width >= n:
                break

            t_local = np.arange(blink_width)
            blink_shape = 120.0 * np.exp(-((t_local - blink_width / 2) ** 2) / (2 * sigma ** 2))

            # Channel-specific weights (fall off from Fp1/Fp2)
            blink_weights = {
                0: 1.00,  # Fp1
                1: 1.00,  # Fp2
                2: 0.50,  # F7
                3: 0.40,  # F3
                4: 0.35,  # Fz
                5: 0.40,  # F4
                6: 0.50,  # F8
            }
            for ch, w in blink_weights.items():
                end = min(t0 + blink_width, n)
                data[ch, t0:end] += w * blink_shape[:end - t0]

        # --- Horizontal saccades ---
        saccade_freq = 0.3   # Hz
        saccade_amp = 50.0   # µV
        saccade = saccade_amp * np.sin(2 * np.pi * saccade_freq * self.times)

        saccade_weights = {
            0:  1.00,   # Fp1  (+)
            1: -1.00,   # Fp2  (−) — opposite polarity for horizontal
            2:  0.55,   # F7
            3:  0.30,   # F3
            4:  0.00,   # Fz — midline, minimal
            5: -0.30,   # F4
            6: -0.55,   # F8
        }
        for ch, w in saccade_weights.items():
            data[ch] += w * saccade

        return data

    # ------------------------------------------------------------------
    # Muscle (EMG) artifacts
    # ------------------------------------------------------------------

    def generate_muscle(self) -> np.ndarray:
        """
        Generate EMG/muscle artifacts: high-frequency bursts on temporal & outer
        frontal channels.

        Each burst is bandlimited white noise (20–80 Hz), shaped with a
        Hann window to avoid sharp edges.  Burst duration: 0.5–2 s.

        Returns
        -------
        np.ndarray, shape (19, n_samples), units µV
        """
        n = self.n_samples
        data = np.zeros((self.N_CHANNELS, n))

        # Channels susceptible to EMG: temporal and outer frontal
        emg_channels = self.TEMPORAL_IDX + [2, 6]  # T3, T4, T5, T6, F7, F8

        # Bandpass filter for EMG (20–80 Hz)
        sos_emg = signal.butter(
            4, [20.0, min(80.0, self.sfreq / 2.0 - 1.0)],
            btype='bandpass', fs=self.sfreq, output='sos'
        )

        n_bursts_per_ch = max(1, int(self.duration / 8.0))

        for ch in emg_channels:
            for _ in range(n_bursts_per_ch):
                burst_dur = self.rng.uniform(0.5, 2.0)  # seconds
                burst_len = int(burst_dur * self.sfreq)
                t_start = self.rng.integers(0, max(1, n - burst_len))

                raw_burst = self.rng.standard_normal(burst_len)
                emg_burst = signal.sosfilt(sos_emg, raw_burst)

                # Hann envelope to avoid sharp edges
                envelope = np.hanning(burst_len)
                amp = self.rng.uniform(15.0, 35.0)
                shaped = amp * envelope * emg_burst

                t_end = min(t_start + burst_len, n)
                data[ch, t_start:t_end] += shaped[:t_end - t_start]

        return data

    # ------------------------------------------------------------------
    # Top-level API
    # ------------------------------------------------------------------

    def generate(self, dbs_freq: float = 7.0,
                 include_dbs: bool = True,
                 include_eyes: bool = True,
                 include_muscle: bool = True) -> dict:
        """
        Generate complete synthetic EEG with all artifact sources.

        Parameters
        ----------
        dbs_freq : float
            DBS fundamental frequency in Hz.
        include_dbs / include_eyes / include_muscle : bool
            Toggle individual artifact sources.

        Returns
        -------
        dict with keys:
            'brain'   — clean brain-only signal (19, n_samples) µV
            'dbs'     — DBS artifact only         (19, n_samples) µV
            'eyes'    — eye artifact only          (19, n_samples) µV
            'muscle'  — EMG artifact only          (19, n_samples) µV
            'mixed'   — all sources summed          (19, n_samples) µV
            'times'   — time axis (n_samples,) s
            'sfreq'   — sampling frequency Hz
            'ch_names'— list of 19 channel names
            'dbs_freq'— DBS fundamental Hz
        """
        brain  = self.generate_brain()
        dbs    = self.generate_dbs(f_dbs=dbs_freq)    if include_dbs    else np.zeros_like(brain)
        eyes   = self.generate_eyes()                  if include_eyes   else np.zeros_like(brain)
        muscle = self.generate_muscle()                if include_muscle else np.zeros_like(brain)

        return {
            'brain':    brain,
            'dbs':      dbs,
            'eyes':     eyes,
            'muscle':   muscle,
            'mixed':    brain + dbs + eyes + muscle,
            'times':    self.times,
            'sfreq':    self.sfreq,
            'ch_names': list(self.CHANNELS),
            'dbs_freq': dbs_freq,
        }

    def to_mne_raw(self, data_dict: dict, which: str = 'mixed') -> mne.io.RawArray:
        """
        Wrap a signal component as an MNE RawArray (with 10-20 montage).

        Parameters
        ----------
        data_dict : dict
            Output of :meth:`generate`.
        which : str
            Key to extract: 'mixed', 'brain', 'dbs', 'eyes', or 'muscle'.

        Returns
        -------
        mne.io.RawArray
        """
        data_uv = data_dict[which]       # µV
        data_v  = data_uv * 1e-6         # V — MNE internal unit

        info = mne.create_info(
            ch_names=data_dict['ch_names'],
            sfreq=data_dict['sfreq'],
            ch_types='eeg',
            verbose=False,
        )
        raw = mne.io.RawArray(data_v, info, verbose=False)

        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_case=False, on_missing='ignore', verbose=False)
        raw.set_eeg_reference('average', projection=True, verbose=False)

        return raw
