"""
Shared constants for the EEG DBS artifact removal pipeline.

All filter method definitions live here so assets, tests, and notebooks
all reference the same canonical list.
"""


# Standard 10-20 channels used throughout the pipeline
STANDARD_CH: list[str] = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3",  "C3",  "Cz", "C4", "T4",
    "T5",  "P3",  "Pz", "P4", "T6",
    "O1",  "O2",
]

# EEG frequency bands (name, lo_hz, hi_hz)
BANDS: list[tuple[str, float, float]] = [
    ("Delta", 0.5,  4.0),
    ("Theta", 4.0,  8.0),
    ("Alpha", 8.0, 13.0),
    ("Beta", 13.0, 30.0),
]

# ---------------------------------------------------------------------------
# DBS removal method registry
# Each entry: (display_label, filter_key, kwargs_factory)
# kwargs_factory is a callable(dbs_freq: float) → dict passed to
# ArtifactFilterFactory.process(filter_key, data, sfreq, **kwargs)
# ---------------------------------------------------------------------------
DBS_METHODS: list[tuple[str, str, object]] = [
    (
        "Spectrum Fit (2 Hz)",
        "spectrum_fit",
        lambda f: dict(f_target=f, bandwidth=2.0),
    ),
    (
        "Comb Notch Q=50",
        "comb_notch",
        lambda f: dict(f0=f, q_factor=50),
    ),
    (
        "Comb Notch Q=200",
        "comb_notch",
        lambda f: dict(f0=f, q_factor=200),
    ),
    (
        "Hampel Freq (2 Hz)",
        "hampel_freq",
        lambda f: dict(window_hz=2.0, n_sigmas=3.0, attenuation_db=-60.0),
    ),
    (
        "Hampel Time",
        "hampel_time",
        lambda f: dict(window_sec=1.0 / f, n_sigmas=3.0),
    ),
    (
        "FFT Spectral Interp",
        "fft_spectral_interp",
        lambda f: dict(f_target=f),
    ),
    (
        "Sinusoidal Regression",
        "sinusoidal_regression",
        lambda f: dict(f_target=f),
    ),
]

# Visual style per method (color, linestyle, linewidth) for comparison plots
METHOD_PALETTE: dict[str, tuple[str, str, float]] = {
    "Spectrum Fit (2 Hz)":   ("sienna",    "--", 1.0),
    "Comb Notch Q=50":       ("crimson",   "-.", 1.1),
    "Comb Notch Q=200":      ("orchid",    "-.", 1.1),
    "Hampel Freq (2 Hz)":    ("peru",      ":",  1.4),
    "Hampel Time":           ("goldenrod", ":",  1.4),
    "FFT Spectral Interp":   ("steelblue", "-",  2.2),
    "Sinusoidal Regression": ("darkcyan",  "-",  1.6),
}
