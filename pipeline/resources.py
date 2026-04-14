"""
Dagster resources for the EEG DBS artifact removal pipeline.

EEGPipelineConfig is a ConfigurableResource that carries all runtime
parameters — paths, DBS frequency, filter settings, ICA settings.
Inject it into any asset or op via its typed parameter.
"""


import pathlib
from dagster import ConfigurableResource
from pydantic import Field


class EEGPipelineConfig(ConfigurableResource):
    """Runtime configuration for the EEG DBS artifact removal pipeline."""

    # ── Paths ───────────────────────────────────────────────────────────
    data_dir: str = Field(
        default="data/raw/XU",
        description="Directory containing raw EDF recordings.",
    )
    figures_dir: str = Field(
        default="figures",
        description="Root directory for output figures.",
    )
    processed_dir: str = Field(
        default="data/processed",
        description="Directory for intermediate .fif files written by pipeline assets.",
    )

    # ── DBS stimulation ─────────────────────────────────────────────────
    dbs_freq: float = Field(
        default=7.0,
        description="DBS fundamental frequency in Hz.",
    )
    target_sfreq: float = Field(
        default=256.0,
        description="Target sampling rate; recordings are resampled to this if needed.",
    )

    # ── Chosen best method (used in ICA stage) ──────────────────────────
    best_method: str = Field(
        default="FFT Spectral Interp",
        description=(
            "Display label of the DBS removal method passed to the ICA stage. "
            "Must match one of the labels defined in pipeline.constants.DBS_METHODS."
        ),
    )

    # ── ICA ─────────────────────────────────────────────────────────────
    n_ica_components: int = Field(
        default=15,
        description="Number of FastICA components.",
    )
    ica_random_state: int = Field(
        default=42,
        description="Random seed for reproducible ICA.",
    )
    ica_max_iter: int = Field(
        default=800,
        description="Maximum ICA iterations.",
    )

    # ── Synthetic signal generation ─────────────────────────────────────
    synthetic_duration: float = Field(
        default=120.0,
        description="Duration (seconds) of synthetic EEG signal.",
    )
    synthetic_sfreq: float = Field(
        default=256.0,
        description="Sampling rate for synthetic signal.",
    )
    synthetic_seed: int = Field(
        default=42,
        description="Random seed for reproducible synthetic signal generation.",
    )
    synthetic_dbs_amplitude: float = Field(
        default=80.0,
        description="Peak amplitude (µV) of synthetic DBS artifact.",
    )

    # ── Real-data EDF filenames ─────────────────────────────────────────
    awake7_file: str = Field(
        default="XUAWAKE7_deidentified.edf",
        description="EDF filename for the DBS-on awake recording.",
    )
    sleep7_file: str = Field(
        default="XUSLEEP7_deidentified.edf",
        description="EDF filename for the DBS-on sleep recording.",
    )
    pre_awake_file: str = Field(
        default="XUAWAKEPRE_deidentified.edf",
        description="EDF filename for the DBS-off awake baseline.",
    )
    pre_sleep_file: str = Field(
        default="XUSLEEP_deidentified.edf",
        description="EDF filename for the DBS-off sleep baseline.",
    )

    # ── Helpers (not fields — computed properties) ───────────────────────
    def data_path(self) -> pathlib.Path:
        return pathlib.Path(self.data_dir)

    def figures_path(self) -> pathlib.Path:
        p = pathlib.Path(self.figures_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def processed_path(self) -> pathlib.Path:
        p = pathlib.Path(self.processed_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def synth_figures_path(self) -> pathlib.Path:
        p = self.figures_path() / "synth"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def real_figures_path(self) -> pathlib.Path:
        p = self.figures_path() / "real"
        p.mkdir(parents=True, exist_ok=True)
        return p
