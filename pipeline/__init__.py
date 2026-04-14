"""
EEG DBS Artifact Removal Pipeline — Dagster Definitions

Entry point for ``dagster dev`` and ``dagster-webserver``.
All assets, resources, jobs, and sensors are registered here.

Usage
-----
From the project root::

    dagster dev -m pipeline

Or via workspace.yaml (recommended for deployment)::

    dagster dev           # picks up workspace.yaml automatically
    dagster-webserver -w workspace.yaml
"""

from dagster import Definitions, FilesystemIOManager

from pipeline.assets import all_assets
from pipeline.jobs import synthetic_job, real_data_job, full_eeg_pipeline, export_edf_job
from pipeline.resources import EEGPipelineConfig
from pipeline.sensors import new_edf_sensor

defs = Definitions(
    assets=all_assets,
    jobs=[
        synthetic_job,
        real_data_job,
        full_eeg_pipeline,
        export_edf_job,
    ],
    resources={
        # Resource key must match the parameter name used in @asset functions
        "eeg_config": EEGPipelineConfig(
            data_dir="data/raw/XU",
            figures_dir="figures",
            processed_dir="data/processed",
            dbs_freq=7.0,
            target_sfreq=256.0,
            best_method="FFT Spectral Interp",
            n_ica_components=15,
            ica_random_state=42,
            synthetic_duration=120.0,
            synthetic_sfreq=256.0,
            synthetic_seed=42,
        ),
        # Default IOManager — serialises asset outputs as pickle files under
        # .dagster/storage/  (swap for S3IOManager, GCSIOManager, etc. in prod)
        "io_manager": FilesystemIOManager(base_dir=".dagster/storage"),
    },
    sensors=[new_edf_sensor],
)
