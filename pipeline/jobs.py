"""
Dagster job definitions for the EEG DBS artifact removal pipeline.

Three jobs are provided:
  synthetic_job       — generate synthetic signal, filter all methods, run ICA
  real_data_job       — load real XU recordings, filter all methods, run ICA
  full_eeg_pipeline   — everything above in a single run
"""

from dagster import define_asset_job, AssetSelection

synthetic_job = define_asset_job(
    name="synthetic_job",
    selection=AssetSelection.groups("synthetic"),
    description=(
        "Full synthetic EEG pipeline: generate contaminated signal, "
        "compare 7 DBS removal methods, run ICA on best method."
    ),
)

real_data_job = define_asset_job(
    name="real_data_job",
    selection=AssetSelection.groups("real_data"),
    description=(
        "Full real-data pipeline on patient XU (7 Hz DBS): "
        "load AWAKE7 + SLEEP7 + PRE baselines, compare 7 DBS removal methods, "
        "run ICA on best method, save final figures."
    ),
)

full_eeg_pipeline = define_asset_job(
    name="full_eeg_pipeline",
    selection=AssetSelection.all(),
    description="Run the synthetic and real-data pipelines back-to-back.",
)
