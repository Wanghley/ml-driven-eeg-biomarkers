"""
Dagster job definitions for the EEG DBS artifact removal pipeline.

Four jobs are provided:
  synthetic_job       — generate synthetic signal, filter all methods, run ICA, export EDF
  real_data_job       — load real XU recordings, filter all methods, run ICA, export EDF
  full_eeg_pipeline   — everything above in a single run
  export_edf_job      — run only the EDF export assets (requires upstream .fif files)
"""

from dagster import define_asset_job, AssetSelection

synthetic_job = define_asset_job(
    name="synthetic_job",
    selection=AssetSelection.groups("synthetic"),
    description=(
        "Full synthetic EEG pipeline: generate contaminated signal, "
        "compare 7 DBS removal methods, run ICA on best method, export EDF files."
    ),
)

real_data_job = define_asset_job(
    name="real_data_job",
    selection=AssetSelection.groups("real_data"),
    description=(
        "Full real-data pipeline on patient XU (7 Hz DBS): "
        "load AWAKE7 + SLEEP7 + PRE baselines, compare 7 DBS removal methods, "
        "run ICA on best method, export EDF files."
    ),
)

full_eeg_pipeline = define_asset_job(
    name="full_eeg_pipeline",
    selection=AssetSelection.all(),
    description="Run the synthetic and real-data pipelines back-to-back.",
)

export_edf_job = define_asset_job(
    name="export_edf_job",
    selection=AssetSelection.assets(["real_edf_export", "synthetic_edf_export"]),
    description=(
        "Export final pipeline-cleaned signals as EDF files. "
        "Requires upstream .fif files produced by the full pipeline runs."
    ),
)
