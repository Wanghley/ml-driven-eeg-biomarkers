from pipeline.assets.synthetic import (
    synthetic_signal,
    synthetic_dbs_filtered,
    synthetic_ica_pipeline,
    synthetic_edf_export,
)
from pipeline.assets.real_data import (
    real_raw_loaded,
    real_dbs_filtered,
    real_ica_pipeline,
    real_edf_export,
)
from pipeline.assets.eeg_assets import (
    ingested_edf,
    cleaned_edf,
    spike_features,
    eeg_ml_features,
)

all_assets = [
    # Legacy XU-specific assets
    synthetic_signal,
    synthetic_dbs_filtered,
    synthetic_ica_pipeline,
    synthetic_edf_export,
    real_raw_loaded,
    real_dbs_filtered,
    real_ica_pipeline,
    real_edf_export,
    # Generic single-file EEG pipeline (target of edf_ingest_sensor)
    ingested_edf,
    cleaned_edf,
    spike_features,
    eeg_ml_features,
]
