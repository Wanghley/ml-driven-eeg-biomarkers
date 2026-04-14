from pipeline.assets.synthetic import (
    synthetic_signal,
    synthetic_dbs_filtered,
    synthetic_ica_pipeline,
)
from pipeline.assets.real_data import (
    real_raw_loaded,
    real_dbs_filtered,
    real_ica_pipeline,
)

all_assets = [
    synthetic_signal,
    synthetic_dbs_filtered,
    synthetic_ica_pipeline,
    real_raw_loaded,
    real_dbs_filtered,
    real_ica_pipeline,
]
