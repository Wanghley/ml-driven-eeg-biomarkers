"""
Dagster sensors for the EEG DBS artifact removal pipeline.

new_edf_sensor
    Watches ``data/raw/XU/`` for new ``*7*.edf`` files (DBS-on recordings)
    and triggers ``real_data_job`` automatically when a new file appears.

    The sensor stores the set of previously-seen filenames in its cursor
    so each new file triggers at most one run.
"""


import pathlib
from dagster import sensor, RunRequest, SensorEvaluationContext, SkipReason

from pipeline.jobs import real_data_job


@sensor(
    job=real_data_job,
    minimum_interval_seconds=60,
    description=(
        "Watches data/raw/XU/ for new *7*.edf DBS-on recordings. "
        "Triggers real_data_job when a new file is detected."
    ),
)
def new_edf_sensor(context: SensorEvaluationContext):
    data_dir = pathlib.Path("data/raw/XU")
    if not data_dir.exists():
        yield SkipReason(f"Data directory {data_dir} does not exist yet.")
        return

    # All DBS-on EDF files (filename contains '7', case-insensitive)
    current_files = {
        p.name for p in data_dir.glob("*.edf")
        if "7" in p.stem and "pre" not in p.stem.lower()
    }

    if not current_files:
        yield SkipReason("No DBS-on EDF files found.")
        return

    # Cursor stores previously-seen filenames as a comma-separated string
    seen: set[str] = set(context.cursor.split(",")) if context.cursor else set()
    new_files = current_files - seen

    if not new_files:
        yield SkipReason(f"No new files since last check. Watching: {sorted(current_files)}")
        return

    for fname in sorted(new_files):
        context.log.info(f"New EDF detected: {fname} — triggering real_data_job")
        yield RunRequest(
            run_key=fname,
            run_config={},
            tags={"triggered_by": "new_edf_sensor", "source_file": fname},
        )

    # Update cursor to include all currently-seen files
    context.update_cursor(",".join(sorted(current_files)))
