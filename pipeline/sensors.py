"""
Dagster sensors for the EEG DBS artifact removal pipeline.

new_edf_sensor
    Watches ``data/raw/XU/`` for new ``*7*.edf`` files (DBS-on recordings)
    and triggers ``real_data_job`` automatically when a new file appears.

    The sensor stores the set of previously-seen filenames in its cursor
    so each new file triggers at most one run.

edf_ingest_sensor
    Watches ``data/raw/`` recursively for ANY new .edf file and triggers
    ``eeg_ingest_job``.  DBS frequency is inferred from the filename:
    a number immediately before "hz" or "Hz" is used (e.g. "AWAKE7" → 7 Hz,
    "SLEEP60" → 60 Hz); default = 7.0 Hz when no number can be parsed.

    Per-run config is passed as::

        run_config = {
            "ops": {
                "ingested_edf": {
                    "config": {
                        "file_path": "<absolute_path>",
                        "dbs_freq": <float>,
                    }
                }
            }
        }
"""


import pathlib
import re
from dagster import sensor, RunRequest, SensorEvaluationContext, SkipReason

from pipeline.jobs import real_data_job, eeg_ingest_job


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


# ──────────────────────────────────────────────────────────────────────────────
# Generic EDF ingest sensor (any new .edf anywhere under data/raw/)
# ──────────────────────────────────────────────────────────────────────────────

_DBS_FREQ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:hz|Hz)", re.IGNORECASE)
_DBS_STEM_RE = re.compile(r"(?<!\d)(\d+)(?!\d)")  # bare integer in filename stem


def _infer_dbs_freq(stem: str) -> float:
    """Infer the DBS frequency from a filename stem.

    Rules (first match wins):
    1. Explicit ``<number>hz`` or ``<number>Hz`` in the stem.
    2. First bare integer in the stem (e.g. ``XUAWAKE7`` → 7.0).
    3. Default: 7.0 Hz.
    """
    m = _DBS_FREQ_RE.search(stem)
    if m:
        return float(m.group(1))
    m = _DBS_STEM_RE.search(stem)
    if m:
        return float(m.group(1))
    return 7.0


@sensor(
    job=eeg_ingest_job,
    minimum_interval_seconds=30,
    description=(
        "Watches data/raw/ recursively for any new .edf file. "
        "Triggers eeg_ingest_job with file_path and inferred dbs_freq per run."
    ),
)
def edf_ingest_sensor(context: SensorEvaluationContext):
    data_root = pathlib.Path("data/raw")
    if not data_root.exists():
        yield SkipReason(f"data/raw/ does not exist yet.")
        return

    # Find all .edf files recursively (case-insensitive)
    current_files: dict[str, str] = {}  # relative_path_str → absolute_path_str
    for pattern in ("*.edf", "*.EDF"):
        for p in data_root.rglob(pattern):
            key = str(p.resolve().relative_to(pathlib.Path.cwd()) if p.is_absolute()
                      else p)
            current_files[key] = str(p.resolve())

    if not current_files:
        yield SkipReason("No .edf files found under data/raw/.")
        return

    seen: set[str] = set(context.cursor.split(",")) if context.cursor else set()
    new_keys = sorted(set(current_files.keys()) - seen)

    if not new_keys:
        yield SkipReason(f"No new EDF files. Tracking {len(current_files)} file(s).")
        return

    for key in new_keys:
        abs_path = current_files[key]
        stem     = pathlib.Path(abs_path).stem
        dbs_freq = _infer_dbs_freq(stem)
        context.log.info(
            f"New EDF detected: {pathlib.Path(abs_path).name}  "
            f"(inferred DBS freq = {dbs_freq} Hz) → triggering eeg_ingest_job"
        )
        yield RunRequest(
            run_key=key,
            run_config={
                "ops": {
                    "ingested_edf": {
                        "config": {
                            "file_path": abs_path,
                            "dbs_freq":  dbs_freq,
                        }
                    }
                }
            },
            tags={
                "triggered_by": "edf_ingest_sensor",
                "source_file":  pathlib.Path(abs_path).name,
                "dbs_freq_hz":  str(dbs_freq),
            },
        )

    context.update_cursor(",".join(sorted(current_files.keys())))
