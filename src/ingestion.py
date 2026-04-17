"""src/ingestion.py — canonical EDF ingestion API.

All EDF loading must go through load_edf(). This module explicitly types EOG,
EMG, and EEG channels so they survive intact to the ICA cleanup stage.

DBS frequency is always an explicit parameter — never inferred from filenames.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import mne

# Canonical capitalisation for the standard 10-20 montage
STANDARD_1020: list[str] = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2",
]

# Lookup from uppercase → canonical 10-20 name
_UPPER_1020: dict[str, str] = {s.upper(): s for s in STANDARD_1020}

# Channel-name patterns for auxiliary channel typing.
# Extend bare_name (before the first "-") matching patterns here as needed.
_EOG_RE = re.compile(
    r"^(eog\d*|e[12]|heog|veog|reog|leog|roc|loc|eye|blink)",
    re.IGNORECASE,
)
_EMG_RE = re.compile(
    r"^(emg\d*|chin\d*|mentalis|submentalis)",
    re.IGNORECASE,
)


# ──────────────────────────────────────────────────────────────────────────────
# Public data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IngestionConfig:
    """Parameters for EDF ingestion.

    Set dbs_freq explicitly — this value is stored in provenance metadata but
    never inferred from the filename.
    """
    max_duration_sec: Optional[float] = None   # None → keep full recording
    target_sfreq: Optional[float] = None        # None → keep original sample rate
    l_freq: Optional[float] = None              # None → skip high-pass
    h_freq: Optional[float] = None              # None → skip low-pass
    apply_avg_ref: bool = False                 # average reference for EEG channels
    dbs_freq: Optional[float] = None            # stored in metadata only
    verbose: Union[bool, str] = False


@dataclass
class IngestionResult:
    """Output of load_edf(), carrying a fully typed Raw object and provenance."""
    raw: mne.io.BaseRaw
    eeg_channels: list[str]
    eog_channels: list[str]
    emg_channels: list[str]
    other_channels: list[str]
    source_path: Path
    metadata: dict


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def normalize_channel_names(raw: mne.io.BaseRaw) -> dict[str, str]:
    """Return a {old: new} rename map from clinical EDF names to 10-20 names.

    Handles decorated names such as "EEG Fp1-REF", "Fp1-LE", or "FP1".
    Non-EEG channels (EOG, EMG) are left as-is and classified separately.
    """
    rename: dict[str, str] = {}
    for ch in raw.ch_names:
        # Fast path: bare uppercase match ("FP1" → "Fp1")
        if ch.upper() in _UPPER_1020:
            rename[ch] = _UPPER_1020[ch.upper()]
            continue
        # Slow path: find a 10-20 label embedded in a decorated name
        for std in STANDARD_1020:
            if re.search(
                rf"(?<![A-Za-z0-9]){re.escape(std)}(?![A-Za-z0-9])",
                ch,
                re.IGNORECASE,
            ):
                rename[ch] = std
                break
    return rename


def classify_channels(
    raw: mne.io.BaseRaw,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Classify channel names into (eeg, eog, emg, other).

    Assumes normalize_channel_names has already been applied so that EEG
    channels carry their canonical 10-20 labels.
    """
    std_set = set(STANDARD_1020)
    eeg: list[str] = []
    eog: list[str] = []
    emg: list[str] = []
    other: list[str] = []

    for ch in raw.ch_names:
        bare = ch.split("-")[0].strip()
        if ch in std_set:
            eeg.append(ch)
        elif _EOG_RE.match(bare):
            eog.append(ch)
        elif _EMG_RE.match(bare):
            emg.append(ch)
        else:
            other.append(ch)

    return eeg, eog, emg, other


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_edf(
    path: Union[str, Path],
    config: Optional[IngestionConfig] = None,
) -> IngestionResult:
    """Load an EDF file into a fully typed MNE Raw object.

    Channel typing is applied before any filtering so that EOG and EMG channels
    survive to the ICA cleanup stage downstream.

    Steps performed in order:
      1. Read EDF with preload.
      2. Optional crop to max_duration_sec.
      3. Rename channels to standard 10-20 labels where possible.
      4. Classify and type EEG / EOG / EMG channels explicitly.
      5. Assign standard_1020 montage to EEG channels.
      6. Optional resample (anti-aliasing handled by MNE).
      7. Optional zero-phase FIR bandpass.
      8. Optional average reference (EEG channels only).

    Args:
        path: Path to the .edf file.
        config: Ingestion parameters; defaults to IngestionConfig() if None.

    Returns:
        IngestionResult with the typed Raw object and channel classification.

    Raises:
        FileNotFoundError: If the EDF file does not exist.
        ValueError: If no standard 10-20 EEG channels are found.
    """
    if config is None:
        config = IngestionConfig()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EDF file not found: {path}")

    raw = mne.io.read_raw_edf(str(path), preload=True, verbose=config.verbose)

    # ── 1. Crop ──────────────────────────────────────────────────────────────
    if config.max_duration_sec is not None:
        raw.crop(tmin=0.0, tmax=min(config.max_duration_sec, raw.times[-1]))

    # ── 2. Rename to 10-20 labels ────────────────────────────────────────────
    rename = normalize_channel_names(raw)
    if rename:
        raw.rename_channels(rename)

    # ── 3. Classify and type channels ────────────────────────────────────────
    eeg_ch, eog_ch, emg_ch, other_ch = classify_channels(raw)

    if not eeg_ch:
        raise ValueError(
            f"No standard 10-20 EEG channels found in {path.name}. "
            f"Available channels: {raw.ch_names}"
        )

    ch_types: dict[str, str] = {}
    ch_types.update({ch: "eeg" for ch in eeg_ch})
    ch_types.update({ch: "eog" for ch in eog_ch})
    ch_types.update({ch: "emg" for ch in emg_ch})
    raw.set_channel_types(ch_types)

    # ── 4. Montage (EEG only; EOG/EMG have no standard positions) ────────────
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="ignore", verbose=False)

    # ── 5. Resample ──────────────────────────────────────────────────────────
    if config.target_sfreq is not None and raw.info["sfreq"] != config.target_sfreq:
        raw.resample(config.target_sfreq, verbose=config.verbose)

    # ── 6. Bandpass filter ───────────────────────────────────────────────────
    if config.l_freq is not None or config.h_freq is not None:
        nyquist = raw.info["sfreq"] / 2.0
        h_safe = config.h_freq
        if h_safe is not None:
            h_safe = min(h_safe, nyquist - 0.5)
        raw.filter(
            l_freq=config.l_freq,
            h_freq=h_safe,
            fir_design="firwin",
            phase="zero",
            verbose=config.verbose,
        )

    # ── 7. Average reference ─────────────────────────────────────────────────
    if config.apply_avg_ref:
        raw.set_eeg_reference("average", projection=True, verbose=False)
        raw.apply_proj()

    metadata = {
        "source_path": str(path),
        "sfreq": float(raw.info["sfreq"]),
        "n_eeg": len(eeg_ch),
        "n_eog": len(eog_ch),
        "n_emg": len(emg_ch),
        "n_other": len(other_ch),
        "duration_sec": float(raw.times[-1]),
    }
    if config.dbs_freq is not None:
        metadata["dbs_freq_hz"] = float(config.dbs_freq)

    return IngestionResult(
        raw=raw,
        eeg_channels=eeg_ch,
        eog_channels=eog_ch,
        emg_channels=emg_ch,
        other_channels=other_ch,
        source_path=path,
        metadata=metadata,
    )


def find_edf_files(directory: Union[str, Path]) -> list[Path]:
    """Recursively find all EDF files under directory, deduplicating by resolved path."""
    directory = Path(directory)
    seen: set[Path] = set()
    files: list[Path] = []
    candidates = sorted(directory.rglob("*.edf")) + sorted(directory.rglob("*.EDF"))
    for f in candidates:
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            files.append(f)
    return files
