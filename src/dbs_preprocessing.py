"""DBS-aware EEG preprocessing pipeline.

This module provides a compact object-oriented pipeline that:
- standardizes EEG/auxiliary channel metadata,
- applies consensus-style 0.5 Hz high-pass with configurable low-pass preprocessing,
- compares three DBS artifact strategies,
- runs automated ICA cleanup with an ICLabel-style fallback,
- exports the final cleaned data to EDF,
- and optionally renders a live Raw-vs-Processed comparison dashboard.

The three DBS strategies are:
- standard high-pass/notch preprocessing,
- perceptual artifact rejection implemented as smooth spectral interpolation,
- adaptive filtering implemented with a baseline-referenced Wiener filter
  when a clean baseline is available, otherwise Zapline+ is used as an
  adaptive fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mne
import numpy as np
from scipy import signal

from src.filters import ArtifactFilterFactory, BaselineReferencedFilter


try:
    from mne_icalabel import label_components as icalabel_components
except Exception:  # pragma: no cover - optional dependency
    icalabel_components = None


STANDARD_1020 = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T3",
    "C3",
    "Cz",
    "C4",
    "T4",
    "T5",
    "P3",
    "Pz",
    "P4",
    "T6",
    "O1",
    "O2",
]


@dataclass(slots=True)
class MethodResult:
    label: str
    key: str
    raw: mne.io.BaseRaw
    harmonic_attenuation_db: float
    spectral_distortion_db: float
    alpha_preservation_pct: float
    beta_preservation_pct: float
    score: float
    ica_excluded: list[int]
    ica_summary: str


class RealtimeComparisonDashboard:
    """Matplotlib dashboard that slides through a Raw-vs-Processed window."""

    def __init__(
        self,
        raw_before: mne.io.BaseRaw,
        raw_after: mne.io.BaseRaw,
        channel_name: Optional[str] = None,
        window_sec: float = 10.0,
        step_sec: float = 1.0,
    ) -> None:
        self.raw_before = raw_before.copy().pick_types(eeg=True, exclude=[])
        self.raw_after = raw_after.copy().pick_types(eeg=True, exclude=[])
        self.window_sec = window_sec
        self.step_sec = step_sec

        if channel_name and channel_name in self.raw_before.ch_names:
            self.channel_name = channel_name
        else:
            self.channel_name = self.raw_before.ch_names[0]

        self.before_idx = self.raw_before.ch_names.index(self.channel_name)
        self.after_idx = self.raw_after.ch_names.index(self.channel_name)
        self.sfreq = float(self.raw_before.info["sfreq"])
        self.window_samples = max(2, int(round(self.window_sec * self.sfreq)))
        self.step_samples = max(1, int(round(self.step_sec * self.sfreq)))

    def show(self) -> tuple[plt.Figure, FuncAnimation]:
        before = self.raw_before.get_data(picks=[self.before_idx])[0]
        after = self.raw_after.get_data(picks=[self.after_idx])[0]
        n_samples = min(before.size, after.size)

        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor("white")
        ax.set_title(f"Raw vs. Processed sliding window - {self.channel_name}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (uV)")
        ax.grid(True, alpha=0.25, linestyle="--")

        raw_line, = ax.plot([], [], color="#6C757D", lw=1.0, label="Raw")
        proc_line, = ax.plot([], [], color="#1D4ED8", lw=1.4, label="Processed")
        window_label = ax.text(0.01, 0.96, "", transform=ax.transAxes, va="top")
        ax.legend(loc="upper right")

        def update(frame: int):
            start = frame * self.step_samples
            stop = min(start + self.window_samples, n_samples)
            if stop <= start:
                return raw_line, proc_line, window_label

            times = np.arange(start, stop) / self.sfreq
            raw_line.set_data(times, before[start:stop] * 1e6)
            proc_line.set_data(times, after[start:stop] * 1e6)
            window_label.set_text(f"Window: {times[0]:.2f} s to {times[-1]:.2f} s")

            combined = np.concatenate([before[start:stop], after[start:stop]]) * 1e6
            if combined.size:
                pad = max(1.0, 0.1 * float(np.ptp(combined)))
                ax.set_ylim(float(combined.min() - pad), float(combined.max() + pad))
                ax.set_xlim(float(times[0]), float(times[-1]))

            return raw_line, proc_line, window_label

        frame_count = max(1, math.ceil((n_samples - self.window_samples) / self.step_samples) + 1)
        anim = FuncAnimation(fig, update, frames=frame_count, interval=150, blit=False)
        update(0)
        return fig, anim


class DBSPreprocessingPipeline:
    """End-to-end DBS-aware EEG preprocessing pipeline built on MNE-Python."""

    EEG_BANDS = {
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "theta": (4.0, 8.0),
    }

    def __init__(
        self,
        input_file: str | Path,
        output_dir: str | Path,
        baseline_file: str | Path | None = None,
        dbs_freq: float = 7.0,
        line_freq: float = 60.0,
        lowpass_freq: float = 45.0,
        target_sfreq: Optional[float] = None,
        final_method: str = "auto",
        ica_n_components: int | float = 0.95,
        ica_random_state: int = 42,
        ica_max_iter: int | str = "auto",
    ) -> None:
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_file = Path(baseline_file) if baseline_file else None
        self.dbs_freq = float(dbs_freq)
        self.line_freq = float(line_freq)
        self.lowpass_freq = float(lowpass_freq)
        self.target_sfreq = target_sfreq
        self.final_method = final_method
        self.ica_n_components = ica_n_components
        self.ica_random_state = ica_random_state
        self.ica_max_iter = ica_max_iter

    def _load_raw(self, path: Path) -> mne.io.BaseRaw:
        raw = mne.io.read_raw_edf(path, preload=True, verbose="WARNING")
        return raw

    def _rename_standard_channels(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        raw = raw.copy()
        rename = {}
        for ch in raw.ch_names:
            ch_norm = ch.strip()
            if ch_norm == "T1":
                rename[ch] = "FT9"
            elif ch_norm == "T2":
                rename[ch] = "FT10"
            elif ch_norm.upper().startswith("EEG "):
                name = ch_norm.split(" ", 1)[1].split("-")[0].strip()
                if name in STANDARD_1020:
                    rename[ch] = name
        if rename:
            raw.rename_channels(rename)
        return raw

    def _set_channel_types(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        raw = raw.copy()
        type_map: dict[str, str] = {}
        for ch in raw.ch_names:
            upper = ch.upper()
            if upper in {"EKG", "ECG"} or "EKG" in upper or "ECG" in upper:
                type_map[ch] = "ecg"
            elif "EMG" in upper:
                type_map[ch] = "emg"
            elif "EOG" in upper or upper in {"LOC1", "LOC2", "ROC1", "ROC2", "HEOG", "VEOG"}:
                type_map[ch] = "eog"
            elif upper in {"TRIG", "TRIGGER", "PHOTIC", "IBI", "BURSTS", "SUPPRESSION"}:
                type_map[ch] = "misc"
            elif ch in STANDARD_1020:
                type_map[ch] = "eeg"

        if type_map:
            raw.set_channel_types(type_map, verbose="WARNING")

        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False, on_missing="ignore", verbose="WARNING")
        return raw

    def _prepare_recording(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        raw = self._rename_standard_channels(raw)
        raw = self._set_channel_types(raw)
        if self.target_sfreq and not math.isclose(float(raw.info["sfreq"]), float(self.target_sfreq)):
            raw.resample(self.target_sfreq, verbose="WARNING")
        return raw

    def _apply_consensus_filter(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        raw = raw.copy()
        sfreq = float(raw.info["sfreq"])
        nyquist = sfreq / 2.0

        h_freq = min(self.lowpass_freq, nyquist - 0.5)
        if h_freq <= 0.5:
            raise ValueError("Sampling rate is too low for the configured 0.5-Hz high-pass and low-pass settings.")

        raw.filter(
            l_freq=0.5,
            h_freq=h_freq,
            fir_design="firwin",
            phase="zero",
            filter_length="auto",
            l_trans_bandwidth=0.25,
            h_trans_bandwidth=min(5.0, max(2.0, h_freq * 0.05)),
            verbose="WARNING",
        )

        raw.set_eeg_reference("average", projection=False, verbose="WARNING")

        notch_freqs = [freq for freq in self._harmonic_series(self.line_freq, nyquist) if freq < h_freq]
        if notch_freqs:
            raw.notch_filter(
                freqs=notch_freqs,
                notch_widths=1.0,
                trans_bandwidth=1.0,
                fir_design="firwin",
                phase="zero",
                verbose="WARNING",
            )

        return raw

    @staticmethod
    def _harmonic_series(f0: float, nyquist: float) -> list[float]:
        if f0 <= 0:
            return []
        return [float(freq) for freq in np.arange(f0, nyquist, f0)]

    @staticmethod
    def _pick_eeg_indices(raw: mne.io.BaseRaw) -> np.ndarray:
        return mne.pick_types(raw.info, eeg=True, exclude=[])

    def _apply_method(self, raw: mne.io.BaseRaw, method_key: str, baseline_raw: Optional[mne.io.BaseRaw]) -> mne.io.BaseRaw:
        method_key = method_key.lower().strip()
        raw_clean = raw.copy()
        eeg_picks = self._pick_eeg_indices(raw_clean)

        if method_key == "standard_notch":
            return raw_clean

        if method_key == "perceptual_artifact_rejection":
            eeg_data = raw_clean.get_data(picks=eeg_picks)
            cleaned = ArtifactFilterFactory.process(
                method="spectrum_fit",
                data=eeg_data,
                sfreq=float(raw_clean.info["sfreq"]),
                f_target=self.dbs_freq,
                bandwidth=2.0,
                attenuation_db=-60.0,
            )
            raw_clean._data[eeg_picks] = cleaned
            return raw_clean

        if method_key == "adaptive_filter":
            if baseline_raw is not None:
                adaptive = BaselineReferencedFilter(
                    baseline_raw=baseline_raw,
                    dbs_freq=self.dbs_freq,
                    harmonic_bandwidth=1.5,
                    n_fft=4096,
                    floor_db=-40.0,
                    alpha=1.5,
                    taper_width=1.0,
                )
                return adaptive.filter(raw_clean)

            eeg_data = raw_clean.get_data(picks=eeg_picks)
            cleaned = ArtifactFilterFactory.process(
                method="zapline",
                data=eeg_data,
                sfreq=float(raw_clean.info["sfreq"]),
                f_target=self.dbs_freq,
                n_harmonics=10,
                threshold_percentile=95.0,
            )
            raw_clean._data[eeg_picks] = cleaned
            return raw_clean

        raise ValueError(
            "Unknown DBS method. Expected one of: standard_notch, perceptual_artifact_rejection, adaptive_filter"
        )

    def _prepare_for_ica(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        raw_for_ica = raw.copy()
        raw_for_ica.filter(
            l_freq=1.0,
            h_freq=None,
            fir_design="firwin",
            phase="zero",
            filter_length="auto",
            l_trans_bandwidth=0.25,
            verbose="WARNING",
        )
        return raw_for_ica

    def _run_auto_ica(self, raw: mne.io.BaseRaw) -> tuple[mne.io.BaseRaw, list[int], str]:
        raw_ica = self._prepare_for_ica(raw)
        eeg_picks = self._pick_eeg_indices(raw_ica)
        if eeg_picks.size < 2:
            return raw.copy(), [], "Insufficient EEG channels for ICA"

        ica = mne.preprocessing.ICA(
            n_components=self.ica_n_components,
            method="fastica",
            random_state=self.ica_random_state,
            max_iter=self.ica_max_iter,
        )
        ica.fit(raw_ica, picks=eeg_picks, verbose="WARNING")

        bads: set[int] = set()
        summary_parts: list[str] = []

        try:
            if icalabel_components is not None:
                label_result = icalabel_components(raw_ica, ica, method="iclabel")
                labels = None
                probabilities = None
                if isinstance(label_result, dict):
                    labels = label_result.get("labels")
                    probabilities = label_result.get("y_pred_proba")
                elif isinstance(label_result, tuple):
                    labels = label_result[0]
                    probabilities = label_result[1] if len(label_result) > 1 else None

                if labels is not None:
                    for idx, label in enumerate(labels):
                        label_text = str(label).lower()
                        if label_text in {"eye blink", "muscle artifact"}:
                            bads.add(idx)
                    summary_parts.append(f"ICLabel labels={list(map(str, labels))}")
                    if probabilities is not None:
                        summary_parts.append("ICLabel probabilities available")
        except Exception as exc:  # pragma: no cover - optional dependency path
            summary_parts.append(f"ICLabel unavailable: {exc}")

        eog_candidates = [ch for ch in ("EOG", "VEOG", "HEOG", "LOC1", "LOC2", "Fp1", "Fp2") if ch in raw_ica.ch_names]
        if eog_candidates:
            try:
                eog_idx, _ = ica.find_bads_eog(raw_ica, ch_name=eog_candidates, verbose="WARNING")
                bads.update(eog_idx)
                summary_parts.append(f"EOG={list(eog_idx)}")
            except Exception as exc:
                summary_parts.append(f"EOG heuristic unavailable: {exc}")

        try:
            muscle_idx, _ = ica.find_bads_muscle(raw_ica, verbose="WARNING")
            bads.update(muscle_idx)
            summary_parts.append(f"EMG={list(muscle_idx)}")
        except Exception as exc:
            summary_parts.append(f"EMG heuristic unavailable: {exc}")

        raw_clean = raw.copy()
        ica.exclude = sorted(bads)
        if ica.exclude:
            ica.apply(raw_clean, verbose="WARNING")

        return raw_clean, sorted(bads), "; ".join(summary_parts) if summary_parts else "ICA completed"

    @staticmethod
    def _welch_psd(raw: mne.io.BaseRaw, fmin: float = 0.5, fmax: float = 100.0, n_fft: int = 2048) -> tuple[np.ndarray, np.ndarray]:
        eeg = raw.copy().pick_types(eeg=True, exclude=[])
        data = eeg.get_data()
        sfreq = float(eeg.info["sfreq"])
        freqs, psd = signal.welch(data, fs=sfreq, nperseg=min(n_fft, data.shape[1]), noverlap=min(n_fft, data.shape[1]) // 2, window="hann", axis=1)
        mask = (freqs >= fmin) & (freqs <= fmax)
        return freqs[mask], psd[:, mask].mean(axis=0)

    def _harmonic_attenuation(self, reference: mne.io.BaseRaw, cleaned: mne.io.BaseRaw, half_bw: float = 1.0) -> float:
        sfreq = float(reference.info["sfreq"])
        freqs, ref_psd = self._welch_psd(reference)
        _, clean_psd = self._welch_psd(cleaned)
        attenuations = []
        for harmonic in self._harmonic_series(self.dbs_freq, sfreq / 2.0):
            if harmonic >= 100.0:
                break
            band = np.abs(freqs - harmonic) <= half_bw
            if not np.any(band):
                continue
            ref_level = float(ref_psd[band].mean())
            clean_level = float(clean_psd[band].mean())
            attenuations.append(10.0 * np.log10((ref_level + 1e-30) / (clean_level + 1e-30)))
        return float(np.mean(attenuations)) if attenuations else 0.0

    def _spectral_distortion(self, reference: mne.io.BaseRaw, cleaned: mne.io.BaseRaw, fmin: float = 0.5, fmax: float = 100.0) -> float:
        freqs, ref_psd = self._welch_psd(reference, fmin=fmin, fmax=fmax)
        _, clean_psd = self._welch_psd(cleaned, fmin=fmin, fmax=fmax)
        ref_db = 10.0 * np.log10(ref_psd + 1e-30)
        clean_db = 10.0 * np.log10(clean_psd + 1e-30)
        return float(np.sqrt(np.mean((clean_db - ref_db) ** 2)))

    def _band_preservation(self, reference: mne.io.BaseRaw, cleaned: mne.io.BaseRaw, band: tuple[float, float]) -> float:
        lo, hi = band
        freqs, ref_psd = self._welch_psd(reference, fmin=lo, fmax=hi)
        _, clean_psd = self._welch_psd(cleaned, fmin=lo, fmax=hi)
        return float(100.0 * (clean_psd.mean() / (ref_psd.mean() + 1e-30)))

    def _evaluate_method(self, label: str, key: str, reference: mne.io.BaseRaw, cleaned: mne.io.BaseRaw, ica_excluded: list[int], ica_summary: str) -> MethodResult:
        harmonic_attenuation_db = self._harmonic_attenuation(reference, cleaned)
        spectral_distortion_db = self._spectral_distortion(reference, cleaned)
        alpha_preservation_pct = self._band_preservation(reference, cleaned, self.EEG_BANDS["alpha"])
        beta_preservation_pct = self._band_preservation(reference, cleaned, self.EEG_BANDS["beta"])
        score = harmonic_attenuation_db - spectral_distortion_db
        return MethodResult(
            label=label,
            key=key,
            raw=cleaned,
            harmonic_attenuation_db=harmonic_attenuation_db,
            spectral_distortion_db=spectral_distortion_db,
            alpha_preservation_pct=alpha_preservation_pct,
            beta_preservation_pct=beta_preservation_pct,
            score=score,
            ica_excluded=ica_excluded,
            ica_summary=ica_summary,
        )

    def compare_methods(self) -> tuple[dict[str, MethodResult], mne.io.BaseRaw, Optional[mne.io.BaseRaw]]:
        raw = self._prepare_recording(self._load_raw(self.input_file))
        baseline_raw = self._prepare_recording(self._load_raw(self.baseline_file)) if self.baseline_file else None

        prepared_raw = self._apply_consensus_filter(raw)
        prepared_baseline = self._apply_consensus_filter(baseline_raw) if baseline_raw is not None else None

        method_specs = [
            ("Standard high-pass/notch", "standard_notch"),
            ("Perceptual Artifact Rejection", "perceptual_artifact_rejection"),
            ("Adaptive filter", "adaptive_filter"),
        ]

        results: dict[str, MethodResult] = {}
        for label, key in method_specs:
            cleaned = self._apply_method(prepared_raw, key, prepared_baseline)
            results[key] = self._evaluate_method(label, key, prepared_raw, cleaned, [], "DBS filtering only")

        return results, prepared_raw, prepared_baseline

    def choose_final_method(self, results: dict[str, MethodResult]) -> MethodResult:
        if self.final_method != "auto":
            normalized = self.final_method.lower().strip()
            for result in results.values():
                if result.key == normalized or result.label.lower() == normalized:
                    return result
            raise ValueError(f"Unknown final method: {self.final_method}")

        return max(results.values(), key=lambda item: item.score)

    def export_edf(self, raw: mne.io.BaseRaw, out_path: Path) -> Path:
        try:
            import pyedflib
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "pyEDFlib is required for EDF export. Install pyedflib or use an environment that includes it."
            ) from exc

        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw_to_export = raw.copy().load_data()
        data_uv = raw_to_export.get_data() * 1e6
        sfreq = float(raw_to_export.info["sfreq"])

        signal_headers = []
        for idx, ch_name in enumerate(raw_to_export.ch_names):
            channel_data = data_uv[idx]
            chan_min = float(np.min(channel_data))
            chan_max = float(np.max(channel_data))
            if math.isclose(chan_min, chan_max):
                chan_min -= 1.0
                chan_max += 1.0
            padding = 0.05 * max(abs(chan_min), abs(chan_max), 1.0)
            signal_headers.append(
                {
                    "label": ch_name[:16],
                    "dimension": "uV",
                    "sample_frequency": sfreq,
                    "physical_min": chan_min - padding,
                    "physical_max": chan_max + padding,
                    "digital_min": -32768,
                    "digital_max": 32767,
                    "transducer": "",
                    "prefilter": "MNE-Python DBS preprocessing",
                }
            )

        header = {
            "technician": "Copilot",
            "recording_additional": "DBS cleaned EEG",
            "patientname": "anonymous",
            "patientcode": "anonymous",
            "equipment": "MNE-Python + pyEDFlib",
        }

        pyedflib.highlevel.write_edf(
            str(out_path),
            signals=[data_uv[idx] for idx in range(data_uv.shape[0])],
            signal_headers=signal_headers,
            header=header,
            digital=False,
            file_type=pyedflib.FILETYPE_EDFPLUS,
        )
        return out_path

    def run(self, show_dashboard: bool = False, dashboard_window_sec: float = 10.0) -> dict[str, object]:
        results, prepared_raw, _ = self.compare_methods()
        final_result = self.choose_final_method(results)

        ica_cleaned, ica_excluded, ica_summary = self._run_auto_ica(final_result.raw)
        final_result = self._evaluate_method(
            final_result.label,
            final_result.key,
            prepared_raw,
            ica_cleaned,
            ica_excluded,
            ica_summary,
        )

        final_stem = self.input_file.stem.replace(".edf", "")
        final_path = self.output_dir / f"{final_stem}_{final_result.key}_clean.edf"
        fif_path = self.output_dir / f"{final_stem}_{final_result.key}_clean.fif"

        ica_cleaned.save(fif_path, overwrite=True, verbose="WARNING")
        self.export_edf(ica_cleaned, final_path)

        dashboard = None
        if show_dashboard:
            dashboard = RealtimeComparisonDashboard(
                prepared_raw,
                ica_cleaned,
                window_sec=dashboard_window_sec,
            )

        summary = {
            "input_file": str(self.input_file),
            "baseline_file": str(self.baseline_file) if self.baseline_file else None,
            "results": {
                key: {
                    "label": result.label,
                    "harmonic_attenuation_db": result.harmonic_attenuation_db,
                    "spectral_distortion_db": result.spectral_distortion_db,
                    "alpha_preservation_pct": result.alpha_preservation_pct,
                    "beta_preservation_pct": result.beta_preservation_pct,
                    "score": result.score,
                }
                for key, result in results.items()
            },
            "final": {
                "label": final_result.label,
                "key": final_result.key,
                "harmonic_attenuation_db": final_result.harmonic_attenuation_db,
                "spectral_distortion_db": final_result.spectral_distortion_db,
                "alpha_preservation_pct": final_result.alpha_preservation_pct,
                "beta_preservation_pct": final_result.beta_preservation_pct,
                "score": final_result.score,
                "ica_excluded": final_result.ica_excluded,
                "ica_summary": final_result.ica_summary,
                "edf_path": str(final_path),
                "fif_path": str(fif_path),
            },
        }

        metrics_path = self.output_dir / f"{final_stem}_comparison_metrics.json"
        metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        rows = []
        for key, result in results.items():
            rows.append(
                {
                    "method_key": key,
                    "method_label": result.label,
                    "harmonic_attenuation_db": result.harmonic_attenuation_db,
                    "spectral_distortion_db": result.spectral_distortion_db,
                    "alpha_preservation_pct": result.alpha_preservation_pct,
                    "beta_preservation_pct": result.beta_preservation_pct,
                    "score": result.score,
                }
            )
        comparison_csv = self.output_dir / f"{final_stem}_comparison_metrics.csv"
        try:
            import pandas as pd

            pd.DataFrame(rows).to_csv(comparison_csv, index=False)
        except Exception:
            pass

        if dashboard is not None:
            fig, anim = dashboard.show()
            plt.show()
            summary["dashboard"] = {"figure": fig, "animation": anim}

        return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DBS-aware EEG preprocessing pipeline")
    parser.add_argument("input_file", type=Path, help="Input EDF file")
    parser.add_argument("--baseline-file", type=Path, default=None, help="Optional clean baseline EDF file")
    parser.add_argument("--output-dir", type=Path, default=Path("results/processed"), help="Output directory")
    parser.add_argument("--dbs-freq", type=float, default=7.0, help="DBS fundamental frequency in Hz")
    parser.add_argument("--line-freq", type=float, default=60.0, help="Power-line frequency in Hz")
    parser.add_argument(
        "--lowpass-freq",
        type=float,
        default=45.0,
        help="Low-pass cutoff in Hz (conventional EEG often 40-45 Hz)",
    )
    parser.add_argument("--target-sfreq", type=float, default=None, help="Optional resampling target in Hz")
    parser.add_argument(
        "--final-method",
        type=str,
        default="auto",
        choices=["auto", "standard_notch", "perceptual_artifact_rejection", "adaptive_filter"],
        help="Final method to export",
    )
    parser.add_argument("--ica-components", type=float, default=0.95, help="ICA n_components value")
    parser.add_argument("--ica-random-state", type=int, default=42, help="ICA random seed")
    parser.add_argument("--ica-max-iter", type=str, default="auto", help="ICA max_iter value")
    parser.add_argument("--dashboard", action="store_true", help="Show live Raw-vs-Processed dashboard")
    parser.add_argument("--dashboard-window-sec", type=float, default=10.0, help="Dashboard window length in seconds")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    ica_components: int | float
    try:
        ica_components = float(args.ica_components)
        if ica_components.is_integer():
            ica_components = int(ica_components)
    except Exception:
        ica_components = args.ica_components

    pipeline = DBSPreprocessingPipeline(
        input_file=args.input_file,
        output_dir=args.output_dir,
        baseline_file=args.baseline_file,
        dbs_freq=args.dbs_freq,
        line_freq=args.line_freq,
        lowpass_freq=args.lowpass_freq,
        target_sfreq=args.target_sfreq,
        final_method=args.final_method,
        ica_n_components=ica_components,
        ica_random_state=args.ica_random_state,
        ica_max_iter=args.ica_max_iter,
    )
    summary = pipeline.run(show_dashboard=args.dashboard, dashboard_window_sec=args.dashboard_window_sec)

    final = summary["final"]
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())