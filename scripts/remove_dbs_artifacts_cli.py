#!/usr/bin/env python3
"""
DBS Artifact Removal - Interactive CLI Tool

Allows selection and application of multiple DBS artifact removal methods.
Supports: Spectrum-Fit, Wiener, Zapline+, Hampel Freq, Hampel Time
"""

import argparse
import sys
from pathlib import Path
import mne
import json

from src.preprocessing import EEGPreprocessor
from src.filters import BaselineReferencedFilter


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_methods():
    """Print available methods."""
    methods = {
        "1": {
            "name": "Spectrum-Fit Multi-Harmonic",
            "key": "spectrum_fit",
            "description": "Fast, strong, no baseline required",
            "attenuation": "-60 dB (tunable -40 to -100)",
            "time": "~1x baseline",
        },
        "2": {
            "name": "Wiener (Baseline-Referenced)",
            "key": "wiener",
            "description": "Best brain preservation, requires baseline",
            "attenuation": "-20 to -40 dB",
            "time": "~3x baseline",
        },
        "3": {
            "name": "Zapline+ (Chen et al., 2022)",
            "key": "zapline",
            "description": "Spectro-spatial filtering, multi-channel",
            "attenuation": "-20 to -50 dB",
            "time": "~5x baseline",
        },
        "4": {
            "name": "Freq-Domain Hampel (Allen et al., 2010)",
            "key": "hampel_freq",
            "description": "Adaptive peak detection",
            "attenuation": "-60 dB (tunable)",
            "time": "~0.8x baseline",
        },
        "5": {
            "name": "Time-Domain Hampel (Allen et al., 2010)",
            "key": "hampel_time",
            "description": "Transient pulse removal, fastest",
            "attenuation": "-20 to -40 dB",
            "time": "~0.5x baseline",
        },
    }

    print("\nAvailable Methods:")
    print("-" * 80)
    for idx, method in methods.items():
        print(f"\n[{idx}] {method['name']}")
        print(f"    Description:  {method['description']}")
        print(f"    Attenuation:  {method['attenuation']}")
        print(f"    Speed:        {method['time']}")

    return methods


def select_method_interactive():
    """Interactively select a method."""
    methods = print_methods()

    while True:
        print("\n" + "-" * 80)
        choice = input("\nSelect method (1-5) or 'q' to quit: ").strip()

        if choice.lower() == "q":
            print("Exiting...")
            sys.exit(0)

        if choice in methods:
            return methods[choice]["key"]
        else:
            print("❌ Invalid choice. Please enter 1-5 or 'q'.")


def select_input_file_interactive(input_dir: Path):
    """Interactively select an input EDF file."""
    edf_files = sorted(input_dir.glob("*.edf"))

    if not edf_files:
        print(f"❌ No EDF files found in {input_dir}")
        sys.exit(1)

    print(f"\n📁 Found {len(edf_files)} EDF file(s):")
    print("-" * 80)

    for idx, f in enumerate(edf_files, 1):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"[{idx}] {f.name:<50} ({size_mb:.1f} MB)")

    while True:
        choice = input(
            f"\nSelect file (1-{len(edf_files)}) or 'a' for all: "
        ).strip()

        if choice.lower() == "a":
            return [f.name for f in edf_files]
        elif choice.isdigit() and 1 <= int(choice) <= len(edf_files):
            return [edf_files[int(choice) - 1].name]
        else:
            print(f"❌ Invalid choice. Please enter 1-{len(edf_files)} or 'a'.")


def get_method_parameters(method: str, preprocessor: EEGPreprocessor) -> dict:
    """Get parameters for the selected method from user."""
    params = {}

    if method == "spectrum_fit":
        print("\nSpectrum-Fit Parameters:")
        print("  - f_target: Fundamental DBS frequency (Hz), e.g., 7.0")
        print("  - bandwidth: Total bandwidth around harmonics (Hz), default 2.0")
        print("  - attenuation_db: Attenuation level in dB, default -60.0 (more negative = stronger)")

        # Get f_target
        while True:
            try:
                f_target = float(input("  Enter f_target [auto-detect]: ") or "0")
                if f_target > 0:
                    params["f_target"] = f_target
                else:
                    params["f_target"] = None
                break
            except ValueError:
                print("  ❌ Invalid. Enter a number or press Enter for auto-detect.")

        # Get bandwidth
        while True:
            try:
                bw = float(input("  Enter bandwidth [2.0]: ") or "2.0")
                if 0.5 <= bw <= 5.0:
                    params["bandwidth"] = bw
                    break
                else:
                    print("  ❌ Range: 0.5-5.0 Hz")
            except ValueError:
                print("  ❌ Invalid. Enter a number (0.5-5.0).")

        # Get attenuation
        while True:
            try:
                att = float(input("  Enter attenuation_db [-60.0]: ") or "-60.0")
                if -100 <= att <= -30:
                    params["attenuation_db"] = att
                    break
                else:
                    print("  ❌ Range: -100 to -30 dB")
            except ValueError:
                print("  ❌ Invalid. Enter a number (-100 to -30).")

    elif method == "wiener":
        print("\nWiener Filter Parameters (requires clean baseline):")
        print("  - baseline_file: Path to clean baseline EDF file")
        print("  - dbs_freq: DBS frequency (Hz), default 7.0")
        print("  - alpha: Oversubtraction factor, default 1.5 (higher = stronger)")

        # Get baseline file
        baseline_file = input("  Enter baseline file path: ").strip()
        if not Path(baseline_file).exists():
            print(f"  ❌ File not found: {baseline_file}")
            return None
        params["baseline_file"] = baseline_file

        # Get dbs_freq
        while True:
            try:
                freq = float(input("  Enter dbs_freq [7.0]: ") or "7.0")
                if 1 <= freq <= 100:
                    params["dbs_freq"] = freq
                    break
                else:
                    print("  ❌ Range: 1-100 Hz")
            except ValueError:
                print("  ❌ Invalid. Enter a number.")

        # Get alpha
        while True:
            try:
                alpha = float(input("  Enter alpha [1.5]: ") or "1.5")
                if 1.0 <= alpha <= 3.0:
                    params["alpha"] = alpha
                    break
                else:
                    print("  ❌ Range: 1.0-3.0")
            except ValueError:
                print("  ❌ Invalid. Enter a number.")

    elif method == "zapline":
        print("\nZapline+ (Chen et al., 2022) Parameters:")
        print("  - f_target: Fundamental DBS frequency (Hz), e.g., 7.0")
        print("  - n_harmonics: Number of harmonics, default 10")
        print("  - threshold_percentile: Detection threshold, default 95.0 (higher = more selective)")

        # Get f_target
        while True:
            try:
                f_target = float(input("  Enter f_target [auto-detect]: ") or "0")
                if f_target > 0:
                    params["f_target"] = f_target
                else:
                    params["f_target"] = None
                break
            except ValueError:
                print("  ❌ Invalid. Enter a number or press Enter for auto-detect.")

        # Get n_harmonics
        while True:
            try:
                n_harm = int(input("  Enter n_harmonics [10]: ") or "10")
                if 1 <= n_harm <= 20:
                    params["n_harmonics"] = n_harm
                    break
                else:
                    print("  ❌ Range: 1-20")
            except ValueError:
                print("  ❌ Invalid. Enter an integer.")

        # Get threshold
        while True:
            try:
                thresh = float(input("  Enter threshold_percentile [95.0]: ") or "95.0")
                if 80 <= thresh <= 99:
                    params["threshold_percentile"] = thresh
                    break
                else:
                    print("  ❌ Range: 80-99")
            except ValueError:
                print("  ❌ Invalid. Enter a number.")

    elif method == "hampel_freq":
        print("\nFreq-Domain Hampel (Allen et al., 2010) Parameters:")
        print("  - window_hz: Frequency window (Hz), default 2.0")
        print("  - n_sigmas: Sensitivity threshold, default 3.0 (lower = more aggressive)")
        print("  - attenuation_db: Attenuation level in dB, default -60.0")

        # Get window_hz
        while True:
            try:
                window = float(input("  Enter window_hz [2.0]: ") or "2.0")
                if 0.5 <= window <= 5.0:
                    params["window_hz"] = window
                    break
                else:
                    print("  ❌ Range: 0.5-5.0 Hz")
            except ValueError:
                print("  ❌ Invalid. Enter a number.")

        # Get n_sigmas
        while True:
            try:
                sigmas = float(input("  Enter n_sigmas [3.0]: ") or "3.0")
                if 1.0 <= sigmas <= 6.0:
                    params["n_sigmas"] = sigmas
                    break
                else:
                    print("  ❌ Range: 1.0-6.0")
            except ValueError:
                print("  ❌ Invalid. Enter a number.")

        # Get attenuation
        while True:
            try:
                att = float(input("  Enter attenuation_db [-60.0]: ") or "-60.0")
                if -100 <= att <= -30:
                    params["attenuation_db"] = att
                    break
                else:
                    print("  ❌ Range: -100 to -30 dB")
            except ValueError:
                print("  ❌ Invalid. Enter a number.")

    elif method == "hampel_time":
        print("\nTime-Domain Hampel (Allen et al., 2010) Parameters:")
        print("  - window_sec: Time window (seconds), default 0.2")
        print("  - n_sigmas: Sensitivity threshold, default 3.0 (lower = more aggressive)")
        print("  - attenuation_factor: Multi-pass factor, default 1.0 (higher = stronger)")

        # Get window_sec
        while True:
            try:
                window = float(input("  Enter window_sec [0.2]: ") or "0.2")
                if 0.05 <= window <= 1.0:
                    params["window_sec"] = window
                    break
                else:
                    print("  ❌ Range: 0.05-1.0 seconds")
            except ValueError:
                print("  ❌ Invalid. Enter a number.")

        # Get n_sigmas
        while True:
            try:
                sigmas = float(input("  Enter n_sigmas [3.0]: ") or "3.0")
                if 1.0 <= sigmas <= 6.0:
                    params["n_sigmas"] = sigmas
                    break
                else:
                    print("  ❌ Range: 1.0-6.0")
            except ValueError:
                print("  ❌ Invalid. Enter a number.")

        # Get attenuation_factor
        while True:
            try:
                att_factor = float(input("  Enter attenuation_factor [1.0]: ") or "1.0")
                if 0.5 <= att_factor <= 5.0:
                    params["attenuation_factor"] = att_factor
                    break
                else:
                    print("  ❌ Range: 0.5-5.0")
            except ValueError:
                print("  ❌ Invalid. Enter a number.")

    return params


def process_file_with_wiener(
    filepath: Path, baseline_file: str, output_dir: Path, **kwargs
):
    """Process file using Wiener filter."""
    print(f"\n📖 Loading baseline: {baseline_file}")
    raw_baseline = mne.io.read_raw_edf(baseline_file, preload=True, verbose="WARNING")

    preprocessor = EEGPreprocessor(output_dir=str(output_dir))
    raw_baseline = preprocessor.standardize_channels(raw_baseline)
    raw_baseline.filter(l_freq=1.0, h_freq=70.0, fir_design="firwin", phase="zero", verbose="WARNING")

    print(f"📖 Loading DBS data: {filepath.name}")
    raw_dbs = mne.io.read_raw_edf(str(filepath), preload=True, verbose="WARNING")
    raw_dbs = preprocessor.standardize_channels(raw_dbs)
    raw_dbs.filter(l_freq=1.0, h_freq=70.0, fir_design="firwin", phase="zero", verbose="WARNING")

    print("🔄 Applying Baseline-Referenced Wiener Filter...")
    filt = BaselineReferencedFilter(
        baseline_raw=raw_baseline,
        dbs_freq=kwargs.get("dbs_freq", 7.0),
        harmonic_bandwidth=kwargs.get("harmonic_bandwidth", 1.5),
        alpha=kwargs.get("alpha", 1.5),
    )
    raw_clean = filt.filter(raw_dbs)

    out_file = output_dir / f"{filepath.stem}-cleaned-raw.fif"
    raw_clean.save(out_file, overwrite=True, verbose="WARNING")
    print(f"✅ Saved: {out_file}")

    return out_file


def process_file_with_method(
    filepath: Path, method: str, output_dir: Path, **kwargs
):
    """Process file with specified method."""
    preprocessor = EEGPreprocessor(output_dir=str(output_dir))

    print(f"📖 Loading: {filepath.name}")
    raw = mne.io.read_raw_edf(str(filepath), preload=True, verbose="WARNING")
    raw = preprocessor.standardize_channels(raw)
    raw.filter(l_freq=1.0, h_freq=70.0, fir_design="firwin", phase="zero", verbose="WARNING")

    print(f"🔄 Applying {method}...")

    if method == "spectrum_fit":
        raw_clean = preprocessor.remove_artifacts_spectrum_fit(
            raw,
            f_target=kwargs.get("f_target"),
            bandwidth=kwargs.get("bandwidth", 2.0),
            attenuation_db=kwargs.get("attenuation_db", -60.0),
        )
    elif method == "zapline":
        raw_clean = preprocessor.remove_artifacts_zapline(
            raw,
            f_target=kwargs.get("f_target"),
            n_harmonics=kwargs.get("n_harmonics", 10),
            threshold_percentile=kwargs.get("threshold_percentile", 95.0),
        )
    elif method == "hampel_freq":
        raw_clean = preprocessor.remove_artifacts_freq_hampel(
            raw,
            window_hz=kwargs.get("window_hz", 2.0),
            n_sigmas=kwargs.get("n_sigmas", 3.0),
            attenuation_db=kwargs.get("attenuation_db", -60.0),
        )
    elif method == "hampel_time":
        raw_clean = preprocessor.remove_artifacts_time_hampel(
            raw,
            window_sec=kwargs.get("window_sec", 0.2),
            n_sigmas=kwargs.get("n_sigmas", 3.0),
            attenuation_factor=kwargs.get("attenuation_factor", 1.0),
        )
    else:
        print(f"❌ Unknown method: {method}")
        return None

    out_file = output_dir / f"{filepath.stem}-cleaned-raw.fif"
    raw_clean.save(out_file, overwrite=True, verbose="WARNING")
    print(f"✅ Saved: {out_file}")

    return out_file


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="DBS Artifact Removal - Interactive CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended)
  python remove_dbs_artifacts_cli.py

  # Direct mode with specific file and method
  python remove_dbs_artifacts_cli.py \\
    --input-file data/XU/XUAWAKE7_deidentified.edf \\
    --method spectrum_fit \\
    --f-target 7.0 \\
    --attenuation-db -60.0

  # Wiener filter with baseline
  python remove_dbs_artifacts_cli.py \\
    --input-file data/XU/XUAWAKE7_deidentified.edf \\
    --method wiener \\
    --baseline-file data/XU/XUAWAKEPRE_deidentified.edf \\
    --alpha 1.5
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/XU/",
        help="Input directory with EDF files (default: data/XU/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/",
        help="Output directory for processed files (default: data/processed/)",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Specific input file (full path). If not provided, interactive selection.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["spectrum_fit", "wiener", "zapline", "hampel_freq", "hampel_time"],
        help="Artifact removal method. If not provided, interactive selection.",
    )
    parser.add_argument(
        "--f-target",
        type=float,
        help="Fundamental DBS frequency (Hz)",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=2.0,
        help="Bandwidth for Spectrum-Fit (Hz, default: 2.0)",
    )
    parser.add_argument(
        "--attenuation-db",
        type=float,
        default=-60.0,
        help="Attenuation for Spectrum-Fit/Hampel Freq (dB, default: -60.0)",
    )
    parser.add_argument(
        "--baseline-file",
        type=str,
        help="Baseline EDF file for Wiener filter",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.5,
        help="Alpha for Wiener filter (default: 1.5)",
    )
    parser.add_argument(
        "--n-harmonics",
        type=int,
        default=10,
        help="Number of harmonics for Zapline+ (default: 10)",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=95.0,
        help="Threshold percentile for Zapline+ (default: 95.0)",
    )
    parser.add_argument(
        "--window-hz",
        type=float,
        default=2.0,
        help="Frequency window for Hampel Freq (Hz, default: 2.0)",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=0.2,
        help="Time window for Hampel Time (sec, default: 0.2)",
    )
    parser.add_argument(
        "--n-sigmas",
        type=float,
        default=3.0,
        help="Sigma threshold for Hampel methods (default: 3.0)",
    )
    parser.add_argument(
        "--attenuation-factor",
        type=float,
        default=1.0,
        help="Attenuation factor for Hampel Time (default: 1.0)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all files in input directory",
    )

    args = parser.parse_args()

    # Prepare directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_header("DBS Artifact Removal - CLI Tool")

    # Select method
    if args.method:
        method = args.method
        print(f"✓ Method selected: {method}")
    else:
        method = select_method_interactive()
        print(f"✓ Method selected: {method}")

    # Get parameters
    if (
        args.method
        and args.f_target is not None
        and (method != "wiener" or args.baseline_file)
    ):
        # Direct mode with CLI args
        params = {
            "f_target": args.f_target,
            "bandwidth": args.bandwidth,
            "attenuation_db": args.attenuation_db,
            "baseline_file": args.baseline_file,
            "alpha": args.alpha,
            "n_harmonics": args.n_harmonics,
            "threshold_percentile": args.threshold_percentile,
            "window_hz": args.window_hz,
            "window_sec": args.window_sec,
            "n_sigmas": args.n_sigmas,
            "attenuation_factor": args.attenuation_factor,
        }
        print("✓ Parameters from CLI args")
    else:
        # Interactive parameter selection
        params = get_method_parameters(method, EEGPreprocessor(output_dir=str(output_dir)))
        if params is None:
            sys.exit(1)
        print("✓ Parameters configured")

    # Select file(s)
    if args.input_file:
        files = [Path(args.input_file).name]
        print(f"✓ File selected: {files[0]}")
    elif args.batch:
        edf_files = list(input_dir.glob("*.edf"))
        files = [f.name for f in edf_files]
        print(f"✓ Batch mode: {len(files)} file(s) selected")
    else:
        files = select_input_file_interactive(input_dir)
        print(f"✓ File(s) selected: {len(files)} file(s)")

    # Process file(s)
    print_header("Processing")
    processed_files = []

    for filename in files:
        filepath = input_dir / filename
        print(f"\n[{files.index(filename) + 1}/{len(files)}] {filename}")

        try:
            if method == "wiener":
                out_file = process_file_with_wiener(
                    filepath, params["baseline_file"], output_dir, **params
                )
            else:
                out_file = process_file_with_method(
                    filepath, method, output_dir, **params
                )
            if out_file:
                processed_files.append(out_file)
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print_header("Summary")
    print(f"✅ Processed: {len(processed_files)}/{len(files)} files")
    print(f"📁 Output directory: {output_dir.resolve()}")

    if processed_files:
        print("\n📋 Output files:")
        for f in processed_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  • {f.name} ({size_mb:.1f} MB)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
