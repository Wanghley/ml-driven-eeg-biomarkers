import os
import re
from pathlib import Path
from typing import List, Optional, Union

import mne
import numpy as np
import matplotlib.pyplot as plt

from src.filters import ArtifactFilterFactory, BaselineReferencedFilter

class EEGPreprocessor:
    """
    A robust, object-oriented pipeline for preprocessing continuous EEG data.
    
    This class handles:
    - Data Ingestion: Recursively finding and loading `.edf` files.
    - Channel Standardization: Filtering out dummy/ECG channels and retaining 
      core 10-20 system channels.
    - Clinical Filtering: Zero-phase FIR bandpass filtering.
    - Artifact Removal: Steep Notch filters for DBS (e.g., 130/160 Hz) 
      and 60 Hz line noise artifacts + harmonics.
    - Exporting: Saving to MNE's native `.fif` format.
    """
    
    # Core 10-20 system EEG channels. We exclude ECG, EOG, and auxiliary channels.
    STANDARD_1020_CHANNELS = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
        'T3', 'C3', 'Cz', 'C4', 'T4', 
        'T5', 'P3', 'Pz', 'P4', 'T6', 
        'O1', 'O2'
    ]

    def __init__(self, 
                 input_dir: Union[str, Path] = "data/XU/", 
                 output_dir: Union[str, Path] = "data/processed/",
                 l_freq: float = 1.0,
                 h_freq: float = 70.0,
                 line_noise_freq: float = 60.0,
                 dbs_freqs: Optional[List[float]] = None,
                 generate_plots: bool = False,
                 plot_intermediate: bool = False,
                 plot_dir: Union[str, Path] = "figures/preprocessing/"):
        """
        Initialize the EEGPreprocessor with pipeline parameters.
        
        Args:
            input_dir (Union[str, Path]): Directory to search for `.edf` files recursively.
            output_dir (Union[str, Path]): Directory to save processed `.fif` files.
            l_freq (float): Lower pass-band edge (Hz) for FIR filter.
            h_freq (float): Upper pass-band edge (Hz) for FIR filter.
            line_noise_freq (float): Frequency of AC line noise (e.g., 60.0 Hz).
            dbs_freqs (Optional[List[float]]): List of primary DBS frequencies (e.g., [130.0, 160.0]).
            generate_plots (bool): Whether to generate PSD visualizations comparing before and after processing.
            plot_intermediate (bool): Whether to generate plots after every intermediate filtering step.
            plot_dir (Union[str, Path]): Directory to save the generated figures.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.line_noise_freq = line_noise_freq
        self.dbs_freqs = dbs_freqs or [130.0, 160.0]
        self.generate_plots = generate_plots
        self.plot_intermediate = plot_intermediate
        self.plot_dir = Path(plot_dir)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.generate_plots:
            self.plot_dir.mkdir(parents=True, exist_ok=True)

    def find_edf_files(self) -> List[Path]:
        """
        Recursively find all .edf files in the input directory.
        
        Returns:
            List[Path]: A list of file paths pointing to the .edf files.
        """
        # Look for lowercase extension
        edf_files = list(self.input_dir.rglob("*.edf"))
        # Also include uppercase extension just in case
        edf_files.extend(list(self.input_dir.rglob("*.EDF")))
        
        # Remove duplicates
        unique_files = list(set(edf_files))
        return unique_files

    def standardize_channels(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Standardize channel names by stripping prefixes/suffixes. 
        Drops dummy and ECG/EOG channels, keeping only standard 10-20 EEG channels.
        
        Args:
            raw (mne.io.Raw): Unprocessed MNE Raw object.
            
        Returns:
            mne.io.Raw: MNE Raw object containing only standardized 10-20 channels.
        """
        # Working with a copy to avoid unintended modifications
        raw = raw.copy()
        
        # Channel naming in clinical EDFs often comes with varied affixes 
        # (e.g., "EEG Fp1-REF", "Fp1-LE", "Fp1"). We map these to standard names.
        rename_mapping = {}
        for ch_name in raw.ch_names:
            for std_ch in self.STANDARD_1020_CHANNELS:
                # Check for standard channel name within the original channel string,
                # isolated by word boundaries or hyphens/spaces.
                pattern = rf"\b{std_ch}\b|[^A-Za-z0-9]{std_ch}[^A-Za-z0-9]"
                if re.search(pattern, ch_name, re.IGNORECASE) or std_ch.lower() == ch_name.lower().split('-')[0].split(' ')[-1]:
                    rename_mapping[ch_name] = std_ch
                    break
                    
        # Apply standard names to the mapped channels
        if rename_mapping:
            raw.rename_channels(rename_mapping)
        
        # Identify channels that successfully matched our 10-20 system list
        ch_to_keep = [ch for ch in raw.ch_names if ch in self.STANDARD_1020_CHANNELS]
        
        if not ch_to_keep:
            raise ValueError("No standard 10-20 channels could be identified in the EDF file.")
            
        # Drop all non-matching (dummy/ECG) channels
        raw.pick(ch_to_keep)
        
        # Explicitly declare the remaining channels as EEG type to MNE
        ch_types = {ch: 'eeg' for ch in ch_to_keep}
        raw.set_channel_types(ch_types)
        
        # Standardize the physical locational montage mapping for plotting and source analysis
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_case=False, on_missing='ignore')
        
        return raw

    def apply_clinical_filter(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Apply a zero-phase FIR bandpass filter. 
        E.g., 1.0 Hz - 70.0 Hz to eliminate slow drifts and ultra-high frequency noise.
        
        Args:
            raw (mne.io.Raw): MNE Raw object to filter.
            
        Returns:
            mne.io.Raw: Bandpass filtered MNE Raw object.
        """
        print(f"Applying zero-phase FIR bandpass filter: {self.l_freq} - {self.h_freq} Hz")
        # FIR design 'firwin' and phase='zero' are typically MNE defaults, explicitly enforced here
        raw.filter(l_freq=self.l_freq, h_freq=self.h_freq, 
                   fir_design='firwin', phase='zero', verbose='WARNING')
        return raw
        
    def remove_artifacts(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Apply specific, steep Notch filters to remove primary DBS hardware artifacts, 
        their respective harmonics, and standardized line noise.
        
        Args:
            raw (mne.io.Raw): MNE Raw object.
            
        Returns:
            mne.io.Raw: Artifact-cleaned MNE Raw object.
        """
        # Nyquist frequency specifies the maximum resolvable frequency
        sfreq = raw.info['sfreq']
        nyquist = sfreq / 2.0
        
        target_freqs = []
        
        # Generate harmonics up to the Nyquist limit for Line Noise
        target_freqs.extend(np.arange(self.line_noise_freq, nyquist, self.line_noise_freq))
        
        # Try to dynamically identify a DBS frequency from the filename (e.g. 'XUAWAKE7' -> 7Hz)
        dynamic_dbs_freqs = list(self.dbs_freqs) if self.dbs_freqs else []
        if raw.filenames and raw.filenames[0]:
            filename_stem = Path(raw.filenames[0]).stem
            match = re.search(r'(\d+)', filename_stem)
            if match:
                f0 = float(match.group(1))
                if f0 not in dynamic_dbs_freqs:
                    dynamic_dbs_freqs.append(f0)

        # Generate harmonics up to the Nyquist limit for each targeted DBS artifact baseline frequency
        for dbs_f in dynamic_dbs_freqs:
            target_freqs.extend(np.arange(dbs_f, nyquist, dbs_f))
            
        # Convert to a unique, sorted list of frequencies to filter
        target_freqs = sorted(list(set(target_freqs)))

        # MNE requires sufficient separation between adjacent notch stop bands.
        # Each notch of width `notch_widths` also needs a transition band on each side.
        # Enforce minimum separation to prevent "Stop bands are not sufficiently separated" errors
        # when DBS harmonics fall close to line noise frequencies (e.g. 7 Hz harmonics near 60 Hz).
        notch_widths = 2.0
        min_sep = notch_widths + 2.0  # conservative: stop band + 1 Hz transition on each side
        deduped_freqs = []
        for f in target_freqs:
            if not deduped_freqs or (f - deduped_freqs[-1]) >= min_sep:
                deduped_freqs.append(f)
        target_freqs = deduped_freqs

        print(f"Applying steep Notch filters at frequencies (Hz): {target_freqs}")

        if target_freqs:
            # We use notch_widths=2.0 for a sharper/steeper cut. Adjust if clipping becomes an issue.
            raw.notch_filter(freqs=target_freqs,
                             fir_design='firwin',
                             phase='zero',
                             notch_widths=notch_widths,
                             verbose='WARNING')
                             
        return raw

    def remove_artifacts_baseline(self, raw: mne.io.Raw, baseline_raw: mne.io.Raw,
                                    dbs_freq: float = 7.0, **kwargs) -> mne.io.Raw:
        """
        Remove DBS artifacts using a clean baseline recording as spectral reference.

        This replaces the notch filter approach (remove_artifacts) with a
        Wiener spectral gating method that preserves endogenous brain activity
        at the DBS fundamental and harmonic frequencies.

        Args:
            raw: DBS-contaminated MNE Raw object.
            baseline_raw: Clean baseline MNE Raw object (same channels/sfreq).
            dbs_freq: DBS stimulation frequency in Hz.
            **kwargs: Passed to BaselineReferencedFilter (harmonic_bandwidth,
                      n_fft, floor_db).

        Returns:
            Cleaned MNE Raw object.
        """
        filt = BaselineReferencedFilter(
            baseline_raw=baseline_raw,
            dbs_freq=dbs_freq,
            **kwargs
        )
        return filt.filter(raw)

    def remove_artifacts_advanced(self, raw: mne.io.Raw, method: str = 'comb_notch', **kwargs) -> mne.io.Raw:
        """
        Apply advanced, heavily vectorized artifact removal mapping.
        
        Args:
            raw (mne.io.Raw): MNE Raw object.
            method (str): Filter method ('hampel_time', 'hampel_freq', 'spectrum_fit', 
                                       'zapline', 'comb_notch').
            **kwargs: Extra parameters for the chosen filter algorithm.
            
        Returns:
            mne.io.Raw: Cleaned MNE Raw object.
        """
        # Create a copy so we don't modify the object in place destructively without intent
        raw_clean = raw.copy()
        raw_clean.load_data()
        
        data = raw_clean.get_data()
        sfreq = raw_clean.info['sfreq']
        
        # Default target freq configuration to DBS if not provided explicitly in kwargs
        if 'f_target' not in kwargs and 'f0' not in kwargs:
            f0 = None
            
            # 1. Try to extract dynamic target freq from the filename first (e.g. 'XuSleep7' -> 7.0)
            if raw.filenames and raw.filenames[0]:
                filename_stem = Path(raw.filenames[0]).stem
                match = re.search(r'(\d+)', filename_stem)
                if match:
                    f0 = float(match.group(1))
                    
            # 2. Fallback to initialized instance default freq
            if f0 is None:
                f0 = self.dbs_freqs[0]
                
            if method.lower() in ['comb_notch', 'spectrum_fit', 'zapline', 'hampel_freq', 'hampel_time']:
                kwargs['f_target'] = f0

        cleaned_data = ArtifactFilterFactory.process(method=method, data=data, sfreq=sfreq, **kwargs)
        
        # Put back into MNE structure
        raw_clean._data = cleaned_data
        return raw_clean

    def remove_artifacts_spectrum_fit(self, raw: mne.io.Raw, f_target: Optional[float] = None, 
                                      bandwidth: float = 2.0, attenuation_db: float = -60.0) -> mne.io.Raw:
        """
        Remove DBS artifacts using Spectrum-Fit Multi-Harmonic Removal.
        
        Applies frequency-domain gain mask to smoothly attenuate harmonics while
        preserving endogenous brain activity.
        
        Args:
            raw: DBS-contaminated MNE Raw object
            f_target: Fundamental frequency (Hz). If None, auto-detected from filename
            bandwidth: Total bandwidth around each harmonic to filter (Hz)
            attenuation_db: Attenuation level in dB (more negative = stronger removal)
            
        Returns:
            Cleaned MNE Raw object
        """
        return self.remove_artifacts_advanced(raw, method='spectrum_fit', 
                                              f_target=f_target, 
                                              bandwidth=bandwidth,
                                              attenuation_db=attenuation_db)

    def remove_artifacts_zapline(self, raw: mne.io.Raw, f_target: Optional[float] = None,
                                 n_harmonics: int = 10, threshold_percentile: float = 95.0) -> mne.io.Raw:
        """
        Remove DBS artifacts using Zapline+ (Chen et al., 2022).
        
        Adaptive notch filtering using spectro-spatial decomposition to identify
        and remove artifact subspace while preserving brain activity.
        
        Args:
            raw: DBS-contaminated MNE Raw object
            f_target: Fundamental frequency (Hz). If None, auto-detected from filename
            n_harmonics: Number of harmonics to consider
            threshold_percentile: Percentile threshold for artifact detection (higher = more selective)
            
        Returns:
            Cleaned MNE Raw object
        """
        return self.remove_artifacts_advanced(raw, method='zapline',
                                              f_target=f_target,
                                              n_harmonics=n_harmonics,
                                              threshold_percentile=threshold_percentile)

    def remove_artifacts_time_hampel(self, raw: mne.io.Raw, 
                                     window_sec: float = 0.2, n_sigmas: float = 3.0,
                                     attenuation_factor: float = 1.0) -> mne.io.Raw:
        """
        Remove DBS artifacts using Time-Domain Hampel Filter (Allen et al., 2010).
        
        Detects and removes impulsive artifacts using median-based outlier detection.
        Effective for sharp DBS pulses and transient artifacts.
        
        Args:
            raw: EEG MNE Raw object
            window_sec: Window duration for median filtering (seconds)
            n_sigmas: Sensitivity threshold in MAD units (higher = more selective)
            attenuation_factor: Strength of removal (>1.0 for multiple passes)
            
        Returns:
            Cleaned MNE Raw object
        """
        return self.remove_artifacts_advanced(raw, method='hampel_time',
                                              window_sec=window_sec,
                                              n_sigmas=n_sigmas,
                                              attenuation_factor=attenuation_factor)

    def remove_artifacts_freq_hampel(self, raw: mne.io.Raw,
                                     window_hz: float = 2.0, n_sigmas: float = 3.0,
                                     attenuation_db: float = -60.0) -> mne.io.Raw:
        """
        Remove DBS artifacts using Frequency-Domain Hampel Filter (Allen et al., 2010).
        
        Detects spectral peaks (harmonics) via FFT magnitude analysis and scales them down
        while preserving phase relationships. Excellent for harmonic suppression.
        
        Args:
            raw: EEG MNE Raw object
            window_hz: Window size for rolling median in frequency (Hz)
            n_sigmas: Sensitivity threshold in MAD units (higher = more selective)
            attenuation_db: Target attenuation for detected peaks (dB, more negative = stronger)
            
        Returns:
            Cleaned MNE Raw object
        """
        return self.remove_artifacts_advanced(raw, method='hampel_freq',
                                              window_hz=window_hz,
                                              n_sigmas=n_sigmas,
                                              attenuation_db=attenuation_db)

    def _save_plot(self, raw: mne.io.Raw, title: str, save_path: Path):
        """
        Helper method to generate and save a Power Spectral Density (PSD) plot.
        """
        if not self.generate_plots:
            return
            
        try:
            # Use fmax = min(Nyquist, 200 Hz) to efficiently capture the relevant DBS frequencies (130-160 Hz)
            sfreq = raw.info['sfreq']
            fmax = min(sfreq / 2.0, 200.0)
            
            # compute_psd is the modern interface in MNE >= 1.3
            fig = raw.compute_psd(fmax=fmax).plot(show=False)
            
            # Format and save
            fig.suptitle(title, fontsize=14)
            fig.tight_layout()
            fig.savefig(save_path)
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not generate plot {title}. Reason: {e}")
        
    def process_file(self, filepath: Path) -> Optional[Path]:
        """
        Full lifecycle processing on a single .edf file.
        
        Args:
            filepath (Path): Input file path.
            
        Returns:
            Optional[Path]: Output .fif file path if successfully completed, else None.
        """
        try:
            print(f"---\nProcessing: {filepath}")
            # Preload is necessary as time-domain filtering requires data to be loaded in memory
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose='WARNING')
            
            # Prepare plotting directory for this specific file
            file_plot_dir = self.plot_dir / filepath.stem
            if self.generate_plots:
                file_plot_dir.mkdir(parents=True, exist_ok=True)
            
            # Application flow:
            # 1. Clean up montage and retain EEG only
            raw = self.standardize_channels(raw)
            
            if self.generate_plots:
                self._save_plot(raw, "PSD - 1. Standardized raw (Before Filtering)", file_plot_dir / "01_before_filtering.png")
            
            # 2. Broadband bandpass limiting
            raw = self.apply_clinical_filter(raw)
            
            if self.generate_plots and self.plot_intermediate:
                self._save_plot(raw, "PSD - 2. After FIR Bandpass Filter", file_plot_dir / "02_after_bandpass.png")
                
            # 3. Clean targeted narrow-frequency artifact noise
            raw = self.remove_artifacts(raw)
            
            if self.generate_plots:
                self._save_plot(raw, "PSD - 3. Final Preprocessed (After Artifact Filters)", file_plot_dir / "03_after_artifacts.png")
            
            # Deduce output name: e.g., 'patient1_record1.EDF' -> 'patient1_record1-raw.fif'
            # FIF extension requires suffix logic usually terminating in 'raw.fif' or 'epo.fif' etc.
            out_filename = self.output_dir / f"{filepath.stem}-raw.fif"
            
            # Save the fully qualified processed raw structure natively preserving all metadata
            raw.save(out_filename, overwrite=True, verbose='WARNING')
            
            print(f"Successfully processed and saved to: {out_filename}")
            return out_filename
            
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
            return None
            
    def run_pipeline(self):
        """
        Entry point to process sequentially all discovered `.edf` data sets.
        """
        files = self.find_edf_files()
        if not files:
            print(f"No .edf files located in {self.input_dir.resolve()}. Exiting.")
            return
            
        print(f"Discovered {len(files)} target (.edf) file(s). Igniting preprocessing pipeline...")
        
        processed_files = []
        for file in files:
            out_path = self.process_file(file)
            if out_path:
                processed_files.append(out_path)
                
        print(f"Pipeline executed. Preprocessed {len(processed_files)} out of {len(files)} files.")


if __name__ == "__main__":
    # Example Initialization and Execution Process
    preprocessor = EEGPreprocessor(
        input_dir="data/XU/", 
        output_dir="data/processed/",
        l_freq=1.0,
        h_freq=70.0,
        line_noise_freq=60.0,
        dbs_freqs=[130.0, 160.0],
        generate_plots=True,
        plot_intermediate=True
    )
    preprocessor.run_pipeline()
