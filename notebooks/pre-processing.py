# %%
"""
EEG preprocessing & visualization notebook (MNE)
Cells are separated with `# %%` so you can paste directly into a Jupyter/Colab / VSCode Python notebook.

Features:
- Load EDF
- Robust channel-typing (maps LOC1/LOC2 -> EOG)
- Set montage & reference
- Filtering (bandpass + notch)
- PSD generation + comparison at every step
- Band-power calculation and topomaps
- ICA-based artifact detection (using LOC1/LOC2 as EOG channels)
- Figures saved to ./figures/

Notes:
- This notebook is based on your uploaded script (I inspected your file and reorganized it). See: uploaded file. 

"""

# %%
# Imports & config
import os
from pathlib import Path
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import psd_array_welch
import pandas as pd
from typing import Optional, List, Tuple, Dict

# plotting defaults
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

OUT_DIR = Path('./figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)

print('MNE version:', mne.__version__)

# %%
# Utility: load EDF (single file) - returns Raw
def load_edf(path: str, preload=True) -> mne.io.Raw:
    raw = mne.io.read_raw_edf(path, preload=preload, verbose='ERROR')
    return raw

# %%
# Utility: Print summary info
def print_eeg_info(raw: mne.io.Raw):
    print('\n' + '='*60)
    subj = raw.info.get('subject_info', {}).get('his_id', 'unknown')
    print(f"EEG DATA INFO: subject={subj}")
    print('='*60)
    print(f"Duration: {raw.times[-1]/60:.2f} min | sfreq: {raw.info['sfreq']} Hz | n_channels: {len(raw.ch_names)}")
    print('\nChannels:')
    print(raw.ch_names)

# %%
# Channel typing: ensure LOC1/LOC2 -> eog and sensible eeg detection
def set_custom_channel_types(raw: mne.io.Raw, loc_names: List[str] = ['LOC1', 'LOC2']):
    ch_types = {}
    for ch in raw.ch_names:
        # if channel appears like LOC* or ROC* mark as eog
        if any(ch.upper().startswith(x.upper()) for x in loc_names) or ch.upper().startswith('ROC'):
            ch_types[ch] = 'eog'
        # some common prefixes
        elif ch.upper().startswith('EMG'):
            ch_types[ch] = 'emg'
        elif ch.upper().startswith('EKG') or ch.upper().startswith('ECG'):
            ch_types[ch] = 'ecg'
        # fallback: try to detect standard 10-20
        else:
            ch_types[ch] = 'eeg'  # conservative: treat as EEG unless known otherwise
    raw.set_channel_types(ch_types)
    print(f"Set channel types. Marked {', '.join(ch for ch,t in ch_types.items() if t=='eog')} as EOG")

# %%
# Montage & reference helpers
def try_set_montage(raw: mne.io.Raw, montage_name: str = 'standard_1020'):
    try:
        montage = mne.channels.make_standard_montage(montage_name)
        raw.set_montage(montage, on_missing='warn')
        print('Montage set:', montage_name)
    except Exception as e:
        print('Could not set montage:', e)

def set_reference(raw: mne.io.Raw, ref_channels: Optional[List[str]] = None):
    if ref_channels is None:
        raw.set_eeg_reference('average')
        print('Set average reference')
    else:
        raw.set_eeg_reference(ref_channels)
        print('Set reference to:', ref_channels)

# %%
# Filtering and notch
def apply_filters(raw: mne.io.Raw, l_freq: Optional[float], h_freq: Optional[float], notch_freqs: Optional[List[float]] = None):
    raw_filtered = raw.copy()
    raw_filtered.load_data()
    if l_freq is not None or h_freq is not None:
        raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')
        print(f'Applied bandpass: {l_freq} - {h_freq} Hz')
    if notch_freqs:
        raw_filtered.notch_filter(freqs=notch_freqs, fir_design='firwin')
        print('Applied notch at:', notch_freqs)
    return raw_filtered

# %%
# PSD computation using psd_array_welch for arrays (fast) but also wrapper for Raw
def compute_psd(raw: mne.io.Raw, fmin=0.5, fmax=40.0, n_fft=2048, n_overlap=None, n_per_seg=None):
    picks = mne.pick_types(raw.info, eeg=True, eog=False, exclude='bads')
    if len(picks) == 0:
        raise RuntimeError('No EEG channels found for PSD')
    data = raw.get_data(picks=picks)
    sfreq = raw.info['sfreq']
    # defaults
    if n_overlap is None:
        n_overlap = n_fft // 2
    psds, freqs = psd_array_welch(data, sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg)
    # psds shape: (n_channels, n_freqs)
    ch_names = [raw.ch_names[p] for p in picks]
    return freqs, psds, ch_names

# %%
# Plot PSD for one channel or mean across channels
def plot_psd(freqs, psds, ch_names, picks: Optional[List[str]] = None, title: str = None, save_name: Optional[str] = None, fmin=0.5, fmax=40.0):
    if picks is None:
        # plot average across channels
        psd_to_plot = psds.mean(axis=0)
        label = 'Average EEG'
    else:
        # map picks names to indices
        indices = [ch_names.index(p) for p in picks if p in ch_names]
        if len(indices) == 0:
            raise RuntimeError('Picked channels not found in ch_names')
        psd_to_plot = psds[indices].mean(axis=0)
        label = ','.join([ch_names[i] for i in indices])
    plt.figure()
    plt.semilogy(freqs, psd_to_plot)
    plt.xlim(fmin, fmax)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (uV^2/Hz)')
    if title:
        plt.title(title)
    else:
        plt.title(f'PSD - {label}')
    if save_name:
        plt.savefig(OUT_DIR / save_name, dpi=200)
    plt.show()

# %%
# Band powers and topomap
def calculate_band_powers(psds: np.ndarray, freqs: np.ndarray, ch_names: List[str], bands: Optional[Dict[str, Tuple[float, float]]] = None) -> pd.DataFrame:
    if bands is None:
        bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 40)
        }
    band_powers = {}
    df_index = ch_names
    for band, (fmin, fmax) in bands.items():
        freq_idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        # integrate PSD across band
        band_powers[band] = np.trapz(psds[:, freq_idx], x=freqs[freq_idx], axis=1)
    df = pd.DataFrame(band_powers, index=df_index)
    df.index.name = 'Channel'
    return df

def plot_band_topomaps(raw: mne.io.Raw, band_power_df: pd.DataFrame, save_name: Optional[str] = None, title: Optional[str] = None):
    """Plot band-power topomaps robustly.

    This function now:
    - selects only channels present in both the DataFrame and the raw
    - restricts to channels that exist in the standard_1020 montage (if available)
    - removes non-EEG channels and duplicate / overlapping electrode positions
    - logs channels that are dropped and proceeds with the remaining ones
    """
    # 1) intersect channels
    chs = [ch for ch in band_power_df.index if ch in raw.ch_names]
    if len(chs) == 0:
        print('No matching channels for topomap')
        return

    # 2) try to get a standard montage and limit to channels that have positions
    try:
        std_montage = mne.channels.make_standard_montage('standard_1020')
        montage_chs = set(std_montage.ch_names)
    except Exception:
        std_montage = None
        montage_chs = set()

    # keep only EEG channels from raw
    picks_eeg = mne.pick_types(raw.info, eeg=True, eog=False, emg=False, exclude='bads')
    eeg_chs = [raw.ch_names[p] for p in picks_eeg]

    # candidate channels: present in df, present in raw EEG
    candidate_chs = [ch for ch in chs if ch in eeg_chs]

    # if we have a standard montage, prefer channels present in montage
    if montage_chs:
        candidate_chs = [ch for ch in candidate_chs if ch in montage_chs]

    if len(candidate_chs) == 0:
        print('No suitable EEG channels with montage positions found for topomap. Channels considered:', chs)
        return

    # 3) build positions array and drop overlapping positions
    # get positions from the standard montage if available, otherwise use raw.get_montage()
    if std_montage is not None:
        ch_pos = std_montage.get_positions()['ch_pos']
    else:
        mont = raw.get_montage()
        if mont is None:
            print('No montage available and no standard montage could be created. Cannot make topomap.')
            return
        ch_pos = mont.get_positions()['ch_pos']

    pos_list = []
    keep_chs = []
    dropped_duplicates = []
    seen = []
    for ch in candidate_chs:
        pos = ch_pos.get(ch)
        if pos is None:
            continue
        # round positions to mm-level tolerance to detect overlapping
        key = tuple(np.round(np.array(pos), 4))
        if key in seen:
            dropped_duplicates.append(ch)
            continue
        seen.append(key)
        keep_chs.append(ch)
        pos_list.append(pos)

    if len(keep_chs) == 0:
        print('After removing duplicates/no-position channels, nothing left to plot. Dropped:', dropped_duplicates)
        return

    if dropped_duplicates:
        print('Dropped channels with overlapping/duplicate positions:', dropped_duplicates)

    # 4) prepare an info object limited to keep_chs
    info_subset = mne.create_info(ch_names=keep_chs, sfreq=raw.info['sfreq'], ch_types=['eeg'] * len(keep_chs))
    # attach the montage to the subset info
    if std_montage is not None:
        info_subset.set_montage(std_montage, on_missing='ignore')
    else:
        info_subset.set_montage(raw.get_montage(), on_missing='ignore')

    # 5) plot topomaps for each band
    fig = plt.figure(figsize=(14, 8))
    n = len(band_power_df.columns)
    for i, band in enumerate(band_power_df.columns):
        ax = fig.add_subplot(1, n, i+1)
        data = band_power_df.loc[keep_chs, band].values
        # mne expects data in order of info_subset.ch_names
        mne.viz.plot_topomap(data, info_subset, axes=ax, show=False)
        ax.set_title(band)
    if title:
        fig.suptitle(title)
    if save_name:
        plt.savefig(OUT_DIR / save_name, dpi=200)
    plt.show()

# %%
# ICA artifact detection using EOG channels (LOC1/LOC2) if present
def run_ica_detect(raw: mne.io.Raw, eog_chs: List[str] = ['LOC1', 'LOC2'], n_components: Optional[int] = 0.95, random_state: int = 97):
    # prepare copy for ICA
    raw_ica = raw.copy()
    picks_eeg = mne.pick_types(raw_ica.info, eeg=True, eog=False, exclude='bads')

    ica = mne.preprocessing.ICA(n_components=n_components, random_state=random_state, max_iter='auto')
    print('Fitting ICA...')
    ica.fit(raw_ica, picks=picks_eeg)
    print('ICA fitted')

    # try to find EOG-related components using provided loc channels
    present_eog_chs = [ch for ch in eog_chs if ch in raw_ica.ch_names]
    eog_indices = []
    eog_scores = None
    if len(present_eog_chs) > 0:
        try:
            eog_indices, eog_scores = ica.find_bads_eog(raw_ica, ch_name=present_eog_chs)
            print('Found EOG ICs:', eog_indices)
        except Exception as e:
            print('find_bads_eog failed:', e)
    else:
        print('No LOC/ROC channels found in data for automatic EOG detection.')

    # Return also a list of EEG channel names used for ICA plotting convenience
    eeg_ch_names = [raw_ica.ch_names[p] for p in picks_eeg]
    return ica, eog_indices, eog_scores, eeg_ch_names

# %%
# Apply ICA exclusion and return cleaned raw
def apply_ica(raw: mne.io.Raw, ica: mne.preprocessing.ICA, exclude: List[int]):
    raw_clean = raw.copy()
    ica.exclude = exclude
    ica.apply(raw_clean)
    print('Applied ICA exclusions:', exclude)
    return raw_clean

# %%
# Full pipeline runner (example usage)
def pipeline_run(edf_path: str, out_prefix: str = 'subject'):
    # 0) load
    raw = load_edf(edf_path)
    print_eeg_info(raw)

    # 1) set types (map LOC1,LOC2)
    set_custom_channel_types(raw, loc_names=['LOC1', 'LOC2'])

    # quick raw plot
    raw.plot(n_channels=20, show=True, block=False)
    plt.savefig(OUT_DIR / f'{out_prefix}_raw.png', dpi=200)

    # 2) set montage & reference
    try_set_montage(raw)
    set_reference(raw, ref_channels=None)  # average

    # 3) initial PSD (before filtering)
    freqs_raw, psds_raw, ch_names = compute_psd(raw, fmin=0.5, fmax=40.0, n_fft=2048)
    plot_psd(freqs_raw, psds_raw, ch_names, picks=None, title='PSD - BEFORE filtering', save_name=f'{out_prefix}_psd_before.png')

    # 4) filtering
    raw_filt = apply_filters(raw, l_freq=0.5, h_freq=40.0, notch_freqs=[50.0, 100.0])
    freqs_filt, psds_filt, ch_names_filt = compute_psd(raw_filt, fmin=0.5, fmax=40.0, n_fft=2048)
    plot_psd(freqs_filt, psds_filt, ch_names_filt, picks=None, title='PSD - AFTER filtering', save_name=f'{out_prefix}_psd_after.png')

    # 5) band-power & topomap (after filtering)
    df_band = calculate_band_powers(psds_filt, freqs_filt, ch_names_filt)
    plot_band_topomaps(raw_filt, df_band, save_name=f'{out_prefix}_topomap_after_filter.png', title='Band powers - after filtering')

    # 6) ICA stage for artifact detection using LOC channels
    raw_ica_ready = raw_filt.copy()
    # highpass a bit for ICA stability
    raw_ica_ready.filter(l_freq=1.0, h_freq=None, fir_design='firwin')
    ica, eog_indices, eog_scores, ica_eeg_chs = run_ica_detect(raw_ica_ready, eog_chs=['LOC1', 'LOC2'])

    # plot components & scores: use a Raw/Info that contains only EEG channels to avoid layout/topomap overlap errors
    inst_for_ica_plot = raw_ica_ready.copy().pick_types(eeg=True, eog=False, exclude='bads')
    try:
        ica.plot_components(inst=inst_for_ica_plot)  # interactive
    except ValueError as e:
        print('ICA component topomap plotting failed:', e)
        print('Attempting to plot components without an `inst` (no topomaps)')
        try:
            ica.plot_components()
        except Exception as e2:
            print('Fallback plotting also failed:', e2)

    if eog_scores is not None:
        try:
            ica.plot_scores(eog_scores)
        except Exception as e:
            print('Could not plot EOG scores:', e)

    # choose to exclude found eog comps (if any). If none found, leave exclusion empty so user can inspect components manually
    exclude = eog_indices if eog_indices is not None else []
    raw_clean = apply_ica(raw_filt, ica, exclude)

    # 7) PSD & topomap after ICA cleanup
    freqs_clean, psds_clean, ch_names_clean = compute_psd(raw_clean, fmin=0.5, fmax=40.0, n_fft=2048)
    plot_psd(freqs_clean, psds_clean, ch_names_clean, picks=None, title='PSD - AFTER ICA cleanup', save_name=f'{out_prefix}_psd_after_ica.png')
    df_band_clean = calculate_band_powers(psds_clean, freqs_clean, ch_names_clean)
    plot_band_topomaps(raw_clean, df_band_clean, save_name=f'{out_prefix}_topomap_after_ica.png', title='Band powers - after ICA')

    # save cleaned Raw if desired
    cleaned_fname = f'{out_prefix}_cleaned_raw.fif'
    raw_clean.save(cleaned_fname, overwrite=True)
    print('Saved cleaned raw to', cleaned_fname)

    return raw, raw_filt, raw_clean

# %%
# If run as a script, example usage (comment out in notebook and use pipeline_run interactively)
if __name__ == '__main__':
    example_path = '/Users/wanghley/Workspace/Projects/ML-driven EEG biomarkers/data/XU/XUAWAKE7.EDF'  # <- change me
    if Path(example_path).exists():
        pipeline_run(example_path, out_prefix='example')
    else:
        print('No example file found. Replace example_path with your EDF.')
