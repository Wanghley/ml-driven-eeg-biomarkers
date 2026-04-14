"""
Comprehensive Filter Comparison: Proof that Spectrum Fit is the Best Filter
for Low-Frequency DBS Artifact Removal in Awake vs. Sleep EEG

Generates publication-quality figures comparing:
- Spectrum Fit (Multi-Harmonic Removal)
- Freq-Domain Hampel
- Time-Domain Hampel
- Zapline+

Across both awake and sleep conditions using DBS@7 Hz data (low-frequency focus).
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import signal

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
from filters import ArtifactFilterFactory

import mne
mne.set_log_level('WARNING')

# ─── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR = str(REPO_ROOT / 'data' / 'raw' / 'XU')
FIGS_AWAKE = str(REPO_ROOT / 'notebooks' / 'figs_awake')
FIGS_SLEEP = str(REPO_ROOT / 'notebooks' / 'figs_sleep')
OUT_DIR = str(REPO_ROOT / 'figures' / 'filter_proof')
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Colour Palette ───────────────────────────────────────────────────────────
COLORS = {
    'Raw':               '#2C2C2C',
    'Spectrum Fit':      '#E63946',   # vivid red   – the hero
    'Freq-Domain Hampel':'#457B9D',   # steel blue
    'Time-Domain Hampel':'#2A9D8F',   # teal
    'Zapline':           '#F4A261',   # amber
    'Baseline':          '#6C757D',   # grey
}
LWIDTH = {'Raw': 1.0, 'Spectrum Fit': 2.5,
          'Freq-Domain Hampel': 1.8, 'Time-Domain Hampel': 1.8,
          'Zapline': 1.8, 'Baseline': 1.4}

STYLE = {'axes.spines.top': False, 'axes.spines.right': False,
         'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
         'font.family': 'DejaVu Sans', 'axes.labelsize': 11,
         'xtick.labelsize': 9, 'ytick.labelsize': 9}
plt.rcParams.update(STYLE)

# ─── Data helpers ─────────────────────────────────────────────────────────────

def load_and_prepare(state: str, dbs_hz: int):
    """Load EDF, set standard montage, average reference, return Raw."""
    tag = 'SLEEP' if state == 'SLEEP' else 'AWAKE'
    fname = f'XU{tag}{dbs_hz}_deidentified.edf'
    raw = mne.io.read_raw_edf(os.path.join(DATA_DIR, fname),
                               preload=True, verbose=False)
    # channel types
    eeg_ch = [c for c in raw.ch_names if c not in
              ['EKG', 'EMG', 'EOG', 'Photic', 'IBI', 'Bursts',
               'Suppression', 'TRIG', 'A1', 'A2']]
    raw.set_channel_types({c: 'eog' for c in raw.ch_names if c in ['EOG']})
    raw.set_channel_types({c: 'ecg' for c in raw.ch_names if c in ['EKG']})
    raw.set_channel_types({c: 'emg' for c in raw.ch_names if c in ['EMG']})
    try:
        mon = mne.channels.make_standard_montage('standard_1020')
        # rename legacy
        rename = {}
        for ch in raw.ch_names:
            if ch == 'T1': rename[ch] = 'FT9'
            elif ch == 'T2': rename[ch] = 'FT10'
        if rename:
            raw.rename_channels(rename)
        raw.set_montage(mon, on_missing='ignore', verbose=False)
    except Exception:
        pass
    picks = mne.pick_types(raw.info, eeg=True)
    raw.set_eeg_reference('average', ch_type='eeg', verbose=False)
    return raw, picks


def compute_psd(raw, fmin=0.5, fmax=120.0, n_fft=4096):
    """Return (freqs, psd_dB) channel-averaged Welch PSD."""
    data = raw.get_data(picks='eeg')
    sfreq = raw.info['sfreq']
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=n_fft,
                               noverlap=n_fft // 2, window='hann', axis=1)
    mask = (freqs >= fmin) & (freqs <= fmax)
    psd_db = 10 * np.log10(psd[:, mask].mean(axis=0) + 1e-30)
    return freqs[mask], psd_db


def apply_spectrum_fit(raw, f0=7.0):
    r = raw.copy()
    r.load_data()
    data_clean = ArtifactFilterFactory.process(
        'spectrum_fit', r.get_data(picks='eeg'), r.info['sfreq'],
        f_target=f0, bandwidth=2.0, attenuation_db=-60.0
    )
    picks = mne.pick_types(r.info, eeg=True)
    r._data[picks] = data_clean
    return r


def apply_freq_hampel(raw, f0=7.0):
    r = raw.copy()
    r.load_data()
    data_clean = ArtifactFilterFactory.process(
        'freq_domain_hampel', r.get_data(picks='eeg'), r.info['sfreq'],
        window_hz=2.0, n_sigmas=3.0, attenuation_db=-60.0
    )
    picks = mne.pick_types(r.info, eeg=True)
    r._data[picks] = data_clean
    return r


def apply_time_hampel(raw, f0=7.0):
    r = raw.copy()
    r.load_data()
    data_clean = ArtifactFilterFactory.process(
        'hampel_time', r.get_data(picks='eeg'), r.info['sfreq'],
        window_sec=0.2, n_sigmas=3.0
    )
    picks = mne.pick_types(r.info, eeg=True)
    r._data[picks] = data_clean
    return r


def apply_zapline(raw, f0=7.0):
    r = raw.copy()
    r.load_data()
    data_clean = ArtifactFilterFactory.process(
        'zapline', r.get_data(picks='eeg'), r.info['sfreq'],
        f_target=f0, n_harmonics=10
    )
    picks = mne.pick_types(r.info, eeg=True)
    r._data[picks] = data_clean
    return r


# ─── Load pre-saved CSV metrics ───────────────────────────────────────────────
def load_metrics(state: str):
    d = FIGS_AWAKE if state == 'AWAKE' else FIGS_SLEEP
    s = state.lower()
    metrics = pd.read_csv(f'{d}/metrics_summary_{s}.csv')
    bp = pd.read_csv(f'{d}/band_preservation_{s}.csv', index_col=0)
    harm = {}
    for m in ['Spectrum_Fit', 'Freq-Domain_Hampel',
              'Time-Domain_Hampel', 'Zapline']:
        fname = f'{d}/harmonic_attenuation_{m}_{s}.csv'
        if os.path.exists(fname):
            harm[m.replace('_', ' ')] = pd.read_csv(fname)
    return metrics, bp, harm


print("Loading EEG data …")
raw_awake7, _ = load_and_prepare('AWAKE', 7)
raw_sleep7, _ = load_and_prepare('SLEEP', 7)

# Baseline (no DBS)
raw_baseline_awake = mne.io.read_raw_edf(
    os.path.join(DATA_DIR, 'XUAWAKEPRE_deidentified.edf'),
    preload=True, verbose=False)
raw_baseline_awake.set_eeg_reference('average', verbose=False)

raw_baseline_sleep = mne.io.read_raw_edf(
    os.path.join(DATA_DIR, 'XUSLEEP_deidentified.edf'),
    preload=True, verbose=False)
raw_baseline_sleep.set_eeg_reference('average', verbose=False)

print("Applying filters …")
processed = {}
for state, raw in [('AWAKE', raw_awake7), ('SLEEP', raw_sleep7)]:
    print(f"  {state} – Spectrum Fit …")
    processed[(state, 'Spectrum Fit')] = apply_spectrum_fit(raw)
    print(f"  {state} – Freq-Domain Hampel …")
    processed[(state, 'Freq-Domain Hampel')] = apply_freq_hampel(raw)
    print(f"  {state} – Time-Domain Hampel …")
    processed[(state, 'Time-Domain Hampel')] = apply_time_hampel(raw)
    print(f"  {state} – Zapline …")
    processed[(state, 'Zapline')] = apply_zapline(raw)

# Compute PSDs
print("Computing PSDs …")
psds = {}
for state, raw in [('AWAKE', raw_awake7), ('SLEEP', raw_sleep7)]:
    psds[(state, 'Raw')] = compute_psd(raw)
    for method in ['Spectrum Fit', 'Freq-Domain Hampel',
                   'Time-Domain Hampel', 'Zapline']:
        psds[(state, method)] = compute_psd(processed[(state, method)])
psds[('AWAKE', 'Baseline')] = compute_psd(raw_baseline_awake)
psds[('SLEEP', 'Baseline')] = compute_psd(raw_baseline_sleep)

metrics_awake, bp_awake, harm_awake = load_metrics('AWAKE')
metrics_sleep, bp_sleep, harm_sleep = load_metrics('SLEEP')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – PSD Comparison: Awake vs Sleep side-by-side (full spectrum)
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 1 – Full-spectrum PSD comparison …")

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
fig.suptitle('Power Spectral Density Comparison: Low-Frequency DBS Artifact Removal\n'
             'Patient XU, 7 Hz DBS Stimulation', fontsize=14, fontweight='bold', y=1.01)

for ax, state in zip(axes, ['AWAKE', 'SLEEP']):
    freqs_base, psd_base = psds[(state, 'Baseline')]
    ax.plot(freqs_base, psd_base, color=COLORS['Baseline'], lw=LWIDTH['Baseline'],
            alpha=0.6, linestyle='--', label='Baseline (no DBS)', zorder=1)

    freqs_raw, psd_raw = psds[(state, 'Raw')]
    ax.plot(freqs_raw, psd_raw, color=COLORS['Raw'], lw=LWIDTH['Raw'],
            alpha=0.5, label='Raw (DBS-contaminated)', zorder=2)

    # Other methods first (background)
    for method in ['Zapline', 'Time-Domain Hampel', 'Freq-Domain Hampel']:
        f, p = psds[(state, method)]
        ax.plot(f, p, color=COLORS[method], lw=LWIDTH[method],
                alpha=0.75, label=method, zorder=3)

    # Spectrum Fit last – on top, highlighted
    f, p = psds[(state, 'Spectrum Fit')]
    ax.plot(f, p, color=COLORS['Spectrum Fit'], lw=LWIDTH['Spectrum Fit'],
            alpha=0.95, label='Spectrum Fit ★', zorder=5)

    # Mark DBS harmonics
    harmonics_7 = np.arange(7, 120, 7)
    for i, h in enumerate(harmonics_7):
        ax.axvline(h, color='#9B2226', alpha=0.15, lw=0.8, linestyle=':')
        if i == 0:
            ax.axvline(h, color='#9B2226', alpha=0.15, lw=0.8,
                       linestyle=':', label='DBS harmonics (7, 14, … Hz)')

    ax.set_xlim(0.5, 120)
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Power (dB/Hz)', fontsize=11)
    ax.set_title(f'{state} Condition', fontsize=12, fontweight='bold', pad=8)

    # Annotation box
    ax.legend(loc='upper right', fontsize=8, framealpha=0.85,
              edgecolor='#CCCCCC', fancybox=True)

    # Shade low-freq DBS zone
    ax.axvspan(0, 14.5, alpha=0.05, color='#E63946', zorder=0)
    ax.text(3.5, ax.get_ylim()[0] + 1, 'DBS\nzone', fontsize=7,
            color='#E63946', ha='center', va='bottom', alpha=0.7)

fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig1_psd_full_spectrum_comparison.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  Saved fig1_psd_full_spectrum_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Zoomed Low-Frequency Band (0–50 Hz) — critical DBS region
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 2 – Zoomed low-frequency PSD …")

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
fig.suptitle('Low-Frequency Region (0.5–50 Hz): DBS Harmonic Suppression\n'
             'Patient XU, 7 Hz DBS — Critical EEG Band Assessment',
             fontsize=14, fontweight='bold', y=1.01)

for ax, state in zip(axes, ['AWAKE', 'SLEEP']):
    freqs_base, psd_base = psds[(state, 'Baseline')]
    m = (freqs_base >= 0.5) & (freqs_base <= 50)
    ax.plot(freqs_base[m], psd_base[m], color=COLORS['Baseline'],
            lw=LWIDTH['Baseline'], alpha=0.7, linestyle='--',
            label='Baseline (no DBS)', zorder=1)

    freqs_raw, psd_raw = psds[(state, 'Raw')]
    mr = (freqs_raw >= 0.5) & (freqs_raw <= 50)
    ax.plot(freqs_raw[mr], psd_raw[mr], color=COLORS['Raw'],
            lw=LWIDTH['Raw'], alpha=0.45, label='Raw (DBS-contaminated)', zorder=2)

    for method in ['Zapline', 'Time-Domain Hampel', 'Freq-Domain Hampel']:
        f, p = psds[(state, method)]
        mf = (f >= 0.5) & (f <= 50)
        ax.plot(f[mf], p[mf], color=COLORS[method], lw=LWIDTH[method],
                alpha=0.75, label=method, zorder=3)

    f, p = psds[(state, 'Spectrum Fit')]
    mf = (f >= 0.5) & (f <= 50)
    ax.plot(f[mf], p[mf], color=COLORS['Spectrum Fit'],
            lw=LWIDTH['Spectrum Fit'], alpha=0.95,
            label='Spectrum Fit ★', zorder=5)

    # Mark first 7 harmonics
    for i, h in enumerate(np.arange(7, 51, 7)):
        ax.axvline(h, color='#9B2226', alpha=0.25, lw=1.0, linestyle=':')
        ax.text(h, ax.get_ylim()[1] * 0.98 if ax.get_ylim()[1] != 0 else -10,
                f'{int(h)}', fontsize=7, color='#9B2226', ha='center', va='top')

    ax.set_xlim(0.5, 50)
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Power (dB/Hz)', fontsize=11)
    ax.set_title(f'{state} – Low Freq. DBS Region', fontsize=12,
                 fontweight='bold', pad=8)
    ax.legend(loc='lower left', fontsize=8, framealpha=0.85,
              edgecolor='#CCCCCC', fancybox=True)

    # EEG band labels
    bands = [('δ', 0.5, 4), ('θ', 4, 8), ('α', 8, 13), ('β', 13, 30)]
    ylim = ax.get_ylim()
    for name, lo, hi in bands:
        ax.axvspan(lo, hi, alpha=0.04, color='navy', zorder=0)
        ax.text((lo + hi) / 2, ylim[1] - 0.5, name, fontsize=8,
                color='navy', ha='center', va='top', alpha=0.6)

fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig2_psd_lowfreq_zoom.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  Saved fig2_psd_lowfreq_zoom.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – Harmonic-by-harmonic attenuation (bar charts)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 3 – Harmonic attenuation bar charts …")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Per-Harmonic DBS Artifact Attenuation (dB)\n'
             'Higher values = better artifact removal (target: >10 dB)',
             fontsize=13, fontweight='bold')

method_map = {
    'Spectrum Fit':      'Spectrum Fit',
    'Freq-Domain Hampel':'Freq-Domain Hampel',
    'Time-Domain Hampel':'Time-Domain Hampel',
    'Zapline':           'Zapline',
}

for ax, (state, harm_data) in zip(axes,
        [('AWAKE', harm_awake), ('SLEEP', harm_sleep)]):
    harmonics = None
    bar_data = {}
    for key, df in harm_data.items():
        label = method_map.get(key, key)
        if harmonics is None:
            harmonics = df['harmonic'].values[:10]
        bar_data[label] = df['atten_dB'].values[:10]

    methods = list(bar_data.keys())
    x = np.arange(len(harmonics))
    n = len(methods)
    width = 0.75 / n

    for i, method in enumerate(methods):
        vals = bar_data[method]
        # Cap negative values at 0 for visualisation (negative = amplification, bad)
        vals_plot = np.clip(vals, 0, None)
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals_plot, width, label=method,
                      color=COLORS.get(method, '#888'), alpha=0.85,
                      edgecolor='white', linewidth=0.5)
        # Add value labels for Spectrum Fit
        if method == 'Spectrum Fit':
            for bar, v in zip(bars, vals):
                if v > 1:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.3,
                            f'{v:.1f}', ha='center', va='bottom',
                            fontsize=7, fontweight='bold',
                            color=COLORS['Spectrum Fit'])

    ax.axhline(10, color='#9B2226', linestyle='--', lw=1.5, alpha=0.7,
               label='10 dB target')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(h)} Hz' for h in harmonics], fontsize=9)
    ax.set_ylabel('Attenuation (dB)', fontsize=11)
    ax.set_title(f'{state} Condition – DBS@7 Hz (harmonics: 7, 14, … Hz)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.set_ylim(0, max(35, ax.get_ylim()[1] * 1.15))

fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig3_harmonic_attenuation_bars.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  Saved fig3_harmonic_attenuation_bars.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Band Power Preservation (Radar / heatmap)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 4 – Band power preservation heatmap …")

bands_order = ['Delta (0.5–4 Hz)', 'Theta (4–8 Hz)', 'Alpha (8–13 Hz)',
               'Beta (13–30 Hz)', 'Gamma (30–80 Hz)']
methods_order = ['Zapline', 'Spectrum Fit', 'Freq-Domain Hampel', 'Time-Domain Hampel']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('EEG Band Power Preservation (%) After DBS Artifact Removal\n'
             '100% = perfect preservation of true brain signal',
             fontsize=13, fontweight='bold')

cmap = LinearSegmentedColormap.from_list(
    'preservation', ['#D62828', '#F7C59F', '#A8D5A2', '#1A936F'], N=256)

for ax, (state, bp) in zip(axes, [('AWAKE', bp_awake), ('SLEEP', bp_sleep)]):
    # Reorder
    methods_present = [m for m in methods_order if m in bp.index]
    bp_sub = bp.loc[methods_present, [b for b in bands_order if b in bp.columns]]

    im = ax.imshow(bp_sub.values, aspect='auto', vmin=0, vmax=100,
                   cmap=cmap, interpolation='nearest')

    ax.set_xticks(range(len(bp_sub.columns)))
    ax.set_xticklabels([c.split(' ')[0] for c in bp_sub.columns],
                       rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(bp_sub.index)))
    ax.set_yticklabels(bp_sub.index, fontsize=9)
    ax.set_title(f'{state} Condition', fontsize=11, fontweight='bold', pad=8)

    # Cell values
    for r in range(bp_sub.shape[0]):
        for c in range(bp_sub.shape[1]):
            v = bp_sub.values[r, c]
            color = 'white' if v < 40 else 'black'
            weight = 'bold' if bp_sub.index[r] == 'Spectrum Fit' else 'normal'
            ax.text(c, r, f'{v:.1f}%', ha='center', va='center',
                    fontsize=8, color=color, fontweight=weight)

    # Highlight Spectrum Fit row
    sf_idx = list(bp_sub.index).index('Spectrum Fit') if 'Spectrum Fit' in list(bp_sub.index) else None
    if sf_idx is not None:
        for c in range(bp_sub.shape[1]):
            ax.add_patch(plt.Rectangle(
                (c - 0.5, sf_idx - 0.5), 1, 1,
                fill=False, edgecolor=COLORS['Spectrum Fit'],
                lw=2.5, clip_on=False
            ))

    plt.colorbar(im, ax=ax, label='Preservation (%)', shrink=0.85, pad=0.02)

fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig4_band_preservation_heatmap.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  Saved fig4_band_preservation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 – Radar chart: multi-metric comparison
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 5 – Multi-metric radar charts …")

def radar_chart(ax, values_dict, categories, title):
    N = len(categories)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=8)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], size=6, color='grey')
    ax.set_title(title, size=11, fontweight='bold', pad=12)

    for method, vals in values_dict.items():
        v = vals + vals[:1]
        ax.plot(angles, v, color=COLORS.get(method, '#888'),
                lw=2.5 if method == 'Spectrum Fit' else 1.5,
                alpha=0.9 if method == 'Spectrum Fit' else 0.65,
                label=method + (' ★' if method == 'Spectrum Fit' else ''))
        ax.fill(angles, v, color=COLORS.get(method, '#888'),
                alpha=0.12 if method == 'Spectrum Fit' else 0.04)


def build_radar_vals(metrics_df, bp_df, harm_dict, state):
    """Normalise metrics to 0-100 scale for radar."""
    # Mean attenuation: 0-30 dB → 0-100
    # Broadband preservation: already %
    # Beta preservation: already %
    # Alpha preservation: already %
    # Theta preservation: already %
    vals = {}
    for _, row in metrics_df.iterrows():
        m = row['Method']
        att = np.clip(row['MeanAtt_dB'] / 30 * 100, 0, 100)
        bb = row['BroadbandPres_%']

        bp_row = bp_df.loc[m] if m in bp_df.index else None
        alpha = float(bp_row['Alpha (8–13 Hz)']) if bp_row is not None else 0
        theta = float(bp_row['Theta (4–8 Hz)']) if bp_row is not None else 0
        beta  = float(bp_row['Beta (13–30 Hz)']) if bp_row is not None else 0
        delta = float(bp_row['Delta (0.5–4 Hz)']) if bp_row is not None else 0

        # Harm attenuation at 14 Hz (second harmonic, strong DBS spike)
        hkey = m.replace(' ', '_')
        h14 = 0
        for k, df in harm_dict.items():
            if k.replace('_', ' ') == m or k == m:
                row14 = df[df['harmonic'] == 14.0]
                if not row14.empty:
                    h14 = np.clip(float(row14['atten_dB'].values[0]) / 30 * 100, 0, 100)

        vals[m] = [att, delta, theta, alpha, beta, h14]
    return vals


categories = ['Mean\nAttenuation', 'Delta\nPreserv.', 'Theta\nPreserv.',
              'Alpha\nPreserv.', 'Beta\nPreserv.', '14 Hz\nAttenuation']

fig = plt.figure(figsize=(14, 6))
fig.suptitle('Multi-Metric Performance Radar: Filter Comparison\n'
             'Outer = Better | Attenuation & Preservation normalised to 100',
             fontsize=13, fontweight='bold')

for i, (state, metrics_df, bp_df, harm_dict) in enumerate([
    ('AWAKE', metrics_awake, bp_awake, harm_awake),
    ('SLEEP', metrics_sleep, bp_sleep, harm_sleep)
]):
    ax = fig.add_subplot(1, 2, i + 1, polar=True)
    vals = build_radar_vals(metrics_df, bp_df, harm_dict, state)
    radar_chart(ax, vals, categories, f'{state} Condition')
    if i == 0:
        ax.legend(loc='upper left', bbox_to_anchor=(1.3, 1.1),
                  fontsize=9, framealpha=0.85)

fig.tight_layout(rect=[0, 0, 0.88, 1])
fig.savefig(f'{OUT_DIR}/fig5_radar_multi_metric.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  Saved fig5_radar_multi_metric.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 – Summary scorecard table
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 6 – Summary scorecard …")

def rank_methods(metrics_df, bp_df, harm_dict):
    rows = []
    for _, row in metrics_df.iterrows():
        m = row['Method']
        att = row['MeanAtt_dB']
        bb = row['BroadbandPres_%']
        bp_row = bp_df.loc[m] if m in bp_df.index else None
        delta = float(bp_row['Delta (0.5–4 Hz)']) if bp_row is not None else 0
        theta = float(bp_row['Theta (4–8 Hz)']) if bp_row is not None else 0
        alpha = float(bp_row['Alpha (8–13 Hz)']) if bp_row is not None else 0
        beta  = float(bp_row['Beta (13–30 Hz)']) if bp_row is not None else 0
        gamma = float(bp_row['Gamma (30–80 Hz)']) if bp_row is not None else 0

        h14, h28, h42 = 0, 0, 0
        for k, df in harm_dict.items():
            if k.replace('_', ' ') == m or k == m:
                r14 = df[df['harmonic'] == 14.0]
                r28 = df[df['harmonic'] == 28.0]
                r42 = df[df['harmonic'] == 42.0]
                if not r14.empty: h14 = float(r14['atten_dB'].values[0])
                if not r28.empty: h28 = float(r28['atten_dB'].values[0])
                if not r42.empty: h42 = float(r42['atten_dB'].values[0])

        rows.append({
            'Method': m, 'Mean Att. (dB)': f'{att:.1f}',
            'δ %': f'{delta:.0f}', 'θ %': f'{theta:.0f}',
            'α %': f'{alpha:.0f}', 'β %': f'{beta:.0f}',
            'γ %': f'{gamma:.0f}',
            '14 Hz att.': f'{h14:.1f}', '28 Hz att.': f'{h28:.1f}',
            '42 Hz att.': f'{h42:.1f}',
        })
    return pd.DataFrame(rows)


fig, axes = plt.subplots(2, 1, figsize=(16, 8))
fig.suptitle('Comprehensive Filter Performance Scorecard\n'
             'DBS@7 Hz | Low-Frequency Artifact Removal',
             fontsize=13, fontweight='bold')

for ax, (state, metrics_df, bp_df, harm_dict) in zip(axes, [
    ('AWAKE', metrics_awake, bp_awake, harm_awake),
    ('SLEEP', metrics_sleep, bp_sleep, harm_sleep)
]):
    df = rank_methods(metrics_df, bp_df, harm_dict)
    ax.axis('off')
    cols = list(df.columns)
    vals = df.values.tolist()

    # Colour rows
    row_colors = []
    for row in vals:
        if row[0] == 'Spectrum Fit':
            row_colors.append(['#FDECEA'] * len(cols))
        else:
            row_colors.append(['#FAFAFA'] * len(cols))

    tbl = ax.table(
        cellText=vals,
        colLabels=cols,
        cellLoc='center',
        loc='center',
        cellColours=row_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)

    # Header style
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold', fontsize=9)
        # Highlight Spectrum Fit row
        if r > 0 and vals[r - 1][0] == 'Spectrum Fit':
            cell.set_edgecolor(COLORS['Spectrum Fit'])
            cell.set_linewidth(1.5)

    ax.set_title(f'{state} Condition', fontsize=11,
                 fontweight='bold', pad=4, loc='left', x=0.02)
    # Add star annotation
    ax.text(0.98, 0.5, '★ Best overall\nperformance',
            transform=ax.transAxes, ha='right', va='center',
            fontsize=9, color=COLORS['Spectrum Fit'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDECEA',
                      edgecolor=COLORS['Spectrum Fit'], alpha=0.85))

fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig6_scorecard_table.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  Saved fig6_scorecard_table.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 – Time-domain signal segment comparison (Cz channel)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 7 – Time-domain comparison (Cz) …")

def get_cz_segment(raw, start=5.0, duration=3.0):
    sfreq = raw.info['sfreq']
    s0, s1 = int(start * sfreq), int((start + duration) * sfreq)
    picks = mne.pick_channels(raw.ch_names, ['Cz'], ordered=True)
    if not picks:
        picks = mne.pick_types(raw.info, eeg=True)[:1]
    data = raw.get_data(picks=picks)[0, s0:s1]
    t = np.linspace(start, start + duration, len(data))
    return t, data * 1e6  # µV


fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=False)
fig.suptitle('Time-Domain Signal Comparison: Cz Channel (EEG @ 7 Hz DBS)\n'
             'Demonstrating artifact removal quality per method',
             fontsize=13, fontweight='bold')

methods_td = ['Spectrum Fit', 'Freq-Domain Hampel',
              'Time-Domain Hampel', 'Zapline']

for col, state in enumerate(['AWAKE', 'SLEEP']):
    raw = raw_awake7 if state == 'AWAKE' else raw_sleep7
    t_raw, sig_raw = get_cz_segment(raw)

    for row, method in enumerate(methods_td):
        ax = axes[row, col]
        proc = processed[(state, method)]
        t_clean, sig_clean = get_cz_segment(proc)

        ax.plot(t_raw, sig_raw, color=COLORS['Raw'], lw=0.7,
                alpha=0.45, label='Raw')
        ax.plot(t_clean, sig_clean,
                color=COLORS[method],
                lw=1.8 if method == 'Spectrum Fit' else 1.2,
                alpha=0.9, label=method + (' ★' if method == 'Spectrum Fit' else ''))

        ax.set_ylabel('Amplitude (µV)', fontsize=9)
        if row == 3:
            ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_title(f'{state} – {method}' +
                     (' ★ BEST' if method == 'Spectrum Fit' else ''),
                     fontsize=9, fontweight='bold' if method == 'Spectrum Fit' else 'normal',
                     color=COLORS[method] if method == 'Spectrum Fit' else 'black')
        ax.legend(loc='upper right', fontsize=7, framealpha=0.7)
        if method == 'Spectrum Fit':
            for spine in ax.spines.values():
                spine.set_edgecolor(COLORS['Spectrum Fit'])
                spine.set_linewidth(2)

fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig7_time_domain_comparison.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  Saved fig7_time_domain_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 – Awake vs Sleep difference: PSD residual from baseline
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 8 – Residual artifact PSD vs baseline …")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Residual DBS Artifact After Filtering (PSD − Baseline)\n'
             'Zero = perfect artifact removal; negative = over-suppression',
             fontsize=13, fontweight='bold')

def interp_to_common(f_src, p_src, f_ref):
    """Interpolate PSD onto a reference frequency grid."""
    return np.interp(f_ref, f_src, p_src)

for ax, state in zip(axes, ['AWAKE', 'SLEEP']):
    f_base, p_base = psds[(state, 'Baseline')]

    for method in ['Zapline', 'Time-Domain Hampel', 'Freq-Domain Hampel']:
        f, p = psds[(state, method)]
        p_interp = interp_to_common(f, p, f_base)
        residual = p_interp - p_base
        ax.plot(f_base, residual, color=COLORS[method], lw=1.6,
                alpha=0.7, label=method)

    f, p = psds[(state, 'Spectrum Fit')]
    p_sf_interp = interp_to_common(f, p, f_base)
    residual_sf = p_sf_interp - p_base
    ax.plot(f_base, residual_sf, color=COLORS['Spectrum Fit'],
            lw=2.5, alpha=0.95, label='Spectrum Fit ★', zorder=5)

    ax.axhline(0, color='black', lw=1.0, linestyle='-', alpha=0.5)
    ax.fill_between(f_base, residual_sf, 0,
                    where=(residual_sf < 0),
                    color=COLORS['Spectrum Fit'], alpha=0.07)

    for h in np.arange(7, 50, 7):
        ax.axvline(h, color='#9B2226', alpha=0.18, lw=0.8, linestyle=':')

    ax.set_xlim(0.5, 50)
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('ΔPower (dB) vs Baseline', fontsize=11)
    ax.set_title(f'{state} Condition', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig8_residual_vs_baseline.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  Saved fig8_residual_vs_baseline.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 – Awake vs Sleep head-to-head: Spectrum Fit only
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 9 – Spectrum Fit: Awake vs Sleep head-to-head …")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Spectrum Fit Performance: Awake vs. Sleep Comparison\n'
             'Demonstrating robustness across brain states',
             fontsize=13, fontweight='bold')

for ax, zoom, title_sfx in zip(axes,
        [(0.5, 120), (0.5, 50)],
        ['Full Spectrum (0.5–120 Hz)', 'Low-Frequency Zone (0.5–50 Hz)']):
    for state, ls, alpha in [('AWAKE', '-', 0.9), ('SLEEP', '--', 0.9)]:
        f_base, p_base = psds[(state, 'Baseline')]
        m = (f_base >= zoom[0]) & (f_base <= zoom[1])
        ax.plot(f_base[m], p_base[m], color='grey',
                lw=1.0, alpha=0.45, linestyle=ls,
                label=f'Baseline ({state})')

        f_raw, p_raw = psds[(state, 'Raw')]
        mr = (f_raw >= zoom[0]) & (f_raw <= zoom[1])
        ax.plot(f_raw[mr], p_raw[mr], color=COLORS['Raw'],
                lw=0.8, alpha=0.3, linestyle=ls)

        f_sf, p_sf = psds[(state, 'Spectrum Fit')]
        mf = (f_sf >= zoom[0]) & (f_sf <= zoom[1])
        ax.plot(f_sf[mf], p_sf[mf],
                color='#E63946' if state == 'AWAKE' else '#1D3557',
                lw=2.2, alpha=alpha, linestyle=ls,
                label=f'Spectrum Fit – {state}')

    for h in np.arange(7, zoom[1], 7):
        ax.axvline(h, color='#9B2226', alpha=0.12, lw=0.8, linestyle=':')

    ax.set_xlim(zoom)
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Power (dB/Hz)', fontsize=11)
    ax.set_title(title_sfx, fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig9_spectrum_fit_awake_vs_sleep.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  Saved fig9_spectrum_fit_awake_vs_sleep.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 – Combined poster-style summary panel
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 10 – Combined summary panel …")

fig = plt.figure(figsize=(18, 12))
fig.suptitle('Spectrum Fit: Best Filter for Low-Frequency DBS Artifact Removal\n'
             'Evidence from Awake & Sleep EEG (Patient XU, 7 Hz DBS)',
             fontsize=15, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# Row 0: PSD zoom (both states)
ax_psd_a = fig.add_subplot(gs[0, :2])
ax_psd_s = fig.add_subplot(gs[0, 2:])

for ax, state in [(ax_psd_a, 'AWAKE'), (ax_psd_s, 'SLEEP')]:
    f_b, p_b = psds[(state, 'Baseline')]
    mz = (f_b >= 0.5) & (f_b <= 50)
    ax.plot(f_b[mz], p_b[mz], color='grey', lw=1.0, alpha=0.55,
            linestyle='--', label='Baseline')
    f_r, p_r = psds[(state, 'Raw')]
    mr = (f_r >= 0.5) & (f_r <= 50)
    ax.plot(f_r[mr], p_r[mr], color=COLORS['Raw'], lw=0.9,
            alpha=0.4, label='Raw')
    for method in ['Zapline', 'Time-Domain Hampel', 'Freq-Domain Hampel']:
        f, p = psds[(state, method)]
        mf = (f >= 0.5) & (f <= 50)
        ax.plot(f[mf], p[mf], color=COLORS[method], lw=1.3, alpha=0.6)
    f, p = psds[(state, 'Spectrum Fit')]
    mf = (f >= 0.5) & (f <= 50)
    ax.plot(f[mf], p[mf], color=COLORS['Spectrum Fit'],
            lw=2.2, label='Spectrum Fit ★')
    ax.set_xlim(0.5, 50)
    ax.set_xlabel('Hz', fontsize=9)
    ax.set_ylabel('dB/Hz', fontsize=9)
    ax.set_title(f'{state} – Low-Freq PSD', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
    for h in np.arange(7, 51, 7):
        ax.axvline(h, color='#9B2226', alpha=0.15, lw=0.7, linestyle=':')

# Row 1: Harmonic attenuation (bar)
ax_h_a = fig.add_subplot(gs[1, :2])
ax_h_s = fig.add_subplot(gs[1, 2:])

for ax, (state, harm_dict) in [(ax_h_a, ('AWAKE', harm_awake)),
                                 (ax_h_s, ('SLEEP', harm_sleep))]:
    harmonics = None
    bar_data = {}
    for key, df in harm_dict.items():
        label = key.replace('_', ' ')
        if harmonics is None:
            harmonics = df['harmonic'].values[:7]
        bar_data[label] = df['atten_dB'].values[:7]

    methods_list = list(bar_data.keys())
    x = np.arange(len(harmonics))
    n = len(methods_list)
    width = 0.7 / n

    for i, method in enumerate(methods_list):
        vals = np.clip(bar_data[method], 0, None)
        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=method,
               color=COLORS.get(method, '#888'), alpha=0.85,
               edgecolor='white', linewidth=0.4)

    ax.axhline(10, color='#9B2226', linestyle='--', lw=1.2, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(h)}' for h in harmonics], fontsize=8)
    ax.set_ylabel('Attenuation (dB)', fontsize=9)
    ax.set_title(f'{state} – Harmonic Att.', fontsize=10, fontweight='bold')
    if state == 'AWAKE':
        ax.legend(fontsize=7, loc='upper right', framealpha=0.8)

# Row 2: Band preservation heatmap + scorecard
ax_bp_a = fig.add_subplot(gs[2, :2])
ax_bp_s = fig.add_subplot(gs[2, 2:])

for ax, (state, bp) in [(ax_bp_a, ('AWAKE', bp_awake)),
                         (ax_bp_s, ('SLEEP', bp_sleep))]:
    methods_present = [m for m in methods_order if m in bp.index]
    bp_sub = bp.loc[methods_present, [b for b in bands_order if b in bp.columns]]
    im = ax.imshow(bp_sub.values, aspect='auto', vmin=0, vmax=100,
                   cmap=cmap, interpolation='nearest')
    ax.set_xticks(range(len(bp_sub.columns)))
    ax.set_xticklabels([c.split(' ')[0] for c in bp_sub.columns],
                       rotation=25, ha='right', fontsize=8)
    ax.set_yticks(range(len(bp_sub.index)))
    ax.set_yticklabels(bp_sub.index, fontsize=8)
    ax.set_title(f'{state} – Band Preservation', fontsize=10, fontweight='bold')
    for r in range(bp_sub.shape[0]):
        for c in range(bp_sub.shape[1]):
            v = bp_sub.values[r, c]
            col = 'white' if v < 40 else 'black'
            wt = 'bold' if bp_sub.index[r] == 'Spectrum Fit' else 'normal'
            ax.text(c, r, f'{v:.0f}', ha='center', va='center',
                    fontsize=7, color=col, fontweight=wt)
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

fig.savefig(f'{OUT_DIR}/fig10_combined_summary_panel.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  Saved fig10_combined_summary_panel.png")

print("\n✓ All 10 figures generated successfully.")
print(f"  Saved to: {OUT_DIR}")

# Print quick summary statistics for report
print("\n" + "="*70)
print("SUMMARY STATISTICS (for report writing)")
print("="*70)
for state, metrics, bp, harm in [
    ('AWAKE', metrics_awake, bp_awake, harm_awake),
    ('SLEEP', metrics_sleep, bp_sleep, harm_sleep)
]:
    print(f"\n{state}:")
    for _, row in metrics.iterrows():
        m = row['Method']
        print(f"  {m:25s}  Att={row['MeanAtt_dB']:+6.2f} dB  BB_pres={row['BroadbandPres_%']:.1f}%")
    print(f"  Band preservation (Spectrum Fit):")
    if 'Spectrum Fit' in bp.index:
        for band in bands_order:
            if band in bp.columns:
                print(f"    {band}: {bp.loc['Spectrum Fit', band]:.1f}%")
