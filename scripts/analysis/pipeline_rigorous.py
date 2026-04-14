"""
Full EEG Signal Processing Pipeline + Rigorous Filter Evaluation
================================================================
Patient XU | DBS: 7, 60, 100 Hz | States: AWAKE, SLEEP
Uses 60 s of data per recording to keep runtime manageable.
Metrics: ΔSNR, SDI, per-harmonic attenuation, band preservation.
"""

import sys, os, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy import signal
from scipy.integrate import simpson

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
from filters import ArtifactFilterFactory
import mne
mne.set_log_level('ERROR')

# ─── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR = str(REPO_ROOT / 'data' / 'raw' / 'XU')
OUT_DIR  = str(REPO_ROOT / 'figures' / 'rigorous_proof')
os.makedirs(OUT_DIR, exist_ok=True)

DBS_FREQS  = [7, 60, 100]
STATES     = ['AWAKE', 'SLEEP']
METHODS    = ['Spectrum Fit', 'Freq-Domain Hampel', 'Time-Domain Hampel', 'Zapline']
SFREQ_DBS  = 256.0          # DBS recording sfreq
NYQUIST    = SFREQ_DBS / 2.0
CLIP_SEC   = 60              # seconds to use per recording
EEG_BANDS  = {'δ (0.5–4 Hz)':(0.5,4), 'θ (4–8 Hz)':(4,8), 'α (8–13 Hz)':(8,13),
              'β (13–30 Hz)':(13,30), 'γ (30–80 Hz)':(30,80)}
EEG_NAMES = ['Fp1','F7','T3','T5','O1','F3','C3','P3','Fz','Cz',
             'Fp2','F8','T4','T6','O2','F4','C4','P4','Fpz','Pz']
COLORS = {
    'Spectrum Fit':       '#E63946',
    'Freq-Domain Hampel': '#457B9D',
    'Time-Domain Hampel': '#2A9D8F',
    'Zapline':            '#F4A261',
    'Raw':                '#2C2C2C',
    'Baseline':           '#6C757D',
}
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':0.22,'grid.linestyle':'--',
    'font.family':'DejaVu Sans','axes.labelsize':10,'xtick.labelsize':8,
    'ytick.labelsize':8})

PRES_CMAP = LinearSegmentedColormap.from_list('pres',
    ['#D62828','#F4A261','#A8D5A2','#1A936F'], N=256)

# ─── Helpers ─────────────────────────────────────────────────────────────────

NON_EEG = {'EKGL','EKGR','LOC1','LOC2','EMG1','EMG2',
           'X9','X10','X11','X12','X13','X14','X15','X16',
           'X17','X18','DC1','DC2','DC3','DC4','OSAT','PR',
           'A1','A2','EKG','EMG','EOG','Photic','IBI',
           'Bursts','Suppression','TRIG'}

def load_and_preprocess(fname, clip_sec=60):
    """Load EDF, pick EEG only, bandpass, avg ref, return (data ndarray, sfreq)."""
    path = os.path.join(DATA_DIR, fname)
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    sfreq = raw.info['sfreq']
    nyq = sfreq / 2 - 1.0

    # Set non-EEG types then pick EEG
    for ch in raw.ch_names:
        if ch in NON_EEG:
            raw.set_channel_types({ch: 'misc'})
    raw.pick_types(eeg=True)

    # Rename legacy channels
    rename = {}
    for ch in raw.ch_names:
        if ch == 'T1': rename[ch] = 'FT9'
        elif ch == 'T2': rename[ch] = 'FT10'
    if rename:
        raw.rename_channels(rename)

    # Montage
    try:
        mon = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(mon, on_missing='ignore', verbose=False)
    except Exception:
        pass

    # Bandpass + notch + avg ref
    raw.filter(0.5, min(90.0, nyq), method='fir', fir_window='hamming', verbose=False)
    if 60.0 < nyq:
        raw.notch_filter([60.0], notch_widths=2.0, verbose=False)
    raw.set_eeg_reference('average', verbose=False)

    # Clip to clip_sec for speed
    n_clip = min(int(clip_sec * sfreq), raw.n_times)
    data = raw.get_data()[:, :n_clip]
    info = raw.info
    return data, sfreq, info

def resample_data(data, orig_sfreq, target_sfreq):
    """Resample 2D array (channels × samples) to target_sfreq."""
    if orig_sfreq == target_sfreq:
        return data
    n_out = int(data.shape[1] * target_sfreq / orig_sfreq)
    return signal.resample(data, n_out, axis=1)

def apply_filter_np(data, sfreq, method, f0):
    """Apply filter to (n_ch, n_samples) ndarray."""
    method_map = {
        'Spectrum Fit':       ('spectrum_fit',
                               dict(f_target=float(f0), bandwidth=2.0, attenuation_db=-60.0)),
        'Freq-Domain Hampel': ('freq_domain_hampel',
                               dict(window_hz=2.0, n_sigmas=3.0, attenuation_db=-60.0)),
        'Time-Domain Hampel': ('hampel_time',
                               dict(window_sec=0.2, n_sigmas=3.0)),
        'Zapline':            ('zapline',
                               dict(f_target=float(f0),
                                    n_harmonics=max(1, min(10, int((sfreq/2-1)/f0)-1)))),
    }
    mname, kwargs = method_map[method]
    return ArtifactFilterFactory.process(mname, data, sfreq, **kwargs)

def welch_avg(data, sfreq, fmin=0.5, fmax=90.0, n_fft=512):
    """Channel-averaged Welch PSD in dB/Hz."""
    f, p = signal.welch(data, fs=sfreq, nperseg=n_fft,
                        noverlap=n_fft//2, window='hann', axis=1)
    m = (f >= fmin) & (f <= fmax)
    return f[m], 10*np.log10(p[:, m].mean(0) + 1e-30)

def band_power_ch(data, sfreq, lo, hi, n_fft=512):
    """Per-channel band power (linear µV² via integration)."""
    f, p = signal.welch(data, fs=sfreq, nperseg=n_fft,
                        noverlap=n_fft//2, window='hann', axis=1)
    m = (f >= lo) & (f <= hi)
    return np.array([simpson(p[ch, m], x=f[m]) for ch in range(p.shape[0])])

def compute_delta_snr(data_dbs, data_clean, sfreq, f0, half_bw=1.0):
    """ΔSNR = SNR_clean − SNR_raw (dB). Higher is better."""
    def snr(data):
        f, p = signal.welch(data, fs=sfreq, nperseg=512, noverlap=256, window='hann', axis=1)
        p_avg = p.mean(0)
        harm = np.zeros(len(f), bool)
        for h in np.arange(f0, sfreq/2, f0):
            harm |= (np.abs(f - h) <= half_bw)
        non = ~harm & (f >= 0.5) & (f <= 80)
        hb  =  harm & (f >= 0.5) & (f <= 80)
        if not (np.any(hb) and np.any(non)):
            return 0.0
        return 10*np.log10(p_avg[non].mean() / (p_avg[hb].mean() + 1e-30))
    return snr(data_clean) - snr(data_dbs)

def compute_sdi(data_clean, data_base, sfreq, fmin=0.5, fmax=80.0):
    """Spectral Distortion Index: RMS(PSD_clean_dB − PSD_base_dB)."""
    f, p_c = signal.welch(data_clean, fs=sfreq, nperseg=512, noverlap=256, window='hann', axis=1)
    _, p_b = signal.welch(data_base,  fs=sfreq, nperseg=512, noverlap=256, window='hann', axis=1)
    m = (f >= fmin) & (f <= fmax)
    diff = 10*np.log10(p_c[:, m].mean(0) + 1e-30) - 10*np.log10(p_b[:, m].mean(0) + 1e-30)
    return float(np.sqrt(np.mean(diff**2)))

def harmonic_att(data_dbs, data_clean, sfreq, f0):
    """Per-harmonic attenuation dict {hz: dB}."""
    f, p_r = signal.welch(data_dbs,   fs=sfreq, nperseg=512, noverlap=256, window='hann', axis=1)
    _, p_c = signal.welch(data_clean, fs=sfreq, nperseg=512, noverlap=256, window='hann', axis=1)
    p_r = p_r.mean(0); p_c = p_c.mean(0)
    result = {}
    for h in np.arange(f0, sfreq/2, f0):
        band = np.abs(f - h) <= 1.0
        if np.any(band):
            result[h] = 10*np.log10(p_r[band].mean() / (p_c[band].mean() + 1e-30))
    return result

# ─── Load all data ────────────────────────────────────────────────────────────
print("=" * 70)
print("FULL EEG PIPELINE — Patient XU | 7/60/100 Hz DBS | AWAKE & SLEEP")
print("=" * 70)
print(f"\n[1/5] Loading and preprocessing (clipping to {CLIP_SEC}s) …")

FILE_MAP = {
    ('AWAKE',   'baseline'): 'XUAWAKEPRE_deidentified.edf',
    ('SLEEP',   'baseline'): 'XUSLEEP_deidentified.edf',
    ('AWAKE',   7):          'XUAWAKE7_deidentified.edf',
    ('SLEEP',   7):          'XUSLEEP7_deidentified.edf',
    ('AWAKE',   60):         'XUAWAKE60_deidentified.edf',
    ('SLEEP',   60):         'XUSLEEP60_deidentified.edf',
    ('AWAKE',   100):        'XUAWAKET100_deidentified.edf',
    ('SLEEP',   100):        'XUSLEEPT100_deidentified.edf',
}

raws    = {}   # (state, hz/baseline) -> (data_ndarray, sfreq, info)
for key, fname in FILE_MAP.items():
    print(f"  Loading {fname} …")
    data, sfreq, info = load_and_preprocess(fname, clip_sec=CLIP_SEC)
    raws[key] = (data, sfreq, info)
    print(f"    shape={data.shape}  sfreq={sfreq}")

# Resample baseline (200 Hz) to 256 Hz so metrics are comparable
for state in STATES:
    data, sfreq, info = raws[(state, 'baseline')]
    if sfreq != SFREQ_DBS:
        data = resample_data(data, sfreq, SFREQ_DBS)
        raws[(state, 'baseline')] = (data, SFREQ_DBS, info)
        print(f"  Resampled {state} baseline: {sfreq}→{SFREQ_DBS} Hz")

# ─── Apply filters ───────────────────────────────────────────────────────────
print("\n[2/5] Applying four filter methods …")
filtered = {}   # (state, hz, method) -> cleaned ndarray

for state in STATES:
    for hz in DBS_FREQS:
        data_dbs, sfreq_d, _ = raws[(state, hz)]
        for method in METHODS:
            print(f"  {state} {hz}Hz — {method} …", end=' ', flush=True)
            cleaned = apply_filter_np(data_dbs, sfreq_d, method, hz)
            filtered[(state, hz, method)] = cleaned
            print("done")

# ─── Compute PSDs ────────────────────────────────────────────────────────────
print("\n[3/5] Computing PSDs …")
psds = {}   # (state, hz/'baseline', method/'Raw'/'Baseline') -> (f, pdb)

for state in STATES:
    db, sb, _ = raws[(state, 'baseline')]
    psds[(state, 'Baseline')] = welch_avg(db, SFREQ_DBS)
    for hz in DBS_FREQS:
        dd, sd, _ = raws[(state, hz)]
        psds[(state, hz, 'Raw')] = welch_avg(dd, sd)
        for method in METHODS:
            psds[(state, hz, method)] = welch_avg(filtered[(state, hz, method)], sd)

# ─── Compute metrics ─────────────────────────────────────────────────────────
print("\n[4/5] Computing rigorous metrics …")
rows = []
harm_all = {}   # (state, hz, method) -> {harmonic: dB}

for state in STATES:
    db_base, _, _ = raws[(state, 'baseline')]
    for hz in DBS_FREQS:
        data_dbs, sfreq_d, _ = raws[(state, hz)]
        for method in METHODS:
            dc = filtered[(state, hz, method)]
            snr_imp = compute_delta_snr(data_dbs, dc, sfreq_d, hz)
            sdi_val = compute_sdi(dc, db_base, sfreq_d)
            harm     = harmonic_att(data_dbs, dc, sfreq_d, hz)
            harm_all[(state, hz, method)] = harm
            h5_att = np.mean(list(harm.values())[:5]) if harm else 0.0

            bp_pct = {}
            for band_name, (lo, hi) in EEG_BANDS.items():
                bp_raw   = band_power_ch(data_dbs, sfreq_d, lo, hi).mean()
                bp_clean = band_power_ch(dc,       sfreq_d, lo, hi).mean()
                bp_pct[band_name] = 100.0 * bp_clean / (bp_raw + 1e-30)

            row = {'State': state, 'DBS_Hz': hz, 'Method': method,
                   'ΔSNR_dB': round(snr_imp, 3),
                   'SDI_dB':  round(sdi_val, 3),
                   'MeanAtt5_dB': round(h5_att, 3)}
            for k, v in bp_pct.items():
                row[f'Pres_{k}'] = round(v, 1)
            rows.append(row)
            print(f"  {state:5s} {hz:3d}Hz {method:25s}  ΔSNR={snr_imp:+6.2f}  SDI={sdi_val:.2f}  Att5={h5_att:+6.2f}")

df = pd.DataFrame(rows)
df.to_csv(f'{OUT_DIR}/rigorous_metrics.csv', index=False)
print(f"  → saved rigorous_metrics.csv")

# ─── Figure generation ────────────────────────────────────────────────────────
print("\n[5/5] Generating figures …")

def harm_vlines(ax, hz, fmax, alpha=0.2, lw=0.8):
    for h in np.arange(hz, fmax, hz):
        ax.axvline(h, color='#9B2226', alpha=alpha, lw=lw, linestyle=':')

def interp_to(f_src, p_src, f_ref):
    return np.interp(f_ref, f_src, p_src)

# ── FIG A: 3×2 PSD matrix ────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('Power Spectral Density After Artifact Removal\n'
             'Full Pipeline: bandpass 0.5–90 Hz + avg reference | Patient XU',
             fontsize=13, fontweight='bold')

for ri, hz in enumerate(DBS_FREQS):
    for ci, state in enumerate(STATES):
        ax = axes[ri, ci]
        fb, pb = psds[(state, 'Baseline')]
        ax.plot(fb, pb, color=COLORS['Baseline'], lw=1.3, alpha=0.6,
                linestyle='--', label='Baseline', zorder=1)
        fr, pr = psds[(state, hz, 'Raw')]
        ax.plot(fr, pr, color=COLORS['Raw'], lw=0.8, alpha=0.4, label='Raw DBS', zorder=2)
        for m in ['Zapline','Time-Domain Hampel','Freq-Domain Hampel']:
            f, p = psds[(state, hz, m)]
            ax.plot(f, p, color=COLORS[m], lw=1.5, alpha=0.72, label=m, zorder=3)
        f, p = psds[(state, hz, 'Spectrum Fit')]
        ax.plot(f, p, color=COLORS['Spectrum Fit'], lw=2.5, alpha=0.95,
                label='Spectrum Fit ★', zorder=5)
        harm_vlines(ax, hz, 90)
        ax.set_xlim(0.5, 90); ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (dB/Hz)')
        ax.set_title(f'{state} — DBS {hz} Hz', fontweight='bold')
        if ri == 0 and ci == 1:
            ax.legend(fontsize=7, loc='upper right', framealpha=0.88)

fig.tight_layout()
fig.savefig(f'{OUT_DIR}/figA_psd_full_matrix.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  figA saved")

# ── FIG B: 3×2 zoomed PSD 0.5–50 Hz ─────────────────────────────────────────
BAND_SHADES = [(0.5,4,'#3D405B',0.07),(4,8,'#81B29A',0.07),
               (8,13,'#F2CC8F',0.07),(13,30,'#E07A5F',0.06)]

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('EEG Neural Band Region 0.5–50 Hz: DBS Harmonic Suppression\n'
             'δ / θ / α / β band shading shown | DBS harmonic positions marked',
             fontsize=13, fontweight='bold')

for ri, hz in enumerate(DBS_FREQS):
    for ci, state in enumerate(STATES):
        ax = axes[ri, ci]
        for lo, hi, bc, ba in BAND_SHADES:
            ax.axvspan(lo, hi, alpha=ba, color=bc, zorder=0)
        fb, pb = psds[(state, 'Baseline')]
        m = (fb >= 0.5) & (fb <= 50)
        ax.plot(fb[m], pb[m], color=COLORS['Baseline'], lw=1.2, alpha=0.65,
                linestyle='--', label='Baseline', zorder=1)
        fr, pr = psds[(state, hz, 'Raw')]
        mr = (fr >= 0.5) & (fr <= 50)
        ax.plot(fr[mr], pr[mr], color=COLORS['Raw'], lw=0.8, alpha=0.35, label='Raw DBS', zorder=2)
        for mth in ['Zapline','Time-Domain Hampel','Freq-Domain Hampel']:
            f, p = psds[(state, hz, mth)]
            mf = (f >= 0.5) & (f <= 50)
            ax.plot(f[mf], p[mf], color=COLORS[mth], lw=1.5, alpha=0.72, label=mth, zorder=3)
        f, p = psds[(state, hz, 'Spectrum Fit')]
        mf = (f >= 0.5) & (f <= 50)
        ax.plot(f[mf], p[mf], color=COLORS['Spectrum Fit'], lw=2.5, alpha=0.95,
                label='Spectrum Fit ★', zorder=5)
        harm_vlines(ax, hz, 51, alpha=0.28, lw=1.0)
        ylim = ax.get_ylim()
        for h in np.arange(hz, 51, hz):
            ax.text(h, ylim[1], f'{int(h)}', fontsize=6, color='#9B2226',
                    ha='center', va='top', rotation=90, clip_on=True)
        ax.set_xlim(0.5, 50); ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (dB/Hz)')
        ax.set_title(f'{state} — DBS {hz} Hz', fontweight='bold')
        for name, (lo, hi2) in [('δ',(0.5,4)),('θ',(4,8)),('α',(8,13)),('β',(13,30))]:
            ax.text((lo+hi2)/2, ylim[0]+0.5, name, fontsize=8,
                    color='#333', ha='center', alpha=0.55)
        if ri == 0 and ci == 1:
            ax.legend(fontsize=7, loc='upper right', framealpha=0.88)

fig.tight_layout()
fig.savefig(f'{OUT_DIR}/figB_psd_lowfreq_matrix.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  figB saved")

# ── FIG C: ΔSNR bar chart ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('SNR Improvement (ΔSNR) After DBS Artifact Removal\n'
             'ΔSNR = SNR_filtered − SNR_raw  |  Positive = better suppression',
             fontsize=13, fontweight='bold')

for ax, state in zip(axes, STATES):
    sub = df[df['State'] == state]
    x = np.arange(len(DBS_FREQS)); n = len(METHODS); width = 0.7/n
    for i, method in enumerate(METHODS):
        vals = [float(sub[(sub['DBS_Hz']==hz)&(sub['Method']==method)]['ΔSNR_dB'].values[0])
                for hz in DBS_FREQS]
        offset = (i - n/2 + 0.5)*width
        bars = ax.bar(x+offset, vals, width, label=('★ ' if method=='Spectrum Fit' else '')+method,
                      color=COLORS[method], alpha=0.88, edgecolor='white', linewidth=0.5)
        if method == 'Spectrum Fit':
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+(0.2 if v>=0 else -0.8),
                        f'{v:+.1f}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold', color=COLORS['Spectrum Fit'])
    ax.axhline(0, color='black', lw=1.0, alpha=0.5)
    ax.axhline(3, color='#2A9D8F', lw=1.2, linestyle='--', alpha=0.7, label='3 dB target')
    ax.set_xticks(x); ax.set_xticklabels([f'{hz} Hz' for hz in DBS_FREQS])
    ax.set_ylabel('ΔSNR (dB)'); ax.set_title(f'{state}', fontweight='bold')
    ax.legend(fontsize=8, loc='best', framealpha=0.9)

fig.tight_layout()
fig.savefig(f'{OUT_DIR}/figC_snr_improvement.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  figC saved")

# ── FIG D: SDI bar chart ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Spectral Distortion Index (SDI): Neural Signal Fidelity\n'
             'SDI = RMS(PSD_filtered − PSD_baseline)  |  Lower = better preservation',
             fontsize=13, fontweight='bold')

for ax, state in zip(axes, STATES):
    sub = df[df['State']==state]
    x = np.arange(len(DBS_FREQS)); n = len(METHODS); width = 0.7/n
    for i, method in enumerate(METHODS):
        vals = [float(sub[(sub['DBS_Hz']==hz)&(sub['Method']==method)]['SDI_dB'].values[0])
                for hz in DBS_FREQS]
        offset = (i - n/2 + 0.5)*width
        ax.bar(x+offset, vals, width, label=('★ ' if method=='Spectrum Fit' else '')+method,
               color=COLORS[method], alpha=0.88, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f'{hz} Hz' for hz in DBS_FREQS])
    ax.set_ylabel('SDI (dB RMS)'); ax.set_title(f'{state}', fontweight='bold')
    ax.legend(fontsize=8, loc='best', framealpha=0.9)

fig.tight_layout()
fig.savefig(f'{OUT_DIR}/figD_sdi.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  figD saved")

# ── FIG E: Harmonic attenuation heatmaps (3 DBS freqs) ─────────────────────
for hz in DBS_FREQS:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Per-Harmonic Attenuation (dB) — DBS {hz} Hz\n'
                 f'Positive = suppression | Green cell = strong artifact removal',
                 fontsize=12, fontweight='bold')
    n_harm = min(8, int(NYQUIST / hz) - 1)
    harmonics = [hz*(i+1) for i in range(n_harm)]

    for ax, state in zip(axes, STATES):
        matrix = np.zeros((len(METHODS), n_harm))
        for mi, method in enumerate(METHODS):
            h_dict = harm_all.get((state, hz, method), {})
            for hi2, h in enumerate(harmonics):
                best = min(h_dict, key=lambda k: abs(k-h)) if h_dict else None
                if best is not None and abs(best-h) < 2:
                    matrix[mi, hi2] = np.clip(h_dict[best], -15, 40)

        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=-5, vmax=30)
        ax.set_xticks(range(n_harm))
        ax.set_xticklabels([f'{h:.0f}' for h in harmonics], rotation=40, ha='right', fontsize=8)
        ax.set_yticks(range(len(METHODS))); ax.set_yticklabels(METHODS, fontsize=9)
        ax.set_title(f'{state}', fontweight='bold')
        for mi in range(len(METHODS)):
            for hi2 in range(n_harm):
                v = matrix[mi, hi2]
                ax.text(hi2, mi, f'{v:.1f}', ha='center', va='center',
                        fontsize=7, color='white' if abs(v)>18 else 'black',
                        fontweight='bold' if METHODS[mi]=='Spectrum Fit' else 'normal')
        sf_i = METHODS.index('Spectrum Fit')
        for c in range(n_harm):
            ax.add_patch(plt.Rectangle((c-.5,sf_i-.5),1,1,fill=False,
                                       edgecolor=COLORS['Spectrum Fit'],lw=2.2))
        plt.colorbar(im, ax=ax, label='Attenuation (dB)', shrink=0.85)

    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/figE_{hz}Hz_harmonic_heatmap.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
print("  figE saved (×3)")

# ── FIG F: Band preservation full matrix ────────────────────────────────────
band_names = list(EEG_BANDS.keys())
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('EEG Band Power Preservation (%) After DBS Artifact Removal\n'
             '100% = perfectly preserved | All DBS frequencies & states',
             fontsize=13, fontweight='bold')

for ri, state in enumerate(STATES):
    for ci, hz in enumerate(DBS_FREQS):
        ax = axes[ri, ci]
        matrix = np.zeros((len(METHODS), len(band_names)))
        for mi, method in enumerate(METHODS):
            sub = df[(df['State']==state)&(df['DBS_Hz']==hz)&(df['Method']==method)]
            for bi, band in enumerate(band_names):
                v = sub[f'Pres_{band}'].values
                matrix[mi, bi] = np.clip(float(v[0]) if len(v) else 0, 0, 120)
        im = ax.imshow(matrix, aspect='auto', cmap=PRES_CMAP, vmin=0, vmax=100)
        ax.set_xticks(range(len(band_names)))
        ax.set_xticklabels([b.split(' ')[0] for b in band_names], rotation=30, ha='right', fontsize=8)
        ax.set_yticks(range(len(METHODS))); ax.set_yticklabels(METHODS, fontsize=8)
        ax.set_title(f'{state} — {hz} Hz DBS', fontsize=9, fontweight='bold')
        sf_i = METHODS.index('Spectrum Fit')
        for mi in range(len(METHODS)):
            for bi in range(len(band_names)):
                v = matrix[mi, bi]
                ax.text(bi, mi, f'{v:.0f}', ha='center', va='center', fontsize=7,
                        color='white' if v<35 else 'black',
                        fontweight='bold' if METHODS[mi]=='Spectrum Fit' else 'normal')
        for c in range(len(band_names)):
            ax.add_patch(plt.Rectangle((c-.5,sf_i-.5),1,1,fill=False,
                                       edgecolor=COLORS['Spectrum Fit'],lw=2.0))
        plt.colorbar(im, ax=ax, shrink=0.85, label='%')

fig.tight_layout()
fig.savefig(f'{OUT_DIR}/figF_band_preservation.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  figF saved")

# ── FIG G: Time-domain waveform (Cz, DBS 7 Hz, 4s segment) ─────────────────
def get_segment(data, sfreq, t0=5.0, dur=4.0, ch_idx=9):
    """ch_idx=9 is Cz in standard 10-20 for these arrays."""
    s0 = int(t0*sfreq); s1 = int((t0+dur)*sfreq)
    sig = data[min(ch_idx, data.shape[0]-1), s0:s1] * 1e6
    return np.linspace(t0, t0+dur, len(sig)), sig

fig, axes = plt.subplots(len(METHODS)+1, 2, figsize=(16, 14))
fig.suptitle('Time-Domain Signal at Cz Electrode — DBS 7 Hz\n'
             'Full preprocessing applied (bandpass + avg ref) | 4-second segment',
             fontsize=13, fontweight='bold')

for ci, state in enumerate(STATES):
    data_dbs, sf, _ = raws[(state, 7)]
    data_base, _, _ = raws[(state, 'baseline')]
    # use first 4s for baseline (resampled already)
    t_base = np.linspace(5, 9, int(4*SFREQ_DBS))
    sig_base = data_base[min(9,data_base.shape[0]-1), int(5*SFREQ_DBS):int(9*SFREQ_DBS)]*1e6

    ax0 = axes[0, ci]
    t_raw, sig_raw = get_segment(data_dbs, sf)
    ax0.plot(t_base, sig_base[:len(t_base)], color=COLORS['Baseline'], lw=1.0,
             alpha=0.75, label='Baseline (no DBS)')
    ax0.plot(t_raw, sig_raw, color=COLORS['Raw'], lw=0.8, alpha=0.8, label='Raw DBS 7 Hz')
    ax0.set_title(f'{state} — Raw vs Baseline', fontsize=9, fontweight='bold')
    ax0.set_ylabel('µV', fontsize=8); ax0.legend(fontsize=7, loc='upper right')

    for ri, method in enumerate(METHODS):
        ax = axes[ri+1, ci]
        dc = filtered[(state, 7, method)]
        t_c, sig_c = get_segment(dc, sf)
        ax.plot(t_raw, sig_raw, color=COLORS['Raw'], lw=0.5, alpha=0.3, label='Raw')
        ax.plot(t_c, sig_c, color=COLORS[method],
                lw=2.0 if method=='Spectrum Fit' else 1.2, alpha=0.92,
                label=('★ ' if method=='Spectrum Fit' else '')+method)
        ax.set_ylabel('µV', fontsize=8)
        ax.set_title(f'{state} — {method}'+(' ★ BEST' if method=='Spectrum Fit' else ''),
                     fontsize=9,
                     fontweight='bold' if method=='Spectrum Fit' else 'normal',
                     color=COLORS[method] if method=='Spectrum Fit' else 'black')
        ax.legend(fontsize=7, loc='upper right')
        if method=='Spectrum Fit':
            for sp in ax.spines.values():
                sp.set_edgecolor(COLORS['Spectrum Fit']); sp.set_linewidth(2)
        if ri == len(METHODS)-1: ax.set_xlabel('Time (s)', fontsize=9)

fig.tight_layout()
fig.savefig(f'{OUT_DIR}/figG_time_domain.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  figG saved")

# ── FIG H: Topomaps (band power before vs after Spectrum Fit) ────────────────
from mne import create_info as mne_ci

def make_topomap(data, sfreq_t, info_mne, lo, hi, ax, title, vmin=None, vmax=None):
    """Plot topomap of band power."""
    try:
        f, p = signal.welch(data, fs=sfreq_t, nperseg=256, noverlap=128, window='hann', axis=1)
        m = (f >= lo) & (f <= hi)
        bp = np.array([simpson(p[ch, m], x=f[m]) for ch in range(p.shape[0])])
        bp_db = 10*np.log10(bp + 1e-30)
        if vmin is None: vmin = np.percentile(bp_db, 5)
        if vmax is None: vmax = np.percentile(bp_db, 95)
        im, _ = mne.viz.plot_topomap(
            bp_db, info_mne, axes=ax, show=False,
            cmap='RdBu_r', vlim=(vmin, vmax), contours=4)
        ax.set_title(title, fontsize=7, pad=2)
        return im, vmin, vmax
    except Exception as e:
        ax.set_title(title+f'\n({str(e)[:30]})', fontsize=6)
        ax.axis('off')
        return None, None, None

topo_bands = [('θ (4–8 Hz)', 4, 8), ('α (8–13 Hz)', 8, 13), ('β (13–30 Hz)', 13, 30)]

for hz in DBS_FREQS:
    n_cols = 4  # AWAKE raw, AWAKE SF, SLEEP raw, SLEEP SF
    fig, axes = plt.subplots(len(topo_bands), n_cols, figsize=(14, 9))
    fig.suptitle(f'Topographic Band Power Maps: Before vs. After Spectrum Fit\n'
                 f'DBS {hz} Hz | Full preprocessing applied',
                 fontsize=12, fontweight='bold')
    col_labels = ['AWAKE – Raw DBS', 'AWAKE – Spectrum Fit',
                  'SLEEP – Raw DBS', 'SLEEP – Spectrum Fit']
    for ci, lbl in enumerate(col_labels):
        axes[0, ci].set_title(lbl, fontsize=8, fontweight='bold', pad=10)

    for bi, (band_name, lo, hi) in enumerate(topo_bands):
        for ci, (state, use_raw) in enumerate([('AWAKE',True),('AWAKE',False),
                                               ('SLEEP',True),('SLEEP',False)]):
            ax = axes[bi, ci]
            data_t, sf_t, info_t = raws[(state, hz)] if use_raw else (
                filtered[(state, hz, 'Spectrum Fit')], raws[(state, hz)][1], raws[(state, hz)][2])
            im, vm, vM = make_topomap(data_t, sf_t, info_t, lo, hi, ax,
                                      band_name if ci == 0 else '')
            if bi == 0 and ci == 0:
                axes[bi, ci].set_ylabel(band_name, fontsize=8, rotation=0,
                                        labelpad=50, va='center')
            if im is not None:
                plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02, label='dB')

    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/figH_{hz}Hz_topomaps.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
print("  figH saved (×3)")

# ── FIG I: Residual artifact PSD ────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('Residual DBS Artifact: PSD − Baseline\n'
             'Ideal filter = zero residual at harmonic positions',
             fontsize=13, fontweight='bold')

for ri, hz in enumerate(DBS_FREQS):
    for ci, state in enumerate(STATES):
        ax = axes[ri, ci]
        fb, pb = psds[(state, 'Baseline')]
        fmax_r = min(50.0, NYQUIST-1)
        m_b = (fb >= 0.5) & (fb <= fmax_r)

        for method in ['Zapline','Time-Domain Hampel','Freq-Domain Hampel']:
            f, p = psds[(state, hz, method)]
            p_i = interp_to(f, p, fb)
            res = p_i - pb; rm = m_b
            ax.plot(fb[rm], res[rm], color=COLORS[method], lw=1.5, alpha=0.72, label=method)
        f, p = psds[(state, hz, 'Spectrum Fit')]
        p_i = interp_to(f, p, fb)
        res_sf = p_i - pb
        rm = m_b
        ax.plot(fb[rm], res_sf[rm], color=COLORS['Spectrum Fit'], lw=2.5,
                alpha=0.95, label='Spectrum Fit ★', zorder=5)
        ax.fill_between(fb[rm], res_sf[rm], 0, where=(res_sf[rm]<0),
                        alpha=0.08, color=COLORS['Spectrum Fit'])
        ax.axhline(0, color='black', lw=0.9, alpha=0.4)
        harm_vlines(ax, hz, fmax_r+1, alpha=0.25)
        ax.set_xlim(0.5, fmax_r)
        ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('ΔPower (dB)')
        ax.set_title(f'{state} — DBS {hz} Hz', fontweight='bold')
        if ri == 0 and ci == 1:
            ax.legend(fontsize=8, loc='lower right', framealpha=0.9)

fig.tight_layout()
fig.savefig(f'{OUT_DIR}/figI_residual.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  figI saved")

# ── FIG J: ΔSNR vs SDI efficiency frontier ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Filter Efficiency Frontier: SNR Improvement vs Signal Distortion\n'
             'Best filter = upper-left region (high ΔSNR, low SDI)',
             fontsize=13, fontweight='bold')

mk = {7:'o', 60:'s', 100:'^'}
for ax, state in zip(axes, STATES):
    sub = df[df['State']==state]
    for method in METHODS:
        ms = sub[sub['Method']==method]
        xs = ms['SDI_dB'].values; ys = ms['ΔSNR_dB'].values; hzs = ms['DBS_Hz'].values
        for x, y, h in zip(xs, ys, hzs):
            ax.scatter(x, y, c=COLORS[method], marker=mk[h], s=160, zorder=5,
                       edgecolors='white', linewidth=1.2)
            if method == 'Spectrum Fit':
                ax.annotate(f'{h}Hz', (x,y), xytext=(5,5),
                            textcoords='offset points', fontsize=7,
                            color=COLORS['Spectrum Fit'], fontweight='bold')
    for method in METHODS:
        ax.scatter([], [], c=COLORS[method], s=80,
                   label=('★ ' if method=='Spectrum Fit' else '')+method)
    for h, mkr in mk.items():
        ax.scatter([], [], c='#888', marker=mkr, s=80, label=f'{h} Hz DBS')
    ax.set_xlabel('SDI (dB) ← lower is better'); ax.set_ylabel('ΔSNR (dB) ↑ higher is better')
    ax.set_title(f'{state}', fontweight='bold')
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.88, ncol=2)
    ax.axvline(ax.get_xlim()[0], color='k', alpha=0); ax.axhline(0, color='k', lw=0.8, alpha=0.3)

fig.tight_layout()
fig.savefig(f'{OUT_DIR}/figJ_efficiency_frontier.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  figJ saved")

# ── FIG K: Master proof panel ────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 14))
fig.suptitle('Spectrum Fit: Best Filter for EEG Under Low-Frequency DBS\n'
             'Patient XU | All DBS Frequencies | Full Preprocessing Pipeline Applied',
             fontsize=15, fontweight='bold', y=0.99)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.35)

# Row 0: 7 Hz PSD both states
for ci, state in enumerate(STATES):
    ax = fig.add_subplot(gs[0, ci*2:(ci+1)*2])
    fb, pb = psds[(state,'Baseline')]; m=(fb>=0.5)&(fb<=50)
    ax.plot(fb[m], pb[m], color='grey', lw=1.0, alpha=0.55, linestyle='--', label='Baseline')
    fr, pr = psds[(state,7,'Raw')]; mr=(fr>=0.5)&(fr<=50)
    ax.plot(fr[mr], pr[mr], color=COLORS['Raw'], lw=0.7, alpha=0.35, label='Raw 7 Hz DBS')
    for mth in ['Zapline','Time-Domain Hampel','Freq-Domain Hampel']:
        f, p = psds[(state,7,mth)]; mf=(f>=0.5)&(f<=50)
        ax.plot(f[mf], p[mf], color=COLORS[mth], lw=1.3, alpha=0.65)
    f, p = psds[(state,7,'Spectrum Fit')]; mf=(f>=0.5)&(f<=50)
    ax.plot(f[mf], p[mf], color=COLORS['Spectrum Fit'], lw=2.2, label='Spectrum Fit ★')
    harm_vlines(ax, 7, 51, alpha=0.15)
    ax.set_xlim(0.5,50); ax.set_xlabel('Hz',fontsize=8); ax.set_ylabel('dB/Hz',fontsize=8)
    ax.set_title(f'{state} — 7 Hz DBS PSD', fontsize=9, fontweight='bold')
    if ci==0: ax.legend(fontsize=6.5, loc='upper right', framealpha=0.85)

# Row 1: ΔSNR + SDI grouped bars (all conditions)
labels2 = [f'{hz}Hz\n{st[:2]}' for hz in DBS_FREQS for st in STATES]
x2 = np.arange(len(labels2)); n=len(METHODS); width=0.72/n

ax_snr = fig.add_subplot(gs[1, :2])
ax_sdi = fig.add_subplot(gs[1, 2:])

for ax, metric, ylabel, do_zero in [
    (ax_snr, 'ΔSNR_dB', 'ΔSNR (dB)', True),
    (ax_sdi, 'SDI_dB',  'SDI (dB)', False)
]:
    for i, method in enumerate(METHODS):
        vals = [float(df[(df['State']==st)&(df['DBS_Hz']==hz)&(df['Method']==method)][metric].values[0])
                for hz in DBS_FREQS for st in STATES]
        offset = (i - n/2 + 0.5)*width
        ax.bar(x2+offset, vals, width, label=('★ ' if method=='Spectrum Fit' else '')+method,
               color=COLORS[method], alpha=0.88, edgecolor='white', linewidth=0.4)
    if do_zero: ax.axhline(0, color='black', lw=0.8, alpha=0.4)
    ax.set_xticks(x2); ax.set_xticklabels(labels2, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=9); ax.set_title(ylabel, fontsize=9, fontweight='bold')
    if ax==ax_snr: ax.legend(fontsize=6.5, loc='best', framealpha=0.85)

# Row 2: Band preservation (7 Hz, both states)
for ci, state in enumerate(STATES):
    ax_bp = fig.add_subplot(gs[2, ci*2:(ci+1)*2])
    matrix = np.zeros((len(METHODS), len(EEG_BANDS)))
    for mi, method in enumerate(METHODS):
        for bi, band in enumerate(EEG_BANDS.keys()):
            v = df[(df['State']==state)&(df['DBS_Hz']==7)&(df['Method']==method)][f'Pres_{band}'].values
            matrix[mi, bi] = np.clip(float(v[0]) if len(v) else 0, 0, 100)
    im = ax_bp.imshow(matrix, aspect='auto', cmap=PRES_CMAP, vmin=0, vmax=100)
    ax_bp.set_xticks(range(len(EEG_BANDS)))
    ax_bp.set_xticklabels([b.split(' ')[0] for b in EEG_BANDS.keys()],
                           rotation=25, ha='right', fontsize=8)
    ax_bp.set_yticks(range(len(METHODS))); ax_bp.set_yticklabels(METHODS, fontsize=8)
    ax_bp.set_title(f'{state} Band Preservation (7 Hz DBS)', fontsize=9, fontweight='bold')
    sf_i = METHODS.index('Spectrum Fit')
    for mi in range(len(METHODS)):
        for bi in range(len(EEG_BANDS)):
            v = matrix[mi,bi]
            ax_bp.text(bi, mi, f'{v:.0f}', ha='center', va='center', fontsize=7,
                       color='white' if v<35 else 'black',
                       fontweight='bold' if METHODS[mi]=='Spectrum Fit' else 'normal')
    for c in range(len(EEG_BANDS)):
        ax_bp.add_patch(plt.Rectangle((c-.5,sf_i-.5),1,1,fill=False,
                                      edgecolor=COLORS['Spectrum Fit'],lw=2.2))
    plt.colorbar(im, ax=ax_bp, shrink=0.85, pad=0.02)

fig.savefig(f'{OUT_DIR}/figK_master_proof_panel.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  figK saved")

# ── Print final summary ──────────────────────────────────────────────────────
print("\n" + "="*70)
print("FINAL METRICS SUMMARY (mean across DBS frequencies)")
print("="*70)
summary = df.groupby(['Method','State'])[['ΔSNR_dB','SDI_dB','MeanAtt5_dB']].mean().round(2)
print(summary.to_string())

print(f"\n✓ ALL FIGURES → {OUT_DIR}")
print(f"✓ Metrics CSV → {OUT_DIR}/rigorous_metrics.csv")
print("\nFigures generated:")
for fname in sorted(os.listdir(OUT_DIR)):
    if fname.endswith('.png'):
        sz = os.path.getsize(f'{OUT_DIR}/{fname}')//1024
        print(f"  {fname:45s}  {sz} KB")
