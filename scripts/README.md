# Scripts Directory

All utility scripts, CLI tools, and analysis scripts for the EEG biomarkers pipeline.

---

## Directory Structure

```
scripts/
├── README.md (this file)
├── CLI-EXAMPLES.sh                # Example CLI commands
├── dbs_remove.sh                  # Shell wrapper for DBS removal
├── remove_dbs_artifacts_cli.py    # Python CLI tool for batch processing
├── analysis/
│   ├── pipeline_rigorous.py       # Main full EEG pipeline (use this)
│   └── generate_filter_proof.py   # Filter proof-of-concept
└── report/
    └── build_report.js            # Report generator (Node.js)
```

---

## CLI Tools

Command-line utilities for batch processing and artifact removal:

- **CLI-EXAMPLES.sh** - Example CLI commands and usage patterns
- **dbs_remove.sh** - Shell script wrapper for DBS artifact removal
- **remove_dbs_artifacts_cli.py** - Python CLI tool for batch DBS artifact removal

### Using CLI Tools

```bash
# See examples
cat CLI-EXAMPLES.sh

# Run DBS removal
./dbs_remove.sh input.edf output.edf

# Or use Python CLI
python remove_dbs_artifacts_cli.py --help
```

---

## Analysis Scripts

### `analysis/pipeline_rigorous.py` — Full Rigorous Pipeline

The primary analysis script. Runs the complete EEG preprocessing and filter comparison pipeline across all conditions.

### What it does
1. Loads all 8 EDFs for patient XU (baseline + 7/60/100 Hz × awake/sleep)
2. Applies full preprocessing: bandpass FIR (0.5–90 Hz), notch (60 Hz), average reference, 60s clip
3. Resamples baseline recordings (200 Hz → 256 Hz) to match DBS recording sample rate
4. Applies all 4 artifact removal methods to every DBS condition
5. Computes rigorous metrics: ΔSNR, SDI, per-harmonic attenuation, band power preservation
6. Generates 16 publication-quality figures
7. Saves `rigorous_metrics.csv`

### Usage
```bash
python scripts/analysis/pipeline_rigorous.py
```

### Output
All figures and metrics saved to:
```
figures/rigorous_proof/
├── figA_psd_full_matrix.png           # Full-band PSD (all conditions)
├── figB_psd_lowfreq_matrix.png        # Low-freq PSD detail (0.5–50 Hz)
├── figC_snr_improvement.png           # ΔSNR bar chart
├── figD_spectral_distortion.png       # SDI bar chart
├── figE_7Hz_harmonic_heatmap.png      # Harmonic attenuation heatmap (7 Hz)
├── figE_60Hz_harmonic_heatmap.png     # Harmonic attenuation heatmap (60 Hz)
├── figE_100Hz_harmonic_heatmap.png    # Harmonic attenuation heatmap (100 Hz)
├── figF_band_preservation.png         # EEG band power preservation
├── figG_time_domain.png               # Time-domain trace comparison
├── figH_7Hz_topomaps.png              # Scalp topomaps (7 Hz)
├── figH_60Hz_topomaps.png             # Scalp topomaps (60 Hz)
├── figH_100Hz_topomaps.png            # Scalp topomaps (100 Hz)
├── figI_residual.png                  # Residual PSD (filtered - baseline)
├── figJ_efficiency_frontier.png       # ΔSNR vs SDI efficiency frontier
├── figK_master_proof_panel.png        # Master summary panel
└── rigorous_metrics.csv               # All 24 conditions, all metrics
```

### Key constants (edit at top of script)
| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `data/XU` | Path to EDF files |
| `OUT_DIR` | `figures/rigorous_proof` | Output directory |
| `DBS_FREQS` | `[7, 60, 100]` | DBS frequencies to analyze |
| `CLIP_SEC` | `60` | Recording clip length (seconds) |
| `METHODS` | all 4 | Filter methods to compare |

### Dependencies
```
mne >= 1.0
scipy
numpy
matplotlib
seaborn
pandas
```
All are in `requirements.txt` / `environment.yml`.

---

## `report/build_report.js` — Thesis Report Generator

Generates the full thesis Word document (`.docx`) with embedded figures, tables, and academic text.

### What it does
- Builds a complete academic report with title page, TOC, abstract, methods, results, discussion, and references
- Embeds all 15 figures from `figures/rigorous_proof/`
- Includes 3 tables: dataset summary, full 24-condition metrics, and mean summary
- Professional formatting with headers/footers, Arial font, page numbers

### Usage
```bash
# Install docx package (first time only)
npm install --prefix ./node_modules_local docx

# Run the builder
node scripts/report/build_report.js
```

### Output
```
docs/report/Senior_Thesis_Report_EEG_Biomarkers_v2.docx
```

### Requirements
- Node.js >= 14
- `docx` npm package (installed to `./node_modules_local/`)
- Figures must already exist in `figures/rigorous_proof/` (run pipeline first)

---

## Workflow

```
1. Run pipeline:
   python scripts/analysis/pipeline_rigorous.py

2. Build report:
   npm install --prefix ./node_modules_local docx
   node scripts/report/build_report.js

3. Open report:
   docs/report/Senior_Thesis_Report_EEG_Biomarkers_v2.docx
```

---

## Metrics Summary (Patient XU)

| Method | Avg ΔSNR | Avg SDI | Recommendation |
|---|---|---|---|
| **Spectrum Fit** | **+7.5 dB** | 11.4 dB | Best for 60/100 Hz DBS |
| Freq-Domain Hampel | +3.5 dB | **8.6 dB** | Best for 7 Hz / neural preservation |
| Time-Domain Hampel | −0.4 dB | 14.3 dB | ❌ Avoid — degrades signal |
| Zapline+ | +2.2 dB | 10.6 dB | Good supplement at 60/100 Hz |
