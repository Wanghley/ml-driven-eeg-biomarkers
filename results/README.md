# Results Directory

Centralized location for all analysis outputs, figures, metrics, and reports.

## Directory Structure

```
results/
├── README.md (this file)
├── figures/                    # All generated visualizations
│   ├── filter_proof/          # Filter performance proof figures
│   ├── rigorous_analysis/     # Rigorous analysis output figures
│   └── time_domain_comparisons/  # Time-domain analysis plots
├── metrics/                    # CSV files and data tables
│   ├── metrics_summary_awake.csv
│   ├── metrics_summary_sleep.csv
│   ├── band_preservation_awake.csv
│   ├── band_preservation_sleep.csv
│   ├── harmonic_attenuation_*.csv
│   └── rigorous_metrics.csv
└── reports/                    # Compiled reports and documentation
    └── (report outputs)
```

## Figures

### Filter Proof Figures (`figures/filter_proof/`)
Early-stage validation of filter implementations:
- PSD comparisons
- Individual method performance
- Proof of attenuation

### Rigorous Analysis Figures (`figures/rigorous_analysis/`)
Publication-quality final analysis figures:
- Full-band PSD matrices
- SNR improvement comparisons
- Spectral distortion metrics
- Harmonic attenuation heatmaps (7 Hz, 60 Hz, 100 Hz)
- Band power preservation plots
- Time-domain traces

### Time Domain Comparisons (`figures/time_domain_comparisons/`)
Detailed time-domain analysis figures:
- Raw signal traces
- Method comparisons in time domain
- Artifact removal visualization

## Metrics

### Summary Metrics
- **metrics_summary_awake.csv** - All metrics for awake conditions
- **metrics_summary_sleep.csv** - All metrics for sleep conditions

### Band Preservation
- **band_preservation_awake.csv** - Brain activity preservation (awake)
- **band_preservation_sleep.csv** - Brain activity preservation (sleep)

### Harmonic Attenuation
- **harmonic_attenuation_Hampel_[awake|sleep].csv** - Hampel method peak attenuation
- **harmonic_attenuation_Spectrum_Fit_[awake|sleep].csv** - Spectrum-fit method
- **harmonic_attenuation_Zapline_[awake|sleep].csv** - Zapline method
- **harmonic_attenuation_Freq-Domain_Hampel_[awake|sleep].csv** - Frequency-domain Hampel

### Complete Analysis
- **rigorous_metrics.csv** - Comprehensive metrics from full pipeline_rigorous.py analysis

## Reports

Compiled reports and documentation:
- PDF exports
- HTML reports
- Analysis summaries
- Thesis chapters

## Using Results

### Accessing Figures
```python
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# Load a figure
fig_path = Path("results/figures/rigorous_analysis/figC_snr_improvement.png")
img = Image.open(fig_path)
plt.imshow(img)
```

### Loading Metrics
```python
import pandas as pd

# Load metrics
metrics = pd.read_csv("results/metrics/rigorous_metrics.csv")
print(metrics.head())

# Filter by condition
awake_metrics = metrics[metrics['condition'] == 'AWAKE']
```

### Analysis Examples
```python
# Compare method performance
band_preservation = pd.read_csv("results/metrics/band_preservation_awake.csv")
print(band_preservation.sort_values('preservation_pct', ascending=False))

# Check harmonic attenuation
harmonics = pd.read_csv("results/metrics/harmonic_attenuation_Spectrum_Fit_awake.csv")
print(f"7 Hz harmonic attenuation: {harmonics.loc[0, 'attenuation_db']} dB")
```

## Generating Results

### Run Rigorous Analysis
```bash
python scripts/analysis/pipeline_rigorous.py
```
Generates all figures, metrics, and summary statistics.

### Run Filter Proof
```bash
python scripts/analysis/generate_filter_proof.py
```
Generates filter validation figures.

### Generate Report
```bash
node scripts/report/build_report.js
```
Compiles report from results.

## File Organization Best Practices

1. **Don't manually delete** - Results are reproducible from source notebooks/scripts
2. **Use subdirectories** - Organize by analysis type
3. **Version output** - Keep timestamped results if comparing runs
4. **Document changes** - Note modifications in results metadata

## Output Naming Convention

Generated files typically follow:
- `fig[A-Z]_[description].png` - Figures
- `[metric]_[condition]_[method].csv` - Metrics
- `[method]_[condition]_results.json` - Raw data

## Storage Notes

- **Figures**: PNG format, 300 dpi for publication
- **Metrics**: CSV format for Excel/pandas compatibility
- **Reports**: HTML + PDF formats
- **Total size**: ~500 MB to 2 GB depending on analysis depth

## Notes

- Results are generally NOT version controlled (see .gitignore)
- All results are reproducible from notebooks and scripts
- Archive complete results for thesis/publication submission
- See main README for integration with other directories
