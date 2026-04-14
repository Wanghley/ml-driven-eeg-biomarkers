# Jupyter Notebooks Guide

This directory contains all Jupyter notebooks for the ML-Driven EEG Biomarkers project, organized by purpose.

## Main Analysis Notebooks

These are the primary analysis and processing notebooks:

1. **01-baseline-referenced-dbs-removal.ipynb** - DBS artifact removal using baseline-referenced Wiener filtering
2. **02-dbs-artifact-removal-comparison.ipynb** - Comprehensive comparison of all 5 DBS removal methods
3. **03-dbs-filter-diagnostic-pipeline.ipynb** - Diagnostic pipeline for DBS filter validation
4. **04-eeg-dbs-signal-pipeline.ipynb** - Complete EEG DBS signal processing pipeline
5. **05-lowfreq-dbs-artifact-removal-comparison.ipynb** - Specific comparison for low-frequency DBS artifact removal with ICA and Hampel filtering
6. **06-pipeline-v2.ipynb** - Version 2 of the processing pipeline
7. **07-poster-optimized.ipynb** - Optimized notebook for presentation/poster generation

## Experiments & Development

Experimental notebooks are located in the `experiments/` subdirectory:

- **EDA-sleep-vs-awake.ipynb** - Exploratory data analysis comparing sleep vs awake EEG
- **exploratory-xu.ipynb** - Exploratory analysis of XU dataset
- **pipeline.ipynb** - Original pipeline notebook
- **pipeline-v2.ipynb** - Development version 2
- **pipeline.test.ipynb** - Test notebook for pipeline validation
- **pre-processing-artifact-removal.ipynb** - Pre-processing and artifact removal exploration
- **Time-frequency-analysis.ipynb.ipynb** - Time-frequency analysis and visualization
- **test.ipynb** - General testing notebook

## How to Use

### Running Notebooks
```bash
jupyter lab
# or
jupyter notebook
```

### Recommended Workflow

1. **Start with:** `02-dbs-artifact-removal-comparison.ipynb` - Best overview of methods
2. **For baseline:** `01-baseline-referenced-dbs-removal.ipynb` - Uses baseline reference
3. **For diagnostics:** `03-dbs-filter-diagnostic-pipeline.ipynb` - Validate your filters
4. **For full pipeline:** `04-eeg-dbs-signal-pipeline.ipynb` - Complete processing

### For Experimentation

Use notebooks in the `experiments/` folder to:
- Test new preprocessing approaches
- Compare different parameters
- Validate improvements
- Generate exploratory visualizations

## Data Access

All notebooks automatically source data from `../data/raw/XU/` and `../data/raw/LI/`.

Output files (figures, metrics) are saved to `../results/`:
- Figures → `../results/figures/`
- Metrics/CSVs → `../results/metrics/`
- Reports → `../results/reports/`

## Project Structure
```
notebooks/
├── README.md (this file)
├── 01-baseline-referenced-dbs-removal.ipynb
├── 02-dbs-artifact-removal-comparison.ipynb
├── ... (other main notebooks)
└── experiments/
    ├── EDA-sleep-vs-awake.ipynb
    ├── pipeline.ipynb
    └── ... (other experimental notebooks)
```

## Notes

- Notebooks assume MNE-Python and all dependencies are installed (see `../environment.yml`)
- Raw data (.edf files) are stored in `../data/raw/`
- Output is centralized in `../results/` for easy organization
- For development, use `experiments/` subdirectory
- For production analysis, use main notebooks
