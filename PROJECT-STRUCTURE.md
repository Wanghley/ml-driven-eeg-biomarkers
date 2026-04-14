# Project Structure Guide

Complete guide to the ML-Driven EEG Biomarkers repository organization.

## High-Level Overview

```
ml-driven-eeg-biomarkers/
├── Documentation & Config (root level)
│   ├── README.md                    # Main project README
│   ├── IMPLEMENTATION-SUMMARY.md    # Implementation details
│   ├── State-of-the-Project.md      # Project status
│   ├── environment.yml              # Conda environment
│   ├── requirements.txt             # PIP requirements
│   ├── PROJECT-STRUCTURE.md         # This file
│   └── .gitignore                   # Version control rules
│
├── src/                             # Production Code
│   ├── __init__.py
│   ├── preprocessing.py             # EEGPreprocessor (Surgical Pipeline API)
│   ├── filters.py                   # High-Fidelity DBS Removal Filters
│   └── README.md                    # Code documentation
│
├── notebooks/                       # Jupyter Analysis
│   ├── README.md                    # Notebook guide
│   ├── 01-baseline-referenced-dbs-removal.ipynb
│   ├── 02-dbs-artifact-removal-comparison.ipynb
│   ├── 03-dbs-filter-diagnostic-pipeline.ipynb
│   ├── 04-eeg-dbs-signal-pipeline.ipynb
│   ├── 05-lowfreq-dbs-artifact-removal-comparison.ipynb
│   ├── 06-pipeline-v2.ipynb
│   ├── 07-poster-optimized.ipynb
│   └── experiments/                 # Experimental notebooks
│       ├── EDA-sleep-vs-awake.ipynb
│       ├── exploratory-xu.ipynb
│       ├── pipeline.ipynb
│       ├── pre-processing-artifact-removal.ipynb
│       ├── Time-frequency-analysis.ipynb
│       └── ...
│
├── scripts/                         # CLI & Analysis Tools
│   ├── README.md                    # Scripts guide
│   ├── CLI-EXAMPLES.sh              # CLI examples
│   ├── dbs_remove.sh                # Shell wrapper
│   ├── remove_dbs_artifacts_cli.py  # Python CLI
│   ├── analysis/                    # Analysis scripts
│   │   ├── pipeline_rigorous.py     # Main analysis
│   │   └── generate_filter_proof.py # Filter validation
│   └── report/
│       └── build_report.js          # Report generator
│
├── data/                            # Data Directory
│   ├── README.md                    # Data guide
│   ├── raw/                         # Raw EDF files (not in git)
│   │   ├── XU/                      # XU patient data
│   │   │   ├── XUAWAKE7_deidentified.edf
│   │   │   ├── XUAWAKE60_deidentified.edf
│   │   │   ├── XUAWAKE100_deidentified.edf
│   │   │   ├── XUAWAKEPRE_deidentified.edf
│   │   │   ├── XUSLEEP7_deidentified.edf
│   │   │   ├── XUSLEEP60_deidentified.edf
│   │   │   ├── XUSLEEP100_deidentified.edf
│   │   │   └── XUSLEEPPRE_deidentified.edf
│   │   └── LI/                      # LI patient data
│   └── processed/                   # Preprocessed data (not in git)
│       └── (processed .fif files)
│
├── results/                         # Analysis Outputs
│   ├── README.md                    # Results guide
│   ├── figures/                     # Generated figures (not in git)
│   │   ├── filter_proof/
│   │   ├── rigorous_analysis/
│   │   └── time_domain_comparisons/
│   ├── metrics/                     # CSV metrics (not in git)
│   │   ├── metrics_summary_awake.csv
│   │   ├── metrics_summary_sleep.csv
│   │   ├── band_preservation_*.csv
│   │   └── harmonic_attenuation_*.csv
│   └── reports/                     # Generated reports (not in git)
│
├── docs/                            # Documentation
│   ├── README.md                    # Docs index and navigation
│   ├── API-REFERENCE.md             # Complete API docs
│   ├── guides/                      # How-to guides
│   │   ├── CLI-GUIDE.md
│   │   ├── CLI-QUICK-REFERENCE.md
│   │   ├── DBS-ARTIFACT-REMOVAL-METHODS.md
│   │   ├── QUICK-START-DBS-REMOVAL.md
│   │   └── VISUAL-METHOD-COMPARISON.md
│   ├── images/                      # Documentation images
│   └── presentation/                # Presentation materials
│
├── tests/                           # Unit Tests
│   ├── README.md                    # Testing guide
│   ├── __init__.py
│   ├── test_filters.py              # Filter unit tests
│   └── test_preprocessing.py        # Preprocessing tests
│
└── config/                          # Configuration Files
    └── (future: settings, parameters)
```

## Directory Purposes

### `/src` - Production Code
**What**: Core Python modules for surgical EEG processing and DBS artifact removal.
**Key Files**: 
- `preprocessing.py` - EEGPreprocessor class with multi-stage surgical suite.
- `filters.py` - Production-grade filters (Sinusoidal Regression, Hampel, Phase Template).
**When to use**: Import these modules for high-fidelity clinical artifact removal.
**Size**: ~1-2 MB

### `/notebooks` - Analysis & Exploration
**What**: Jupyter notebooks for analysis and development
**Key Files**: 
- `01-02-*.ipynb` - Main analysis notebooks (use these)
- `experiments/` - Experimental notebooks (for development)
**When to use**: Exploratory analysis, visualization, reporting
**Size**: ~100-300 MB (compressed)

### `/scripts` - Utilities & Tools
**What**: Command-line tools and analysis scripts
**Key Files**:
- `remove_dbs_artifacts_cli.py` - CLI tool for batch processing
- `analysis/pipeline_rigorous.py` - Rigorous analysis pipeline
**When to use**: Batch processing, command-line workflows
**Size**: ~50 KB

### `/data` - EEG Data Storage
**What**: Raw and processed EEG data
**Not in Git**: Large .edf files (see .gitignore)
**Structure**: 
- `raw/[PATIENT]/[FILENAME].edf` - Raw recordings
- `processed/` - Cleaned data in MNE format
**Size**: ~2-10 GB (not version controlled)

### `/results` - Analysis Outputs
**What**: Generated figures, metrics, reports
**Not in Git**: Can be recreated (see .gitignore)
**Structure**:
- `figures/` - PNG/PDF plots
- `metrics/` - CSV data tables
- `reports/` - HTML/PDF reports
**Size**: Varies, typically 1-5 GB

### `/docs` - Documentation
**What**: User guides, API reference, tutorials
**Key Files**:
- `API-REFERENCE.md` - Complete API documentation
- `guides/` - How-to guides and tutorials
**When to use**: Learning, reference, troubleshooting
**Size**: ~2-5 MB

### `/tests` - Unit Tests
**What**: Pytest-based unit and integration tests
**Key Files**:
- `test_filters.py` - Filter implementation tests
- `test_preprocessing.py` - Pipeline tests
**When to use**: Validate changes, ensure code quality
**Size**: ~100 KB

### `/config` - Configuration
**What**: Settings, parameters, future configuration files
**Currently**: Empty (for future use)
**Size**: ~1 KB

## Quick Navigation

| Task | Location | Notes |
|------|----------|-------|
| **Get started** | `docs/guides/QUICK-START-DBS-REMOVAL.md` | 5-minute intro |
| **Understand methods** | `docs/guides/DBS-ARTIFACT-REMOVAL-METHODS.md` | Detailed guide |
| **Run analysis** | `notebooks/02-dbs-artifact-removal-comparison.ipynb` | Main notebook |
| **Use CLI** | `scripts/` | See `scripts/README.md` |
| **Reference API** | `docs/API-REFERENCE.md` | Complete docs |
| **Experiment** | `notebooks/experiments/` | Development |
| **View results** | `results/figures/` | Output figures |
| **Check metrics** | `results/metrics/` | CSV data |
| **Run tests** | `pytest tests/` | Validation |

## File Organization Principles

### 1. **Separation of Concerns**
- **Code** → `src/`
- **Analysis** → `notebooks/`
- **Tools** → `scripts/`
- **Data** → `data/`
- **Output** → `results/`
- **Docs** → `docs/`

### 2. **Consistency**
- All notebooks numbered: `01-`, `02-`, etc.
- All data organized by patient: `data/raw/[PATIENT]/`
- All outputs in `results/` with clear subdirectories

### 3. **Discoverability**
- Every directory has a `README.md`
- Clear naming conventions
- Documented structure

### 4. **Reproducibility**
- Source code in version control
- Data not in version control (see `.gitignore`)
- Results recreatable from code

### 5. **Scalability**
- Easy to add new patients to `data/raw/`
- Easy to add new notebooks to `notebooks/`
- Easy to add new scripts to `scripts/`

## Version Control Strategy

### What's in Git
- ✓ Source code (`src/`)
- ✓ Notebooks (`notebooks/`)
- ✓ Scripts (`scripts/`)
- ✓ Documentation (`docs/`)
- ✓ Tests (`tests/`)
- ✓ Configuration files (`.gitignore`, `requirements.txt`)

### What's NOT in Git
- ✗ Large data files (`data/raw/*.edf`)
- ✗ Processed data (`data/processed/*.fif`)
- ✗ Generated figures (`results/figures/`)
- ✗ Generated metrics (`results/metrics/*.csv`)
- ✗ Python cache (`__pycache__/`, `.ipynb_checkpoints`)
- ✗ Virtual environments (`venv/`, `env/`)

## Workflow Example

### Typical Analysis Workflow

1. **Setup**
   ```bash
   conda env create -f environment.yml
   conda activate eeg-biomarkers
   ```

2. **Explore Data**
   ```bash
   jupyter lab notebooks/01-baseline-referenced-dbs-removal.ipynb
   ```

3. **Run Analysis**
   ```bash
   python scripts/analysis/pipeline_rigorous.py
   ```

4. **Review Results**
   ```bash
   # Check figures
   ls results/figures/rigorous_analysis/
   
   # Check metrics
   pandas results/metrics/rigorous_metrics.csv
   ```

5. **Commit Progress**
   ```bash
   git add notebooks/
   git commit -m "Add new analysis notebook"
   git push
   ```

## Adding New Content

### Adding a New Notebook
```bash
# Create new notebook with next number
notebooks/08-my-analysis.ipynb

# See: notebooks/README.md for naming guide
```

### Adding New Data
```bash
# Create folder for new patient
data/raw/[NEW_PATIENT]/

# Add EDF files following naming convention
data/raw/[NEW_PATIENT]/[ID][STATE][FREQ]_deidentified.edf

# See: data/README.md for details
```

### Adding New Script
```bash
# For CLI tools
scripts/my_tool.py

# For analysis
scripts/analysis/my_analysis.py

# See: scripts/README.md for organization
```

## Directory Size Reference

| Directory | Size | Typical | Notes |
|-----------|------|---------|-------|
| `src/` | ~1-2 KB | Small | Production code |
| `notebooks/` | ~100-300 MB | Medium | Uncompressed without outputs |
| `scripts/` | ~50 KB | Tiny | Utilities |
| `docs/` | ~2-5 MB | Small | Documentation |
| `tests/` | ~100 KB | Tiny | Test code |
| `data/raw/` | ~2-10 GB | Large | **Not in git** |
| `data/processed/` | ~1-5 GB | Large | **Not in git** |
| `results/` | 1-10 GB | Large | **Not in git** |

## Best Practices

### ✓ Do's
- Keep `src/` clean and focused on production code
- Use notebooks for exploration and communication
- Save scripts for repeatable automation
- Organize data by subject/patient
- Use consistent naming conventions
- Keep README.md files updated
- Document in docstrings

### ✗ Don'ts
- Don't commit large data files
- Don't put analysis code in `src/`
- Don't mix different data types in one folder
- Don't create deeply nested directory structures
- Don't forget to document directory purpose
- Don't update only one part of the structure

## Future Expansion

This structure supports:
- ✓ Multiple patients (add to `data/raw/`)
- ✓ Multiple analysis methods (add to `notebooks/`)
- ✓ Multiple preprocessing variants (add to `scripts/`)
- ✓ Expanded testing (add to `tests/`)
- ✓ Better organization as project grows

## Related Files

- `README.md` - Project overview
- `IMPLEMENTATION-SUMMARY.md` - Implementation details
- `State-of-the-Project.md` - Project status
- `.gitignore` - Version control rules
- `environment.yml` - Conda environment
- `requirements.txt` - Pip requirements
