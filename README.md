<!-- PROJECT SHIELDS -->
<a name="readme-top"></a>
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/wanghley)

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/wanghley/ml-driven-eeg-biomarkers">
    <img src="eeg_analysis.gif" alt="Logo" width="280">
  </a>
  <h3 align="center">ML-Driven EEG Biomarkers</h3>
  <p align="center">
    Advanced machine learning pipeline for extracting clinically relevant biomarkers from EEG signals
    <br />
    <a href="#"><strong>Explore the code »</strong></a>
    <br />
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## 🚀 High-Fidelity DBS Artifact Removal Suite
The core library (`src/`) now features a production-grade surgical pipeline designed to remove 7 Hz stimulation artifacts while preserving endogenous brain biomarkers:

- **Stage I: Foundation**: 0.1-100 Hz Filtering, Bad Channel Interpolation, CAR.
- **Stage II: Hybrid Surgical Suite**: 
    - *Chunked Sinusoidal Regression* (4s windows) for non-stationary DBS.
    - *Complex Spectral Hampel* for narrowband residual cleaning.
    - *Phase Template Subtraction* for waveform jitter compensation.
- **Stage III: Automated ICA**: SNR-driven rejection of harmonic components.

The master `pipeline.py` script utilizes this suite to generate comprehensive validation reports, including PSD overlays, time-domain comparisons, and biomarker integrity scores.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="python" style="vertical-align:top; margin:4px"> <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="tensorflow" style="vertical-align:top; margin:4px"> <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="sklearn" style="vertical-align:top; margin:4px"> <img src="https://img.shields.io/badge/MNE-Python-326CE5?style=for-the-badge" alt="mne" style="vertical-align:top; margin:4px"> <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="numpy" style="vertical-align:top; margin:4px">

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- PROJECT STRUCTURE -->
## 📁 Project Organization

This repository is organized for clarity and maintainability. **See [PROJECT-STRUCTURE.md](PROJECT-STRUCTURE.md) for the complete directory guide.**

```
ml-driven-eeg-biomarkers/
├── src/                    # ✨ Production code (use this!)
├── notebooks/              # 📓 Analysis & exploration
├── scripts/                # 🔧 CLI tools & utilities
├── data/                   # 📊 Raw & processed EEG data
├── results/                # 📈 Generated figures, metrics, reports
├── docs/                   # 📚 Comprehensive documentation
├── tests/                  # ✅ Unit & integration tests
└── config/                 # ⚙️ Configuration (future)
```

### 🚀 Quick Start Paths

| I want to... | Go to... |
|---|---|
| **Get started quickly** | [docs/guides/QUICK-START-DBS-REMOVAL.md](docs/guides/QUICK-START-DBS-REMOVAL.md) |
| **Run analysis** | [notebooks/02-dbs-artifact-removal-comparison.ipynb](notebooks/02-dbs-artifact-removal-comparison.ipynb) |
| **Understand methods** | [docs/guides/DBS-ARTIFACT-REMOVAL-METHODS.md](docs/guides/DBS-ARTIFACT-REMOVAL-METHODS.md) |
| **Use command-line tools** | [scripts/README.md](scripts/README.md) |
| **View all notebooks** | [notebooks/README.md](notebooks/README.md) |
| **Explore API** | [docs/API-REFERENCE.md](docs/API-REFERENCE.md) |
| **See project structure** | [PROJECT-STRUCTURE.md](PROJECT-STRUCTURE.md) ← Start here! |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

To get started with the ML-Driven EEG Biomarkers pipeline, follow these steps:

### Prerequisites

* Python 3.8 or higher
* pip or conda package manager
* ~10 GB free disk space (for data and results)

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/wanghley/ml-driven-eeg-biomarkers.git
   cd ml-driven-eeg-biomarkers
   ```

2. Create and activate virtual environment
   ```bash
   # Using conda (recommended)
   conda env create -f environment.yml
   conda activate eeg-biomarkers
   
   # OR using pip + venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Verify installation
   ```bash
   python -c "import mne; import src; print('✓ Setup successful')"
   ```

### First Steps

#### Option 1: Run a Notebook (Recommended for Learning)
```bash
# Start Jupyter
jupyter lab

# Open and run:
# notebooks/02-dbs-artifact-removal-comparison.ipynb
```

#### Option 2: Use Python API
```python
from src.preprocessing import EEGPreprocessor
import mne

# Load data
raw = mne.io.read_raw_edf("data/raw/XU/XUAWAKE7_deidentified.edf")

# Remove DBS artifacts
preprocessor = EEGPreprocessor()
clean = preprocessor.remove_artifacts_spectrum_fit(raw)

print("✓ Artifact removal complete!")
```

#### Option 3: Use Command-Line Tool
```bash
# See available commands
cat scripts/CLI-EXAMPLES.sh

# Remove DBS artifacts
python scripts/remove_dbs_artifacts_cli.py \
  --input data/raw/XU/XUAWAKE7.edf \
  --output results/XUAWAKE7_clean.edf \
  --method spectrum_fit
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 📖 Usage

### Core API: EEGPreprocessor

```python
from src.preprocessing import EEGPreprocessor
import mne

preprocessor = EEGPreprocessor()

# Load and preprocess
raw = mne.io.read_raw_edf("data/raw/XUAWAKE7.edf")

# Choose your removal method:

# 1. Spectrum-Fit (strong harmonic removal)
clean = preprocessor.remove_artifacts_spectrum_fit(
    raw, f_target=7.0, attenuation_db=-60.0
)

# 2. Zapline+ (adaptive removal)
clean = preprocessor.remove_artifacts_zapline(
    raw, f_target=7.0, n_harmonics=5
)

# 3. Hampel Frequency-Domain
clean = preprocessor.remove_artifacts_freq_hampel(
    raw, window_hz=3.0, attenuation_db=-60.0
)

# 4. Hampel Time-Domain
clean = preprocessor.remove_artifacts_time_hampel(
    raw, window_sec=0.02, attenuation_factor=2.0
)

# Or use generic interface
clean = preprocessor.remove_artifacts_advanced(
    raw, method='spectrum_fit', f_target=7.0, attenuation_db=-60.0
)

# Extract features
psd, freqs = preprocessor.extract_spectral_features(clean)
```

### Running Analysis

```bash
# Comprehensive method comparison
python scripts/analysis/pipeline_rigorous.py

# Generate filter validation
python scripts/analysis/generate_filter_proof.py

# See results
ls results/figures/
ls results/metrics/
```

### Complete Workflow Example

See [notebooks/02-dbs-artifact-removal-comparison.ipynb](notebooks/02-dbs-artifact-removal-comparison.ipynb) for a complete example including:
- Data loading
- Method application
- Visualization
- Metrics computation

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## ✨ Key Features

### 5 DBS Artifact Removal Methods
- ✓ **Spectrum-Fit** - Strong, tunable harmonic removal (DEFAULT)
- ✓ **Zapline+** - Adaptive spectro-spatial filtering (Chen et al., 2022)
- ✓ **Hampel Frequency-Domain** - Outlier-based removal (Allen et al., 2010)
- ✓ **Hampel Time-Domain** - Non-stationary noise suppression
- ✓ **Wiener Baseline-Referenced** - Optimal with baseline data

### Comprehensive Analysis Tools
- Full preprocessing pipeline with MNE-Python
- Automated feature extraction (PSD, band power)
- Method comparison and validation
- Publication-quality visualizations
- Detailed metrics and reports

### Well-Organized Codebase
- Production code in `src/`
- Analysis notebooks in `notebooks/`
- CLI tools in `scripts/`
- Comprehensive documentation in `docs/`

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 📚 Documentation

All documentation is centralized in the `docs/` folder:

- **[API-REFERENCE.md](docs/API-REFERENCE.md)** - Complete API documentation
- **[guides/QUICK-START-DBS-REMOVAL.md](docs/guides/QUICK-START-DBS-REMOVAL.md)** - 5-minute quick start
- **[guides/DBS-ARTIFACT-REMOVAL-METHODS.md](docs/guides/DBS-ARTIFACT-REMOVAL-METHODS.md)** - Detailed method guide
- **[guides/CLI-GUIDE.md](docs/guides/CLI-GUIDE.md)** - Command-line reference
- **[PROJECT-STRUCTURE.md](PROJECT-STRUCTURE.md)** - Repository organization guide

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 📁 Directory Guide

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| **src/** | Production code | `preprocessing.py`, `filters.py` |
| **notebooks/** | Analysis & exploration | `02-dbs-artifact-removal-comparison.ipynb`, `experiments/` |
| **scripts/** | CLI tools & utilities | `remove_dbs_artifacts_cli.py`, `analysis/` |
| **data/raw/** | Raw EDF recordings | `XU/`, `LI/` (not in git) |
| **data/processed/** | Cleaned EEG data | (not in git) |
| **results/figures/** | Generated plots | `filter_proof/`, `rigorous_analysis/` (not in git) |
| **results/metrics/** | Analysis data | `*.csv` files (not in git) |
| **docs/** | Documentation | Guides, API reference, tutorials |
| **tests/** | Unit tests | `test_filters.py`, `test_preprocessing.py` |

👉 **See [PROJECT-STRUCTURE.md](PROJECT-STRUCTURE.md) for complete directory breakdown**

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 🧪 Testing

Run tests to validate the codebase:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 📊 Example Results

After running analysis, you'll get:

- **Figures**: PSD comparisons, SNR improvements, harmonic attenuation, band preservation
- **Metrics**: CSV tables with quantitative results
- **Reports**: HTML/PDF summaries

See [results/README.md](results/README.md) for details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 🛠️ Common Tasks

### Task: Remove DBS artifacts from my own EDF
```python
from src.preprocessing import EEGPreprocessor
import mne

# Load your data
raw = mne.io.read_raw_edf("my_file.edf")

# Remove artifacts
clean = EEGPreprocessor().remove_artifacts_spectrum_fit(raw)

# Save cleaned data
clean.save("my_file_clean.fif", overwrite=True)
```

### Task: Compare all methods on my data
See: [notebooks/02-dbs-artifact-removal-comparison.ipynb](notebooks/02-dbs-artifact-removal-comparison.ipynb)

### Task: Batch process multiple files
```bash
python scripts/remove_dbs_artifacts_cli.py --batch-mode \
  --input-dir data/raw/XU/ \
  --output-dir results/cleaned/ \
  --method spectrum_fit
```

### Task: Add a new analysis notebook
1. Create: `notebooks/XX-my-analysis.ipynb`
2. See: [notebooks/README.md](notebooks/README.md) for guidance

<p align="right">(<a href="#readme-top">back to top</a>)</p>
