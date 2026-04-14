# Source Code

Core production code for the ML-Driven EEG Biomarkers project.

## Directory Structure

```
src/
├── __init__.py                  # Package initialization
├── preprocessing.py             # EEG preprocessing pipeline
├── filters.py                   # Artifact removal filter implementations
└── README.md (this file)
```

## Core Modules

### 0. `dbs_preprocessing.py`

Object-oriented DBS preprocessing pipeline for real EDF recordings.

**Key Classes**:
- `DBSPreprocessingPipeline` - prepares channels, applies consensus filtering, compares DBS methods, runs ICA, and exports EDF.
- `RealtimeComparisonDashboard` - live Matplotlib Raw-vs-Processed sliding window.

**Entry Point**:
```python
from src.dbs_preprocessing import DBSPreprocessingPipeline

pipeline = DBSPreprocessingPipeline(
    input_file="data/raw/XU/XUAWAKE7_deidentified.edf",
    baseline_file="data/raw/XU/XUAWAKEPRE_deidentified.edf",
    output_dir="results/processed",
    lowpass_freq=45.0,
)
summary = pipeline.run(show_dashboard=False)
```

Default low-pass in this module is 45 Hz (within the common 30-70 Hz clinical EEG range).

### 1. `preprocessing.py`

Main preprocessing pipeline and convenience functions.

**Key Class**: `EEGPreprocessor`

```python
from src.preprocessing import EEGPreprocessor

preprocessor = EEGPreprocessor()

# Load and preprocess
raw = preprocessor.load_and_preprocess("data/raw/XUAWAKE7.edf")

# Remove artifacts (multiple methods available)
clean = preprocessor.remove_artifacts_spectrum_fit(
    raw, f_target=7.0, attenuation_db=-60.0
)
```

**Main Methods**:
- `load_and_preprocess()` - Load EDF and apply baseline preprocessing
- `remove_artifacts_spectrum_fit()` - Spectrum-fit multi-harmonic removal
- `remove_artifacts_zapline()` - Zapline+ method
- `remove_artifacts_freq_hampel()` - Frequency-domain Hampel
- `remove_artifacts_time_hampel()` - Time-domain Hampel
- `remove_artifacts_advanced()` - Generic method interface
- `extract_spectral_features()` - Extract PSD and band power
- `apply_ica()` - Apply ICA for artifact rejection

### 2. `filters.py`

Low-level filter implementations and artifact removal methods.

**Key Classes**:
- `ArtifactFilterFactory` - Router for different removal methods
- `BaselineReferencedFilter` - Wiener filtering with baseline reference
- `TimeHampelFilter` - Time-domain Hampel implementation
- `FrequencyHampelFilter` - Frequency-domain Hampel implementation
- `SpectralInterpolationFilter` - Spectrum-fit harmonic removal
- `ZaplineFilter` - Zapline+ spectro-spatial removal

**Static Methods**:
```python
from src.filters import ArtifactFilterFactory

# Route to appropriate method
clean = ArtifactFilterFactory.process(
    method='spectrum_fit',
    data=raw_data,
    sfreq=256,
    f_target=7.0,
    attenuation_db=-60.0
)
```

## Methods Reference

### Available Removal Methods

| Method | Class | Best For | Needs Baseline |
|--------|-------|----------|----------------|
| **spectrum_fit** | `SpectralInterpolationFilter` | Strong harmonic removal | No |
| **zapline** | `ZaplineFilter` | Adaptive spectro-spatial removal | No |
| **hampel_freq** | `FrequencyHampelFilter` | Frequency-domain outlier removal | No |
| **hampel_time** | `TimeHampelFilter` | Time-domain outlier removal | No |
| **wiener** | `BaselineReferencedFilter` | Best preservation + removal | Yes ✓ |

## Usage Examples

### Basic Usage
```python
from src.preprocessing import EEGPreprocessor
import mne

# Initialize
preprocessor = EEGPreprocessor()

# Load raw data
raw = mne.io.read_raw_edf("data/raw/XUAWAKE7.edf")

# Remove DBS artifacts
clean = preprocessor.remove_artifacts_spectrum_fit(
    raw,
    f_target=7.0,
    bandwidth=2.0,
    attenuation_db=-60.0
)

# Extract features
psd, freqs = preprocessor.extract_spectral_features(clean)
```

### Advanced: Multiple Passes
```python
# Primary removal
clean = preprocessor.remove_artifacts_spectrum_fit(raw, attenuation_db=-50.0)

# Refined peak removal
clean = preprocessor.remove_artifacts_freq_hampel(clean, attenuation_db=-40.0)

# Optional: baseline refinement
baseline = mne.io.read_raw_edf("data/raw/XUAWAKEPRE.edf")
from src.filters import BaselineReferencedFilter
filt = BaselineReferencedFilter(baseline, dbs_freq=7.0)
clean = filt.filter(clean)
```

### ICA for Muscle/Eye Artifacts
```python
# Apply ICA
clean_ica = preprocessor.apply_ica(raw, n_components=25)
# Manually inspect and remove components
# Then remove ICA on DBS recording
clean_dbs = preprocessor.apply_ica(raw_dbs, ica_model=clean_ica)
```

## Architecture

### Pipeline Flow
```
Raw EDF
    ↓
Load [read_raw_edf]
    ↓
Standardize Channels [auto-typing]
    ↓
Apply Montage [10-20 standard]
    ↓
Bandpass Filter [0.5-100 Hz FIR]
    ↓
Notch Filter [60 Hz]
    ↓
ICA Preprocessing [optional]
    ↓
DBS Removal [choose method]
    ↓
Feature Extraction [PSD, band power]
    ↓
Analysis/ML
```

## Configuration

### Default Parameters
```python
# Preprocessing defaults
SFREQ = 256  # Resample to this
HPF = 0.5    # High-pass frequency
LPF = 100.0  # Low-pass frequency
NOTCH = 60.0 # Notch frequency (power line)

# Method-specific defaults
SPECTRUM_FIT_BANDWIDTH = 2.0
SPECTRUM_FIT_ATTENUATION_DB = -60.0
HAMPEL_N_SIGMAS = 4.0
ZAPLINE_THRESHOLD_PERCENTILE = 95.0
```

## Integration Points

### MNE-Python
All methods work with MNE Raw objects:
```python
raw = mne.io.read_raw_edf(...)  # MNE Raw
clean = preprocessor.remove_artifacts_spectrum_fit(raw)  # Still MNE Raw
```

### NumPy Arrays
Lower-level filters work with raw arrays:
```python
from src.filters import ArtifactFilterFactory
clean_data = ArtifactFilterFactory.process(
    method='spectrum_fit',
    data=raw_array,  # NumPy array
    sfreq=256
)
```

## Dependencies

```
mne
numpy
scipy
pandas
matplotlib
scikit-learn (for ICA)
```

See `../requirements.txt` or `../environment.yml`

## Testing

Run unit tests:
```bash
python -m pytest tests/
```

Validate syntax:
```bash
python -m py_compile src/preprocessing.py src/filters.py
```

## File Organization

- **preprocessing.py** - User-facing API (use this first)
- **filters.py** - Implementation details (lower-level)
- Both files are well-documented with docstrings

## Quick Start

```python
from src.preprocessing import EEGPreprocessor

# One liner for quick artifact removal
clean = EEGPreprocessor().remove_artifacts_spectrum_fit(
    mne.io.read_raw_edf("data/raw/XUAWAKE7.edf")
)
```

## Need Help?

- See: `../docs/API-REFERENCE.md` - Complete API documentation
- See: `../docs/guides/QUICK-START-DBS-REMOVAL.md` - Quick start examples
- See: `../docs/guides/DBS-ARTIFACT-REMOVAL-METHODS.md` - Deep dive on methods
- Check docstrings: `help(EEGPreprocessor.remove_artifacts_spectrum_fit)`
