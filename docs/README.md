# Documentation

Comprehensive guides, API references, and usage documentation for the ML-Driven EEG Biomarkers project.

## Directory Structure

```
docs/
├── README.md (this file)
├── API-REFERENCE.md                  # Complete API documentation
├── PIPELINE-VALIDATION-REPORT.md    # ← Validation results, conclusions & lit. comparison
├── guides/                            # How-to guides and tutorials
│   ├── CLI-GUIDE.md                  # Command-line interface guide
│   ├── CLI-QUICK-REFERENCE.md        # Quick CLI reference
│   ├── DBS-ARTIFACT-REMOVAL-METHODS.md  # Detailed method explanations
│   ├── QUICK-START-DBS-REMOVAL.md       # Quick start examples
│   └── VISUAL-METHOD-COMPARISON.md      # Visual comparisons
├── images/                            # Documentation images and diagrams
└── presentation/                      # Presentation materials
```

## Quick Navigation

| Need | Document |
|------|----------|
| **Pipeline results & why they're good** | [**PIPELINE-VALIDATION-REPORT.md**](PIPELINE-VALIDATION-REPORT.md) |
| Get started quickly | [QUICK-START-DBS-REMOVAL.md](guides/QUICK-START-DBS-REMOVAL.md) |
| Detailed method info | [DBS-ARTIFACT-REMOVAL-METHODS.md](guides/DBS-ARTIFACT-REMOVAL-METHODS.md) |
| API details | [API-REFERENCE.md](API-REFERENCE.md) |
| CLI commands | [CLI-GUIDE.md](guides/CLI-GUIDE.md) OR [CLI-QUICK-REFERENCE.md](guides/CLI-QUICK-REFERENCE.md) |
| Visual comparison | [VISUAL-METHOD-COMPARISON.md](guides/VISUAL-METHOD-COMPARISON.md) |

## Main Guides

### 1. Quick Start - DBS Artifact Removal
**File**: `guides/QUICK-START-DBS-REMOVAL.md`

Best for:
- First-time users
- Need quick examples
- Choose a method fast
- Copy-paste ready code

### 2. DBS Artifact Removal Methods (Detailed)
**File**: `guides/DBS-ARTIFACT-REMOVAL-METHODS.md`

Contains:
- Theory behind each method
- Complete parameter explanations
- Tuning recommendations
- Comparison tables
- Troubleshooting guide
- Best practices

### 3. CLI Guide
**File**: `guides/CLI-GUIDE.md`

For:
- Batch processing
- Command-line workflows
- Integration with other tools
- Scripting examples

### 4. Visual Method Comparison
**File**: `guides/VISUAL-METHOD-COMPARISON.md`

Shows:
- Side-by-side method comparisons
- Visual differences
- Use case recommendations
- Interactive comparisons

## API Reference

**File**: `API-REFERENCE.md`

Complete documentation:
- Function signatures
- Parameter descriptions
- Return types
- Usage examples
- Error handling

### Key Classes

```python
from src.filters import ArtifactFilterFactory, BaselineReferencedFilter
from src.preprocessing import EEGPreprocessor
```

## For Different Use Cases

### I want to remove DBS artifacts...

**One-liner:**
```python
from src.preprocessing import EEGPreprocessor
clean = EEGPreprocessor().remove_artifacts_spectrum_fit(raw)
```
→ See: [QUICK-START-DBS-REMOVAL.md](guides/QUICK-START-DBS-REMOVAL.md)

### I want to understand the methods...

→ See: [DBS-ARTIFACT-REMOVAL-METHODS.md](guides/DBS-ARTIFACT-REMOVAL-METHODS.md)

### I want to use command-line tools...

→ See: [CLI-GUIDE.md](guides/CLI-GUIDE.md)

### I want detailed API documentation...

→ See: [API-REFERENCE.md](API-REFERENCE.md)

### I want to compare methods visually...

→ See: [VISUAL-METHOD-COMPARISON.md](guides/VISUAL-METHOD-COMPARISON.md)

## Code Examples Quick Links

### Python API Examples

**Time-Domain Hampel (Allen et al.)**
```python
clean = preprocessor.remove_artifacts_time_hampel(
    raw, window_sec=0.02, n_sigmas=4.0, attenuation_factor=2.0
)
```

**Frequency-Domain Hampel (Allen et al.)**
```python
clean = preprocessor.remove_artifacts_freq_hampel(
    raw, window_hz=3.0, n_sigmas=5.0, attenuation_db=-60.0
)
```

**Spectrum-Fit Multi-Harmonic**
```python
clean = preprocessor.remove_artifacts_spectrum_fit(
    raw, f_target=7.0, bandwidth=2.0, attenuation_db=-60.0
)
```

**Zapline+ (Chen et al., 2022)**
```python
clean = preprocessor.remove_artifacts_zapline(
    raw, f_target=7.0, n_harmonics=5, threshold_percentile=95.0
)
```

### Command-Line Examples

```bash
# Simple removal
python remove_dbs_artifacts_cli.py --input file.edf --output clean.edf

# Specify method
python remove_dbs_artifacts_cli.py \
  --input file.edf \
  --method spectrum_fit \
  --frequency 7.0 \
  --attenuation -80.0
```

## File Descriptions

- **API-REFERENCE.md** - Function/class documentation, signatures, examples
- **guides/CLI-GUIDE.md** - Command-line tool usage and examples
- **guides/CLI-QUICK-REFERENCE.md** - Quick CLI command reference
- **guides/DBS-ARTIFACT-REMOVAL-METHODS.md** - Method theory, parameters, tuning
- **guides/QUICK-START-DBS-REMOVAL.md** - Get started quickly with examples
- **guides/VISUAL-METHOD-COMPARISON.md** - Visual side-by-side method comparison
- **images/** - Diagrams and documentation illustrations
- **presentation/** - Presentation/poster materials

## Contributing to Documentation

When adding new features:
1. Update API-REFERENCE.md with new functions/classes
2. Add examples to appropriate guide
3. Include visual comparisons if applicable
4. Test all code examples

## Project Documentation Structure

```
docs/
├── User Guides (guides/ folder)
│   └── Task-based how-tos
├── API Reference
│   └── Function/class details
├── Images & Diagrams
│   └── Visual documentation
└── Presentables
    └── Slides, posters, figures
```

## Related Documentation

- **Main README**: `../README.md` - Project overview
- **Getting Started**: See QUICK-START-DBS-REMOVAL.md
- **Notebook Guides**: `../notebooks/README.md`
- **Script Guides**: `../scripts/README.md`
- **Data Guide**: `../data/README.md`
- **Results Guide**: `../results/README.md`
