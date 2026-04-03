# DBS Artifact Removal - Quick CLI Guide

## Setup (One-time)

### Option A: Create Conda Environment (Recommended)
```bash
# Create environment from file
cd /Users/student/Workspace/ml-driven-eeg-biomarkers
conda env create -f environment.yml -n eeg_thesis_v1

# Activate environment
conda activate eeg_thesis_v1
```

### Option B: Install Packages to Existing Environment
```bash
pip install mne scipy numpy pandas matplotlib
```

---

## Usage

Once environment is activated, use the CLI tool:

```bash
# Show help
python remove_dbs_artifacts_cli.py --help

# Interactive mode (recommended - easiest)
python remove_dbs_artifacts_cli.py

# Or specify arguments directly
python remove_dbs_artifacts_cli.py --input-file data/XU/XUAWAKE7_deidentified.edf --method spectrum_fit
```

---

## Common Commands

### 1. **Interactive Selection (Recommended)**
Start here if you're unsure. The tool guides you through every step:
```bash
python remove_dbs_artifacts_cli.py
```

What happens:
- Selects processing method interactively (1-5 options shown)
- Enters method-specific parameters via prompts
- Selects input file(s) from GUI menu
- Runs processing
- Shows results

---

### 2. **Spectrum-Fit (Recommended Default)**
Fast, strong, doesn't need baseline:
```bash
python remove_dbs_artifacts_cli.py \
  --input-file data/XU/XUAWAKE7_deidentified.edf \
  --method spectrum_fit
```

**Parameters to customize:**
```bash
# Light removal
python remove_dbs_artifacts_cli.py \
  --input-file data/XU/XUAWAKE7_deidentified.edf \
  --method spectrum_fit \
  --bandwidth 3.0 \
  --attenuation-db -40.0

# Maximum removal
python remove_dbs_artifacts_cli.py \
  --input-file data/XU/XUAWAKE7_deidentified.edf \
  --method spectrum_fit \
  --bandwidth 1.5 \
  --attenuation-db -80.0
```

---

### 3. **Wiener Filter (Best Brain Preservation)**
Use if you have clean baseline (XUAWAKEPRE):
```bash
python remove_dbs_artifacts_cli.py \
  --input-file data/XU/XUAWAKE7_deidentified.edf \
  --method wiener \
  --baseline-file data/XU/XUAWAKEPRE_deidentified.edf
```

**Adjust strength:**
```bash
# Lighter removal
--alpha 1.0

# Stronger removal
--alpha 2.0
```

---

### 4. **Zapline+ (Chen et al., 2022)**
Good for multi-channel recordings:
```bash
python remove_dbs_artifacts_cli.py \
  --input-file data/XU/XUAWAKE7_deidentified.edf \
  --method zapline
```

---

### 5. **Hampel Freq (Spectral Peak Detection)**
Adaptive detection of sharp peaks:
```bash
python remove_dbs_artifacts_cli.py \
  --input-file data/XU/XUAWAKE7_deidentified.edf \
  --method hampel_freq \
  --window-hz 2.0 \
  --n-sigmas 3.0 \
  --attenuation-db -60.0
```

---

### 6. **Hampel Time (Fastest)**
Quick transient pulse removal:
```bash
python remove_dbs_artifacts_cli.py \
  --input-file data/XU/XUAWAKE7_deidentified.edf \
  --method hampel_time
```

---

### 7. **Batch Process All Files**
Process everything in `data/XU/`:
```bash
python remove_dbs_artifacts_cli.py \
  --batch \
  --method spectrum_fit
```

---

### 8. **Custom Output Directory**
Save to different location:
```bash
python remove_dbs_artifacts_cli.py \
  --input-file data/XU/XUAWAKE7_deidentified.edf \
  --output-dir results/cleaned_eeg/ \
  --method spectrum_fit
```

---

## Decision Tree: Which Method to Use?

```
Do you know which method you want?
├─ NO → Run: python remove_dbs_artifacts_cli.py
│       (Interactive mode - answers all questions)
│
└─ YES
   ├─ Want FASTEST & STRONGEST removal?
   │  └─ YES → spectrum_fit
   │         Command: --method spectrum_fit --attenuation-db -60.0
   │
   ├─ Have CLEAN BASELINE (XUAWAKEPRE)?
   │  └─ YES → wiener (best brain preservation)
   │         Command: --method wiener --baseline-file data/XU/XUAWAKEPRE*.edf
   │
   ├─ Working with MULTI-CHANNEL data?
   │  └─ YES → zapline
   │         Command: --method zapline
   │
   ├─ Have SHARP, well-defined HARMONICS?
   │  └─ YES → hampel_freq
   │         Command: --method hampel_freq --attenuation-db -60.0
   │
   └─ Need MAXIMUM SPEED?
      └─ YES → hampel_time (fastest)
              Command: --method hampel_time
```

---

## Parameter Guides

### Spectrum-Fit Parameters
| Parameter | Light | Balanced | Strong |
|-----------|-------|----------|--------|
| `bandwidth` | 3.0 Hz | 2.0 Hz | 1.5 Hz |
| `attenuation-db` | -40 | -60 | -80 |
| **Removal** | 99% | 99.9% | 99.99% |

### Wiener Parameters
| Parameter | Light | Balanced | Strong |
|-----------|-------|----------|--------|
| `alpha` | 1.0 | 1.5 | 2.0 |
| **Removal** | 90% | 95% | 98% |
| **Brain Preservation** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### Hampel Freq Parameters
| Parameter | Light | Balanced | Strong |
|-----------|-------|----------|--------|
| `n-sigmas` | 4.0 | 3.0 | 2.0 |
| `attenuation-db` | -40 | -60 | -80 |
| **Detection** | Selective | Balanced | Aggressive |

### Hampel Time Parameters
| Parameter | Light | Balanced | Strong |
|-----------|-------|----------|--------|
| `attenuation-factor` | 1.0 | 1.5 | 2.0+ |
| **Passes** | 1 | 1.5 | 2+ |

---

## Examples by Scenario

### Scenario 1: Clinical Data (Quick Screening)
```bash
# Fast screening with strong removal
python remove_dbs_artifacts_cli.py \
  --batch \
  --method spectrum_fit \
  --attenuation-db -60.0
```

### Scenario 2: Research (Preserve Brain Activity)
```bash
# Best brain preservation approach
python remove_dbs_artifacts_cli.py \
  --input-file data/XU/XUAWAKE7_deidentified.edf \
  --method wiener \
  --baseline-file data/XU/XUAWAKEPRE_deidentified.edf \
  --alpha 1.5
```

### Scenario 3: Maximum Artifact Removal
```bash
# Two-step approach for maximum removal
# Step 1: Strong Spectrum-Fit
python remove_dbs_artifacts_cli.py \
  --input-file data/XU/XUAWAKE7_deidentified.edf \
  --method spectrum_fit \
  --attenuation-db -80.0 \
  --output-dir temp/

# Step 2: Refinement with Hampel Freq
python remove_dbs_artifacts_cli.py \
  --input-file temp/XUAWAKE7_deidentified-cleaned-raw.fif \
  --method hampel_freq \
  --attenuation-db -40.0
```

### Scenario 4: Multi-Channel Analysis
```bash
python remove_dbs_artifacts_cli.py \
  --batch \
  --method zapline \
  --n-harmonics 10
```

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'mne'"
```bash
# Activate environment
conda activate eeg_thesis_v1

# Or install MNE
pip install mne scipy
```

### Error: "FileNotFoundError: data/XU/..."
```bash
# Check files exist
ls -la data/XU/

# Update path in command if needed
python remove_dbs_artifacts_cli.py \
  --input-dir data/XU/ \
  --input-file XUAWAKE7_deidentified.edf
```

### Error: "Baseline and DBS data must have same channels"
```bash
# Ensure both files are preprocessed the same way
# Use XUAWAKEPRE (pre-stim) with XUAWAKE7 (7 Hz stim)
python remove_dbs_artifacts_cli.py \
  --input-file data/XU/XUAWAKE7_deidentified.edf \
  --method wiener \
  --baseline-file data/XU/XUAWAKEPRE_deidentified.edf
```

---

## Output

All cleaned files are saved to `data/processed/` (or custom `--output-dir`):
```
data/processed/
├─ XUAWAKE7_deidentified-cleaned-raw.fif
├─ XUAWAKE60_deidentified-cleaned-raw.fif
├─ XUSLEEP_deidentified-cleaned-raw.fif
└─ ...
```

Files can be loaded in Python:
```python
import mne

raw = mne.io.read_raw_fif("data/processed/XUAWAKE7_deidentified-cleaned-raw.fif")
print(raw.info)
raw.plot(duration=5)
```

---

## Full CLI Help Reference

Run this to see all available options:
```bash
python remove_dbs_artifacts_cli.py --help
```

Or see: `remove_dbs_artifacts_cli.py --help 2>&1 | less`

---

## Tips & Tricks

### Auto-detect DBS Frequency
```bash  
# Frequency auto-detected from filename (XUAWAKE7 → 7 Hz)
python remove_dbs_artifacts_cli.py \
  --input-file data/XU/XUAWAKE7_deidentified.edf \
  --method spectrum_fit
  # (no need to specify --f-target 7.0)
```

### Run Multiple Methods for Comparison
```bash
# Process same file with all methods
for method in spectrum_fit wiener zapline hampel_freq hampel_time; do
  python remove_dbs_artifacts_cli.py \
    --input-file data/XU/XUAWAKE7_deidentified.edf \
    --method $method \
    --output-dir "results/${method}/"
done
```

### Dry Run (See what would happen)
```bash
# Just run help to verify parameters
python remove_dbs_artifacts_cli.py --help | grep -A 20 "method"
```

### Monitor Progress
```bash
# Process with verbose output
python remove_dbs_artifacts_cli.py \
  --batch \
  --method spectrum_fit 2>&1 | tee processing.log
```

---

## Next Steps

1. **Setup environment** (one-time):
   ```bash
   conda activate eeg_thesis_v1
   # or: pip install mne scipy
   ```

2. **Try interactive mode**:
   ```bash
   python remove_dbs_artifacts_cli.py
   ```

3. **Or use direct command** (substitute your file):
   ```bash
   python remove_dbs_artifacts_cli.py \
     --input-file data/XU/XUAWAKE7_deidentified.edf \
     --method spectrum_fit
   ```

4. **Compare results**:
   - Open the output files in Python or visualization tool
   - Check PSDs before/after
   - Pick your preferred method

5. **Use in pipeline**:
   - Add CLI command to your processing scripts
   - Match parameters to your needs
   - Run on all your data

---

For more details, see:
- `docs/QUICK-START-DBS-REMOVAL.md` - Code examples
- `docs/API-REFERENCE.md` - Complete API reference
- `notebooks/DBS-Artifact-Removal-Multi-Method-Comparison.ipynb` - Test all methods
