# CLI Quick Reference Card

## 🚀 Getting Started (FASTEST WAY)

```bash
# 1. Activate environment (one-time)
conda activate eeg_thesis_v1

# 2. Run interactive (asks everything)
python remove_dbs_artifacts_cli.py

# 3. Or use wrapper (even simpler)
./dbs_remove.sh -i
```

---

## 📋 Method Selection

### Use THIS for Each Scenario:

| Goal | Command | Parameters |
|------|---------|-----------|
| **Fastest Setup** | `./dbs_remove.sh -i` | Interactive |
| **Default/Quick** | `./dbs_remove.sh -f FILE` | Spectrum-Fit auto |
| **Best Quality** | `./dbs_remove.sh -m wiener -f FILE -b BASELINE` | Wiener filter |
| **Maximum Removal** | `./dbs_remove.sh -m spectrum_fit -f FILE --attenuation-db -80.0` | Spectrum-Fit strong |
| **Multi-channel** | `./dbs_remove.sh --batch -m zapline` | Zapline+ all files |
| **Very Fast** | `./dbs_remove.sh -m hampel_time -f FILE` | Hampel Time |

---

## 🎯 One-Liners

```bash
# Interactive (recommended first time)
python remove_dbs_artifacts_cli.py

# Default - Spectrum-Fit on one file
python remove_dbs_artifacts_cli.py --input-file data/XU/XUAWAKE7_deidentified.edf

# Strong attenuation
python remove_dbs_artifacts_cli.py --input-file data/XU/XUAWAKE7_deidentified.edf --attenuation-db -80.0

# With baseline (best)
python remove_dbs_artifacts_cli.py --input-file data/XU/XUAWAKE7_deidentified.edf --method wiener --baseline-file data/XU/XUAWAKEPRE_deidentified.edf

# Batch process
python remove_dbs_artifacts_cli.py --batch --method spectrum_fit

# Use wrapper (shorter)
./dbs_remove.sh -i
./dbs_remove.sh -f data/XU/XUAWAKE7_deidentified.edf
./dbs_remove.sh --batch
```

---

## 🔧 Tuning Parameters

### For STRONGER Removal:
```bash
# Spectrum-Fit
--attenuation-db -80.0     # was -60.0
--bandwidth 1.5            # was 2.0

# Wiener
--alpha 2.0                # was 1.5

# Hampel Freq
--n-sigmas 2.0             # was 3.0

# Hampel Time
--attenuation-factor 2.0   # was 1.0
```

### For LIGHTER Removal (Brain Preservation):
```bash
# Spectrum-Fit
--attenuation-db -40.0     # was -60.0
--bandwidth 3.0            # was 2.0

# Wiener
--alpha 1.0                # was 1.5

# Hampel
--n-sigmas 4.0             # was 3.0
```

---

## ⚡ Quick Methods Reference

| Method | Speed | Strength | Use Case |
|--------|-------|----------|----------|
| `spectrum_fit` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **DEFAULT - Fast & Strong** |
| `hampel_time` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Real-time, pulses |
| `hampel_freq` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Peak detection |
| `wiener` | ⭐⭐ | ⭐⭐⭐ | Best brain preservation |
| `zapline` | ⭐ | ⭐⭐⭐⭐ | Multi-channel |

Default attenuation strengths:
- `spectrum_fit`: -60 dB (tunable -40 to -100)
- `wiener`: -20 to -40 dB
- `hampel_freq`: -60 dB (tunable)
- `hampel_time`: -20 to -40 dB
- `zapline`: -20 to -50 dB

---

## 📁 File Locations

```
data/XU/                              # Input EDF files
├── XUAWAKE7_deidentified.edf         # With DBS
├── XUAWAKEPRE_deidentified.edf       # Baseline (no DBS)
├── XUAWAKE60_deidentified.edf
├── XUSLEEP_deidentified.edf
└── ...

data/processed/                        # Output (default)
├── XUAWAKE7_deidentified-cleaned-raw.fif
├── XUAWAKE60_deidentified-cleaned-raw.fif
└── ...
```

---

## ❌ Troubleshooting

### "ModuleNotFoundError: mne"
```bash
conda activate eeg_thesis_v1
# or: pip install mne scipy
```

### "No EDF files found"
```bash
# Check files exist
ls data/XU/
# Use correct path if different
python remove_dbs_artifacts_cli.py --input-dir /path/to/files/
```

### "Baseline and DBS must match"
```bash
# Use same state baseline with DBS
# ✓ XUAWAKEPRE + XUAWAKE7
# ✓ XUSLEEP + XUSLEEP7  
# ✗ XUAWAKEPRE + XUSLEEP
```

---

## 💡 Tips

### Auto-detect DBS frequency from filename
```bash
# XUAWAKE7 → 7 Hz auto-detected (no --f-target needed)
python remove_dbs_artifacts_cli.py --input-file data/XU/XUAWAKE7_deidentified.edf
```

### Save to custom output folder
```bash
python remove_dbs_artifacts_cli.py \
  --input-file data/XU/XUAWAKE7_deidentified.edf \
  --output-dir results/my_experiment/
```

### Process multiple files
```bash
# Method 1: Batch
python remove_dbs_artifacts_cli.py --batch

# Method 2: Loop
for file in data/XU/XUAWAKE*.edf; do
  python remove_dbs_artifacts_cli.py --input-file "$file"
done
```

### Compare methods in parallel
```bash
# Process with all methods in background
python remove_dbs_artifacts_cli.py --batch -m spectrum_fit &
python remove_dbs_artifacts_cli.py --batch -m wiener &
wait
```

---

## 📊 Decision Tree

```
Ready to run?
├─ NO → python remove_dbs_artifacts_cli.py (interactive)
└─ YES
   ├─ Have baseline? 
   │  ├─ YES → ./dbs_remove.sh -m wiener -f FILE -b BASELINE
   │  └─ NO → ./dbs_remove.sh -f FILE (uses spectrum_fit)
   │
   ├─ Need maximum removal?
   │  └─ YES → --attenuation-db -80.0
   │
   ├─ Processing MANY files?
   │  └─ YES → ./dbs_remove.sh --batch
   │
   └─ Want fastest speed?
      └─ YES → -m hampel_time
```

---

## 🎓 Learn More

- Full help: `python remove_dbs_artifacts_cli.py --help`
- Guide: `docs/CLI-GUIDE.md`
- Methods: `docs/DBS-ARTIFACT-REMOVAL-METHODS.md`
- API: `docs/API-REFERENCE.md`
- Notebook: `notebooks/DBS-Artifact-Removal-Multi-Method-Comparison.ipynb`

---

## ✅ Checklist

Before first run:
- [ ] Activate conda environment: `conda activate eeg_thesis_v1`
- [ ] Check files exist: `ls data/XU/`
- [ ] Verify Python: `python --version`

First run:
- [ ] Try interactive: `python remove_dbs_artifacts_cli.py`
- [ ] Or simpler: `./dbs_remove.sh -i`
- [ ] Select method (1-5)
- [ ] Set parameters
- [ ] Choose file(s)
- [ ] Wait...
- [ ] Check output in `data/processed/`

Next run:
- [ ] Use direct command with your preferred settings
- [ ] Batch process if many files
- [ ] Compare PSDs before/after

---

**TL;DR**: `python remove_dbs_artifacts_cli.py` then follow prompts ✨
