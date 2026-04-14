#!/bin/bash
# Quick CLI examples for DBS artifact removal

# Make the script executable
chmod +x remove_dbs_artifacts_cli.py

echo "Usage Examples:"
echo "==============="
echo ""

# Example 1: Interactive mode (recommended)
echo "1️⃣  INTERACTIVE MODE (Recommended for first-time):"
echo "   python remove_dbs_artifacts_cli.py"
echo ""

# Example 2: Direct - Spectrum-Fit (fastest, no baseline needed)
echo "2️⃣  SPECTRUM-FIT (Fast, strong, no baseline):"
echo "   python remove_dbs_artifacts_cli.py \\"
echo "     --input-file data/XU/XUAWAKE7_deidentified.edf \\"
echo "     --method spectrum_fit \\"
echo "     --f-target 7.0 \\"
echo "     --bandwidth 2.0 \\"
echo "     --attenuation-db -60.0"
echo ""

# Example 3: Maximum attenuation
echo "3️⃣  SPECTRUM-FIT (Maximum Attenuation):"
echo "   python remove_dbs_artifacts_cli.py \\"
echo "     --input-file data/XU/XUAWAKE7_deidentified.edf \\"
echo "     --method spectrum_fit \\"
echo "     --f-target 7.0 \\"
echo "     --bandwidth 1.5 \\"
echo "     --attenuation-db -80.0"
echo ""

# Example 4: Wiener with baseline (best preservation)
echo "4️⃣  WIENER FILTER (Best brain preservation, needs baseline):"
echo "   python remove_dbs_artifacts_cli.py \\"
echo "     --input-file data/XU/XUAWAKE7_deidentified.edf \\"
echo "     --method wiener \\"
echo "     --baseline-file data/XU/XUAWAKEPRE_deidentified.edf \\"
echo "     --alpha 1.5"
echo ""

# Example 5: Zapline+
echo "5️⃣  ZAPLINE+ (Chen et al., 2022):"
echo "   python remove_dbs_artifacts_cli.py \\"
echo "     --input-file data/XU/XUAWAKE7_deidentified.edf \\"
echo "     --method zapline \\"
echo "     --f-target 7.0 \\"
echo "     --n-harmonics 10 \\"
echo "     --threshold-percentile 95.0"
echo ""

# Example 6: Hampel Freq
echo "6️⃣  HAMPEL FREQ (Spectral peak detection):"
echo "   python remove_dbs_artifacts_cli.py \\"
echo "     --input-file data/XU/XUAWAKE7_deidentified.edf \\"
echo "     --method hampel_freq \\"
echo "     --window-hz 2.0 \\"
echo "     --n-sigmas 3.0 \\"
echo "     --attenuation-db -60.0"
echo ""

# Example 7: Hampel Time
echo "7️⃣  HAMPEL TIME (Pulse removal, fastest):"
echo "   python remove_dbs_artifacts_cli.py \\"
echo "     --input-file data/XU/XUAWAKE7_deidentified.edf \\"
echo "     --method hampel_time \\"
echo "     --window-sec 0.2 \\"
echo "     --n-sigmas 3.0 \\"
echo "     --attenuation-factor 1.5"
echo ""

# Example 8: Batch processing all files
echo "8️⃣  BATCH MODE (Process all files in data/XU/):"
echo "   python remove_dbs_artifacts_cli.py \\"
echo "     --batch \\"
echo "     --method spectrum_fit \\"
echo "     --attenuation-db -60.0"
echo ""

# Example 9: Different output directory
echo "9️⃣  CUSTOM OUTPUT DIRECTORY:"
echo "   python remove_dbs_artifacts_cli.py \\"
echo "     --input-file data/XU/XUAWAKE7_deidentified.edf \\"
echo "     --output-dir results/cleaned_eeg/ \\"
echo "     --method spectrum_fit"
echo ""

echo "═══════════════════════════════════════════════════════════"
echo ""
echo "METHOD QUICK REFERENCE:"
echo "├─ spectrum_fit      → Fast, strong, no baseline needed (RECOMMENDED)"
echo "├─ wiener            → Best brain preservation (needs baseline)"
echo "├─ zapline           → Multi-channel spatial filtering"
echo "├─ hampel_freq       → Adaptive spectral peak detection"
echo "└─ hampel_time       → Transient pulse removal (fastest)"
echo ""
echo "DEFAULT PARAMETERS:"
echo "├─ bandwidth: 2.0 Hz"
echo "├─ attenuation-db: -60.0 dB"
echo "├─ window-hz: 2.0 Hz"
echo "├─ window-sec: 0.2 seconds"
echo "├─ n-sigmas: 3.0"
echo "├─ n-harmonics: 10"
echo "├─ threshold-percentile: 95.0"
echo "├─ attenuation-factor: 1.0"
echo "└─ alpha: 1.5"
echo ""
echo "═══════════════════════════════════════════════════════════"
