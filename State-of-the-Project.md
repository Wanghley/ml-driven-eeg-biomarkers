# State of the Project: Automated EEG Machine Learning Pipeline
**Thesis Status Report** | Senior Engineering Thesis | **April 2026**

## 1. Executive Summary

The project is currently in the late-prototyping phase, featuring a robust, modular MNE-Python preprocessing pipeline. We have successfully implemented automated EDF loading, standardized 10-20 montages, and advanced frequency-domain artifact removal specifically targeting non-stationary DBS (Deep Brain Stimulation) noise. The architecture currently supports high-fidelity biomarker extraction (PSD and Topographic Band Power) and is positioned for transition into an end-to-end Machine Learning classification framework for spike/seizure detection.

---

## 2. Completed Milestones (Logical Modules)

### A. Data Ingestion & Metadata Standardization
*   **EDF/EDF+ Compatibility**: Automated parsing of clinical EEG recordings with `mne.io.read_raw_edf`.
*   **Channel Mapping**: Implementation of heuristic typing for non-EEG sensors (e.g., LOC/ROC to `eog`, EKGL/R to `ecg`, and EMG filters).
*   **Montage Application**: Dynamic application of standard 10-20 montages (`standard_1020`) with alignment verification via topographic previews.

### B. Artifact Removal & Signal Conditioning
*   **Classical Filtering**: Phase-neutral FIR bandpass (0.5–100 Hz) and notch filters (60 Hz) for baseline cleaning.
*   **DBS Artifact Mitigation**: 
    *   **Harmonic Notch**: Automated removal of primary 7Hz (and harmonics) DBS interference.
    *   **Zapline Integration**: Implementation of spectral line removal for persistent power-line and DBS harmonics.
    *   **Hampel Filtering**: Both Time-Domain and Frequency-Domain Hampel filters for outlier detection and non-stationary noise suppression.
*   **ICA (Independent Component Analysis)**: Semi-automated ICA using EOG channel correlations to identify and reject ocular/muscular artifacts.

### C. Feature Engineering & Visualization
*   **Spectral Analysis**: PSD (Power Spectral Density) computation via Welch’s method with optimized windowing for low-frequency biomarkers.
*   **Band Power Quantification**: Integration of power across clinical bands (Delta, Theta, Alpha, Beta, Gamma) using Simpson's rule.
*   **Topographic Mapping**: Generation of high-resolution 2D Topomaps to visualize spatial distribution of frequency-band biomarkers.

---

## 3. Gap Analysis

Despite the robustness of the preprocessing layer, the following components remain as critical gaps for the clinical-grade pipeline:

| Component | Status | Missing Requirements |
| :--- | :--- | :--- |
| **DBS Filtering** | Partial | Adaptive filtering methods to handle DBS frequency drift during long-term recordings. |
| **ICA Rejection** | Basic | Transition from manual/EOG inspection to **mne-icalabel** for automated, high-confidence artifact classification. |
| **Orchestration** | Fragmented | Consolidation of Jupyter-based experiments into a unified `EEG_Pipeline` Python class for batch processing. |
| **ML Classification** | Planning | Integration of **EEGNet** (PyTorch) or **RandomForest** (Scikit-learn) for biomarker-based seizure prediction. |
| **Evaluations** | Missing | Standardized cross-validation loops, confusion matrices, and clinical sensitivity/specificity metrics. |

---

## 4. Prioritized Action Plan (Next Steps)

### Phase 1: Pipeline Orchestration (Immediate)
*   **Refactor to Class-Based Architecture**: Transition `pre-processing.py` into a stateful `EEG_Processor` class that can manage patient sessions from raw EDF to processed feature vectors.
*   **Automate ICA**: Integrate the `ICALabel` extension to automatically categorize and drop noise components.

### Phase 2: Feature & Model Integration
*   **Feature Expansion**: Implement Hjorth parameter extraction (Activity, Mobility, Complexity) to provide temporal features alongside spectral power.
*   **Deep Learning Baseline**: Implement a standard EEGNet architecture in PyTorch, utilizing a `DataLoader` that consumes cleaned MNE data segments (Epochs).

### Phase 3: Evaluation & Reporting
*   **Central Metric Hub**: Build a standardized evaluation harness to track ROC/AUC, F1-scores, and seizure latency.
*   **Clinical Report Generator**: Create an automated PDF summary output for each patient session, including cleaned PSDs and topomaps for advisor review.

---

> [!NOTE]
> *Technological Context*: The current environment is built on **MNE 1.10.2**, **NumPy 2.2**, and **Scikit-learn**. Transition to **PyTorch** for the classification layer is the primary architectural goal for the upcoming sprint.
