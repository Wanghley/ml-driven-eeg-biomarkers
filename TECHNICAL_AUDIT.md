# Technical Audit: ML-Driven EEG Biomarkers

## Executive Summary

The repository contains a credible research-grade toolkit for DBS artifact suppression, but it is not yet a coherent end-to-end neuroinformatics pipeline for:

- EDF ingestion
- DBS, EOG/blink, and EMG/muscle artifact removal
- spike morphology biomarker extraction

The strongest area is DBS removal. The weakest area is biomarker extraction: there is no production implementation for spike morphology features, and the artifact-removal workflows are fragmented across multiple entry points with inconsistent preprocessing assumptions.

Overall readiness: research prototype, not yet production-grade.

## Repository Mapping

### Core Processing

- [src/preprocessing.py](src/preprocessing.py) provides the main `EEGPreprocessor` class with EDF loading, channel standardization, clinical FIR filtering, notch filtering, and several DBS-removal wrappers.
- [src/filters.py](src/filters.py) contains the lower-level artifact-removal implementations, including:
  - baseline-referenced Wiener filtering
  - spectral interpolation
  - sinusoidal regression
  - complex Hampel variants
  - Zapline-like spectro-spatial filtering
  - comb notch filtering
- [src/dbs_preprocessing.py](src/dbs_preprocessing.py) is the most complete object-oriented pipeline. It includes standardized channel preparation, consensus filtering, method comparison, ICA-based cleanup, and EDF export.

### Orchestration and CLI

- [pipeline.py](pipeline.py) is a monolithic orchestration script that loads EDFs, runs a surgical DBS pipeline, computes PSD-based metrics, and generates plots.
- [scripts/remove_dbs_artifacts_cli.py](scripts/remove_dbs_artifacts_cli.py) is an interactive CLI for choosing among the artifact-removal methods.
- [scripts/analysis/pipeline_rigorous.py](scripts/analysis/pipeline_rigorous.py) is a comparative analysis script for evaluating methods across conditions.

### Dagster Pipeline

- [pipeline/__init__.py](pipeline/__init__.py) registers the Dagster definitions.
- [pipeline/assets/real_data.py](pipeline/assets/real_data.py) implements real-data assets for loading, filtering, ICA, and EDF export.
- [pipeline/assets/synthetic.py](pipeline/assets/synthetic.py) implements a synthetic benchmark pipeline.
- [pipeline/constants.py](pipeline/constants.py) centralizes method definitions and plotting styles.
- [pipeline/sensors.py](pipeline/sensors.py) watches for new EDF files and triggers a real-data job.

### Supporting Material

- [README.md](README.md) documents the project, but several claims are ahead of the actual implementation.
- [PROJECT-STRUCTURE.md](PROJECT-STRUCTURE.md) outlines the repo organization, but some entries are stale.
- [tests/README.md](tests/README.md) describes test files that do not exist.
- [docs/API-REFERENCE.md](docs/API-REFERENCE.md) is currently empty.

## Gap Analysis

### 1. Data Ingestion

What exists:

- EDF files are read with MNE in several modules.
- Channel renaming and montage assignment are implemented.

What is missing:

- A single canonical ingestion API.
- Consistent handling of non-EEG channels across the codebase.
- Explicit support for metadata normalization, provenance tracking, and batch validation.

Current risk:

- The same EDF can be processed differently depending on which script or asset is used.

### 2. DBS Artifact Removal

What exists:

- Strong method variety in [src/filters.py](src/filters.py).
- Baseline-referenced Wiener filtering when a clean baseline exists.
- Multiple spectral methods that are useful for narrowband DBS harmonics.

What is missing:

- One canonical default strategy.
- Strong quality gates that decide when a method is acceptable.
- Unified parameter management across CLI, scripts, and Dagster assets.

Current risk:

- The repository currently emphasizes artifact attenuation metrics more than morphology preservation.

### 3. EOG and EMG Removal

What exists:

- A more advanced ICA path in [src/dbs_preprocessing.py](src/dbs_preprocessing.py) can identify EOG and muscle components.

What is missing:

- The main preprocessing path used by [pipeline.py](pipeline.py) does not robustly preserve auxiliary channels long enough for reliable EOG/EMG classification.
- There is no unified ocular/muscle cleaning stage in the primary production flow.

Current risk:

- EOG and EMG contamination can survive the pipeline or be implicitly removed with too little traceability.

### 4. Spike Morphology Biomarkers

What exists:

- Band-power and harmonic-power integrity metrics.

What is missing:

- Spike detection.
- Spike morphology feature extraction.
- Channel-level and event-level biomarker summaries.
- Validation that artifact removal preserves spike shape, width, and asymmetry.

Current risk:

- The repository cannot yet support a biomarker pipeline centered on spike morphology because the feature layer does not exist.

## Code Quality and Technical Flaws

### High Priority

1. Pipeline fragmentation

There are multiple overlapping processing stacks: [pipeline.py](pipeline.py), [src/preprocessing.py](src/preprocessing.py), [src/dbs_preprocessing.py](src/dbs_preprocessing.py), and the Dagster asset graph. This creates maintenance overhead and makes results dependent on the entry point.

2. Missing EOG/EMG handling in the main path

The primary script path drops down to EEG-only processing too early and then applies ICA logic that is focused on DBS harmonics rather than ocular or muscle artifact rejection.

3. Double filtering

Some code paths apply broad filtering twice: once in the outer script and again inside the surgical pipeline. This can distort spike morphology and bandpower estimates.

### Medium Priority

4. Overly aggressive notch-style cleanup

Dense harmonic suppression can flatten physiological structure if used as a default instead of a targeted fallback.

5. Brittle DBS frequency inference

Some code infers the DBS frequency from the first number in the filename, which is fragile and easy to mis-target.

6. Performance bottlenecks

Several filters rely on repeated Python loops over channels, harmonics, and bins. The code is workable for research-sized files but will not scale efficiently without optimization.

7. Documentation drift

The README and supporting docs reference APIs and test files that are missing or incomplete.

## Actionable Recommendations

### Artifact Removal

1. Choose one canonical real-data pipeline and deprecate duplicate orchestration paths.
2. Preserve EOG and EMG channels until ICA or another supervised rejection stage has completed.
3. Use baseline-referenced Wiener filtering when a matched clean baseline exists.
4. Use spectral interpolation or sinusoidal regression as the default no-baseline DBS method.
5. Restrict comb-style notching to line noise cleanup or emergency fallback use.
6. Make DBS frequency an explicit configuration value, not a filename heuristic.

### Spike Morphology

1. Add a new spike-feature module that computes morphology descriptors such as amplitude, half-width, rise time, decay time, sharpness, asymmetry, and area.
2. Add spike-preservation tests using synthetic spike injections before and after artifact removal.
3. Report morphology distortion alongside attenuation and PSD metrics.

### Validation and Testing

1. Create real unit tests for each filter family.
2. Add integration tests for the canonical pipeline.
3. Add regression tests for phase distortion and morphology preservation.
4. Populate [docs/API-REFERENCE.md](docs/API-REFERENCE.md) from the actual codebase.

### Performance and Maintainability

1. Reduce save/reload loops in Dagster assets.
2. Cache repeated spectral computations where possible.
3. Consolidate parameter defaults into a single configuration layer.
4. Define one public API for ingestion, filtering, and export.

## Recommended Target Architecture

```text
EDF ingestion
  -> channel typing and montage normalization
  -> consensus clinical filtering
  -> DBS removal
  -> EOG/EMG ICA cleanup
  -> spike detection and morphology extraction
  -> biomarker QC and report generation
```

Suggested module boundaries:

- `src/ingestion.py` for EDF loading and metadata normalization
- `src/artifact_removal.py` for DBS, EOG, and EMG suppression
- `src/spike_features.py` for morphology features and event summaries
- `src/pipeline.py` or `src/dbs_preprocessing.py` for orchestration only

## Bottom Line

The repository is strongest in DBS artifact-removal experimentation, but the current codebase is not yet a complete neuroinformatics production pipeline. To reach the stated goal, the main work is consolidation, robust EOG/EMG handling, and a dedicated spike morphology feature layer.