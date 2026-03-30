# CODEBASE_GUIDE.md

This file provides guidance for developers working with code in this repository.

## Project Overview

Tierpsy Tracker is a multi-animal tracking system for phenotypic analysis of *C. elegans* and other animals. It processes video files through a sequential pipeline of image analysis steps, outputting HDF5 files containing trajectories, skeletons, and behavioral features.

## Installation & Build

```bash
# Install from source (compiles Cython extensions)
pip install -e .

# Build Cython extensions only (without full install)
python setup.py build_ext --inplace
```

Cython extensions are in `tierpsy/analysis/ske_create/segWormPython/cython_files/` (`circCurvature.pyx`, `curvspace.pyx`) and must be compiled before use.

Platform-specific conda dependencies: `requirements-macos.txt` or `requirements-ubuntu.txt`.

## Running

```bash
tierpsy_gui        # Main Qt5 GUI launcher
tierpsy_process    # Batch CLI processing
tierpsy_gui_simple # Lightweight HDF5 video player
```

## Tests

```bash
tierpsy_tests --download_examples   # Download test data from Zenodo
tierpsy_tests                       # Run all tests
tierpsy_tests [test_name]           # Run a specific test
```

Tests are defined in `tierpsy/tests/run_tests.py` using a custom runner (not pytest). Test data is downloaded from Zenodo on first run.

## Architecture

### Analysis Pipeline

Processing happens as a series of named **checkpoints** executed in order. Each checkpoint module in `tierpsy/analysis/<step>/` defines an `args_()` function returning a dict with `func`, `argkws`, `input_files`, `output_files`, and `requirements`. The orchestrator in `tierpsy/processing/` uses this to build and execute the DAG.

Checkpoint sequence:
```
COMPRESS → TRAJ_CREATE → TRAJ_JOIN → SKE_INIT → BLOB_FEATS
         → SKE_CREATE → SKE_FILT → SKE_ORIENT → INT_PROFILE → INT_SKE_ORIENT
         → FEAT_INIT → FEAT_TIERPSY   (Tierpsy route)
         → FEAT_CREATE                 (OpenWorm route)
```

Optional checkpoints: `FOOD_CNT`, `STAGE_ALIGMENT`, `CONTOUR_ORIENT`, `WCON_EXPORT`, `NN_BGND`.

### Key Modules

- **`tierpsy/processing/`** — Orchestration: `ProcessLocal.py` runs a single file; `ProcessMultipleFilesFun.py` handles batch jobs with multiprocessing; `AnalysisPoints.py` maps checkpoint names to module `args_()` calls.
- **`tierpsy/analysis/`** — ~27 analysis step modules implementing each checkpoint.
- **`tierpsy/helper/params/tracker_param.py`** — `TrackerParams` class: parses and validates all processing parameters from JSON files. Default parameter sets live in `tierpsy/extras/param_files/`.
- **`tierpsy/gui/`** — PyQt5 widgets. `SelectApp.py` is the main launcher; each tool (batch processing, viewers, summarizer) has its own widget.
- **`tierpsy/summary/`** — Post-processing: `collect.py` aggregates features across videos; `process_tierpsy.py` / `process_ow.py` compute per-experiment statistics.
- **`tierpsy/features/`** — Feature extraction libraries: `tierpsy_features/` (internal) and `open_worm_analysis_toolbox/` (OpenWorm integration).

### Data Format

All intermediate and final results are stored as **HDF5** files (via PyTables/h5py):
- `*_masked_image.hdf5` — Compressed video frames + metadata
- `*_skeletons.hdf5` — Trajectories (`trajectories_data`, `plate_worms` tables), skeleton coordinates, blob features
- `*_featuresN.hdf5` — Tierpsy features
- `*_features.hdf5` — OpenWorm features

### Analysis Type Flags

Parameter sets follow naming conventions that determine which checkpoints run:
- `BASE*` — skeleton only, no features
- `TIERPSY*` — full tierpsy feature extraction
- `OPENWORM*` — OpenWorm Analysis Toolbox features
- `*WT2` — Worm Tracker 2.0 (single-worm, stage-tracked)
- `*SINGLE` — single-worm mode
- `*AEX` — includes food contour tracking and neural network background filtering

### Adding a New Analysis Step

1. Create `tierpsy/analysis/<step_name>/__init__.py` with an `args_(file_names, param)` function returning the standard dict (`func`, `argkws`, `input_files`, `output_files`, `requirements`).
2. Register the checkpoint name in `tierpsy/processing/AnalysisPoints.py`.
3. Add it to the relevant analysis type lists in the parameter/checkpoint logic.
