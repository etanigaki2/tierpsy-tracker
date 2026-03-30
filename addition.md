# Single-Worm Trajectory Consolidation and Camera Adjustment Detection: A Software Addition to Tierpsy Tracker

## Overview

This document describes the software features added to the Tierpsy Tracker codebase as part of a dissertation project. Two new analysis checkpoints were introduced: `DETECT_CAM_ADJUST`, which identifies and neutralises frames corrupted by camera position adjustments, and `CONSOLIDATE_TRAJ`, which merges all worm trajectory fragments into a single unified identity. Together they form two new end-to-end analysis pipelines — `TIERPSY_CONSOLIDATE` and `OPENWORM_CONSOLIDATE`. The motivation, design decisions, technical implementation, and comparison with existing approaches are all covered below.

---

## 1. Background and Motivation

### 1.1 What Tierpsy Tracker Does

Tierpsy Tracker is an open-source multi-animal tracking platform developed at the MRC Laboratory of Medical Sciences at Imperial College London. It was designed primarily for high-throughput *C. elegans* phenotyping — processing video recordings of worm plates, extracting per-animal trajectories and skeletons frame-by-frame, and computing a large set of behavioural features (speed, body curvature, bending frequency, and many more) for downstream statistical analysis.

The core of the software is a sequential processing **pipeline** composed of named checkpoints. Each checkpoint is a discrete analysis step: video compression, blob detection, trajectory creation, skeleton extraction, and feature calculation. A video file enters the pipeline raw and exits as an HDF5 file containing cleaned skeletons and behavioural features ready for analysis.

### 1.2 The Trajectory Fragmentation Problem

In practice, even in a video that contains only a single worm, the tracking system frequently produces **multiple trajectory fragments** rather than one continuous track. This happens for several reasons:

- **Occlusion**: the worm passes behind or under a piece of debris, temporarily vanishing and reappearing as a "new" object.
- **Coiling or looping**: the worm's body doubles back on itself, causing its bounding box area to change sharply; the tracker's area-ratio constraint then breaks the link.
- **Poor illumination**: regions of uneven lighting cause the worm to fall below the detection threshold for a few frames, splitting a trajectory at the gap.
- **Spurious detections**: dust particles, bubbles, or condensation on the imaging surface are detected as blobs and assigned their own trajectory IDs, cluttering the output.
- **Camera position adjustments**: in many imaging rigs, the camera or stage is repositioned periodically during recording. This produces 1–2 frames of global image motion in which the worm appears disfigured or split into multiple blobs. After the adjustment the worm reappears correctly but the tracker assigns it a new trajectory ID, as if it were a completely different animal. In a 60-video dataset with approximately 40 adjustments per video, this alone generates roughly 2,400 trajectory breaks that would otherwise require manual correction.

```
Without consolidation — typical multi-fragment output:

Frame:    0   10   20   30   40   50   60   70   80   90  100
           |    |    |    |    |    |    |    |    |    |    |
Traj 1:  [===========================]
Traj 2:                        [===========]
Traj 3:                    [==]
Traj 4 (noise):   [=]
Traj 5:                                          [===========]

With consolidation — single unified output:

Frame:    0   10   20   30   40   50   60   70   80   90  100
           |    |    |    |    |    |    |    |    |    |    |
Worm 1:  [=====================================================]
```

When a researcher wants to analyse the full behavioural record of a single animal, having the trajectory split across multiple IDs is a significant problem. Downstream feature extraction is performed **per trajectory ID**, so a worm that was split into five fragments will produce five separate (and short) feature records instead of one complete behavioural profile. Many behavioural metrics — particularly those based on frequency or long-range movement — require long continuous trajectories to be meaningful.

### 1.3 Existing Approaches and Their Limitations

Tierpsy Tracker already provides two single-worm modes, both triggered by setting `analysis_type` in the parameter file:

- **`TIERPSY_SINGLE`**: Runs the standard pipeline but with `is_one_worm=True` during `TRAJ_JOIN`. Inside that step, a function called `correctSingleWormCase` selects one blob per frame based on area similarity to the median, subject to a strict inter-frame movement threshold (at most one quarter of the worm's body length). If the worm moves faster or is absent for even a short gap, the chain breaks and frames are discarded entirely.

- **`TIERPSY_WT2`**: Designed for the Worm Tracker 2.0 hardware, which has a motorised stage that keeps the worm centred. It uses the same `correctSingleWormCase` logic but selects the blob closest to the spatial centre of the field of view. Only applicable when that specific hardware is used.

Both modes operate at the level of **raw frame-to-frame blobs**, before any gap-joining has been attempted. This means they cannot recover trajectories that were already fragmented by the blob-linking step. If the worm disappears for thirty frames and reappears, both existing modes treat the second half as a separate, independent sequence and may discard it entirely.

The new `CONSOLIDATE_TRAJ` checkpoint is designed to address this gap. By running *after* the multi-worm `TRAJ_JOIN` step — which already bridges short gaps between trajectory fragments — it benefits from the richer, pre-linked trajectory structure and can recover detections that neither `SINGLE` nor `WT2` mode would keep.

---

## 2. Software Architecture

### 2.1 The Checkpoint System

Before describing the new feature in detail, it is useful to understand how Tierpsy Tracker's pipeline is structured. Each analysis step is implemented as a Python package under `tierpsy/analysis/<step_name>/`. A mandatory `__init__.py` file in each package exposes a single function, `args_()`, which returns a dictionary describing how to run that step:

```python
def args_(file_names, param):
    return {
        'func':         <callable to execute>,
        'argkws':       <keyword arguments dict>,
        'input_files':  [<list of HDF5 files that must exist>],
        'output_files': [<list of HDF5 files that will be written>],
        'requirements': [<list of checkpoint names that must have finished>],
    }
```

The orchestrator — `tierpsy/processing/AnalysisPoints.py` — discovers these packages dynamically by name. To add a new checkpoint called `CONSOLIDATE_TRAJ`, you create the package `tierpsy/analysis/consolidate_traj/` and register the checkpoint name in the pipeline sequences defined in `tierpsy/helper/params/docs_analysis_points.py`. No changes to the orchestrator itself are needed.

### 2.2 The Full Modified Pipeline (TIERPSY_CONSOLIDATE)

The standard `TIERPSY` analysis type executes checkpoints sequentially from `COMPRESS` through to `FEAT_TIERPSY`. The new `TIERPSY_CONSOLIDATE` pipeline inserts two additional checkpoints:

```
COMPRESS
    │
    ▼
TRAJ_CREATE      ──── Detects blobs frame-by-frame; writes plate_worms table
    │
    ▼
DETECT_CAM_ADJUST ─── NEW: flags camera adjustment frames; invalidates their blobs
    │
    ▼
TRAJ_JOIN        ──── Links blobs into trajectories; bridges across flagged gaps
    │
    ▼
CONSOLIDATE_TRAJ ──── NEW: merges all fragments into worm_index_joined = 1
    │
    ▼
SKE_INIT         ──── Smooths and interpolates trajectories; writes trajectories_data
    │
    ▼
BLOB_FEATS → SKE_CREATE → SKE_FILT → SKE_ORIENT → INT_PROFILE → INT_SKE_ORIENT
    │
    ▼
FEAT_INIT → FEAT_TIERPSY
```

`DETECT_CAM_ADJUST` runs before `TRAJ_JOIN` so that the gap-joiner never sees the corrupted adjustment frames and simply bridges across them. `CONSOLIDATE_TRAJ` runs after `TRAJ_JOIN` to handle any remaining fragmentation from other causes. Everything downstream is unchanged.

---

## 3. The `plate_worms` Table: Central Data Structure

Understanding the new feature requires understanding the central HDF5 table it operates on. After `TRAJ_CREATE` runs, a table called `/plate_worms` is written to the skeletons HDF5 file. Each row represents one detected blob in one video frame:

| Column | Type | Description |
|---|---|---|
| `frame_number` | int | Video frame index |
| `worm_index_blob` | int | Frame-to-frame tracking ID assigned during blob linking |
| `worm_index_joined` | int | Final trajectory ID assigned during gap-joining |
| `coord_x`, `coord_y` | float32 | Centre of mass (pixels) |
| `area` | float32 | Blob area (pixels²) |
| `box_length`, `box_width` | float32 | Bounding box dimensions |
| `threshold` | float32 | Pixel intensity threshold used for this blob |

The column `worm_index_joined` is the key field. After `TRAJ_JOIN` completes:
- Rows belonging to valid, accepted trajectories have `worm_index_joined > 0` (values 1, 2, 3, …)
- Rows rejected as noise or too-short fragments have `worm_index_joined = -1`

All downstream checkpoints (starting with `SKE_INIT`) read only rows where `worm_index_joined > 0` and group by that column to process each animal separately. By the time `CONSOLIDATE_TRAJ` finishes, only `worm_index_joined = 1` and `worm_index_joined = -1` will exist in the table — so the entire downstream pipeline will see exactly one animal.

---

## 4. Implementation

### 4.1 File Structure

Five files were created or modified:

```
tierpsy/
├── analysis/
│   ├── detect_cam_adjust/              ← NEW PACKAGE
│   │   ├── __init__.py                 ← checkpoint interface
│   │   └── detectCameraAdjustment.py   ← implementation
│   └── consolidate_traj/               ← NEW PACKAGE
│       ├── __init__.py                 ← checkpoint interface
│       └── consolidateTrajectories.py  ← implementation
└── helper/
    └── params/
        └── docs_analysis_points.py     ← MODIFIED: two new analysis types added
```

### 4.2 The `consolidateTrajectories` Function

The core logic lives in `tierpsy/analysis/consolidate_traj/consolidateTrajectories.py`. The full function is reproduced below with inline commentary:

```python
def consolidateTrajectories(skeletons_file):
    # Load the plate_worms table from the skeletons HDF5 file
    with pd.HDFStore(skeletons_file, 'r') as fid:
        plate_worms = fid['/plate_worms']

    # Filter to only rows that TRAJ_JOIN accepted (worm_index_joined > 0)
    valid_mask = plate_worms['worm_index_joined'] > 0
    valid_worms = plate_worms[valid_mask]

    # Compute a robust global reference area.
    # We take the maximum area per frame first, then the median across frames.
    # This is more robust than a simple median of all rows because frames with
    # multiple spurious blobs would otherwise pull the median downward.
    max_area_per_frame = valid_worms.groupby('frame_number')['area'].max()
    median_area = float(np.median(max_area_per_frame))

    # For each frame that has at least one valid detection, pick the single
    # blob whose area is closest to the global reference area.
    def _pick_best_row(frame_group):
        cost = (frame_group['area'] - median_area).abs()
        return frame_group.index[cost.argmin()]

    selected_indices = (
        valid_worms
        .groupby('frame_number', group_keys=False)
        .apply(_pick_best_row)
        .values
    )

    # Build a new worm_index_joined array: 1 for winners, -1 for everything else
    worm_index_joined = np.full(len(plate_worms), -1, dtype=np.int32)
    worm_index_joined[selected_indices] = 1

    # Write back to the HDF5 file in-place
    with tables.open_file(skeletons_file, mode='r+') as fid:
        tbl = fid.get_node('/plate_worms')
        tbl.modify_column(colname='worm_index_joined', column=worm_index_joined)
        fid.flush()
```

The algorithm can be expressed in plain language as three steps:

1. **Filter**: ignore everything already marked as noise (`worm_index_joined = -1`).
2. **Score**: for every remaining detection, compute a scalar cost equal to the absolute difference between its area and the global median area.
3. **Select**: for each frame, keep only the lowest-cost detection and set its `worm_index_joined` to 1.

### 4.3 Why Area?

Area is chosen as the selection criterion for several reasons:

- **Invariance to position**: unlike centre-of-mass coordinates, area does not depend on where the worm is in the frame. This is important for imaging setups without a motorised stage.
- **Biological stability**: the cross-sectional area of a healthy adult *C. elegans* (roughly 300–2000 pixels² depending on magnification) is relatively stable across its locomotive repertoire. The worm's area changes somewhat when it coils — but the change is bounded.
- **Discriminability**: dust particles and bubbles are typically either much smaller or (if aggregated) much larger than the worm. The area criterion thus naturally separates the animal from most common sources of false detection.
- **Computational cost**: computing the absolute difference from a scalar reference is O(n) in the number of rows, making the step negligible in the overall pipeline runtime.

### 4.4 Reference Area Calculation

The reference area is computed as follows:

```
For each frame f:
    max_area[f] = maximum blob area detected in frame f

median_area = median( max_area[f] for all f )
```

Taking the **per-frame maximum** before the **global median** is a deliberate robustness choice. Consider a frame where the real worm is present (area ≈ 800 px²) alongside a small dust particle (area ≈ 50 px²). A naive median over all rows would be pulled downward by the many small spurious blobs. By first taking the maximum per frame, we ensure that the worm — being the largest object in most frames — dominates the reference value.

```
Naive median of all areas:       [50, 50, 800, 50, 50, 800, ...] → ~50 px²  (WRONG)

Per-frame max then global median: [800, 800, 800, 800, ...]       → ~800 px² (CORRECT)
```

### 4.5 The `DETECT_CAM_ADJUST` Checkpoint

#### Motivation

In the dataset used for this project, the imaging rig repositions the camera or stage approximately 40 times per video. Each adjustment lasts 1–2 frames and produces a global shift of the entire image. During these frames the worm appears disfigured or split into multiple disconnected blobs. After the adjustment the worm reappears correctly, but because the tracker lost continuity, it assigns the worm a new trajectory ID. Across 60 videos this creates approximately 2,400 trajectory breaks — far too many to correct manually.

#### How Camera Adjustments Are Distinguished from Fast Worm Movement

The key insight is that camera adjustments produce **global** image motion — every pixel in the frame shifts simultaneously. Genuine fast worm movement is **local** — only the pixels in the worm's region change, while the background remains stable.

This is measured using the **mean absolute frame difference**: the average per-pixel absolute change in intensity between consecutive masked frames.

```
Camera adjustment frame:
  All background pixels shift → mean absolute difference is LARGE (spike)

Worm moving fast:
  Only worm pixels change → mean absolute difference stays LOW
```

#### Algorithm

```python
def detectCameraAdjustment(masked_image_file, skeletons_file,
                           cam_adjust_thresh=5.0,
                           cam_adjust_extra_frames=1):

    # 1. Read masked frames and compute frame-to-frame mean absolute difference
    for i in range(1, n_frames):
        frame_diffs[i] = mean(abs(frame[i] - frame[i-1]))

    # 2. Flag frames where the difference exceeds the threshold
    adjustment_frames = where(frame_diffs > cam_adjust_thresh)

    # 3. Expand by ±cam_adjust_extra_frames to cover the full adjustment duration
    expanded = {f + delta for f in adjustment_frames
                for delta in range(-extra, extra+1)}

    # 4. Invalidate all blobs detected in those frames
    worm_index_blob[rows where frame_number in expanded] = -1

    # 5. Write back to plate_worms in the skeletons HDF5 file
```

After this step, `TRAJ_JOIN` sees clean 1–2 frame gaps at every camera adjustment and bridges across them using its existing gap-joining logic. The worm's real positions immediately before and after the adjustment are preserved, and `SKE_INIT` interpolates just the flagged frames — which is acceptable because what the camera captured during a physical shift is not meaningful tracking data.

#### Chunked Frame Reading for Performance

The initial implementation read frames one at a time inside a Python `for` loop, incurring Python interpreter overhead on every iteration. For a typical video with 3,000 frames this meant 3,000 individual loop cycles. The implementation was subsequently optimised to read frames in chunks of 256 using NumPy's vectorised operations:

```python
chunk_size = 256

for chunk_start in range(0, n_frames, chunk_size):
    chunk = mask[chunk_start:chunk_end].astype(np.float32)  # shape: (256, H, W)
    chunk_with_prev = np.concatenate([prev_last[np.newaxis], chunk], axis=0)

    # one vectorised operation across all 256 frame pairs simultaneously
    diffs = np.abs(np.diff(chunk_with_prev, axis=0))
    frame_diffs[chunk_start:chunk_end] = diffs.mean(axis=(1, 2))
```

Instead of 3,000 Python loop iterations, the same video requires only ~12 chunk iterations. The actual difference computation (`np.diff`, `np.abs`, `.mean`) executes in C-speed NumPy rather than Python, resulting in significantly faster execution. The `chunk_size=256` value balances speed against RAM usage — each chunk loads 256 frames into memory simultaneously.

#### Tunable Parameters

| Parameter | Default | Effect |
|---|---|---|
| `cam_adjust_thresh` | `5.0` | Mean pixel intensity change to trigger flagging. Lower = more sensitive. |
| `cam_adjust_extra_frames` | `1` | Extra frames flagged either side of a detected adjustment. |

Both parameters can be set in the JSON parameter file:

```json
{
    "cam_adjust_thresh": 5.0,
    "cam_adjust_extra_frames": 1
}
```

### 4.6 The Checkpoint Interface

The `__init__.py` file wires the function into the checkpoint system:

```python
from .consolidateTrajectories import consolidateTrajectories

def args_(fn, param):
    return {
        'func': consolidateTrajectories,
        'argkws': {
            'skeletons_file': fn['skeletons'],
        },
        'input_files':  [fn['skeletons']],
        'output_files': [fn['skeletons']],
        'requirements': ['TRAJ_JOIN'],
    }
```

The `requirements` field declares that `TRAJ_JOIN` must have successfully completed before this step runs. The orchestrator's `CheckFinished` class enforces this. Both `input_files` and `output_files` point to the same HDF5 file because the step modifies a column in-place rather than creating a new file.

### 4.7 New Analysis Types

Two new entries were added to `tierpsy/helper/params/docs_analysis_points.py`:

```python
_consolidate_base = [
    'COMPRESS',
    'TRAJ_CREATE',
    'DETECT_CAM_ADJUST',  # ← invalidates camera adjustment frames
    'TRAJ_JOIN',
    'CONSOLIDATE_TRAJ',   # ← merges remaining fragments
    'SKE_INIT',
    'BLOB_FEATS',
    'SKE_CREATE',
    'SKE_FILT',
    'SKE_ORIENT',
    'INT_PROFILE',
    'INT_SKE_ORIENT',
]

dflt_analysis_points['TIERPSY_CONSOLIDATE']  = _consolidate_base + ['FEAT_INIT', 'FEAT_TIERPSY']
dflt_analysis_points['OPENWORM_CONSOLIDATE'] = _consolidate_base + ['FEAT_CREATE']
```

`TIERPSY_CONSOLIDATE` uses Tierpsy's own feature extraction (the recommended choice). `OPENWORM_CONSOLIDATE` uses the OpenWorm Analysis Toolbox features for researchers who need compatibility with that pipeline.

---

## 5. Comparison with Existing Single-Worm Modes

The table below summarises the key differences between the three single-worm approaches now available in the software:

| Property | `TIERPSY_SINGLE` | `TIERPSY_WT2` | `TIERPSY_CONSOLIDATE` |
|---|---|---|---|
| **When selection occurs** | During `TRAJ_JOIN`, on raw blobs | During `TRAJ_JOIN`, on raw blobs | After `TRAJ_JOIN`, on gap-joined trajectories |
| **Selection criterion** | Area proximity + movement threshold | Spatial proximity to frame centre | Area proximity only |
| **Movement threshold** | Strict: ≤ ¼ body length per frame | Strict: ≤ ¼ body length per frame | None |
| **Gap tolerance** | None — gaps break the chain | None — gaps break the chain | Benefits from `TRAJ_JOIN` gap-bridging |
| **Hardware requirement** | None | Motorised stage (WT2) | None |
| **Handles spurious blobs** | Yes (area filter) | Yes (centre filter) | Yes (area filter) |
| **Best for** | Clean videos, consistent detection | WT2 hardware only | Fragmented detections, occlusions |

The fundamental architectural difference is illustrated below:

```
TIERPSY_SINGLE / TIERPSY_WT2 — selection happens at the raw-blob stage:

TRAJ_CREATE              TRAJ_JOIN
    │                        │
    │    plate_worms          │  correctSingleWormCase()
    │    (all blobs,          │    ↓ strict movement filter
    │     worm_index_         │    ↓ discards blobs > ¼ body length away
    │     joined = -1)   ─────┤    ↓ result: some frames with NO selected blob
    │                         │
    └──────────────────────► SKE_INIT (trajectory has gaps)


TIERPSY_CONSOLIDATE — selection happens after gap-joining:

TRAJ_CREATE       TRAJ_JOIN            CONSOLIDATE_TRAJ
    │                 │                       │
    │    plate_worms  │  joinGapsTraj()       │  area-based selection
    │    (all blobs)  │    ↓ links fragments  │    ↓ no movement threshold
    │                 │    ↓ worm_index_joined │    ↓ every frame gets a winner
    └────────────────►│    = 1,2,3,4,...  ────┤    → all reassigned to 1
                       │                      │
                       └─────────────────────► SKE_INIT (denser trajectory)
```

---

## 6. Data Flow Through the Modified Pipeline

The following diagram traces the state of the `worm_index_joined` column through the pipeline for a hypothetical video containing one worm and some spurious detections:

```
After TRAJ_CREATE:
  All rows: worm_index_joined = -1   (placeholder, not yet assigned)
  All rows: worm_index_blob = 1..N  (frame-to-frame blob IDs)

After TRAJ_JOIN (joinGapsTrajectories):
  Fragment A (frames 0–45):   worm_index_joined = 1
  Fragment B (frames 47–89):  worm_index_joined = 2
  Fragment C (frames 92–100): worm_index_joined = 5
  Noise blob (frames 12–14):  worm_index_joined = 3
  Short fragment (3 frames):  worm_index_joined = -1  (too short, rejected)

After CONSOLIDATE_TRAJ:
  Selected detections:        worm_index_joined = 1   (all merged)
  Rejected detections:        worm_index_joined = -1
  → Single trajectory ID=1, spanning frames 0–100

After SKE_INIT (getSmoothedTraj):
  trajectories_data table:    one group, worm_index_joined = 1
  skeleton_id = 0, 1, 2, ...  (sequential across all frames)
  Gaps filled by linear interpolation

After FEAT_TIERPSY:
  One complete feature record for the single animal
```

---

## 7. How to Use the Feature

### 7.1 Via the Parameter JSON File

Create or edit a parameter JSON file and set `analysis_type` to the new value:

```json
{
    "analysis_type": "TIERPSY_CONSOLIDATE"
}
```

All other parameters (`traj_max_allowed_dist`, `traj_area_ratio_lim`, etc.) continue to work as usual since the new step runs after `TRAJ_JOIN` and does not change how that step behaves.

### 7.2 Via the Command Line

```bash
tierpsy_process \
    --video_dir_root /path/to/videos \
    --mask_dir_root  /path/to/masks \
    --results_dir    /path/to/results \
    --json_file      /path/to/params.json
```

### 7.3 Via the GUI

Open the Tierpsy GUI:

```bash
tierpsy_gui
```

In the **Batch Processing** or **Get Mask Params** panel, locate the `analysis_type` dropdown. `TIERPSY_CONSOLIDATE` and `OPENWORM_CONSOLIDATE` will appear as selectable options alongside the existing types.

### 7.4 Re-running the Step on Existing Files

Because `CONSOLIDATE_TRAJ` modifies only the `worm_index_joined` column of `plate_worms`, and because Tierpsy's `CheckFinished` system tracks which checkpoints have completed, it is possible to re-run from `CONSOLIDATE_TRAJ` onwards without reprocessing the video compression and trajectory creation steps. This makes it straightforward to apply the new step to files that were previously processed with a standard analysis type:

1. Set `analysis_type` to `TIERPSY_CONSOLIDATE` in the parameter file.
2. Delete or rename the existing `*_skeletons.hdf5` checkpoint marker if present, or use the GUI's "reprocess from" option.
3. Re-run the pipeline; `TRAJ_CREATE` and `TRAJ_JOIN` will be skipped (already finished), and execution will resume from `CONSOLIDATE_TRAJ`.

---

## 8. GUI Compatibility Fixes

During development and testing of the new pipeline on macOS with Python 3.10 and modern versions of NumPy and PyQt5, several pre-existing compatibility bugs in Tierpsy Tracker's GUI were encountered and fixed. These are unrelated to the new tracking features but were necessary to make the software usable.

### 8.1 C Source File Compatibility (K&R Style Parameters)

Two C source files used by the Cython skeleton extraction extensions used 1970s-era K&R style function declarations that are rejected by Apple's modern clang compiler:

```c
// Before — rejected by clang on macOS arm64
inline int ind(s1, s2) { ... }

// After — C99 compliant
inline int ind(int s1, int s2) { ... }
```

Files fixed: `c_circCurvature.c`, `c_curvspace.c`.

### 8.2 Python 3.10 `collections.Iterable` Removal

Python 3.10 moved abstract base classes from `collections` to `collections.abc`. The OpenWorm Analysis Toolbox submodule used the old import:

```python
# Before
from collections import namedtuple, Iterable, OrderedDict

# After
from collections import namedtuple, OrderedDict
from collections.abc import Iterable
```

File fixed: `tierpsy/features/open_worm_analysis_toolbox/prefeatures/basic_worm.py`.

### 8.3 PyQt5 Integer Type Strictness

PyQt5's drawing methods (`drawRect`, `drawLine`, `fillRect`) require plain Python `int` arguments. Newer NumPy returns `numpy.float64` or `numpy.int64` values from arithmetic operations, which PyQt5 rejects. Explicit `int()` casts were added at all affected call sites.

Files fixed: `tierpsy/gui/MWTrackerViewer.py`, `tierpsy/gui/GetAllParameters.py`.

---

## 9. Limitations and Future Work

### 9.1 No Temporal Continuity in Selection

The current implementation selects the best blob per frame **independently** — it does not use the position of the selected blob in the previous frame as a prior. This means that in a video with two similarly-sized objects (for example, a worm and a piece of debris of similar area), the algorithm may occasionally "jump" between them if their areas happen to cross the reference value.

A natural extension would be to incorporate a continuity term: once a blob is selected in frame `t`, prefer the spatially closest acceptable blob in frame `t+1`. This would be analogous to the frame-to-frame linking already performed in `TRAJ_JOIN`, but applied as a post-selection refinement pass.

### 9.2 Single-Blob-Per-Frame Assumption

The downstream pipeline (`SKE_INIT` → `SKE_CREATE` → features) is designed around the assumption of at most one detection per `worm_index_joined` per frame. The current implementation enforces this by construction (one winner per frame), but does not handle the case where the researcher actually wants to track **two worms** and merge them — that would require a fundamentally different approach (multi-label assignment rather than winner-takes-all).

### 9.3 No Adaptive Reference Area

The reference area is computed once globally. For very long recordings where the worm's size changes (for example, due to gradual changes in illumination shifting the segmentation threshold), a sliding-window or frame-weighted version of the reference area might produce better selection.

### 9.4 Camera Adjustment Threshold is Fixed

The `cam_adjust_thresh` parameter defaults to 5.0 and must currently be set manually in the JSON file. A future improvement would be to estimate the threshold automatically from the distribution of frame differences in each video, making the detector self-calibrating across different imaging rigs and lighting conditions.

---

## 10. Summary

This addition introduces five new components to Tierpsy Tracker:

1. **`tierpsy/analysis/detect_cam_adjust/detectCameraAdjustment.py`** — detects camera adjustment frames by computing mean absolute frame differences, flags the 1–2 affected frames per adjustment event, and invalidates their blob detections so that `TRAJ_JOIN` bridges across them cleanly.

2. **`tierpsy/analysis/detect_cam_adjust/__init__.py`** — the checkpoint interface for `DETECT_CAM_ADJUST`, declaring `TRAJ_CREATE` as its prerequisite.

3. **`tierpsy/analysis/consolidate_traj/consolidateTrajectories.py`** — the core function that reads the `plate_worms` table, computes a robust area reference, selects one blob per frame, and writes back a single unified `worm_index_joined = 1`.

4. **`tierpsy/analysis/consolidate_traj/__init__.py`** — the checkpoint interface for `CONSOLIDATE_TRAJ`, declaring `TRAJ_JOIN` as its prerequisite.

5. **Two new entries in `docs_analysis_points.py`** — `TIERPSY_CONSOLIDATE` and `OPENWORM_CONSOLIDATE` — which define complete end-to-end pipelines that include both new checkpoints in the correct positions.

Together these features are designed for researchers working with single-animal recordings on imaging rigs that perform periodic camera or stage adjustments. The combination of pre-join frame invalidation (`DETECT_CAM_ADJUST`) and post-join fragment merging (`CONSOLIDATE_TRAJ`) eliminates the need for manual trajectory editing that would otherwise be required thousands of times across a large dataset. The result is a single, dense trajectory record covering the full duration of each recording, enabling more complete and reliable behavioural feature extraction.
