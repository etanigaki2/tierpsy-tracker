# -*- coding: utf-8 -*-
"""
Detect camera adjustment frames and invalidate their blob detections.

Camera adjustments produce a sudden global shift of the entire image for 1-2
frames. This is distinguished from genuine fast worm movement by the fact that
the shift is GLOBAL — the mean absolute difference between consecutive masked
frames spikes sharply. Fast worm movement only changes a small local region,
keeping the global mean low.

Blobs detected in adjustment frames are marked worm_index_blob = -1 so that
TRAJ_JOIN treats them as gaps and bridges across them using the existing
gap-joining logic.
"""

import numpy as np
import pandas as pd
import tables

from tierpsy.helper.misc import print_flush


def detectCameraAdjustment(masked_image_file,
                           skeletons_file,
                           cam_adjust_thresh=5.0,
                           cam_adjust_extra_frames=1):
    """
    Detect camera adjustment frames and invalidate their blobs in plate_worms.

    Parameters
    ----------
    masked_image_file : str
        Path to the _masked.hdf5 file containing the /mask dataset.
    skeletons_file : str
        Path to the _skeletons.hdf5 file containing plate_worms.
    cam_adjust_thresh : float
        Threshold for the mean absolute frame difference (in pixel intensity
        units 0-255). Frames where the difference exceeds this value are
        flagged as adjustment frames. Default 5.0 works for most rigs; lower
        it if adjustments are being missed, raise it if normal fast movement
        is being incorrectly flagged.
    cam_adjust_extra_frames : int
        Number of extra frames to invalidate on each side of a detected
        adjustment frame. Default 1 covers the typical 1-2 frame duration.
    """
    base_name = skeletons_file.rpartition('_skeletons')[0].rpartition('/')[-1]
    print_flush(base_name + ' Detecting camera adjustment frames...')

    # --- Step 1: compute frame-to-frame mean absolute difference ----------
    # Frames are read in chunks so NumPy computes the diff across an entire
    # block of frames in one vectorised operation rather than one-by-one in
    # a Python loop. chunk_size controls how many frames are loaded at once;
    # larger chunks are faster but use more RAM.
    chunk_size = 256

    with tables.open_file(masked_image_file, 'r') as fid:
        mask = fid.get_node('/mask')
        n_frames = mask.shape[0]

        if n_frames < 2:
            print_flush(base_name + ' Too few frames — skipping.')
            return

        frame_diffs = np.zeros(n_frames, dtype=np.float32)

        # seed: last frame of the previous chunk, needed to diff across
        # chunk boundaries
        prev_last = mask[0].astype(np.float32)

        for chunk_start in range(0, n_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_frames)
            # shape: (chunk_len, height, width)
            chunk = mask[chunk_start:chunk_end].astype(np.float32)

            # prepend the last frame of the previous chunk so we can diff
            # across the boundary without an extra loop
            chunk_with_prev = np.concatenate(
                [prev_last[np.newaxis], chunk], axis=0
            )

            # vectorised abs diff across all consecutive pairs in the chunk
            # result shape: (chunk_len, height, width)
            diffs = np.abs(np.diff(chunk_with_prev, axis=0))

            # mean over spatial dimensions → one scalar per frame
            frame_diffs[chunk_start:chunk_end] = diffs.mean(axis=(1, 2))

            prev_last = chunk[-1]

    # --- Step 2: flag adjustment frames -----------------------------------
    # A frame is an adjustment frame if its diff exceeds the threshold.
    # We also flag ±cam_adjust_extra_frames neighbours to cover the full
    # 1-2 frame duration of the adjustment.
    adjustment_frames = set(np.where(frame_diffs > cam_adjust_thresh)[0].tolist())

    if not adjustment_frames:
        print_flush(base_name + ' No camera adjustment frames detected.')
        return

    # expand by extra frames on each side
    expanded = set()
    for f in adjustment_frames:
        for delta in range(-cam_adjust_extra_frames, cam_adjust_extra_frames + 1):
            neighbour = f + delta
            if 0 <= neighbour < n_frames:
                expanded.add(neighbour)

    print_flush(
        base_name +
        f' Detected {len(adjustment_frames)} adjustment frames '
        f'({len(expanded)} frames total after expansion).'
    )

    # --- Step 3: invalidate blobs in those frames -------------------------
    with pd.HDFStore(skeletons_file, 'r') as fid:
        plate_worms = fid['/plate_worms']

    if len(plate_worms) == 0:
        return

    bad_rows = plate_worms['frame_number'].isin(expanded)
    n_invalidated = bad_rows.sum()

    if n_invalidated == 0:
        print_flush(base_name + ' No blobs fall in adjustment frames.')
        return

    worm_index_blob = plate_worms['worm_index_blob'].values.copy()
    worm_index_blob[bad_rows.values] = -1

    with tables.open_file(skeletons_file, mode='r+') as fid:
        tbl = fid.get_node('/plate_worms')
        tbl.modify_column(colname='worm_index_blob', column=worm_index_blob)
        fid.flush()

    print_flush(
        base_name +
        f' Invalidated {n_invalidated} blobs in camera adjustment frames.'
    )
