# -*- coding: utf-8 -*-
"""
Consolidate all trajectory fragments into a single worm ID.

After TRAJ_JOIN, a single-worm video may have multiple trajectory fragments
(worm_index_joined = 1, 2, 3, ...) due to occlusions, detection gaps, or
spurious blobs. This step selects the best blob per frame (by area proximity
to the global median) and reassigns all selected blobs to worm_index_joined=1.
"""

import numpy as np
import pandas as pd
import tables

from tierpsy.helper.misc import print_flush


def consolidateTrajectories(skeletons_file):
    """
    Merge all trajectory fragments into a single worm ID (worm_index_joined=1).

    For each frame that has detections, the blob whose area is closest to the
    global median area is kept; all others are discarded (set to -1). This
    produces a single continuous trajectory suitable for single-worm analysis.

    Parameters
    ----------
    skeletons_file : str
        Path to the _skeletons.hdf5 file produced by TRAJ_CREATE / TRAJ_JOIN.
    """
    base_name = skeletons_file.rpartition('_skeletons')[0].rpartition('/')[-1]
    print_flush(base_name + ' Consolidating trajectories to single worm ID...')

    with pd.HDFStore(skeletons_file, 'r') as fid:
        plate_worms = fid['/plate_worms']

    if len(plate_worms) == 0:
        print_flush(base_name + ' No detections found — nothing to consolidate.')
        return

    # Work only with rows that were accepted by TRAJ_JOIN
    valid_mask = plate_worms['worm_index_joined'] > 0
    valid_worms = plate_worms[valid_mask]

    if len(valid_worms) == 0:
        print_flush(base_name + ' No valid trajectories found — nothing to consolidate.')
        return

    # Global reference area: median of per-frame max area
    # (robust against outlier frames with noise blobs)
    max_area_per_frame = valid_worms.groupby('frame_number')['area'].max()
    median_area = float(np.median(max_area_per_frame))

    # For each frame, pick the blob whose area is closest to median_area
    def _pick_best_row(frame_group):
        cost = (frame_group['area'] - median_area).abs()
        return frame_group.index[cost.argmin()]

    selected_indices = (
        valid_worms
        .groupby('frame_number', group_keys=False)
        .apply(_pick_best_row)
        .values
    )

    # Build updated worm_index_joined column: 1 for selected, -1 for the rest
    worm_index_joined = np.full(len(plate_worms), -1, dtype=np.int32)
    worm_index_joined[selected_indices] = 1

    with tables.open_file(skeletons_file, mode='r+') as fid:
        tbl = fid.get_node('/plate_worms')
        tbl.modify_column(colname='worm_index_joined', column=worm_index_joined)
        fid.flush()

    n_frames = len(selected_indices)
    print_flush(
        base_name +
        f' Consolidation complete: {n_frames} frames assigned to worm_index_joined=1.'
    )
