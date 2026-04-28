# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:46:20 2015

@author: ajaver
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from .linearSkeleton import linearSkeleton
from .getHeadTail import getHeadTail, rollHead2FirstIndex
from .cython_files.segWorm_cython import circComputeChainCodeLengths
from .cleanWorm import circSmooth, extremaPeaksCircDist

# wrappers around C functions
from .cython_files.circCurvature import circCurvature


def _extract_longest_skeleton_path(thin):
    """
    Given a binary thinned skeleton image, return (x, y) skeleton points as
    the longest path between the two geometrically farthest endpoints.

    Uses two-pass BFS (graph diameter on a tree) so short stub branches that
    appear at the pinch point of an omega turn are automatically excluded.
    Returns None if fewer than 3 skeleton pixels are found.

    Ref: Zhang & Suen (1984); Cormen et al. (2009)
    """
    ys, xs = np.where(thin)
    if len(xs) < 3:
        return None

    pt_set = set(zip(ys.tolist(), xs.tolist()))

    def _neighbors(r, c):
        return [(r + dr, c + dc)
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0) and (r + dr, c + dc) in pt_set]

    adj = {pt: _neighbors(*pt) for pt in pt_set}

    endpoints = [pt for pt, nbrs in adj.items() if len(nbrs) == 1]
    if len(endpoints) < 2:
        pts_list = list(pt_set)
        endpoints = [pts_list[0], pts_list[len(pts_list) // 2]]

    def _bfs(start):
        parent = {start: None}
        queue = [start]
        tail = start
        for curr in queue:
            tail = curr
            for nbr in adj[curr]:
                if nbr not in parent:
                    parent[nbr] = curr
                    queue.append(nbr)
        return tail, parent

    end1, _ = _bfs(endpoints[0])
    end2, parents = _bfs(end1)

    path = []
    curr = end2
    while curr is not None:
        path.append(curr)
        curr = parents[curr]
    path.reverse()

    # (row, col) -> (x, y) to match OpenCV contour convention
    return np.array([(c, r) for r, c in path], dtype=np.float64)


def _compute_sides_and_widths(skeleton_rs, worm_mask):
    """
    For each resampled skeleton point cast perpendicular rays outward until
    the worm mask boundary is reached, recording side contact points and width.

    Fully vectorised over all skeleton points and ray steps using NumPy array
    indexing — no Python loops.  Ray step size is 0.5 px.

    Two fixes for omega turns:
      1. Tangent computed over a wider window (±4 points instead of ±1) so the
         perpendicular direction remains stable at tight curves.  At a 1-step
         central difference, points on opposite sides of a sharp bend give a
         tangent that points *across* the curve rather than along it, sending
         rays in the wrong direction entirely.
      2. Ray length capped at 3× the distance-transform half-width at each
         skeleton point.  The old cap of max(h,w)/2 ≈ 125 px let wrongly-
         directed rays travel 7× the worm body width and exit far outside the
         body, which is why contour lines appeared outside the worm in Image 9.

    Ref: normal ray intersection — Yemini et al. (2013)
    """
    from scipy.ndimage import distance_transform_edt

    h, w = worm_mask.shape
    n = len(skeleton_rs)

    # --- tangents over a wider window to stabilise tight curves ---
    win = min(4, (n - 1) // 2)          # ±win points; shrink near endpoints
    tangents = np.empty_like(skeleton_rs)
    for i in range(n):
        lo = max(0, i - win)
        hi = min(n - 1, i + win)
        tangents[i] = skeleton_rs[hi] - skeleton_rs[lo]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    tangents /= norms
    perps = np.column_stack([-tangents[:, 1], tangents[:, 0]])  # (n, 2)

    px = skeleton_rs[:, 0]   # col  (n,)
    py = skeleton_rs[:, 1]   # row  (n,)
    dx = perps[:, 0]
    dy = perps[:, 1]

    # --- per-point ray cap from distance transform ---
    # dist[r,c] = distance to nearest background pixel = local half-width
    dist = distance_transform_edt(worm_mask)
    col_sk = np.clip(np.round(px).astype(np.int32), 0, w - 1)
    row_sk = np.clip(np.round(py).astype(np.int32), 0, h - 1)
    half_widths = dist[row_sk, col_sk]                   # (n,)
    # allow up to 3× the local half-width so small misalignments are handled
    max_per_pt  = np.maximum(half_widths * 3.0, 5.0)    # (n,) at least 5 px

    # --- ray sampling grid ---
    # Use the largest per-point cap as the global grid length
    global_max = float(np.max(max_per_pt))
    t_vals = np.arange(0.5, global_max, 0.5)   # (T,)
    T = len(t_vals)

    # Broadcast to (n, T)
    t_grid = t_vals[np.newaxis, :]            # (1, T)

    def _find_exit(sign):
        """
        Cast rays in direction sign*(dx, dy) for all n skeleton points.
        Returns (d, side_pts) where d[i] is the distance to the boundary
        and side_pts[i] is the (x, y) contact point.
        """
        ray_x = px[:, np.newaxis] + sign * dx[:, np.newaxis] * t_grid   # (n, T)
        ray_y = py[:, np.newaxis] + sign * dy[:, np.newaxis] * t_grid   # (n, T)

        col = np.clip(np.round(ray_x).astype(np.int32), 0, w - 1)       # (n, T)
        row = np.clip(np.round(ray_y).astype(np.int32), 0, h - 1)       # (n, T)

        # A ray step is "inside" only when in-bounds, within the mask,
        # AND within the per-point distance cap (prevents rays from
        # travelling through the interior of omega loops).
        in_bounds  = ((ray_x >= 0) & (ray_x < w) &
                      (ray_y >= 0) & (ray_y < h))                        # (n, T)
        within_cap = t_grid <= max_per_pt[:, np.newaxis]                 # (n, T)
        inside     = in_bounds & within_cap & (worm_mask[row, col] > 0)  # (n, T)

        exited   = np.any(~inside, axis=1)                               # (n,)
        exit_idx = np.where(exited, np.argmax(~inside, axis=1), T - 1)
        d        = t_vals[exit_idx]                                       # (n,)

        side_x   = px + sign * dx * d
        side_y   = py + sign * dy * d
        side_pts = np.column_stack([side_x, side_y])

        return d, side_pts

    d1, cnt_side1 = _find_exit(+1)
    d2, cnt_side2 = _find_exit(-1)
    cnt_widths    = d1 + d2

    return cnt_side1, cnt_side2, cnt_widths


def _omega_skeleton_fallback(worm_mask, prev_skeleton, resampling_N):
    """
    Morphological-thinning fallback for omega turns and other self-touching
    poses where contour2Skeleton fails (errors 104, 105, or 106).

    During an omega turn the worm folds its head close to its mid-body,
    making one contour side much shorter than the other and triggering
    error 106 (isHeadTailTouching).  This fallback:
      1. Skeletonizes the binary mask via morphological thinning (Lee et al., 1994).
      2. Extracts the longest path between skeleton endpoints with two-pass BFS,
         discarding short stub branches at the pinch point.
      3. Resamples to resampling_N equidistant points.
      4. Computes contour sides and widths via perpendicular ray casting.
      5. Orients the skeleton consistently with prev_skeleton.

    Returns the same 6-tuple as getSkeleton, or None on failure.
    """
    try:
        from skimage.morphology import skeletonize as morph_skeletonize
    except ImportError:
        return None

    thin = morph_skeletonize(worm_mask > 0)
    skeleton_raw = _extract_longest_skeleton_path(thin)
    if skeleton_raw is None or len(skeleton_raw) < resampling_N:
        return None

    skeleton, ske_len, _ = resample_curve(skeleton_raw, resampling_N)
    if skeleton is None:
        return None

    if prev_skeleton.size > 0 and prev_skeleton.shape == skeleton.shape:
        # During an omega turn the head sweeps a large arc (80+ px per frame)
        # while the tail is nearly stationary (~4 px).  Comparing the whole
        # skeleton (as orientWorm does) or just the head region is unreliable
        # because the head has moved so far.  Instead compare BOTH endpoints
        # simultaneously against the previous frame's head AND tail:
        #   correct:  skel[0]≈prev_head  AND  skel[-1]≈prev_tail
        #   flipped:  skel[0]≈prev_tail  AND  skel[-1]≈prev_head
        # The tail barely moves during an omega turn, giving a very strong signal.
        prev_head = prev_skeleton[0]
        prev_tail = prev_skeleton[-1]
        A = skeleton[0]
        B = skeleton[-1]
        score_correct = np.sum((A - prev_head) ** 2) + np.sum((B - prev_tail) ** 2)
        score_flipped = np.sum((B - prev_head) ** 2) + np.sum((A - prev_tail) ** 2)
        if score_flipped < score_correct:
            skeleton = skeleton[::-1]

    cnt_side1, cnt_side2, cnt_widths = _compute_sides_and_widths(skeleton, worm_mask)
    cnt_area = float(np.sum(worm_mask))

    return (skeleton.astype(np.float32),
            ske_len,
            cnt_side1.astype(np.float32),
            cnt_side2.astype(np.float32),
            cnt_widths.astype(np.float32),
            cnt_area)


errMsg = {104 : '''The worm has 3 or more low-frequency sampled convexities
        sharper than 90 degrees (possible head/tail points).''',
          105 : '''The worm contour has less than 2 high-frequency sampled
        convexities sharper than 60 degrees (the head and tail).
        Therefore, the worm is coiled or obscured and cannot be segmented.''',
          106: '''The worm length, from head to tail, is more than
        twice as large on one side than it is on the other.
        Therefore, the worm is coiled or obscured and cannot be segmented.'''
          }


def get_contour_angles(contour, cnt_chain_code_len, cnt_worm_segments, edge_len_hi_freq):
    
    cnt_ang_hi_freq = circCurvature(
        contour, edge_len_hi_freq, cnt_chain_code_len)

    edge_len_low_freq = 2 * edge_len_hi_freq
    cnt_ang_low_freq = circCurvature(
        contour, edge_len_low_freq, cnt_chain_code_len)

    #% Blur the contour's local high-frequency curvature.
    #% Note: on a small scale, noise causes contour imperfections that shift an
    #% angle from its correct location. Therefore, blurring angles by averaging
    #% them with their neighbors can localize them better.
    worm_seg_size = contour.shape[0] / cnt_worm_segments
    blur_size_hi_freq = np.ceil(worm_seg_size / 2)
    cnt_ang_hi_freq = circSmooth(cnt_ang_hi_freq, blur_size_hi_freq)
    
    #% Compute the contour's local high/low-frequency curvature maxima.
    maxima_hi_freq, maxima_hi_freq_ind = extremaPeaksCircDist(
        1, cnt_ang_hi_freq, edge_len_hi_freq, cnt_chain_code_len)

    maxima_low_freq, maxima_low_freq_ind = extremaPeaksCircDist(
        1, cnt_ang_low_freq, edge_len_low_freq, cnt_chain_code_len)

    
    #pack output
    
    hi_freq_output = cnt_ang_hi_freq, maxima_hi_freq, maxima_hi_freq_ind, edge_len_hi_freq
    low_freq_output = cnt_ang_low_freq, maxima_low_freq, maxima_low_freq_ind, edge_len_low_freq
    
    return hi_freq_output, low_freq_output


def contour2Skeleton(contour, ske_worm_segments = 24, head_angle_thresh=60):
    # contour must be a Nx2 numpy array
    assert isinstance(
        contour,
        np.ndarray) and contour.ndim == 2 and contour.shape[1] == 2

    if contour.dtype != np.double:
        contour = contour.astype(np.double)

    #% The worm is roughly divided into 24 segments of musculature (i.e., hinges
    #% that represent degrees of freedom) on each side. Therefore, 48 segments
    #% around a 2-D contour.
    #% Note: "In C. elegans the 95 rhomboid-shaped body wall muscle cells are
    #% arranged as staggered pairs in four longitudinal bundles located in four
    #% quadrants. Three of these bundles (DL, DR, VR) contain 24 cells each,
    #% whereas VL bundle contains 23 cells." - www.wormatlas.org
    cnt_worm_segments = 2 * ske_worm_segments

    # this line does not really seem to be useful
    #contour = cleanWorm(contour, cnt_worm_segments)

    #% The contour is too small.
    if contour.shape[0] < cnt_worm_segments:
        err_msg = 'Contour is too small'
        return 4 * [np.zeros(0)] + [err_msg]

    # make sure the contours are in the counter-clockwise direction
    # head tail indentification will not work otherwise
    # x1y2 - x2y1(http://mathworld.wolfram.com/PolygonArea.html)
    signed_area = np.sum(
        contour[:-1, 0] * contour[1:, 1] - contour[1:, 0] * contour[:-1, 1]) / 2
    if signed_area > 0:
        contour = np.ascontiguousarray(contour[::-1, :])

    # make sure the array is C_continguous. Several functions required this.
    if not contour.flags['C_CONTIGUOUS']:
        contour = np.ascontiguousarray(contour)

    #% Compute the contour's local high/low-frequency curvature.
    #% Note: worm body muscles are arranged and innervated as staggered pairs.
    #% Therefore, 2 segments have one theoretical degree of freedom (i.e. one
    #% approximation of a hinge). In the head, muscles are innervated
    #% individually. Therefore, we sample the worm head's curvature at twice the
    #% frequency of its body.
    #% Note 2: we ignore Nyquist sampling theorem (sampling at twice the
    #% frequency) since the worm's cuticle constrains its mobility and practical
    #% degrees of freedom.

    cnt_chain_code_len = circComputeChainCodeLengths(contour)
    worm_seg_length = (cnt_chain_code_len[
                       0] + cnt_chain_code_len[-1]) / cnt_worm_segments

    #calculate contour angles
    hi_freq_output, low_freq_output = get_contour_angles(contour, 
                                                         cnt_chain_code_len, 
                                                         cnt_worm_segments, 
                                                         worm_seg_length)

    #unpack data
    cnt_ang_hi_freq, maxima_hi_freq, maxima_hi_freq_ind, edge_len_hi_freq = hi_freq_output
    cnt_ang_low_freq, maxima_low_freq, maxima_low_freq_ind, edge_len_low_freq = low_freq_output
    
    #identify head/tail
    head_ind, tail_ind, err_msg = getHeadTail(cnt_ang_low_freq, 
                                              maxima_low_freq_ind, 
                                              cnt_ang_hi_freq, 
                                              maxima_hi_freq_ind, 
                                              cnt_chain_code_len, 
                                              head_angle_thresh)

    if err_msg != 0:
        return 4 * [np.zeros(0)] + [err_msg]

    # change arrays so the head correspond to the first position
    head_ind, tail_ind, contour, cnt_chain_code_len, cnt_ang_low_freq, maxima_low_freq_ind = \
        rollHead2FirstIndex(head_ind, 
                            tail_ind, 
                            contour, 
                            cnt_chain_code_len, 
                            cnt_ang_low_freq, 
                            maxima_low_freq_ind)

    #% Compute the contour's local low-frequency curvature minima.
    minima_low_freq, minima_low_freq_ind = extremaPeaksCircDist(-1, 
                                                                cnt_ang_low_freq, 
                                                                edge_len_low_freq, 
                                                                cnt_chain_code_len)

    #% Compute the worm's skeleton.
    skeleton, cnt_widths = linearSkeleton(head_ind, 
                                          tail_ind, 
                                          minima_low_freq, 
                                          minima_low_freq_ind,
                                          maxima_low_freq, 
                                          maxima_low_freq_ind, 
                                          contour.copy(), 
                                          worm_seg_length, 
                                          cnt_chain_code_len)

    # The head must be in position 0
    assert head_ind == 0

    # Get the contour for each side.
    cnt_side1 = contour[:tail_ind + 1, :].copy()
    cnt_side2 = np.vstack([contour[0, :], contour[:tail_ind - 1:-1, :]])

    assert np.all(cnt_side1[0] == cnt_side2[0])
    assert np.all(cnt_side1[-1] == cnt_side2[-1])
    assert np.all(skeleton[-1] == cnt_side1[-1])
    assert np.all(skeleton[0] == np.round(cnt_side2[0]))

    return (skeleton, cnt_side1, cnt_side2, cnt_widths, '')


def orientWorm(skeleton, prev_skeleton, cnt_side1, cnt_side2, cnt_widths):
    if skeleton.size == 0:
        return skeleton, cnt_side1, cnt_side2, cnt_widths, np.float(0)

    # orient head tail with respect to hte previous worm
    if prev_skeleton.size > 0:
        #dist2prev_head = np.sum((skeleton[0:3,:]-prev_skeleton[0:3,:])**2)
        #dist2prev_tail = np.sum((skeleton[0:3,:]-prev_skeleton[-3:,:])**2)

        # if the skeleton is wrongly oriented switching it must decrease the
        # error by a lot.
        dist2prev_head = np.sum((skeleton - prev_skeleton)**2)
        dist2prev_tail = np.sum((skeleton - prev_skeleton[::-1, :])**2)
        if dist2prev_head > dist2prev_tail:
            # the skeleton is switched
            skeleton = skeleton[::-1, :]
            cnt_widths = cnt_widths[::-1]
            cnt_side1 = cnt_side1[::-1, :]
            cnt_side2 = cnt_side2[::-1, :]

    # make sure the contours are in the counter-clockwise direction
    # x1y2 - x2y1(http://mathworld.wolfram.com/PolygonArea.html)
    contour = np.vstack((cnt_side1, cnt_side2[::-1, :]))
    signed_area = np.sum(
        contour[:-1, 0] * contour[1:, 1] - contour[1:, 0] * contour[:-1, 1]) / 2
    if signed_area < 0:
        cnt_side1, cnt_side2 = cnt_side2, cnt_side1

    return skeleton, cnt_side1, cnt_side2, cnt_widths, np.abs(signed_area)


def resample_curve(curve, resampling_N=49, widths=np.zeros(0)):
    '''Resample curve to have resampling_N equidistant segments'''

    # calculate the cumulative length for each segment in the curve
    dx = np.diff(curve[:, 0])
    dy = np.diff(curve[:, 1])
    dr = np.sqrt(dx * dx + dy * dy)

    lengths = np.cumsum(dr)
    lengths = np.hstack((0, lengths))  # add the first point
    tot_length = lengths[-1]

    # Verify array lengths
    if len(lengths) < 2 or len(curve) < 2:
        return None, None, None

    fx = interp1d(lengths, curve[:, 0])
    fy = interp1d(lengths, curve[:, 1])

    subLengths = np.linspace(0 + np.finfo(float).eps, tot_length, resampling_N)

    # I add the epsilon because otherwise the interpolation will produce nan
    # for zero
    try:
        resampled_curve = np.zeros((resampling_N, 2))
        resampled_curve[:, 0] = fx(subLengths)
        resampled_curve[:, 1] = fy(subLengths)
        if widths.size > 0:
            fw = interp1d(lengths, widths)
            widths = fw(subLengths)
    except ValueError:
        resampled_curve = np.full((resampling_N, 2), np.nan)
        widths = np.full(resampling_N, np.nan)

    return resampled_curve, tot_length, widths


def smooth_curve(curve, window=5, pol_degree=3):
    '''smooth curves using the savgol_filter'''

    if curve.shape[0] < window:
        # nothing to do here return an empty array
        return np.full_like(curve, np.nan)

    # consider the case of one (widths) or two dimensions (skeletons, contours)
    if curve.ndim == 1:
        smoothed_curve = savgol_filter(curve, window, pol_degree)
    else:
        smoothed_curve = np.zeros_like(curve)
        for nn in range(curve.ndim):
            smoothed_curve[:, nn] = savgol_filter(
                curve[:, nn], window, pol_degree)

    return smoothed_curve


def resampleAll(skeleton, cnt_side1, cnt_side2, cnt_widths, resampling_N):
    '''I am only resample for the moment'''
    # resample data
    skeleton, ske_len, cnt_widths = resample_curve(
        skeleton, resampling_N, cnt_widths)
    cnt_side1, _, _ = resample_curve(cnt_side1, resampling_N)
    cnt_side2, _, _ = resample_curve(cnt_side2, resampling_N)

    #skeleton = smooth_curve(skeleton)
    #cnt_widths = smooth_curve(cnt_widths)
    #cnt_side1 = smooth_curve(cnt_side1)
    #cnt_side2 = smooth_curve(cnt_side2)

    return skeleton, ske_len, cnt_side1, cnt_side2, cnt_widths


def getSkeleton(worm_cnt, prev_skeleton=np.zeros(0), resampling_N=49,
                num_segments=24, head_angle_thresh=60, worm_mask=None):
    '''
    resampling_N -> The final number of points the skeleton, and each contour will have.
    num_segments -> number of segments used to calculate the skeleton curvature
        (or half the number of segments used for the contour curvature).
        Reduced for rounder objects and decreased for sharper organisms.

    head_angle_thresh -> the threshold to consider a peak on the curvature as the head or tail.

    worm_mask -> optional binary mask of the worm (same ROI coordinates as worm_cnt).
        When provided and contour2Skeleton fails (e.g. during an omega turn), a
        morphological-thinning fallback is attempted before returning empty arrays.
    '''
    n_output_param = 6  # number of expected output parameters

    if worm_cnt.size == 0:
        return (n_output_param) * [np.zeros(0)]

    assert isinstance(
        worm_cnt,
        np.ndarray) and worm_cnt.ndim == 2 and worm_cnt.shape[1] == 2

    # make sure the worm contour is float
    worm_cnt = worm_cnt.astype(np.float32)
    skeleton, cnt_side1, cnt_side2, cnt_widths, err_msg = \
        contour2Skeleton(worm_cnt, num_segments, head_angle_thresh)

    if skeleton.size == 0:
        if worm_mask is not None:
            fallback = _omega_skeleton_fallback(worm_mask, prev_skeleton, resampling_N)
            if fallback is not None:
                return fallback
        return (n_output_param) * [np.zeros(0)]

    # resample curves
    skeleton, ske_len, cnt_side1, cnt_side2, cnt_widths = \
        resampleAll(skeleton, cnt_side1, cnt_side2, cnt_widths, resampling_N)

    # orient skeleton with respect to the previous skeleton
    skeleton, cnt_side1, cnt_side2, cnt_widths, cnt_area = \
        orientWorm(skeleton, prev_skeleton, cnt_side1, cnt_side2, cnt_widths)

    output_data = (
        skeleton,
        ske_len,
        cnt_side1,
        cnt_side2,
        cnt_widths,
        cnt_area)
    assert len(output_data) == n_output_param
    return output_data
