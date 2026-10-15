# -*- coding: utf-8 -*-
# Copyright 2016-2025 The pyXem developers
#
# This file is part of pyXem.
#
# pyXem is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyXem is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyXem.  If not, see <http://www.gnu.org/licenses/>.

# Vendored from SpyDE (https://github.com/directelectron/spyde) at 68dfa52038,
# spyde/models/refine.py; see vendor note in __init__.py.

"""Stage-2 refinement: reduce a greedy low-threshold candidate list to real vectors.

Propose-then-refine (owner design): stage 1 over-detects at a LOW confidence
threshold (high recall); stage 2 here encodes the physical constraints to reject
false positives while keeping faint real peaks:
  - PERSISTENCE: real peaks recur in adjacent real-space scan positions.
  - OVERLAP MERGE: collapse double-detections of the same spot.
  - FRIEDEL (soft): peaks with a +g/-g partner about the centre score higher
    (a soft prior — amorphous/disordered legitimately breaks Friedel, so NOT a
    hard filter).
Each candidate gets a score combining model confidence + these signals; a final
threshold on the combined score keeps the real ones.
"""
from __future__ import annotations

import numpy as np


def merge_overlaps(peaks, min_sep=3.0):
    """Merge candidates closer than ``min_sep`` (same spot detected twice): keep the
    highest-confidence one, intensity-weighted position. peaks: (N,>=3) [y,x,score]."""
    peaks = np.asarray(peaks, np.float64)
    if len(peaks) <= 1:
        return peaks.astype(np.float32)
    order = np.argsort(-peaks[:, 2])
    kept = []
    used = np.zeros(len(peaks), bool)
    for oi in order:
        if used[oi]:
            continue
        p = peaks[oi]
        d = np.hypot(peaks[:, 0] - p[0], peaks[:, 1] - p[1])
        grp = (d <= min_sep) & (~used)
        used[grp] = True
        # score col2 may be a disk-mean INTENSITY (GPU batch path), which can be
        # 0 (or negative on background-subtracted data) for every group member ->
        # np.average(weights=...) raises. Fall back to an unweighted mean.
        w = np.clip(peaks[grp, 2], 0, None)
        if w.sum() <= 0:
            w = None
        y = np.average(peaks[grp, 0], weights=w)
        x = np.average(peaks[grp, 1], weights=w)
        kept.append([y, x, peaks[grp, 2].max()])
    return np.asarray(kept, np.float32)


def persistence_score(peaks, neighbor_peaks, tol=3.0):
    """Per-peak fraction of scan neighbours that contain a matching peak (0..1)."""
    peaks = np.asarray(peaks, np.float64)
    if len(peaks) == 0 or not neighbor_peaks:
        return np.zeros(len(peaks))
    out = np.zeros(len(peaks))
    for i, p in enumerate(peaks):
        hits = sum(
            1
            for nb in neighbor_peaks
            if len(nb)
            and np.min(
                np.hypot(np.asarray(nb)[:, 0] - p[0], np.asarray(nb)[:, 1] - p[1])
            )
            <= tol
        )
        out[i] = hits / len(neighbor_peaks)
    return out


def friedel_score(peaks, center, tol=3.0, min_radius=8.0):
    """Per-peak soft Friedel signal: 1 if a partner exists near -g about ``center``."""
    peaks = np.asarray(peaks, np.float64)
    center = np.asarray(center, np.float64)[:2]
    if len(peaks) == 0:
        return np.zeros(0)
    out = np.zeros(len(peaks))
    rel = peaks[:, :2] - center
    r = np.linalg.norm(rel, axis=1)
    for i, p in enumerate(peaks):
        if r[i] < min_radius:
            out[i] = 1.0  # central spots: don't penalise
            continue
        mirror = center - rel[i]
        d = np.hypot(peaks[:, 0] - mirror[0], peaks[:, 1] - mirror[1])
        out[i] = 1.0 if np.min(d) <= tol else 0.0
    return out


def refine(
    peaks,
    neighbor_peaks,
    center=None,
    frame_shape=None,
    min_sep=3.0,
    persist_tol=3.0,
    friedel_tol=3.0,
    w_conf=0.3,
    w_persist=0.9,
    w_friedel=0.3,
    keep_score=0.5,
    min_persist=None,
):
    """Reduce greedy candidates to real vectors. Returns (M,3) [y,x,score].

    Combined score = w_conf*conf + w_persist*persistence + w_friedel*friedel
    (conf is the candidate's own column-2 score, assumed ~0..1). Persistence is the
    dominant term (the strongest real/FP discriminator). Keep candidates whose
    combined score >= ``keep_score``. If ``min_persist`` is set, ALSO require at least
    that persistence fraction (a hard floor; e.g. 0.5 = seen in >=half the neighbours).
    """
    peaks = np.asarray(peaks, np.float32).reshape(-1, 3)
    if len(peaks) == 0:
        return peaks
    peaks = merge_overlaps(peaks, min_sep)
    pers = persistence_score(peaks, neighbor_peaks, persist_tol)
    if center is None and frame_shape is not None:
        # brightest peak near frame centre = direct beam
        mid = np.array(frame_shape[:2]) / 2.0
        near = (
            np.hypot(peaks[:, 0] - mid[0], peaks[:, 1] - mid[1]) < 0.2 * frame_shape[0]
        )
        center = (
            peaks[near][np.argmax(peaks[near, 2]), :2]
            if near.any()
            else peaks[np.argmax(peaks[:, 2]), :2]
        )
    fried = (
        friedel_score(peaks, center, friedel_tol)
        if center is not None
        else np.zeros(len(peaks))
    )
    conf = np.clip(peaks[:, 2] / (peaks[:, 2].max() + 1e-9), 0, 1)
    score = w_conf * conf + w_persist * pers + w_friedel * fried
    keep = score >= keep_score
    if min_persist is not None:
        keep &= pers >= min_persist
    return peaks[keep]
