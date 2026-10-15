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

# SpyDE's spyde/models/infer.py (SpyDE 68dfa52038) with the network run by
# ONNX Runtime and the preprocessing in numpy; see the note in __init__.py.

"""Parameter-free inference: rescale -> local normalization -> network ->
decode -> map back.

The disk size is estimated from the pattern and the pattern resampled so its
disks match the size the network was trained on; predicted positions are
mapped back to the original pattern's pixels.
"""

import os

import numpy as np

from .center import decode_batch_centers, sigmoid
from .preprocess import (
    CANONICAL_DIAMETER,
    estimate_disk_diameter,
    normalize_input,
    scale_to_canonical,
)

#: Local-normalization high-pass scale (px). Very diffuse backgrounds (beam
#: stop, amorphous halo) need a smaller value, set by :func:`calibrate`.
DEFAULT_BG_SIGMA = 12.0


def _mask_radius(work_diam):
    """Expected disk radius at working resolution, for the mask-centroid decode."""
    if np.isfinite(work_diam) and work_diam > 0:
        return 0.5 * float(work_diam)
    return 0.5 * CANONICAL_DIAMETER


def _pad_to_multiple(normalized, levels):
    """Reflect-pad H, W up to a multiple of 2**levels so the U-Net
    pool/upsample line up."""
    height, width = normalized.shape
    multiple = 2**levels
    pad_rows = (multiple - height % multiple) % multiple
    pad_columns = (multiple - width % multiple) % multiple
    if pad_rows or pad_columns:
        normalized = np.pad(
            normalized, ((0, pad_rows), (0, pad_columns)), mode="reflect"
        )
    return normalized


def _estimate_work_diam(frame, factor, spot_diameter):
    """Disk diameter at working (post-scale) resolution -- drives the
    background-subtraction scale and the suppression window for big disks.
    The autocorrelation estimator saturates on big disks, so iterate with a
    size-adaptive high-pass, then correct its measured ~0.83 undershoot."""
    if spot_diameter:
        return float(spot_diameter) * factor
    diameter = estimate_disk_diameter(frame)
    if np.isfinite(diameter) and diameter > 20:
        for _ in range(3):
            refined = estimate_disk_diameter(frame, hp_sigma=max(20.0, diameter))
            if abs(refined - diameter) < 1:
                diameter = refined
                break
            diameter = max(diameter, refined)
        diameter = diameter / 0.83
    return diameter * factor


def _big_disk_params(min_distance, bg_sigma, work_diam):
    """The suppression window and background sigma for the working disk size.

    Canonical-size data (work_diam <= 20) is untouched. On big disks a fixed
    window fires a cloud of detections across one broad disk, and a fixed
    12 px high-pass carves the flat interior into a ring; an explicitly passed
    ``bg_sigma`` is respected."""
    big = np.isfinite(work_diam) and work_diam > 20
    if big:
        min_distance = max(min_distance, int(round(work_diam * 0.55)))
    if bg_sigma is None:
        bg_sigma = max(DEFAULT_BG_SIGMA, 0.6 * work_diam) if big else DEFAULT_BG_SIGMA
    return min_distance, float(bg_sigma)


def auto_bg_sigma(model, work, candidates=(4, 5, 6, 8, 10, 12, 16, 20)):
    """The local-normalization ``bg_sigma`` that maximizes the network's
    heat-map confidence. Returns ``(best_bg_sigma, best_confidence)``."""
    best_confidence, best_sigma = -1.0, DEFAULT_BG_SIGMA
    for sigma in candidates:
        normalized = _pad_to_multiple(
            normalize_input(
                work, local=True, bg_sigma=float(sigma), mad_floor=model.mad_floor
            ),
            model.levels,
        )
        heatmap = model.run(normalized[None, None].astype(np.float32))[0]
        confidence = float(sigmoid(heatmap).max())
        if confidence > best_confidence:
            best_confidence, best_sigma = confidence, float(sigma)
    return best_sigma, best_confidence


def _to_frame_coordinates(peaks, frame_shape, factor):
    """Map peak ``(y, x)`` from the rescaled working frame back to the frame.

    The rescale aligns the first and last pixel centers, so working pixel
    ``i`` sits at frame position ``i * (n - 1) / (out - 1)`` with ``out =
    round(n * factor)``, not at ``i / factor``."""
    if factor == 1.0 or len(peaks) == 0:
        return peaks
    peaks = peaks.copy()
    for axis, n in enumerate(frame_shape[:2]):
        out = int(round(n * factor))
        peaks[:, axis] = peaks[:, axis] * ((n - 1) / (out - 1))
    return peaks


def _width_radius(work_diam, min_distance):
    """Window the response width is taken over (working px): the disk's
    radius, but never past the suppression distance."""
    radius = (
        0.5 * float(work_diam)
        if np.isfinite(work_diam) and work_diam > 0
        else float(min_distance)
    )
    return float(max(1.5, min(radius, min_distance)))


def _frame_rows(rows, frame_shape, factor, with_width):
    """``decode_batch_centers`` rows of one frame -> ``[y, x, score(, width)]``
    in original-frame pixels."""
    columns = [1, 2, 3, 4] if with_width else [1, 2, 3]
    out = _to_frame_coordinates(
        rows[:, columns].astype(np.float32, copy=True), frame_shape, factor
    )
    if with_width and len(out) and factor != 1.0:
        # a width, not a position: the end-pixel correction is negligible
        out[:, 3] = out[:, 3] / factor
    return out


def _neural_sub_batch_size():
    """Patterns per network run (``PYXEM_NEURAL_BATCH``, default 64): bounds
    the activation memory without changing the result."""
    try:
        size = int(os.environ.get("PYXEM_NEURAL_BATCH", "64"))
    except ValueError:
        size = 64
    return max(1, size)


def _build_input_stack(frames, factor, bg_sigma, levels, auto_scale, mad_floor=None):
    """``(N, H, W)`` frames -> the padded ``(N, 1, H', W')`` network input."""
    normalized = []
    for frame in frames:
        if auto_scale and factor != 1.0:
            from scipy.ndimage import zoom

            frame = zoom(frame, factor, order=1)
        normalized.append(
            _pad_to_multiple(
                normalize_input(
                    frame, local=True, bg_sigma=bg_sigma, mad_floor=mad_floor
                ),
                levels,
            )
        )
    return np.stack(normalized, 0)[:, None].astype(np.float32)


def detect_batch(
    model,
    frames,
    thresh=0.3,
    min_distance=4,
    auto_scale=True,
    bg_sigma=None,
    spot_diameter=None,
    with_width=False,
):
    """Detect spots in a stack of frames.

    Returns a list of ``(N_i, 3)`` ``[y, x, score]`` arrays (``(N_i, 4)``
    ``[y, x, score, width]`` with ``with_width``) in original frame
    coordinates. Every frame in a scan has the same physical disk size, so
    the rescale factor, suppression window and background sigma are resolved
    once from ``frames[0]``; the network runs in sub-batches of
    ``PYXEM_NEURAL_BATCH`` frames.
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim == 2:
        frames = frames[None]
    if len(frames) == 0:
        return []
    factor = 1.0
    if auto_scale:
        _, factor = scale_to_canonical(frames[0], diameter=spot_diameter)
    md = max(2, int(round(min_distance * factor)))
    work_diam = _estimate_work_diam(frames[0], factor, spot_diameter)
    md, bg_sigma = _big_disk_params(md, bg_sigma, work_diam)
    radius = _width_radius(work_diam, md)
    mask_radius = _mask_radius(work_diam)

    per_frame = []
    size = _neural_sub_batch_size()
    for start in range(0, len(frames), size):
        chunk = frames[start : start + size]
        stack = _build_input_stack(
            chunk, factor, bg_sigma, model.levels, auto_scale, mad_floor=model.mad_floor
        )
        outputs = model.run(stack)
        rows = decode_batch_centers(
            outputs[0],
            outputs[1],
            thresh=thresh,
            min_distance=md,
            radius=radius,
            mask_logits=outputs[2] if len(outputs) > 2 else None,
            mask_radius=mask_radius,
        )
        batch = rows[:, 0].astype(np.int64)
        for index in range(len(chunk)):
            per_frame.append(
                _frame_rows(rows[batch == index], frames.shape[1:], factor, with_width)
            )
    return per_frame


def detect(model, frame, **kwargs):
    """:func:`detect_batch` of one frame."""
    return detect_batch(model, np.asarray(frame, np.float32)[None], **kwargs)[0]


def calibrate(model, sample_frames, tune_threshold=True, spot_diameter=None):
    """Tune the dataset-dependent settings once on a few representative frames.

    Picks the ``bg_sigma`` that maximizes the network's confidence (median
    over the sample) and, when even the best confidence is weak (faint data),
    lowers the threshold below the model's default. Returns a dict with
    ``bg_sigma``, ``thresh``, ``scale_factor`` and ``confidence``.
    """
    frames = [np.asarray(f, np.float32) for f in sample_frames]
    if not frames:
        return {"bg_sigma": DEFAULT_BG_SIGMA, "thresh": 0.3, "scale_factor": 1.0}
    _, factor = scale_to_canonical(frames[0], diameter=spot_diameter)
    candidates = [4, 5, 6, 8, 10, 12, 16, 20]
    work_diam = _estimate_work_diam(frames[0], factor, spot_diameter)
    if np.isfinite(work_diam) and work_diam > 20:
        candidates += [round(r * work_diam, 1) for r in (0.5, 0.6, 0.7)]
    sigmas, confidences = [], []
    for frame in frames:
        if factor != 1.0:
            from scipy.ndimage import zoom

            frame = zoom(frame, factor, order=1)
        sigma, confidence = auto_bg_sigma(model, frame, candidates=tuple(candidates))
        sigmas.append(sigma)
        confidences.append(confidence)
    confidence = float(np.median(confidences))
    default = float(model.default_threshold)
    thresh = default
    if tune_threshold and confidence < 0.45:
        thresh = float(np.clip(0.5 * confidence, 0.1, default))
    return {
        "bg_sigma": float(np.median(sigmas)),
        "thresh": thresh,
        "scale_factor": float(factor),
        "confidence": confidence,
    }
