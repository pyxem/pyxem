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
# spyde/models/centre_refine.py; see vendor note in __init__.py.

"""The center steps: re-place each detected disk on the raw frame at native resolution.

The detector finds disks on a rescaled, locally normalized working image, so its
center is only as good as that image. Two steps look again at the raw counts:

1. **Refine** (:data:`CENTER_MODEL`, the bundled center network F5): each disk's
   own window, from the detector's center.
2. **Friedel partner** (:data:`FRIEDEL_MODEL`, P4), on top of the refined
   centers and optional: each disk's window together with the window at its
   Friedel mirror point ``2 x beam - p``, which shows the other dynamical fill
   of the same reflection. It works on spots of 6-16 px radius and leaves
   others where the refine step put them.

Both are network refiners (:mod:`pyxem.utils._neural_spots.center_network`) applied by
:func:`refine_centers`, which cuts the windows, hands a chunk's windows to the
network as one batch, and moves each disk to the refined center. A refiner is
any object with

    crop_half_width(spot_radius) -> int
    refine(crops, centers, spot_radius) -> (centers, sigma)

``crops`` is ``(M, S, S)`` float32 raw counts with ``S = 2 * crop_half_width + 1``,
cut around each center's nearest pixel; ``centers`` is ``(M, 2)`` ``[y, x]`` in
crop pixels. It returns the refined ``(M, 2)`` centers in the same crop pixels
and an ``(M,)`` positional uncertainty in pixels, or ``None`` when it has none.
A refiner declines a disk by returning NaN for it. It may carry
``min_spot_radius`` / ``max_spot_radius``, outside which every center is kept,
and ``needs_partner``, which makes :func:`refine_centers` also cut each disk's
Friedel-mirror window.

The step keeps the incoming center for a declined disk and for any center that
moves more than half a spot radius: a refiner that wandered onto a neighbor is
worse than the center it replaced.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Protocol, Sequence

import numpy as np

log = logging.getLogger(__name__)

#: The refine step's network, and the Friedel-partner step's (registry ids).
CENTER_MODEL = "center-fast-f5-v1"
FRIEDEL_MODEL = "center-partner-p4-v1"

# The largest move a step accepts, as a fraction of the spot radius.
MAX_SHIFT_FRACTION = 0.5

# Below this spot radius (native px) a refiner keeps the incoming centers unless
# it says otherwise: on disks this small the window holds too few pixels to beat
# the detector (on SPED-Ag, R ~ 3 px, refining raised the in-grain speckle from
# 0.036 to 0.047).
DEFAULT_MIN_SPOT_RADIUS = 5.0

# Windows refined per batch. A chunk can hold 10^5 disks; this bounds the
# window stack (and the network's activations) whatever the chunk size.
DEFAULT_BATCH = 4096


class CenterRefiner(Protocol):
    def crop_half_width(self, spot_radius: float) -> int: ...

    def refine(
        self, crops: np.ndarray, centers: np.ndarray, spot_radius: float
    ) -> tuple[np.ndarray, Optional[np.ndarray]]: ...


def masked_median(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``np.median(values[i][mask[i]])`` for every row in one sort (0 for an
    empty row). A frame can hold hundreds of disks, and a Python loop of
    ``np.median`` calls cost ten times the detector network itself."""
    count = mask.sum(1)
    ordered = np.sort(np.where(mask, values, np.inf), axis=1)
    rows = np.arange(len(values))
    low = ordered[rows, np.maximum(count - 1, 0) // 2]
    high = ordered[rows, count // 2 - (count == 0)]
    return np.where(count > 0, 0.5 * (low.astype(np.float64) + high), 0.0)


def _window_indices(centers: np.ndarray, half_width: int, shape):
    """Rows, columns and inside-mask of the ``2 * half_width + 1`` px windows
    around each center's nearest pixel, and the centers in window pixels."""
    height, width = shape
    nearest = np.rint(centers).astype(np.intp)
    nearest[:, 0] = np.clip(nearest[:, 0], 0, height - 1)
    nearest[:, 1] = np.clip(nearest[:, 1], 0, width - 1)
    offsets = np.arange(-half_width, half_width + 1)
    rows = nearest[:, 0, None] + offsets
    columns = nearest[:, 1, None] + offsets
    inside = ((rows >= 0) & (rows < height))[:, :, None] & (
        (columns >= 0) & (columns < width)
    )[:, None, :]
    return (
        np.clip(rows, 0, height - 1),
        np.clip(columns, 0, width - 1),
        inside,
        centers - (nearest - half_width),
    )


def extract_crops(frame: np.ndarray, centers: np.ndarray, half_width: int):
    """Windows of ``2 * half_width + 1`` px around each center's nearest pixel,
    zero outside the frame. Returns ``(crops, local)``: the ``(M, S, S)`` float32
    stack and the centers in crop pixels."""
    frame = np.asarray(frame, np.float32)
    centers = np.asarray(centers, np.float64).reshape(-1, 2)
    size = 2 * half_width + 1
    if len(centers) == 0:
        return np.zeros((0, size, size), np.float32), np.zeros((0, 2))
    rows, columns, inside, local = _window_indices(centers, half_width, frame.shape)
    # Gather only the windows. Padding the frame instead copies all of it for
    # every frame, which on 512 x 512 frames cost four times the network.
    crops = frame[rows[:, :, None], columns[:, None, :]]
    return np.where(inside, crops, np.float32(0)), local


def _extract_crops_on_device(
    stack, frame_of: np.ndarray, centers: np.ndarray, half_width: int
):
    """:func:`extract_crops` for a group of frames at once, gathered on the GPU
    from ``stack`` (the group's frames already moved there): the same windows."""
    import torch

    device = stack.device
    rows, columns, inside, local = _window_indices(centers, half_width, stack.shape[1:])
    crops = stack[
        torch.as_tensor(frame_of, device=device)[:, None, None],
        torch.as_tensor(rows, device=device)[:, :, None],
        torch.as_tensor(columns, device=device)[:, None, :],
    ]
    return crops * torch.as_tensor(inside, device=device), local


#: On a GPU, frames are moved to the device a group at a time; this bounds a
#: group's bytes (a few 4096 x 4096 frames, hundreds of 512 x 512 ones).
DEVICE_FRAME_BYTES = 256 * 2**20


def frame_beams(positions, frame_shape, spot_radius) -> np.ndarray:
    """Per frame, the center of Friedel symmetry of its ``(n, 2)`` centers:
    start at the disk nearest the mean of all disks, pair each disk with the
    one nearest its mirror (within ``max(0.4 R, 2 px)``), re-center on the mean
    pair midpoint; three times. A frame with fewer than 4 disks or 2 pairs takes
    the median of the other frames' beams (descanned data), or the frame center
    when none has one — so a lone frame in the live preview can fall back
    differently from the same frame inside a chunk."""
    tolerance = max(0.4 * float(spot_radius), 2.0)
    beams = np.full((len(positions), 2), np.nan)
    for number, points in enumerate(positions):
        points = np.asarray(points, np.float64).reshape(-1, 2)
        if len(points) < 4:
            continue
        center = points[np.argmin(np.hypot(*(points - points.mean(0)).T))]
        found = False
        for _ in range(3):
            mirror = 2 * center - points
            distance = np.hypot(
                mirror[:, None, 0] - points[None, :, 0],
                mirror[:, None, 1] - points[None, :, 1],
            )
            np.fill_diagonal(distance, np.inf)
            partner = distance.argmin(1)
            index = np.arange(len(points))
            paired = (distance[index, partner] < tolerance) & (index < partner)
            if paired.sum() < 2:
                found = False
                break
            center = 0.5 * (points[paired] + points[partner[paired]]).mean(0)
            found = True
        if found:
            beams[number] = center
    if np.isfinite(beams).any():
        fallback = np.nanmedian(beams, 0)
    else:
        fallback = np.asarray(frame_shape, np.float64) / 2
    return np.where(np.isfinite(beams), beams, fallback)


def refine_centers(
    frames: Sequence[np.ndarray],
    positions: Sequence[np.ndarray],
    spot_radius: float,
    refiner: CenterRefiner,
    batch_size: int = DEFAULT_BATCH,
):
    """Refine every frame's ``(n, 2)`` ``[y, x]`` detections in ``frames``.

    Crops are cut and refined a batch at a time across frames, so a chunk's
    disks go through the refiner together without ever holding all their crops
    at once. Returns ``(positions, sigmas)``: the refined ``(n, 2)`` float32
    positions per frame, and per frame the refiner's ``(n,)`` uncertainty
    (NaN where the detection was kept) or ``None`` when the refiner reports
    none. Each disk's result depends only on its own crop, so refining one
    frame alone gives the same answer as refining it inside a chunk."""
    radius = float(spot_radius)
    refined = [np.asarray(p, np.float32).reshape(-1, 2).copy() for p in positions]
    if not (
        float(getattr(refiner, "min_spot_radius", 0.0))
        <= radius
        <= float(getattr(refiner, "max_spot_radius", math.inf))
    ):
        return refined, None
    half_width = int(refiner.crop_half_width(radius))
    sigmas: list = [np.full(len(p), np.nan, np.float32) for p in refined]
    has_sigma = True
    needs_partner = bool(getattr(refiner, "needs_partner", False))
    beams = (
        frame_beams(refined, np.asarray(frames[0]).shape, radius)
        if needs_partner
        else None
    )
    device = _gpu_device(refiner)

    def _cut(pending, points_of, stack):
        """The windows of every pending frame's ``points_of(frame_index)``;
        on the GPU from ``stack``, the pending frames moved there once."""
        if stack is None:
            cut = [
                extract_crops(frames[i], points_of(i), half_width) for i, _ in pending
            ]
            return (
                np.concatenate([c for c, _ in cut]),
                np.concatenate([l for _, l in cut]),
            )
        frame_of = np.concatenate(
            [np.full(len(points), n) for n, (_, points) in enumerate(pending)]
        )
        centers = np.concatenate([points_of(i) for i, _ in pending]).astype(np.float64)
        return _extract_crops_on_device(stack, frame_of, centers, half_width)

    def _flush(pending):
        nonlocal has_sigma
        stack = None
        if device is not None:
            import torch

            from .device_lock import accelerator_lock

            with accelerator_lock(device):
                stack = torch.stack(
                    [
                        torch.as_tensor(
                            np.asarray(frames[i], np.float32), device=device
                        )
                        for i, _ in pending
                    ]
                )
        crops, local = _cut(pending, lambda i: refined[i].astype(np.float64), stack)
        if needs_partner:
            # the window at each disk's Friedel mirror point, 2 x beam - p
            mirror_crops, mirror_local = _cut(
                pending, lambda i: 2 * beams[i] - refined[i].astype(np.float64), stack
            )
            new_local, sigma = refiner.refine(
                crops, local, radius, mirror_crops, mirror_local
            )
        else:
            new_local, sigma = refiner.refine(crops, local, radius)
        new_local = np.asarray(new_local, np.float64).reshape(-1, 2)
        shift = np.hypot(*(new_local - local).T)
        accepted = np.isfinite(new_local).all(1) & (
            shift <= MAX_SHIFT_FRACTION * radius
        )
        if sigma is None:
            has_sigma = False
        start = 0
        for frame_index, points in pending:
            stop = start + len(points)
            keep = accepted[start:stop]
            moved = refined[frame_index] + (new_local[start:stop] - local[start:stop])
            refined[frame_index][keep] = moved[keep]
            if sigma is not None:
                sigmas[frame_index][keep] = np.asarray(sigma[start:stop])[keep]
            start = stop

    pending, pending_count, pending_bytes = [], 0, 0
    for frame_index, (frame, points) in enumerate(zip(frames, refined)):
        if len(points) == 0:
            continue
        # the incoming positions, before any of this batch's refinement
        pending.append((frame_index, points.copy()))
        pending_count += len(points)
        pending_bytes += 4 * np.asarray(frame).size
        if pending_count >= batch_size or (
            device is not None and pending_bytes >= DEVICE_FRAME_BYTES
        ):
            _flush(pending)
            pending, pending_count, pending_bytes = [], 0, 0
    if pending:
        _flush(pending)
    return refined, (sigmas if has_sigma else None)


def _gpu_device(refiner):
    """The refiner's torch device when windows should be cut there (a GPU), else None."""
    device = getattr(refiner, "device", None)
    try:
        import torch

        device = torch.device(device) if device is not None else None
    except (ImportError, TypeError, RuntimeError):
        return None
    return device if device is not None and device.type in ("cuda", "mps") else None


def load_step(model_id: str, device=None):
    """The network for one of the center steps, from the registry; ``None``
    with a warning when it cannot load, so detection still completes with the
    centers it has."""
    from . import weights as registry

    try:
        return registry.get_refiner(model_id, device)
    except Exception as error:
        log.warning(
            "[models] center network %r unavailable (%s); keeping the "
            "centers it would have refined",
            model_id,
            error,
        )
        return None
