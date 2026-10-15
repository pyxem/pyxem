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

# A numpy port of SpyDE's spyde/models/centre.py (SpyDE 68dfa52038), which
# computes the same thing in torch; see the note in __init__.py.

"""Where a detected disk's center is, and how well that is known.

The center is the network's own decode: the heat map's maximum pixel plus the
offset head. A model with a mask head, which predicts how much of each pixel
the disk's aperture covers, then moves that position to the centroid of the
predicted coverage (:func:`mask_centroid`): the aperture's center, however the
intensity is distributed inside it. The center networks (:mod:`.networks`)
take the same :func:`coverage_centroid` of their own coverage map.

Each detection also gets:

* ``confidence`` -- the heat-map peak (sigmoid), 0..1: how sure the network is
  that there is a disk here. A faded reflection scores low.
* ``width`` -- the spread (one standard deviation, working pixels) of the
  softmax of the heat-map logits over the disk. Turned into a positional
  uncertainty by :func:`positional_sigma` once the disk's counts are known.

Arithmetic is float32 throughout, as in the torch original.
"""

import math

import numpy as np
from scipy.ndimage import maximum_filter


def sigmoid(x):
    x = np.asarray(x, np.float32)
    return (np.float32(1) / (np.float32(1) + np.exp(-x))).astype(np.float32)


def coverage_centroid(
    coverage, rows, columns, center_y, center_x, radius, iterations=2
):
    """The centroid of a disk-coverage map (0..1) around each starting center.

    ``coverage`` is ``(M, S, S)``; ``rows`` and ``columns`` are the coordinates
    of its pixels, broadcastable to ``(M, S, 1)`` and ``(M, 1, S)``;
    ``center_y`` and ``center_x`` are ``(M,)`` starting centers in the same
    coordinates. The window is a soft disk (1 px ramp), so the centroid moves
    continuously with the start: the first pass reaches ``1.3 * radius +
    1.5``, the next the radius the coverage itself implies (area / pi) plus 2,
    centered on the previous centroid. A window with no coverage keeps its
    center. Returns the centers and the covered area, each ``(M,)``.
    """
    f32 = np.float32
    center_y = np.asarray(center_y, f32).copy()
    center_x = np.asarray(center_x, f32).copy()
    window = np.full_like(center_y, f32(1.3 * radius + 1.5))
    area = np.zeros_like(center_y)
    for _ in range(iterations):
        distance = np.sqrt(
            (rows - center_y[:, None, None]) ** 2
            + (columns - center_x[:, None, None]) ** 2
        )
        weight = np.clip(window[:, None, None] - distance + f32(0.5), 0, 1) * coverage
        area = weight.sum((1, 2), dtype=f32)
        found = area > f32(1e-3)
        safe = np.maximum(area, f32(1e-6))
        center_y = np.where(
            found, (weight * rows).sum((1, 2), dtype=f32) / safe, center_y
        )
        center_x = np.where(
            found, (weight * columns).sum((1, 2), dtype=f32) / safe, center_x
        )
        window = np.sqrt(area / f32(math.pi)) + f32(2.0)
    return center_y.astype(f32), center_x.astype(f32), area.astype(f32)


def mask_centroid(mask_logits, batch, y, x, radius, iterations=2):
    """:func:`coverage_centroid` of a mask head's ``(B, 1, H, W)`` output
    around each seed ``(batch, y, x)`` (working px), ``radius`` the expected
    disk radius. Returns the centered ``(y, x)`` and the covered area, each
    ``(M,)``."""
    coverage = sigmoid(mask_logits)
    _, _, height, width = coverage.shape
    half = int(np.ceil(1.6 * radius + 3))
    steps = np.arange(-half, half + 1, dtype=np.float32)
    rows = np.round(y)[:, None] + steps[None]
    columns = np.round(x)[:, None] + steps[None]
    inside = ((rows >= 0) & (rows < height))[:, :, None] & (
        (columns >= 0) & (columns < width)
    )[:, None, :]
    patch = (
        coverage[
            batch[:, None, None],
            0,
            np.clip(rows.astype(np.int64), 0, height - 1)[:, :, None],
            np.clip(columns.astype(np.int64), 0, width - 1)[:, None, :],
        ]
        * inside
    )
    return coverage_centroid(
        patch, rows[:, :, None], columns[:, None, :], y, x, radius, iterations
    )


def _center_on_mask(mask_logits, batch, y, x, radius):
    """Move each offset-head position to its mask centroid, unless the mask
    there is too small to be a disk (< 15 % of the expected area) or its
    centroid lands more than 1.5 px away (a neighbor's mask, or no mask)."""
    center_y, center_x, area = mask_centroid(mask_logits, batch, y, x, radius)
    trusted = (np.hypot(center_y - y, center_x - x) < 1.5) & (
        area > 0.15 * np.pi * radius**2
    )
    return np.where(trusted, center_y, y), np.where(trusted, center_x, x)


def decode_batch_centers(
    hm_logits,
    off,
    thresh=0.3,
    min_distance=3,
    radius=3.0,
    mask_logits=None,
    mask_radius=4.5,
):
    """Peaks of a ``(B, 1, H, W)`` heat-map batch, as ``(M, 5)``
    ``[batch, y, x, confidence, width]`` in working pixels.

    A peak is a pixel equal to the maximum of its ``2 * min_distance + 1``
    neighborhood and at least ``thresh``; its position is the pixel plus the
    offset head. ``radius`` (working px) is the window the response width is
    taken over -- the disk's own radius. ``mask_logits`` (a mask-head model's
    third output) moves each position to its mask centroid, with
    ``mask_radius`` the expected disk radius in working px.
    """
    hm_logits = np.asarray(hm_logits, np.float32)
    hm = sigmoid(hm_logits)
    _, _, height, width = hm.shape
    size = 2 * min_distance + 1
    pooled = maximum_filter(hm, size=(1, 1, size, size), mode="constant", cval=-np.inf)
    batch, ys, xs = np.nonzero(((hm == pooled) & (hm >= thresh))[:, 0])
    if len(batch) == 0:
        return np.zeros((0, 5), np.float32)
    confidence = hm[batch, 0, ys, xs]

    half = max(1, int(math.ceil(radius)))
    steps = np.arange(-half, half + 1)
    step_y, step_x = np.meshgrid(steps, steps, indexing="ij")
    inside = (step_y * step_y + step_x * step_x) <= radius * radius + 1e-6
    grid_y = np.clip(ys[:, None, None] + step_y, 0, height - 1)
    grid_x = np.clip(xs[:, None, None] + step_x, 0, width - 1)
    logits = hm_logits[batch[:, None, None], 0, grid_y, grid_x]
    valid = (
        inside
        & (grid_y == ys[:, None, None] + step_y)
        & (grid_x == xs[:, None, None] + step_x)
    )
    logits = np.where(valid, logits, np.float32(-np.inf)).reshape(len(batch), -1)
    weights = np.exp(logits - logits.max(1, keepdims=True))
    weights = (weights / weights.sum(1, keepdims=True)).reshape(grid_y.shape)
    grid_y = grid_y.astype(np.float32)
    grid_x = grid_x.astype(np.float32)
    mean_y = (weights * grid_y).sum((1, 2))
    mean_x = (weights * grid_x).sum((1, 2))
    variance = 0.5 * (
        (weights * (grid_y - mean_y[:, None, None]) ** 2).sum((1, 2))
        + (weights * (grid_x - mean_x[:, None, None]) ** 2).sum((1, 2))
    )
    spread = np.sqrt(np.maximum(variance, 0))
    off = np.asarray(off, np.float32)
    y = ys.astype(np.float32) + off[batch, 0, ys, xs]
    x = xs.astype(np.float32) + off[batch, 1, ys, xs]
    if mask_logits is not None:
        y, x = _center_on_mask(
            np.asarray(mask_logits, np.float32), batch, y, x, mask_radius
        )
    return np.stack([batch.astype(np.float32), y, x, confidence, spread], 1).astype(
        np.float32
    )


def positional_sigma(width_px, signal, background_per_px, pixel=1.0):
    """One standard deviation of a center, in detector pixels.

    The localization-precision formula for a spot of standard deviation ``s``
    measured with ``N`` signal counts over a background of ``b**2`` counts per
    pixel, pixel size ``a`` (Thompson, Larson & Webb, Biophys. J. 82, 2775
    (2002))::

        sigma**2 = (s**2 + a**2 / 12) / N  +  8 pi s**4 b**2 / (a**2 N**2)

    Here ``s`` is the width of the network's response to the disk (``width``
    from :func:`decode_batch_centers`, in detector pixels), ``N`` the disk's
    background-subtracted counts and ``b**2`` its local background level
    (Poisson, so the variance equals the mean). A bright, sharp disk gets a
    small sigma; a faint or diffuse one a large sigma. NaN where the disk has
    no net signal.
    """
    s = np.maximum(np.asarray(width_px, np.float64), 0.5)
    n = np.asarray(signal, np.float64)
    b2 = np.clip(np.asarray(background_per_px, np.float64), 0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = (s * s + pixel * pixel / 12.0) / n + 8 * np.pi * s**4 * b2 / (
            pixel * pixel * n * n
        )
        sigma = np.sqrt(variance)
    return np.where(n > 0, sigma, np.nan).astype(np.float32)
