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
# spyde/models/centre.py; see vendor note in __init__.py.

"""Where a detected disk's center is, and how well that is known.

The center is the network's own decode: the heatmap's maximum pixel plus the
offset head, exactly the vendored ``decode.decode_batch``. A model with a mask
head, which predicts how much of each pixel the disk's aperture covers, then
moves that position to the centroid of the predicted coverage
(:func:`mask_centroid`): the aperture's center, however the intensity is
distributed inside it. The center networks (:mod:`.center_network`) take the
same :func:`coverage_centroid` of their own coverage map.

Each detection also gets:

* ``confidence`` — the heatmap peak (sigmoid), 0..1: how sure the network is that
  there is a disk here. A faded reflection scores low.
* ``width`` — the spread (one standard deviation, working pixels) of the
  softmax of the heatmap logits over the disk. Turned into a positional
  uncertainty by :func:`positional_sigma` once the disk's counts are known.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def coverage_centroid(
    coverage, rows, columns, center_y, center_x, radius: float, iterations: int = 2
):
    """The centroid of a disk-coverage map (0..1) around each starting center.

    ``coverage`` is ``(M, S, S)``; ``rows`` and ``columns`` are the coordinates
    of its pixels, broadcastable to ``(M, S, 1)`` and ``(M, 1, S)``; ``center_y``
    and ``center_x`` are ``(M,)`` starting centers in the same coordinates. The
    window is a soft disk (1 px ramp), so the centroid moves continuously with
    the start: the first pass reaches ``1.3 * radius + 1.5``, the next the
    radius the coverage itself implies (area / pi) plus 2, centered on the
    previous centroid. A window with no coverage keeps its center. Returns the
    centers and the covered area, each ``(M,)``."""
    window = torch.full_like(center_y, 1.3 * radius + 1.5)
    area = torch.zeros_like(center_y)
    for _ in range(iterations):
        distance = torch.sqrt(
            (rows - center_y[:, None, None]) ** 2
            + (columns - center_x[:, None, None]) ** 2
        )
        weight = (window[:, None, None] - distance + 0.5).clamp(0, 1) * coverage
        area = weight.sum((1, 2))
        found = area > 1e-3
        center_y = torch.where(
            found, (weight * rows).sum((1, 2)) / area.clamp_min(1e-6), center_y
        )
        center_x = torch.where(
            found, (weight * columns).sum((1, 2)) / area.clamp_min(1e-6), center_x
        )
        window = torch.sqrt(area / math.pi) + 2.0
    return center_y, center_x, area


def mask_centroid(mask_logits, batch, y, x, radius, iterations=2):
    """:func:`coverage_centroid` of a mask head's ``(B, 1, H, W)`` output
    around each seed ``(batch, y, x)`` (working px), ``radius`` the expected
    disk radius. Returns the centered ``(y, x)`` and the covered area, each ``(M,)``."""
    coverage = torch.sigmoid(mask_logits)
    _, _, height, width = coverage.shape
    half = int(np.ceil(1.6 * radius + 3))
    steps = torch.arange(-half, half + 1, device=coverage.device, dtype=torch.float32)
    rows = y.round()[:, None] + steps[None]  # (M, S)
    columns = x.round()[:, None] + steps[None]
    inside = ((rows >= 0) & (rows < height))[:, :, None] & (
        (columns >= 0) & (columns < width)
    )[:, None, :]
    patch = (
        coverage[
            batch[:, None, None],
            0,
            rows.long().clamp(0, height - 1)[:, :, None],
            columns.long().clamp(0, width - 1)[:, None, :],
        ]
        * inside
    )
    return coverage_centroid(
        patch,
        rows[:, :, None],
        columns[:, None, :],
        y.clone(),
        x.clone(),
        radius,
        iterations,
    )


def _center_on_mask(mask_logits, batch, y, x, radius):
    """Move each offset-head position to its mask centroid, unless the mask there is
    too small to be a disk (< 15 % of the expected area) or its centroid lands more
    than 1.5 px away (a neighbor's mask, or no mask at all)."""
    center_y, center_x, area = mask_centroid(mask_logits, batch, y, x, radius)
    trusted = (torch.hypot(center_y - y, center_x - x) < 1.5) & (
        area > 0.15 * np.pi * radius**2
    )
    return torch.where(trusted, center_y, y), torch.where(trusted, center_x, x)


@torch.no_grad()
def decode_batch_centers(
    hm_logits,
    off,
    thresh=0.3,
    min_distance=3,
    radius=3.0,
    mask_logits=None,
    mask_radius=4.5,
):
    """Peaks of a ``(B,1,H,W)`` heatmap batch, as ``(M,5)``
    ``[batch, y, x, confidence, width]`` in working pixels.

    The peaks and their positions are exactly ``decode.decode_batch``'s (same
    max-pool, same threshold, argmax pixel + offset head). ``radius`` (working
    px) is the window the response width is taken over — the disk's own radius.
    ``mask_logits`` (a mask-head model's third output) moves each position to
    its mask centroid (:func:`_center_on_mask`), with ``mask_radius`` the
    expected disk radius in working px."""
    hm = torch.sigmoid(hm_logits)
    B, _, H, W = hm.shape
    k = 2 * min_distance + 1
    pooled = F.max_pool2d(hm, k, stride=1, padding=min_distance)
    idx = ((hm == pooled) & (hm >= thresh)).squeeze(1).nonzero(as_tuple=False)
    if idx.numel() == 0:
        return torch.zeros((0, 5), device=hm.device)
    b, ys, xs = idx[:, 0], idx[:, 1], idx[:, 2]
    confidence = hm[b, 0, ys, xs]

    half = max(1, int(math.ceil(radius)))
    d = torch.arange(-half, half + 1, device=hm.device)
    dy, dx = torch.meshgrid(d, d, indexing="ij")
    inside = (dy * dy + dx * dx) <= radius * radius + 1e-6
    gy = (ys[:, None, None] + dy).clamp(0, H - 1)
    gx = (xs[:, None, None] + dx).clamp(0, W - 1)
    logits = hm_logits[b[:, None, None], 0, gy, gx]
    logits = torch.where(
        inside & (gy == ys[:, None, None] + dy) & (gx == xs[:, None, None] + dx),
        logits,
        torch.full_like(logits, -float("inf")),
    )
    w = torch.softmax(logits.flatten(1), 1).reshape(logits.shape)
    my = (w * gy).sum((1, 2))
    mx = (w * gx).sum((1, 2))
    var = 0.5 * (
        (w * (gy - my[:, None, None]) ** 2).sum((1, 2))
        + (w * (gx - mx[:, None, None]) ** 2).sum((1, 2))
    )
    width = torch.sqrt(var.clamp_min(0))
    y = ys.float() + off[b, 0, ys, xs]
    x = xs.float() + off[b, 1, ys, xs]
    if mask_logits is not None:
        y, x = _center_on_mask(mask_logits, b, y, x, mask_radius)
    return torch.stack([b.float(), y, x, confidence, width], 1)


def positional_sigma(width_px, signal, background_per_px, pixel=1.0):
    """One standard deviation of a center, in detector pixels.

    The localization-precision formula for a spot of standard deviation ``s``
    measured with ``N`` signal counts over a background of ``b²`` counts per
    pixel, pixel size ``a`` (Thompson, Larson & Webb, Biophys. J. 82, 2775 (2002)):

        sigma² = (s² + a²/12) / N  +  8 π s⁴ b² / (a² N²)

    Here ``s`` is the width of the network's response to the disk (``width`` from
    :func:`decode_batch_centers`, in detector pixels), ``N`` the disk's
    background-subtracted counts and ``b²`` its local background level (Poisson,
    so the variance equals the mean). A bright, sharp disk gets a small sigma; a
    faint or diffuse one a large sigma. NaN where the disk has no net signal."""
    s = np.maximum(np.asarray(width_px, np.float64), 0.5)
    n = np.asarray(signal, np.float64)
    b2 = np.clip(np.asarray(background_per_px, np.float64), 0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (s * s + pixel * pixel / 12.0) / n + 8 * np.pi * s**4 * b2 / (
            pixel * pixel * n * n
        )
        sigma = np.sqrt(var)
    return np.where(n > 0, sigma, np.nan).astype(np.float32)
