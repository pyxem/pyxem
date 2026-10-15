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

"""Input normalization + automatic, parameter-free disk-size scale normalization.

Vendored verbatim from the ``yoloDiffraction`` research project
(``normalize_input`` from ``yolodiffraction/data/dataset.py`` and the scale-norm
helpers from ``yolodiffraction/data/scale_norm.py``) so pyxem preprocesses model
inputs EXACTLY as the checkpoints were trained — DO NOT change the maths here
(train/infer parity).

Within a dataset all diffraction disks are the same physical size (the central
beam only looks bigger because it's brighter). So one model trained at a CANONICAL
disk size suffices if we resample each dataset so its disks hit that size. The
disk size is estimated WITHOUT parameters from the pattern's autocorrelation: a
disk autocorrelated with itself gives a central peak whose width ≈ the disk
diameter.

Usage at inference:
    diam = estimate_disk_diameter(frame)
    scaled, factor = scale_to_canonical(frame, diam)        # disks -> ~CANONICAL px
    # run model on `scaled`, then map predicted positions back with /factor
"""
from __future__ import annotations

import numpy as np

# Canonical disk size the detector prefers, synced 2026-07-16 from yoloDiffraction
# scale_norm.py: with autocorrelation-measured diameters (CuNb/ZrNb ~16-18px,
# SPED-Ag ~8px) canonical=9 + DOWNSAMPLE reproduces the KNOWN-GOOD per-dataset
# scales (CuNb 0.56, ZrNb 0.50, SPED-Ag ~1.1). The previous canonical=20 +
# upsample-only made every scale wrong (CuNb ran UPSAMPLED 1.25x instead of
# downsampled 0.56x -> the model saw disks >2x its trained size -> noise-peak
# flood; wrong scale is the #1 FP-flood cause). Big-disk subpixel is protected by
# SCALE_CLIP, not by refusing to downsample.
CANONICAL_DIAMETER = 9.0
DOWNSAMPLE = True  # allow downscale (large disks shrink toward canonical)

# Bound the resample factor: a pathological diameter estimate (beam stop, empty or
# saturated frame) must not trigger an extreme resample. Same clip the
# yoloDiffraction truth harness uses; also keeps very large disks from being
# crushed to 9px (they run at the 0.3 floor instead, where the size-scaled
# NMS/background parameters in infer.py take over).
SCALE_CLIP = (0.3, 3.0)


def normalize_input(
    frame: np.ndarray, local: bool = True, bg_sigma: float = 12.0
) -> np.ndarray:
    """log1p + robust standardization. Shared train+infer.

    log1p compresses the huge spot/background dynamic range; median/MAD is robust to
    the bright central beam and hot pixels.

    ``local`` (default) additionally subtracts a LOCAL background estimate (large
    Gaussian blur) before standardizing — the learned analogue of WNCC's window
    normalization. This makes a spot a local bump above ITS surroundings regardless
    of a spatially-varying diffuse background across the FOV (the hard case). bg_sigma
    should be a few times the disk size so it removes background, not the spots.
    """
    x = np.log1p(np.clip(frame, 0, None).astype(np.float32))
    if local:
        from scipy.ndimage import gaussian_filter

        x = x - gaussian_filter(x, bg_sigma)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    return (x - med) / (1.4826 * mad)


def estimate_disk_diameter(frame: np.ndarray, hp_sigma: float = 20.0) -> float:
    """Estimate disk diameter (px) from the autocorrelation central-peak FWHM.

    High-passes the frame first (removes smooth background and tames the bright
    central beam), then measures the half-max width of the autocorrelation peak,
    which equals the disk diameter. Robust across sizes; no thresholds/params.
    """
    from scipy.ndimage import gaussian_filter

    f = np.clip(frame.astype(np.float64), 0, None)
    f = np.clip(f - gaussian_filter(f, hp_sigma), 0, None)
    if f.max() <= 0:
        return CANONICAL_DIAMETER
    F = np.fft.fft2(f)
    ac = np.fft.fftshift(np.real(np.fft.ifft2(F * np.conj(F))))
    H, W = ac.shape
    cy, cx = H // 2, W // 2
    peak = ac[cy, cx]
    if peak <= 0:
        return CANONICAL_DIAMETER

    def half_max_width(profile, centre):
        """Width of the central peak where it falls below half its maximum."""
        left = right = centre
        while left > 0 and profile[left] > 0.5:
            left -= 1
        while right < profile.size - 1 and profile[right] > 0.5:
            right += 1
        return max(right - left, 1)

    # The two central-line widths, averaged — NOT the two profiles. They only
    # have the same length on a square detector, so adding them raised on any
    # other: a cropped or composed pattern (a multi-angle sum is 507 x 501 when
    # its reciprocal offsets differ between the axes) took the whole neural
    # detector down. Averaging the widths is what "robust to anisotropy" meant
    # and it holds for any shape.
    return float(
        0.5 * (half_max_width(ac[cy] / peak, cx) + half_max_width(ac[:, cx] / peak, cy))
    )


def scale_factor(frame: np.ndarray, target: float = CANONICAL_DIAMETER) -> float:
    """Resample factor so this frame's disks land at ~``target`` px (both
    directions, bounded by SCALE_CLIP)."""
    f = target / estimate_disk_diameter(frame)
    if not DOWNSAMPLE:
        f = max(f, 1.0)
    return float(np.clip(f, *SCALE_CLIP))


def scale_to_canonical(
    frame: np.ndarray, diameter: float | None = None, target: float = CANONICAL_DIAMETER
):
    """Resample ``frame`` so its disks land at ~``target`` px. Returns
    (scaled, factor), bounded by SCALE_CLIP (see the constants above for why the
    old upsample-only policy was wrong). A predicted position p in the scaled
    frame maps back as p / factor. ``diameter`` may be passed to reuse one
    estimate across a stack."""
    from scipy.ndimage import zoom

    if diameter is None:
        diameter = estimate_disk_diameter(frame)
    factor = target / diameter
    if not DOWNSAMPLE:
        factor = max(factor, 1.0)  # never shrink
    factor = float(np.clip(factor, *SCALE_CLIP))
    if abs(factor - 1.0) < 0.05:
        return frame.astype(np.float32), 1.0
    scaled = zoom(frame.astype(np.float32), factor, order=1)
    return scaled, factor
