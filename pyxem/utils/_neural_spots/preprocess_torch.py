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
# spyde/models/preprocess_torch.py; see vendor note in __init__.py.

"""On-device (torch) batched equivalents of ``preprocess.py``'s numpy/scipy pipeline.

``detect_batch`` used to normalize / scale / pad every frame on the HOST with
numpy + scipy regardless of the model device, then transfer the stack to the GPU
for the forward. That host preprocessing is a FIXED cost that doesn't shrink when
the forward moves to the GPU, so once the forward is fast it dominates end-to-end
time (Amdahl): measured ~1 s of CPU normalize for a 256-frame batch vs ~250 ms for
the MPS forward. These functions do the SAME math batched on the model device so
the whole preprocess → forward → decode pipeline stays on the GPU.

PARITY IS LOAD-BEARING. The checkpoints were trained with ``preprocess.py``'s exact
math (see its "DO NOT change the math" note), so these must match it numerically.
The couplings that actually bite (and are pinned by ``test_preprocess_torch_parity``):

  - Gaussian high-pass: ``scipy.ndimage.gaussian_filter`` defaults to
    ``truncate=4.0`` and ``mode='reflect'`` which is HALF-sample symmetric (the edge
    pixel is duplicated: ``d c b a | a b c d | d c b a``) — NOT torch's whole-sample
    ``F.pad(mode='reflect')``. We build scipy's exact kernel and a half-sample
    reflect index map by hand (``_reflect_index_map``, valid for any radius, incl.
    radius > frame for big-disk bg_sigma).
  - Median / MAD: ``np.median`` AVERAGES the two middle order statistics on an
    even-length input; ``torch.median`` returns the lower one. We sort and average
    the two middles (``_median_lastdim``) to match numpy exactly, and it avoids
    ``torch.quantile`` (which can silently fall back to CPU on MPS).
  - ``scale_to_canonical`` zoom: ``scipy.ndimage.zoom(order=1)`` with the default
    ``grid_mode=False`` endpoint-aligns and sizes the output ``round(n*factor)``;
    reproduced with ``F.interpolate(size=round(n*factor), mode='bilinear',
    align_corners=True)``.
  - ``_pad_to_multiple``: uses ``np.pad(mode='reflect')`` which is WHOLE-sample
    symmetric = torch ``F.pad(mode='reflect')`` (note: a DIFFERENT reflect from the
    gaussian's — numpy 'reflect' ≠ scipy.ndimage 'reflect').

Everything here is numpy/scipy-free: a frame stack goes in as a device tensor and
comes out as a device tensor ready for the U-Net. ``preprocess.py`` stays the
canonical CPU reference and is untouched.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .preprocess import CANONICAL_DIAMETER, DOWNSAMPLE, SCALE_CLIP


def _gaussian_kernel1d(sigma: float, radius: int, device, dtype) -> torch.Tensor:
    """scipy ``_gaussian_kernel1d`` (order 0): normalized ``exp(-x^2/2sigma^2)`` for
    ``x in [-radius, radius]``. Symmetric, so correlate (scipy) == convolve (torch)."""
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / float(sigma)) ** 2)
    return k / k.sum()


def _reflect_index_map(n: int, k: int, device) -> torch.Tensor:
    """Half-sample-symmetric reflect indices for a length-``n`` axis padded by ``k``
    each side (scipy.ndimage 'reflect' — edge pixel duplicated). Valid for any ``k``
    (incl. ``k >= n``, which happens for big-disk ``bg_sigma``): the triangle-wave
    fold handles multiple reflections. Returns a length ``n + 2k`` LongTensor."""
    p = torch.arange(-k, n + k, device=device)
    period = 2 * n
    p = torch.remainder(p, period)  # → [0, 2n)
    p = torch.where(p >= n, period - 1 - p, p)  # fold the upper half back down
    return p


def gaussian_blur2d(
    x: torch.Tensor, sigma: float, truncate: float = 4.0
) -> torch.Tensor:
    """Separable Gaussian blur matching ``scipy.ndimage.gaussian_filter`` (default
    ``truncate=4.0``, half-sample reflect). ``x`` is ``(N, 1, H, W)``."""
    r = int(truncate * float(sigma) + 0.5)
    if r < 1:
        return x
    w = _gaussian_kernel1d(sigma, r, x.device, x.dtype)
    _, _, H, W = x.shape
    # Blur along H then W (separable — order is immaterial for a symmetric kernel).
    xh = x.index_select(2, _reflect_index_map(H, r, x.device))
    xh = F.conv2d(xh, w.view(1, 1, -1, 1))  # 'valid' → back to H
    xw = xh.index_select(3, _reflect_index_map(W, r, x.device))
    return F.conv2d(xw, w.view(1, 1, 1, -1))  # → back to W


def _median_lastdim(flat: torch.Tensor) -> torch.Tensor:
    """Per-row median matching ``np.median`` (averages the two middles on even
    length). ``flat`` is ``(N, M)``; returns ``(N, 1)``. Sort-based (works on MPS;
    avoids torch.quantile's possible CPU fallback there)."""
    M = flat.shape[1]
    xs, _ = flat.sort(dim=1)
    if M % 2 == 1:
        med = xs[:, M // 2]
    else:
        med = 0.5 * (xs[:, M // 2 - 1] + xs[:, M // 2])
    return med.unsqueeze(1)


def normalize_input_batch(
    x: torch.Tensor,
    bg_sigma: float = 12.0,
    local: bool = True,
    mad_floor: float | None = None,
) -> torch.Tensor:
    """Batched, on-device ``preprocess.normalize_input`` (log1p + robust standardize,
    optional local-background subtraction). ``x`` is ``(N, H, W)`` float; returns the
    same shape. Matches the numpy math (see module docstring), including the
    optional floor on the robust scale (``preprocess.normalize_input``)."""
    x = torch.log1p(x.clamp(min=0))
    if local:
        x = x - gaussian_blur2d(x.unsqueeze(1), float(bg_sigma)).squeeze(1)
    flat = x.reshape(x.shape[0], -1)
    med = _median_lastdim(flat)
    scale = 1.4826 * (_median_lastdim((flat - med).abs()) + 1e-6)
    if mad_floor is not None:
        scale = scale.clamp_min(mad_floor)
    return ((flat - med) / scale).reshape_as(x)


def scale_batch(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Batched, on-device ``scipy.ndimage.zoom(order=1)`` (bilinear, grid_mode=False):
    output size ``round(n*factor)`` per axis, endpoint-aligned. ``x`` is ``(N, H, W)``.
    Returns ``(N, round(H*factor), round(W*factor))``. Caller maps predicted positions
    back with ``infer._to_frame_coordinates`` exactly as with the scipy path."""
    _, H, W = x.shape
    out_h = int(round(H * factor))
    out_w = int(round(W * factor))
    y = F.interpolate(
        x.unsqueeze(1), size=(out_h, out_w), mode="bilinear", align_corners=True
    )
    return y.squeeze(1)


def pad_to_multiple_batch(x: torch.Tensor, levels: int) -> torch.Tensor:
    """Batched ``infer._pad_to_multiple``: whole-sample reflect-pad H,W up to a
    multiple of ``2**levels`` (numpy 'reflect' == torch 'reflect'). ``x`` is
    ``(N, H, W)``; returns ``(N, H', W')``."""
    _, H, W = x.shape
    mult = 2**levels
    ph = (mult - H % mult) % mult
    pw = (mult - W % mult) % mult
    if ph or pw:
        # F.pad last-dim-first: (W_left, W_right, H_top, H_bottom).
        x = F.pad(x.unsqueeze(1), (0, pw, 0, ph), mode="reflect").squeeze(1)
    return x


def canonical_scale_factor_from_diameter(
    diameter: float, target: float = CANONICAL_DIAMETER
) -> float:
    """The exact factor ``scale_to_canonical`` derives from a (already-estimated)
    disk diameter — bounded by SCALE_CLIP, with the <5% no-op short-circuit — so the
    GPU path scales identically to the CPU path. The diameter itself is still
    estimated once per batch on the host (``estimate_disk_diameter``, cheap)."""
    factor = target / diameter
    if not DOWNSAMPLE:
        factor = max(factor, 1.0)
    factor = float(min(max(factor, SCALE_CLIP[0]), SCALE_CLIP[1]))
    if abs(factor - 1.0) < 0.05:
        return 1.0
    return factor
