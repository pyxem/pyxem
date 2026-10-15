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

"""Parameter-free inference: rescale -> normalise -> U-Net -> decode -> map back.

The disk size is estimated from the pattern itself and the pattern resampled
so its disks match the size the network was trained on; positions are mapped
back to the original pixel grid. The maths in :func:`detect`,
:func:`detect_batch` and :func:`calibrate` must stay faithful to how the
checkpoints were trained.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading

import numpy as np
import torch

from .decode import decode, decode_batch, split_by_batch
from .preprocess import estimate_disk_diameter, normalize_input, scale_to_canonical
from .unet import SpotUNet

_logger = logging.getLogger(__name__)

#: Local-normalisation high-pass scale in pixels. Very diffuse backgrounds
#: (beam stop, amorphous halo) need a smaller value, which :func:`calibrate`
#: picks automatically.
DEFAULT_BACKGROUND_SIGMA = 12.0
DEFAULT_THRESHOLD = 0.3
#: Patterns per forward pass. Bounds the activation memory of a whole chunk.
SUB_BATCH_SIZE = 64

# Apple MPS is not thread safe: two threads submitting to Metal at once can
# crash the process with an uncatchable native fault. Every MPS submission in
# this package takes this lock; CUDA and CPU never do.
_DEVICE_LOCK = threading.RLock()

# An unsupported MPS operation then runs on the CPU instead of raising. It only
# takes effect if set before torch first touches MPS, hence import time.
if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def device_lock(device):
    """The process-wide lock for ``device`` (a null context off MPS)."""
    if torch.device(device).type == "mps":
        return _DEVICE_LOCK
    return contextlib.nullcontext()


def default_device():
    """Best available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(checkpoint, device=None):
    """Build a :class:`SpotUNet` from a loaded checkpoint dictionary.

    Accepts the architecture either flat (``{"state_dict", "base", "in_ch",
    "levels"}``) or nested (``{"state_dict", "config": {...}}``).
    """
    device = torch.device(device) if device is not None else default_device()
    config = dict(checkpoint.get("config") or {})
    for key in ("base", "in_ch", "levels"):
        if key in checkpoint and key not in config:
            config[key] = checkpoint[key]
    in_channels = int(config.get("in_ch", 1))
    model = SpotUNet(
        in_ch=in_channels,
        base=int(config.get("base", 16)),
        levels=int(config.get("levels", 2)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    with device_lock(device):
        model = model.to(device)
        model.eval()
        if device.type == "mps":
            # An MPS build missing an operation this network uses should fall
            # back to the CPU once, here, not fail on every chunk.
            try:
                with torch.no_grad():
                    model(torch.zeros(1, in_channels, 32, 32, device=device))
            except Exception:
                device = torch.device("cpu")
                model = model.to(device)
    return model, device


def load_checkpoint(path):
    # weights_only: a checkpoint is a state dict plus scalar hyperparameters;
    # never unpickle arbitrary objects from a downloaded file.
    return torch.load(path, map_location="cpu", weights_only=True)


def _pad_to_multiple(normalised, levels):
    """Reflect-pad so both sides are a multiple of ``2**levels``."""
    height, width = normalised.shape
    multiple = 2**levels
    pad_rows = (multiple - height % multiple) % multiple
    pad_columns = (multiple - width % multiple) % multiple
    if pad_rows or pad_columns:
        normalised = np.pad(
            normalised, ((0, pad_rows), (0, pad_columns)), mode="reflect"
        )
    return normalised


def _working_diameter(pattern, factor, spot_diameter):
    """Disk diameter after rescaling. The autocorrelation estimate saturates
    on big disks, so iterate with a size-adaptive high-pass and correct its
    measured ~0.83 undershoot."""
    if spot_diameter:
        return float(spot_diameter) * factor
    diameter = estimate_disk_diameter(pattern)
    if np.isfinite(diameter) and diameter > 20:
        for _ in range(3):
            refined = estimate_disk_diameter(pattern, hp_sigma=max(20.0, diameter))
            if abs(refined - diameter) < 1:
                diameter = refined
                break
            diameter = max(diameter, refined)
        diameter = diameter / 0.83
    return diameter * factor


def _big_disk_parameters(min_distance, background_sigma, working_diameter):
    """Scale the suppression window and background sigma for big disks.

    Below a 20 px working diameter nothing changes. Above it a fixed window
    fires a cloud of detections across one broad disk, and a fixed 12 px
    high-pass carves the flat disk interior into a ring.
    """
    big = np.isfinite(working_diameter) and working_diameter > 20
    if big:
        min_distance = max(min_distance, int(round(working_diameter * 0.55)))
    if background_sigma is None:
        background_sigma = (
            max(DEFAULT_BACKGROUND_SIGMA, 0.6 * working_diameter)
            if big
            else DEFAULT_BACKGROUND_SIGMA
        )
    return min_distance, float(background_sigma)


def _forward(model, device, x):
    """Run the model; on a device error retry on the CPU and stay there."""
    try:
        with device_lock(device):
            heatmap, offset = model(x.to(device))
        return heatmap, offset, model, device
    except RuntimeError as error:
        if device.type == "cpu":
            raise
        _logger.warning(
            "%s forward pass failed (%s); continuing on the CPU", device.type, error
        )
        cpu = torch.device("cpu")
        model = model.to(cpu)
        heatmap, offset = model(x.to(cpu))
        return heatmap, offset, model, cpu


@torch.no_grad()
def detect(
    model,
    pattern,
    device,
    threshold=DEFAULT_THRESHOLD,
    min_distance=4,
    background_sigma=None,
    spot_diameter=None,
):
    """Detect spots in one pattern.

    Returns ``(N, 3)`` float32 ``[row, column, confidence]`` in the original
    pixel grid. ``spot_diameter`` (px) overrides the automatic disk-size
    estimate that drives the rescale.
    """
    return detect_batch(
        model,
        np.asarray(pattern, dtype=np.float32)[None],
        device,
        threshold=threshold,
        min_distance=min_distance,
        background_sigma=background_sigma,
        spot_diameter=spot_diameter,
    )[0]


@torch.no_grad()
def detect_batch(
    model,
    patterns,
    device,
    threshold=DEFAULT_THRESHOLD,
    min_distance=4,
    background_sigma=None,
    spot_diameter=None,
):
    """Detect spots in a stack of patterns with batched forward passes.

    Every pattern in a scan has the same physical disk size, so the rescale
    factor, suppression window and background sigma are resolved once from
    the first pattern; that is also what keeps the stack a single shape.

    Returns a list of ``(N_i, 3)`` float32 ``[row, column, confidence]``.
    """
    patterns = np.asarray(patterns, dtype=np.float32)
    if patterns.ndim == 2:
        patterns = patterns[None]
    if len(patterns) == 0:
        return []

    _, factor = scale_to_canonical(patterns[0], diameter=spot_diameter)
    min_distance = max(2, int(round(min_distance * factor)))
    min_distance, background_sigma = _big_disk_parameters(
        min_distance,
        background_sigma,
        _working_diameter(patterns[0], factor, spot_diameter),
    )
    levels = int(getattr(model, "levels", 2))

    def prepare(pattern):
        if factor != 1.0:
            from scipy.ndimage import zoom

            pattern = zoom(pattern, factor, order=1)
        normalised = normalize_input(pattern, local=True, bg_sigma=background_sigma)
        return _pad_to_multiple(normalised, levels)

    results = []
    for start in range(0, len(patterns), SUB_BATCH_SIZE):
        stack = np.stack([prepare(p) for p in patterns[start : start + SUB_BATCH_SIZE]])
        x = torch.from_numpy(stack[:, None].astype(np.float32))
        heatmap, offset, model, device = _forward(model, device, x)
        decoded = decode_batch(
            heatmap, offset, thresh=threshold, min_distance=min_distance
        )
        results.extend(split_by_batch(decoded, len(stack)))

    if factor != 1.0:
        for i, peaks in enumerate(results):
            if len(peaks):
                peaks = peaks.copy()
                peaks[:, :2] /= factor
                results[i] = peaks
    return results


@torch.no_grad()
def _best_background_sigma(model, working_pattern, device, candidates):
    """The background sigma that maximises the network's peak confidence."""
    levels = int(getattr(model, "levels", 2))
    best_confidence, best_sigma = -1.0, DEFAULT_BACKGROUND_SIGMA
    for sigma in candidates:
        normalised = _pad_to_multiple(
            normalize_input(working_pattern, local=True, bg_sigma=float(sigma)), levels
        )
        x = torch.from_numpy(normalised[None, None].astype(np.float32)).to(device)
        with device_lock(device):
            heatmap, _ = model(x)
        confidence = float(torch.sigmoid(heatmap).max())
        if confidence > best_confidence:
            best_confidence, best_sigma = confidence, float(sigma)
    return best_sigma, best_confidence


@torch.no_grad()
def calibrate(model, sample_patterns, device, spot_diameter=None):
    """Tune the dataset-dependent settings once on a few representative patterns.

    Picks the local-normalisation ``background_sigma`` that maximises the
    network's confidence (median over the sample), and lowers the threshold
    for faint-peak data where even the best confidence is weak.

    Returns a dict with ``background_sigma``, ``threshold``, ``scale_factor``
    and ``confidence``.
    """
    patterns = [np.asarray(p, np.float32) for p in sample_patterns]
    if not patterns:
        return {
            "background_sigma": DEFAULT_BACKGROUND_SIGMA,
            "threshold": DEFAULT_THRESHOLD,
            "scale_factor": 1.0,
            "confidence": float("nan"),
        }
    _, factor = scale_to_canonical(patterns[0], diameter=spot_diameter)
    candidates = [4, 5, 6, 8, 10, 12, 16, 20]
    working_diameter = _working_diameter(patterns[0], factor, spot_diameter)
    if np.isfinite(working_diameter) and working_diameter > 20:
        candidates += [round(r * working_diameter, 1) for r in (0.5, 0.6, 0.7)]

    sigmas, confidences = [], []
    for pattern in patterns:
        if factor != 1.0:
            from scipy.ndimage import zoom

            pattern = zoom(pattern, factor, order=1)
        sigma, confidence = _best_background_sigma(model, pattern, device, candidates)
        sigmas.append(sigma)
        confidences.append(confidence)
    confidence = float(np.median(confidences))
    threshold = DEFAULT_THRESHOLD
    if confidence < 0.45:
        threshold = float(np.clip(0.5 * confidence, 0.1, DEFAULT_THRESHOLD))
    return {
        "background_sigma": float(np.median(sigmas)),
        "threshold": threshold,
        "scale_factor": float(factor),
        "confidence": confidence,
    }


__all__ = [
    "SpotUNet",
    "build_model",
    "calibrate",
    "decode",
    "default_device",
    "detect",
    "detect_batch",
    "device_lock",
    "load_checkpoint",
]
