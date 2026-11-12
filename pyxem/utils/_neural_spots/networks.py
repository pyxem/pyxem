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

"""The networks, run with ONNX Runtime.

Each ``.onnx`` file describes itself through its metadata: the detector's
pooling depth, mask head, normalization floor and default threshold, and the
center network's window contract. Everything outside the network -- window
resampling, normalization, padding, the coverage centroid -- is numpy here,
a port of SpyDE's spyde/models/centre_network.py (SpyDE 68dfa52038).

An ``InferenceSession`` may be run from several threads at once, so the
sessions are shared without a lock.
"""

import math

import numpy as np

from .center import coverage_centroid, sigmoid
from .center_refine import DEFAULT_MIN_SPOT_RADIUS, MAX_SHIFT_FRACTION

#: Execution providers by the ``device=`` names pyxem accepts.
PROVIDERS = {
    "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "mps": "CoreMLExecutionProvider",
    "cpu": "CPUExecutionProvider",
}
#: Tried in order by ``device=None`` / ``"auto"``.
AUTOMATIC_ORDER = ("cuda", "directml", "coreml")

#: Windows per run of a center network, to bound its activations.
FORWARD_BATCH = 1024


def providers_for(device=None):
    """ONNX Runtime providers for ``device``: ``None`` or ``"auto"`` picks the
    first available of CUDA, DirectML and CoreML; the CPU always comes last as
    the fallback."""
    import onnxruntime

    available = onnxruntime.get_available_providers()
    name = "auto" if device is None else str(device).lower()
    if name == "auto":
        chosen = [PROVIDERS[n] for n in AUTOMATIC_ORDER if PROVIDERS[n] in available]
        chosen = chosen[:1]
    elif name in PROVIDERS:
        if PROVIDERS[name] not in available:
            raise ValueError(
                f"device={device!r} needs ONNX Runtime's {PROVIDERS[name]}, which this "
                f"installation does not have (it has {available})."
            )
        chosen = [PROVIDERS[name]]
    else:
        raise ValueError(
            f"device must be None, 'auto' or one of {sorted(PROVIDERS)}, not {device!r}"
        )
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    return chosen


def _session(path, providers):
    import onnxruntime

    return onnxruntime.InferenceSession(str(path), providers=providers)


def _metadata(session):
    return dict(session.get_modelmeta().custom_metadata_map)


class Detector:
    """A DE SpotNet detector: ``run(x)`` maps a normalized ``(N, 1, H, W)``
    float32 stack (H and W multiples of ``2 ** levels``) to the heat-map
    logits, offsets and, with a mask head, mask logits."""

    def __init__(self, path, providers):
        self.session = _session(path, providers)
        metadata = _metadata(self.session)
        self.levels = int(metadata.get("levels", 2))
        self.mask_head = bool(int(metadata.get("mask_head", 0)))
        mad_floor = float(metadata.get("mad_floor", 0.0))
        self.mad_floor = mad_floor if mad_floor > 0 else None
        self.default_threshold = float(metadata.get("threshold", 0.3))
        self.providers = self.session.get_providers()

    def run(self, x):
        return self.session.run(None, {"input": np.ascontiguousarray(x, np.float32)})


class CenterRefiner:
    """A center network: segments each disk in a resampled, normalized window
    of the raw pattern and takes the segment's centroid.

    The refiner interface :func:`.center_refine.refine_centers` expects:
    ``crop_half_width(spot_radius)`` and ``refine(crops, centers, spot_radius,
    partner_crops=None, partner_centers=None) -> (centers, sigma)``. A disk is
    declined (NaN) when it moved half a spot radius or more, when its coverage
    is under a fifth of the expected disk, or when its sigma is
    ``max_sigma_fraction`` of the spot radius or more.
    """

    def __init__(self, path, providers):
        self.session = _session(path, providers)
        metadata = _metadata(self.session)
        self.channels = int(metadata.get("channels", 1))
        self.levels = int(metadata.get("levels", 2))
        self.crop_radius = float(metadata.get("crop_radius", 10.0))
        self.crop_half = int(metadata.get("crop_half", 16))
        self.annulus = tuple(
            float(v) for v in metadata.get("annulus", "0.78,1.0").split(",")
        )
        self.passes = max(1, int(metadata.get("passes", 2)))
        self.recrop_over = float(metadata.get("recrop_over", 0.15))
        self.max_sigma_fraction = float(metadata.get("max_sigma_fraction", 0.25))
        self.sigma_scale = float(metadata.get("sigma_scale", 1.0))
        self.min_spot_radius = float(
            metadata.get("min_spot_radius", DEFAULT_MIN_SPOT_RADIUS)
        )
        self.max_spot_radius = float(metadata.get("max_spot_radius", math.inf))
        #: The network reads a second channel: the disk's Friedel-mirror window.
        self.needs_partner = self.channels == 2
        self.providers = self.session.get_providers()

    def crop_half_width(self, spot_radius):
        # The resampled grid reaches crop_half / crop_radius radii from the
        # center, which sits up to half a pixel off the window center and, on
        # a re-crop pass, may have moved by up to the largest accepted shift.
        # One more pixel keeps the bilinear samples inside the window.
        reach = self.crop_half / self.crop_radius * float(spot_radius)
        if self.passes > 1:
            reach += MAX_SHIFT_FRACTION * float(spot_radius)
        return int(math.ceil(reach + 0.5)) + 1

    def _offsets(self):
        return np.arange(-self.crop_half, self.crop_half + 1, dtype=np.float32)

    def _resample(self, crops, centers, step):
        """Bilinear samples of each window on a ``(2 crop_half + 1)**2`` grid
        around its center, zero outside the window: what
        ``F.grid_sample(mode="bilinear", padding_mode="zeros",
        align_corners=True)`` returns, coordinates included.

        The grid is separable, so the bilinear sampling is two matrix
        products, ``rows @ window @ columns.T``, each interpolation matrix
        holding the two weights of every sample and nothing outside the
        window."""
        crops = np.asarray(crops, np.float32)
        size = crops.shape[-1]
        offsets = self._offsets()
        step = np.float32(step)
        scale = np.float32(size - 1)
        pixels = np.arange(size)

        def interpolation(coordinates):
            # grid_sample's normalized coordinates and back, in float32
            coordinates = ((coordinates / scale) * 2 - 1 + 1) / 2 * scale
            lower = np.floor(coordinates)
            fraction = coordinates - lower
            lower = lower.astype(np.int64)[..., None]
            return (
                np.where(pixels == lower, 1 - fraction[..., None], 0)
                + np.where(pixels == lower + 1, fraction[..., None], 0)
            ).astype(np.float32)

        rows = interpolation(centers[:, 0, None] + offsets[None] * step)
        columns = interpolation(centers[:, 1, None] + offsets[None] * step)
        return (rows @ crops @ columns.transpose(0, 2, 1))[:, None]

    def _normalize(self, sampled):
        count = sampled.shape[0]
        offsets = self._offsets()
        distance = np.sqrt(offsets[:, None] ** 2 + offsets[None] ** 2).reshape(-1)
        inner, outer = self.annulus
        ring = (distance > inner * self.crop_half) & (
            distance <= outer * self.crop_half
        )
        disk = distance <= self.crop_radius
        flat = sampled.reshape(count, -1)
        background = flat[:, ring].mean(1)
        level = np.maximum(np.abs(flat[:, disk].mean(1) - background), np.float32(1e-3))
        normalized = (flat - background[:, None]) / level[:, None]
        return np.clip(normalized, -4, 8).reshape(sampled.shape).astype(np.float32)

    def _forward(self, windows):
        """Coverage logits and log sigma of ``(M, C, S, S)`` windows; the
        windows are replicate-padded to a multiple of ``2 ** levels`` for the
        poolings, and the logits cropped back."""
        size = windows.shape[-1]
        pad_total = (-size) % (2**self.levels)
        before = pad_total // 2
        padding = (
            (0, 0),
            (0, 0),
            (before, pad_total - before),
            (before, pad_total - before),
        )
        logits, log_sigmas = [], []
        for start in range(0, len(windows), FORWARD_BATCH):
            padded = np.pad(
                windows[start : start + FORWARD_BATCH], padding, mode="edge"
            )
            output, log_sigma = self.session.run(None, {"input": padded})
            logits.append(output[:, :, before : before + size, before : before + size])
            log_sigmas.append(log_sigma)
        return np.concatenate(logits), np.concatenate(log_sigmas)

    def _coverage_centroid(self, logits):
        """The coverage map's centroid, as an offset from the window center."""
        offsets = self._offsets()
        start = np.zeros(len(logits), np.float32)
        center_y, center_x, area = coverage_centroid(
            sigmoid(logits[:, 0]),
            offsets[None, :, None],
            offsets[None, None, :],
            start,
            start,
            self.crop_radius,
        )
        return np.stack([center_y, center_x], 1), area

    def _pass(self, crops, center, step, partner=None):
        """One pass at ``center``: the new center, coverage area and sigma
        (native px). ``partner`` is ``(mirror windows, mirror point)`` for a
        partner network: that window, sampled the same way and rotated 180
        degrees, is the second channel."""
        sampled = self._normalize(self._resample(crops, center, step))
        if partner is not None:
            partner_crops, partner_center = partner
            mirror = self._normalize(
                self._resample(partner_crops, partner_center, step)
            )
            sampled = np.concatenate([sampled, mirror[:, :, ::-1, ::-1]], 1)
        logits, log_sigma = self._forward(np.ascontiguousarray(sampled))
        offset, area = self._coverage_centroid(logits)
        sigma = np.exp(log_sigma) * np.float32(step) * np.float32(self.sigma_scale)
        return center + offset * np.float32(step), area, sigma.astype(np.float32)

    def refine(
        self, crops, centers, spot_radius, partner_crops=None, partner_centers=None
    ):
        if len(crops) == 0:
            return np.zeros((0, 2)), np.zeros(0, np.float32)
        if self.needs_partner and partner_crops is None:
            raise ValueError(
                "this center network needs each disk's Friedel-mirror window"
            )
        radius = float(spot_radius)
        step = radius / self.crop_radius
        crops = np.asarray(crops, np.float32)
        seeds = np.asarray(centers, np.float32)
        partner = None
        if self.needs_partner:
            partner_crops = np.asarray(partner_crops, np.float32)
            mirror_seeds = np.asarray(partner_centers, np.float32)
            partner = (partner_crops, mirror_seeds)
        center, area, sigma = self._pass(crops, seeds, step, partner)
        for _ in range(1, self.passes):
            again = np.linalg.norm(center - seeds, axis=1) > self.recrop_over * radius
            if not again.any():
                break
            # a partner window follows the mirror of the moved center
            moved_partner = (
                None
                if partner is None
                else (
                    partner_crops[again],
                    mirror_seeds[again] - (center[again] - seeds[again]),
                )
            )
            moved, moved_area, moved_sigma = self._pass(
                crops[again], center[again], step, moved_partner
            )
            center[again], area[again], sigma[again] = moved, moved_area, moved_sigma
        declined = (
            (np.linalg.norm(center - seeds, axis=1) >= MAX_SHIFT_FRACTION * radius)
            | (area <= 0.2 * math.pi * self.crop_radius**2)
            | (sigma >= self.max_sigma_fraction * radius)
        )
        center[declined] = np.nan
        sigma[declined] = np.nan
        return center.astype(np.float64), sigma.astype(np.float32)
