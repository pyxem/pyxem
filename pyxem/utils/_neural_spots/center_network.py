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
# spyde/models/centre_network.py; see vendor note in __init__.py.

"""The center networks: segment the disk in a raw window, take the segment's centroid.

A small U-Net turns a normalized window into a disk-coverage map, and the
center is that map's centroid. A convolution moves with the disk, so the center
does too, and the centroid can never leave the disk the way a regressed offset
can. A sigma head — a linear layer on the mean-pooled bottleneck — predicts the
log of the center's standard deviation.

The refine step's network (F5) reads one channel, the disk's own window. The
Friedel-partner network (P4) reads two: the second is the window at the disk's
mirror point ``2 x beam - p``, sampled and normalized like the first and
rotated 180 degrees, so both channels show the same offset with the two
different dynamical fills of ``g`` and ``-g``. The channel count is read from
the checkpoint's first convolution; a two-channel network sets
``needs_partner`` and :func:`~pyxem.utils._neural_spots.center_refine.refine_centers` cuts
the mirror windows.

Checkpoint contract (``torch.load(weights_only=True)``): ``state_dict`` plus the
scalars ``base``, ``levels``, ``depthwise``, ``sigma_head``, ``crop_radius``,
``crop_half`` and optionally ``sigma_scale``, ``min_spot_radius``,
``max_spot_radius``. A registry entry with ``"kind": "refiner"`` may override
them under ``arch`` / ``input``.

Input contract, all relative to the spot radius ``R`` in native pixels:

* the window is resampled (bilinear, zeros outside the frame) so the disk spans
  ``crop_radius`` window pixels: a ``(2 * crop_half + 1)²`` grid at
  ``R / crop_radius`` native px per step, centered on the current center;
* normalization: subtract the mean over the annulus ``annulus[0]-annulus[1] x
  crop_half`` (default 0.78-1.0), divide by ``|mean inside crop_radius -
  background|`` (at least 1e-3), clamp to [-4, 8].

Inference: a second pass re-crops at the refined center for the disks the first
moved by more than ``recrop_over x R`` (0.15). Sigma is ``exp(log_sigma) x
sigma_scale x R / crop_radius``; it ranks centers well, but its absolute scale
is uncertain to about 2x, so use it to rank or weight vectors, not as a
calibrated error bar. A disk is declined (NaN, so the incoming center is kept)
when it moved half a spot radius or more, when its coverage is under a fifth of
the expected disk, or when its sigma is ``max_sigma_fraction`` (0.25) of the
spot radius or more. Outside ``min_spot_radius``-``max_spot_radius`` the step
is skipped altogether.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .center import coverage_centroid
from .center_refine import DEFAULT_MIN_SPOT_RADIUS, MAX_SHIFT_FRACTION

DEFAULT_CROP_RADIUS = 10.0
DEFAULT_CROP_HALF = 16
DEFAULT_ANNULUS = (0.78, 1.0)
DEFAULT_PASSES = 2
DEFAULT_RECROP_OVER = 0.15
DEFAULT_MAX_SIGMA_FRACTION = 0.25

# Windows per forward, to bound the network's activations on the device.
FORWARD_BATCH = 1024


def _pad_to_multiple(x, multiple):
    """Replicate-pad a square window to a side divisible by ``multiple`` (the
    poolings must line up); returns the padded window and the leading pad."""
    size = x.shape[-1]
    pad_total = (-size) % multiple
    before, after = pad_total // 2, pad_total - pad_total // 2
    return F.pad(x, (before, after, before, after), mode="replicate"), before


def _convolution(channels_in, channels_out, depthwise):
    if depthwise and channels_in > 1:
        return nn.Sequential(
            nn.Conv2d(
                channels_in, channels_in, 3, padding=1, groups=channels_in, bias=False
            ),
            nn.Conv2d(channels_in, channels_out, 1, bias=False),
        )
    return nn.Conv2d(channels_in, channels_out, 3, padding=1, bias=False)


def _block(channels_in, channels_out, depthwise):
    return nn.Sequential(
        _convolution(channels_in, channels_out, depthwise),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(inplace=True),
        _convolution(channels_out, channels_out, depthwise),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(inplace=True),
    )


class FastCenterNet(nn.Module):
    """U-Net with ``levels`` poolings: coverage logits and, with a sigma head,
    the log of the center's standard deviation in window pixels. Layer names
    are the checkpoint's state-dict keys; do not rename them."""

    def __init__(
        self,
        base: int = 8,
        levels: int = 2,
        depthwise: bool = False,
        sigma_head: bool = True,
        channels: int = 1,
    ):
        super().__init__()
        self.levels = levels
        widths = [base * 2**level for level in range(levels + 1)]
        self.encoders = nn.ModuleList(
            [
                _block(
                    channels if level == 0 else widths[level - 1],
                    widths[level],
                    depthwise,
                )
                for level in range(levels + 1)
            ]
        )
        self.ups = nn.ModuleList(
            [
                nn.ConvTranspose2d(widths[level], widths[level - 1], 2, stride=2)
                for level in range(levels, 0, -1)
            ]
        )
        self.decoders = nn.ModuleList(
            [
                _block(2 * widths[level - 1], widths[level - 1], depthwise)
                for level in range(levels, 0, -1)
            ]
        )
        self.head = nn.Conv2d(base, 1, 1)
        self.sigma_head = nn.Linear(widths[-1], 1) if sigma_head else None

    def forward(self, x):
        size = x.shape[-1]
        x, before = _pad_to_multiple(x, 2**self.levels)
        features = []
        hidden = x
        for level, encoder in enumerate(self.encoders):
            hidden = encoder(hidden if level == 0 else F.max_pool2d(hidden, 2))
            features.append(hidden)
        decoded = features[-1]
        for step, (up, decoder) in enumerate(zip(self.ups, self.decoders)):
            decoded = decoder(
                torch.cat([up(decoded), features[self.levels - 1 - step]], 1)
            )
        logits = self.head(decoded)[
            :, :, before : before + size, before : before + size
        ]
        log_sigma = (
            self.sigma_head(features[-1].mean((2, 3)))[:, 0]
            if self.sigma_head is not None
            else None
        )
        return logits, log_sigma


class NetworkRefiner:
    """A :class:`~pyxem.utils._neural_spots.center_refine.CenterRefiner` backed by a
    :class:`FastCenterNet` with a sigma head."""

    def __init__(
        self,
        net,
        device,
        crop_radius: float = DEFAULT_CROP_RADIUS,
        crop_half: int = DEFAULT_CROP_HALF,
        annulus=DEFAULT_ANNULUS,
        passes: int = DEFAULT_PASSES,
        recrop_over: float = DEFAULT_RECROP_OVER,
        max_sigma_fraction: float = DEFAULT_MAX_SIGMA_FRACTION,
        sigma_scale: float = 1.0,
        min_spot_radius: float = DEFAULT_MIN_SPOT_RADIUS,
        max_spot_radius: float = math.inf,
        needs_partner: bool = False,
    ):
        self.net = net
        self.device = device
        self.crop_radius = float(crop_radius)
        self.crop_half = int(crop_half)
        self.annulus = (float(annulus[0]), float(annulus[1]))
        self.passes = max(1, int(passes))
        self.recrop_over = float(recrop_over)
        self.max_sigma_fraction = float(max_sigma_fraction)
        self.sigma_scale = float(sigma_scale)
        self.min_spot_radius = float(min_spot_radius)
        self.max_spot_radius = float(max_spot_radius)
        #: The network reads a second channel: the disk's Friedel-mirror window.
        self.needs_partner = bool(needs_partner)

    def crop_half_width(self, spot_radius: float) -> int:
        # The resampled grid reaches crop_half / crop_radius radii from the
        # center, which sits up to half a pixel off the window center and, on a
        # re-crop pass, may have moved by up to the largest accepted shift.
        # One more pixel keeps the bilinear samples inside the window.
        reach = self.crop_half / self.crop_radius * float(spot_radius)
        if self.passes > 1:
            reach += MAX_SHIFT_FRACTION * float(spot_radius)
        return int(math.ceil(reach + 0.5)) + 1

    def _offsets(self):
        return torch.arange(
            -self.crop_half, self.crop_half + 1, device=self.device, dtype=torch.float32
        )

    def _resample(self, crops, centers, step):
        size = crops.shape[-1]
        offsets = self._offsets()
        rows = centers[:, 0, None] + offsets[None] * step  # (M, S') window px
        columns = centers[:, 1, None] + offsets[None] * step
        samples = rows.shape[1]
        grid = torch.stack(
            [
                (columns[:, None, :].expand(-1, samples, -1) / (size - 1)) * 2 - 1,
                (rows[:, :, None].expand(-1, -1, samples) / (size - 1)) * 2 - 1,
            ],
            -1,
        )
        return F.grid_sample(
            crops[:, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    def _normalize(self, sampled):
        count = sampled.shape[0]
        offsets = self._offsets()
        distance = torch.sqrt(offsets[:, None] ** 2 + offsets[None] ** 2).reshape(-1)
        inner, outer = self.annulus
        ring = (distance > inner * self.crop_half) & (
            distance <= outer * self.crop_half
        )
        disk = distance <= self.crop_radius
        flat = sampled.reshape(count, -1)
        background = flat[:, ring].mean(1)
        level = (flat[:, disk].mean(1) - background).abs().clamp_min(1e-3)
        return (
            ((flat - background[:, None]) / level[:, None])
            .reshape_as(sampled)
            .clamp(-4, 8)
        )

    def _coverage_centroid(self, logits):
        """The coverage map's centroid, as an offset from the window center."""
        offsets = self._offsets()
        start = torch.zeros(logits.shape[0], device=logits.device)
        center_y, center_x, area = coverage_centroid(
            torch.sigmoid(logits[:, 0]),
            offsets[None, :, None],
            offsets[None, None, :],
            start,
            start.clone(),
            self.crop_radius,
        )
        return torch.stack([center_y, center_x], 1), area

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
            sampled = torch.cat([sampled, mirror.flip(-1, -2)], 1)
        logits, log_sigmas = [], []
        for start in range(0, len(sampled), FORWARD_BATCH):
            output, log_sigma = self.net(sampled[start : start + FORWARD_BATCH])
            logits.append(output)
            log_sigmas.append(log_sigma)
        offset, area = self._coverage_centroid(torch.cat(logits))
        sigma = torch.exp(torch.cat(log_sigmas)) * step * self.sigma_scale
        return center + offset * step, area, sigma

    @torch.no_grad()
    def refine(
        self, crops, centers, spot_radius, partner_crops=None, partner_centers=None
    ):
        """The refiner interface; a partner network also takes each disk's
        mirror window and the mirror point in that window's pixels."""
        from .device_lock import accelerator_lock

        if len(crops) == 0:
            return np.zeros((0, 2)), np.zeros(0, np.float32)
        if self.needs_partner and partner_crops is None:
            raise ValueError(
                "this center network needs each disk's Friedel-mirror window"
            )
        radius = float(spot_radius)
        step = radius / self.crop_radius
        with accelerator_lock(self.device):
            crop_tensor = _as_device_tensor(crops, self.device)
            seeds = torch.as_tensor(np.asarray(centers, np.float32), device=self.device)
            partner = None
            if self.needs_partner:
                partner_tensor = _as_device_tensor(partner_crops, self.device)
                mirror_seeds = torch.as_tensor(
                    np.asarray(partner_centers, np.float32), device=self.device
                )
                partner = (partner_tensor, mirror_seeds)
            center, area, sigma = self._pass(crop_tensor, seeds, step, partner)
            for _ in range(1, self.passes):
                again = (center - seeds).norm(dim=1) > self.recrop_over * radius
                if not again.any():
                    break
                # a partner window follows the mirror of the moved center
                moved_partner = (
                    None
                    if partner is None
                    else (
                        partner_tensor[again],
                        mirror_seeds[again] - (center[again] - seeds[again]),
                    )
                )
                moved, moved_area, moved_sigma = self._pass(
                    crop_tensor[again], center[again], step, moved_partner
                )
                center[again], area[again], sigma[again] = (
                    moved,
                    moved_area,
                    moved_sigma,
                )
            declined = (
                ((center - seeds).norm(dim=1) >= MAX_SHIFT_FRACTION * radius)
                | (area <= 0.2 * math.pi * self.crop_radius**2)
                | (sigma >= self.max_sigma_fraction * radius)
            )
            center[declined] = float("nan")
            sigma[declined] = float("nan")
            result = center.double().cpu().numpy()
            sigma = sigma.float().cpu().numpy()
        return result, sigma


def _as_device_tensor(crops, device):
    """Windows as a float32 tensor on ``device``: already there when the step
    cut them on the GPU, else moved from numpy."""
    if torch.is_tensor(crops):
        return crops.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(crops, np.float32), device=device)


def input_channels(state_dict) -> int:
    """Input channels, from the first convolution's weight: 1 for a window
    alone, 2 for a window and its Friedel-mirror window."""
    for key in ("encoders.0.0.weight", "encoders.0.0.0.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[1])
    raise ValueError("no first convolution in this center network's state dict")


def load_refiner(path, device, arch: dict | None = None, contract: dict | None = None):
    """A :class:`NetworkRefiner` from a checkpoint; registry ``arch`` / ``input``
    values win over the checkpoint's own scalars."""
    from .device_lock import accelerator_lock

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    arch = arch or {}
    contract = contract or {}

    def setting(key, default):
        return arch.get(key, checkpoint.get(key, default))

    def option(key, default):
        return contract.get(key, checkpoint.get(key, default))

    channels = input_channels(checkpoint["state_dict"])
    net = FastCenterNet(
        int(setting("base", 8)),
        int(setting("levels", 2)),
        bool(setting("depthwise", 0)),
        bool(setting("sigma_head", 1)),
        channels=channels,
    )
    if net.sigma_head is None:
        raise ValueError("a center network needs a sigma head")
    net.load_state_dict(checkpoint["state_dict"])
    with accelerator_lock(device):
        net = net.to(device).eval()
    return NetworkRefiner(
        net,
        device,
        crop_radius=float(option("crop_radius", DEFAULT_CROP_RADIUS)),
        crop_half=int(option("crop_half", DEFAULT_CROP_HALF)),
        annulus=tuple(contract.get("annulus", DEFAULT_ANNULUS)),
        passes=int(contract.get("passes", DEFAULT_PASSES)),
        recrop_over=float(contract.get("recrop_over", DEFAULT_RECROP_OVER)),
        max_sigma_fraction=float(
            contract.get("max_sigma_fraction", DEFAULT_MAX_SIGMA_FRACTION)
        ),
        sigma_scale=float(option("sigma_scale", 1.0)),
        min_spot_radius=float(option("min_spot_radius", DEFAULT_MIN_SPOT_RADIUS)),
        max_spot_radius=float(option("max_spot_radius", math.inf)),
        needs_partner=channels == 2,
    )
