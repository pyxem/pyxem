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
# spyde/models/unet.py; see vendor note in __init__.py.

"""Small U-Net with heatmap + subpixel-offset heads (CenterNet-style).

Vendored verbatim from the ``yoloDiffraction`` research project
(``yolodiffraction/model/unet.py``) so SpyDE carries its own copy of the
inference architecture — DO NOT edit the math here; it must stay bit-faithful
to the trained checkpoints (train/infer parity).

Output heads (all at input resolution):
  - heatmap  : 1 channel, spot-presence score (sigmoid). Trained against a
               Gaussian blob at each GT spot (focal/MSE loss).
  - offset   : 2 channels, subpixel (dy, dx) in (-0.5, 0.5], trained with L1 ONLY
               at GT spot pixels. This is the subpixel-accuracy core.
  - mask     : optional (``mask_head=True``), 1 channel, the pixel coverage of
               each disk's aperture (sigmoid). Its centroid is the disk's
               geometric center wherever the fill sits inside it, which the
               offset head misses on dynamically filled disks (decode.py).

The input is (B, C, H, W). C = number of stacked frames; C=1 is single-frame,
C=9 is a 3x3 neighbor stack (multi-frame coupling drops in with no architecture
change). Kept small (~1-2 M params) for the "small & fast" requirement and a
CPU-fast inference path.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class SpotUNet(nn.Module):
    """Compact U-Net with a CONTROLLED receptive field. ``in_ch`` = stacked frames.

    ``levels`` = number of downsampling steps (default 2). This is the key knob
    against the symmetry-overfitting risk: the receptive field must see roughly ONE
    disk + immediate surround (~30-40 px for a 12 px disk / 21 px lattice), NOT the
    whole periodic lattice. levels=3 gives a ~128 px RF (sees ~6 lattice cells ->
    can learn the lattice/symmetry instead of the spot); levels=2 keeps it local;
    levels=1 is a pure local conv stack. The model must be a LOCAL spot detector so
    it generalizes to defects, grain boundaries, isolated spots, and amorphous rings.
    """

    def __init__(
        self, in_ch: int = 1, base: int = 16, levels: int = 2, mask_head: bool = False
    ):
        super().__init__()
        self.levels = levels
        chs = [base * (2**i) for i in range(levels + 1)]
        self.pool = nn.MaxPool2d(2)
        self.enc = nn.ModuleList()
        cin = in_ch
        for c in chs:
            self.enc.append(_conv_block(cin, c))
            cin = c
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(levels, 0, -1):
            self.up.append(nn.ConvTranspose2d(chs[i], chs[i - 1], 2, stride=2))
            self.dec.append(_conv_block(chs[i - 1] * 2, chs[i - 1]))
        self.head_hm = nn.Conv2d(base, 1, 1)
        self.head_off = nn.Conv2d(base, 2, 1)
        nn.init.constant_(self.head_hm.bias, -4.0)
        self.head_mask = nn.Conv2d(base, 1, 1) if mask_head else None

    def forward(self, x):
        feats = []
        h = x
        for i, enc in enumerate(self.enc):
            h = enc(h if i == 0 else self.pool(h))
            feats.append(h)
        d = feats[-1]
        for j, (up, dec) in enumerate(zip(self.up, self.dec)):
            skip = feats[self.levels - 1 - j]
            d = dec(torch.cat([up(d), skip], 1))
        hm = self.head_hm(d)
        off = torch.tanh(self.head_off(d)) * 1.6
        if self.head_mask is not None:
            return hm, off, self.head_mask(d)
        return hm, off

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
