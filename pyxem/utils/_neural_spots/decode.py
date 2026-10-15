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

"""Heatmap+offset decoding for the SpotUNet detector.

Vendored from the ``yoloDiffraction`` research project
(``yolodiffraction/model/targets.py``) — only the DECODE side is carried over
(the training target-rendering and losses are not needed for inference). DO NOT
change the maths here; it must match how the checkpoints were trained.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def decode(hm_logits, off, thresh=0.3, min_distance=3, topk=None):
    """Decode one frame's (1,H,W) logits + (2,H,W) offsets -> (N,3) [y,x,score].

    Heatmap local-max via max-pool (>= thresh), then add the predicted subpixel
    offset at each peak pixel.
    """
    hm = torch.sigmoid(hm_logits)
    if hm.dim() == 3:
        hm = hm.unsqueeze(0)
        off = off.unsqueeze(0)
    B, _, H, W = hm.shape
    k = 2 * min_distance + 1
    pooled = F.max_pool2d(hm, k, stride=1, padding=min_distance)
    peak = (hm == pooled) & (hm >= thresh)
    out = []
    for b in range(B):
        ys, xs = torch.where(peak[b, 0])
        if topk is not None and len(ys) > topk:
            scores = hm[b, 0, ys, xs]
            sel = torch.argsort(scores, descending=True)[:topk]
            ys, xs = ys[sel], xs[sel]
        dy = off[b, 0, ys, xs]
        dx = off[b, 1, ys, xs]
        s = hm[b, 0, ys, xs]
        res = torch.stack([ys.float() + dy, xs.float() + dx, s], dim=1)
        out.append(res.detach().cpu().numpy())
    return out if B > 1 else out[0]


@torch.no_grad()
def decode_batch(hm_logits, off, thresh=0.3, min_distance=3, return_numpy=True):
    """Fully-batched, GPU-resident decode. Returns (M,4) [batch, y, x, score] for ALL
    peaks across the whole batch — ONE host transfer, no per-frame python loop.

    This is the high-throughput path (the old per-frame `decode` with torch.where +
    .cpu() per frame caps throughput at a few hundred fps; this stays on-GPU and is
    ~50x+ faster batched). Split back per frame on the host via the batch column.

    hm_logits: (B,1,H,W) logits. off: (B,2,H,W). Apply subpixel offset at each peak.
    """
    hm = torch.sigmoid(hm_logits)
    B, _, H, W = hm.shape
    k = 2 * min_distance + 1
    pooled = F.max_pool2d(hm, k, stride=1, padding=min_distance)
    mask = (hm == pooled) & (hm >= thresh)  # (B,1,H,W) bool
    idx = mask.squeeze(1).nonzero(as_tuple=False)  # (M,3) [b, y, x] — ONE op
    if idx.numel() == 0:
        empty = torch.zeros((0, 4), device=hm.device)
        return empty.cpu().numpy() if return_numpy else empty
    b, ys, xs = idx[:, 0], idx[:, 1], idx[:, 2]
    dy = off[b, 0, ys, xs]
    dx = off[b, 1, ys, xs]
    s = hm[b, 0, ys, xs]
    res = torch.stack([b.float(), ys.float() + dy, xs.float() + dx, s], dim=1)
    return res.cpu().numpy() if return_numpy else res


def split_by_batch(res, B):
    """Split (M,4) [batch,y,x,score] from decode_batch into a list of B (Ni,3)
    [y,x,score] arrays (host-side, cheap)."""
    out = [np.zeros((0, 3), np.float32) for _ in range(B)]
    if len(res) == 0:
        return out
    res = np.asarray(res)
    order = np.argsort(res[:, 0], kind="stable")
    res = res[order]
    bounds = np.searchsorted(res[:, 0], np.arange(B + 1) - 0.5)
    for i in range(B):
        if bounds[i + 1] > bounds[i]:
            out[i] = res[bounds[i] : bounds[i + 1], 1:].astype(np.float32)
    return out
