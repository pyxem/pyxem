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

"""Normalised cross-correlation of a whole stack of patterns at once.

:func:`match_template_stack` computes exactly what
:func:`skimage.feature.match_template` computes for each pattern (Lewis's
fast normalised cross-correlation: one FFT correlation, window statistics
from integral images, zero padding by the template size), but for a stack of
patterns in one call: the template's transform is computed once and the
pattern transforms are batched through torch, on the CPU (all cores in one
call) or a GPU. It matches scikit-image to floating-point rounding: about
1e-14 in float64; in float32 up to a few 1e-4 on large patterns, which is
the size of scikit-image's own float32 error (its running window sums lose
precision; on 256 x 256 patterns scikit-image is 1.7e-4 from the float64
result and this is 7e-5).

A batched numpy version was tried and is SLOWER than calling scikit-image
per pattern (5.0 vs 3.5 ms per 256 x 256 pattern): the stack's
intermediates fall out of cache. The speed comes from torch's multi-threaded
and GPU transforms, not from batching itself.

Ported from SpyDE (https://github.com/directelectron/spyde).
"""

import threading

import dask.array as da
import numpy as np

#: Patterns per torch call: bounds the intermediates (about eight arrays of
#: the padded stack) to a few hundred MB for 256 x 256 float64 patterns.
SUB_BATCH_SIZE = 64

# torch already spreads one call over every core (or owns the GPU), so dask
# threads take turns rather than oversubscribing it.
_DEVICE_LOCK = threading.Lock()


def float_type(dtype):
    """The floating-point type scikit-image computes ``dtype`` in."""
    dtype = np.dtype(dtype)
    if dtype in (np.float32, np.float64):
        return dtype
    if dtype == np.float16:
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def _window_sum(image, window_shape, cumsum):
    """Sum over every window of ``window_shape`` of the last two axes, as
    :func:`skimage.feature.template._window_sum_2d` computes it."""
    rows, columns = window_shape
    summed = cumsum(image, -2)
    summed = summed[..., rows:-1, :] - summed[..., : -rows - 1, :]
    summed = cumsum(summed, -1)
    return summed[..., :, columns:-1] - summed[..., :, : -columns - 1]


def _crop(response, template_shape, pattern_shape, pad_input):
    rows, columns = template_shape
    height, width = pattern_shape
    if pad_input:
        row0, column0 = (rows - 1) // 2, (columns - 1) // 2
        return response[..., row0 : row0 + height, column0 : column0 + width]
    row0, column0 = rows - 1, columns - 1
    return response[
        ..., row0 : row0 + height - rows + 1, column0 : column0 + width - columns + 1
    ]


def match_template_stack(
    stack,
    template,
    pad_input=True,
    mode="constant",
    constant_values=0,
    device=None,
):
    """Normalised cross-correlation of every pattern in ``stack`` with ``template``.

    Equivalent to ``np.stack([skimage.feature.match_template(p, template,
    pad_input, mode, constant_values) for p in stack])``.

    Parameters
    ----------
    stack : numpy.ndarray
        ``(N, rows, columns)`` patterns.
    template : numpy.ndarray
        The 2-D template.
    pad_input, mode, constant_values
        As for :func:`skimage.feature.match_template`.
    device : str or None
        The torch device (``"cpu"``, ``"cuda"``, ``"mps"``); ``None`` picks
        CUDA, then Apple MPS, then the CPU.

    Returns
    -------
    numpy.ndarray
        ``(N, rows', columns')`` correlation in [-1, 1], in the floating-point
        type scikit-image would use for ``stack``.
    """
    stack = np.asarray(stack)
    dtype = float_type(stack.dtype)
    stack = stack.astype(dtype, copy=False)
    template = np.asarray(template, dtype=dtype)
    if np.any(np.less(stack.shape[1:], template.shape)):
        raise ValueError("Image must be larger than template.")
    padding = ((0, 0),) + tuple((width, width) for width in template.shape)
    if mode == "constant":
        image = np.pad(stack, padding, mode=mode, constant_values=constant_values)
    else:
        image = np.pad(stack, padding, mode=mode)
    response = _correlate_torch(image, template, default_device(device))
    return _crop(response, template.shape, stack.shape[1:], pad_input)


def default_device(device=None):
    import torch

    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _correlate_torch(image, template, device):
    import torch

    dtype = image.dtype
    rows, columns = template.shape
    height, width = image.shape[1:]
    image_tensor = torch.from_numpy(np.ascontiguousarray(image)).to(device)
    template_tensor = torch.from_numpy(np.ascontiguousarray(template[::-1, ::-1])).to(
        device
    )
    window_sum = _window_sum(image_tensor, template.shape, torch.cumsum)
    window_sum_squared = _window_sum(image_tensor**2, template.shape, torch.cumsum)
    shape = (height + rows - 1, width + columns - 1)
    full = torch.fft.irfft2(
        torch.fft.rfft2(image_tensor, s=shape)
        * torch.fft.rfft2(template_tensor, s=shape),
        s=shape,
    )
    xcorr = full[..., rows : height - 1, columns : width - 1]
    del full, image_tensor

    template_mean = float(template.mean())
    template_ssd = float(np.sum((template - template_mean) ** 2))
    numerator = xcorr - window_sum * template_mean
    denominator = window_sum_squared - window_sum * window_sum / template.size
    denominator = torch.sqrt(torch.clamp(denominator * template_ssd, min=0))
    valid = denominator > np.finfo(dtype).eps
    response = torch.where(
        valid, numerator / torch.where(valid, denominator, 1), torch.zeros_like(xcorr)
    )
    return response.cpu().numpy()


def _template_match_block(
    block, template, subtract_min, pad_input, mode, constant_values, device
):
    pattern_shape = block.shape[-2:]
    patterns = block.reshape((-1,) + pattern_shape)
    results = []
    for start in range(0, len(patterns), SUB_BATCH_SIZE):
        with _DEVICE_LOCK:
            results.append(
                match_template_stack(
                    patterns[start : start + SUB_BATCH_SIZE],
                    template,
                    pad_input=pad_input,
                    mode=mode,
                    constant_values=constant_values,
                    device=device,
                )
            )
    response = np.concatenate(results) if results else np.zeros((0,) + pattern_shape)
    if subtract_min:
        response = response - response.min(axis=(-2, -1), keepdims=True)
    return response.reshape(block.shape[:-2] + response.shape[-2:])


def template_match_array(
    data,
    template,
    subtract_min=True,
    pad_input=True,
    mode="constant",
    constant_values=0,
    device=None,
):
    """:func:`pyxem.utils.diffraction.normalize_template_match` on every
    pattern of ``data`` (any navigation shape), one chunk of patterns per torch
    call. Returns a dask array; the signal axes must not be split into chunks
    (they are merged if they are)."""
    data = data if isinstance(data, da.Array) else da.from_array(data)
    if data.numblocks[-2:] != (1, 1):
        data = data.rechunk({data.ndim - 2: -1, data.ndim - 1: -1})
    template = np.asarray(template)
    if pad_input:
        output_shape = data.shape[-2:]
    else:
        output_shape = tuple(
            size - width + 1 for size, width in zip(data.shape[-2:], template.shape)
        )
    return data.map_blocks(
        _template_match_block,
        template=template,
        subtract_min=subtract_min,
        pad_input=pad_input,
        mode=mode,
        constant_values=constant_values,
        device=device,
        chunks=data.chunks[:-2] + tuple((size,) for size in output_shape),
        dtype=float_type(data.dtype),
        meta=np.empty((0,) * data.ndim, dtype=float_type(data.dtype)),
    )
