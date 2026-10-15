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

"""``get_diffraction_vectors(method="neural")``: the neural spot detector run
over every pattern of a dataset.

The dataset is processed one navigation chunk at a time. Each chunk is padded
in navigation space (so the optional navigation blur and the neighbour check
see across chunk seams), blurred, detected, and trimmed back. The full dataset
is never materialised: a lazy signal only ever computes one padded chunk.

Every detection's intensity is the mean of the raw pattern over a small disk
around it, not the network's confidence. A beam stop is a rejection region,
not an image edit: filling the masked pixels would make a sharp edge that the
detector fires on, so detections inside the mask are dropped instead.

Ported from SpyDE (https://github.com/directelectron/spyde).
"""

import math

import dask.array as da
import numpy as np
from dask.array.overlap import ensure_minimum_chunksize
from scipy.ndimage import gaussian_filter, maximum_filter

METHOD_NEURAL = "neural"
#: The ``get_diffraction_vectors`` methods this module handles; every other
#: method goes to hyperspy's ``find_peaks``.
METHODS = (METHOD_NEURAL,)
DEFAULT_THRESHOLD = 0.3
DEFAULT_MIN_DISTANCE = 4
DEFAULT_MASK_DILATION = 5
#: Navigation chunk side used when an in-memory signal is processed.
EAGER_NAVIGATION_CHUNK = 32
#: Radius over which a detection's intensity is averaged.
INTENSITY_RADIUS = 5.0

#: Keyword arguments of ``get_diffraction_vectors`` that configure the finder.
FINDER_ARGUMENTS = (
    "threshold",
    "min_distance",
    "spot_radius",
    "background_sigma",
    "model_path",
    "model_version",
    "device",
)
#: Keyword arguments that configure how the finder runs over the dataset.
RUN_ARGUMENTS = ("navigation_sigma", "mask", "mask_dilation", "neighbour_refine")


def empty_peaks():
    return np.zeros((0, 3), dtype=np.float32)


def disk_mean_intensity(pattern, rows, columns, radius):
    """Mean of ``pattern`` over a disk of ``radius`` px around each position.

    A single-pixel sample is dominated by shot noise and by where the subpixel
    centre happened to land, which bands an intensity-weighted virtual image;
    the disk mean tracks the true spot brightness.
    """
    rows = np.asarray(rows, dtype=np.float64)
    columns = np.asarray(columns, dtype=np.float64)
    radius = int(max(1, round(float(radius))))
    if rows.size == 0:
        return np.zeros(0, dtype=np.float32)
    height, width = pattern.shape
    offset_rows, offset_columns = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    inside = offset_rows**2 + offset_columns**2 <= radius**2
    sample_rows = np.rint(rows).astype(np.intp)[:, None] + offset_rows[inside][None]
    sample_columns = (
        np.rint(columns).astype(np.intp)[:, None] + offset_columns[inside][None]
    )
    valid = (
        (sample_rows >= 0)
        & (sample_rows < height)
        & (sample_columns >= 0)
        & (sample_columns < width)
    )
    samples = pattern.astype(np.float32, copy=False)[
        np.clip(sample_rows, 0, height - 1), np.clip(sample_columns, 0, width - 1)
    ]
    samples = np.where(valid, samples, np.nan)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(samples, axis=1)
    return np.nan_to_num(mean, nan=0.0).astype(np.float32)


def dilate_mask(mask, pixels):
    """Grow a boolean mask by ``pixels`` in every direction."""
    if mask is None or pixels <= 0:
        return mask
    return maximum_filter(mask.astype(np.uint8), size=2 * int(pixels) + 1) > 0


def detect_beam_stop(mean_pattern, fraction=0.15, dilation=DEFAULT_MASK_DILATION):
    """Find a beam stop in a scan-averaged pattern.

    A beam stop is a stable, connected low-intensity region in the average
    pattern. It is thresholded at ``fraction`` of the mean and then dilated:
    the brightest feature in a beam-stopped pattern is the halo hugging the
    stop edge, so the mask has to reach a few pixels past the geometric stop.

    Returns a boolean mask, or ``None`` when no plausible stop is found.
    """
    mean_pattern = np.asarray(mean_pattern, dtype=np.float64)
    if mean_pattern.ndim != 2 or mean_pattern.size == 0:
        return None
    mean_value = float(mean_pattern.mean())
    if mean_value <= 0:
        return None
    mask = mean_pattern < mean_value * fraction
    # Almost nothing means no stop; almost everything means a blank pattern.
    if not 0.001 <= mask.mean() <= 0.45:
        return None
    return dilate_mask(mask, dilation)


def _nearest_pixels(peaks, shape):
    rows = np.clip(np.rint(peaks[:, 0]).astype(np.intp), 0, shape[0] - 1)
    columns = np.clip(np.rint(peaks[:, 1]).astype(np.intp), 0, shape[1] - 1)
    return rows, columns


def _drop_masked(peaks, mask):
    peaks = np.asarray(peaks, dtype=np.float32).reshape(-1, 3)
    if mask is None or len(peaks) == 0 or not mask.any():
        return peaks
    rows, columns = _nearest_pixels(peaks, mask.shape)
    return peaks[~mask[rows, columns]]


class SpotFinder:
    """The neural spot detector with its settings resolved.

    Parameters are the ``method="neural"`` keyword arguments of
    :meth:`pyxem.signals.Diffraction2D.get_diffraction_vectors`.
    """

    def __init__(
        self,
        threshold=DEFAULT_THRESHOLD,
        min_distance=DEFAULT_MIN_DISTANCE,
        spot_radius=None,
        background_sigma=None,
        model_path=None,
        model_version=None,
        device=None,
    ):
        from pyxem import TORCH_INSTALLED

        if not TORCH_INSTALLED:
            raise ImportError(
                "method='neural' requires torch: `pip install pyxem[neural]`."
            )
        self.threshold = float(threshold)
        self.min_distance = int(min_distance)
        self.spot_radius = float(spot_radius) if spot_radius else None
        self.background_sigma = (
            None if background_sigma is None else float(background_sigma)
        )
        self.model_path = model_path
        self.model_version = model_version
        self.device = device

    @property
    def parameters(self):
        return {
            "method": METHOD_NEURAL,
            "threshold": self.threshold,
            "min_distance": self.min_distance,
            "spot_radius": self.spot_radius,
            "background_sigma": self.background_sigma,
        }

    def marker_radius(self):
        """Radius in pixels at which a detection is drawn."""
        return self.spot_radius or INTENSITY_RADIUS

    def model(self):
        from pyxem.utils._neural_spots import get_model

        return get_model(self.model_path, self.model_version, self.device)

    def _spot_diameter(self):
        return 2.0 * self.spot_radius if self.spot_radius else None

    def calibrate(self, sample_patterns):
        """Tune the background sigma (and a weak-data threshold) in place."""
        from pyxem.utils._neural_spots.inference import calibrate

        model, device = self.model()
        result = calibrate(
            model, sample_patterns, device, spot_diameter=self._spot_diameter()
        )
        self.background_sigma = result["background_sigma"]
        return result

    def detect_stack(self, patterns, mask=None):
        """Detect spots in a ``(N, rows, columns)`` stack.

        The third column is the network's confidence; :meth:`with_intensity`
        replaces it by the intensity.
        """
        from pyxem.utils._neural_spots.inference import detect_batch

        model, device = self.model()
        found = detect_batch(
            model,
            patterns,
            device,
            threshold=self.threshold,
            min_distance=self.min_distance,
            background_sigma=self.background_sigma,
            spot_diameter=self._spot_diameter(),
        )
        return [_drop_masked(peaks, mask) for peaks in found]

    def detect(self, pattern, mask=None):
        """Detect spots in one pattern.

        Returns ``(response, peaks)``: an image with each detection's
        confidence painted at its pixel (the network returns no dense map),
        and ``(N, 3)`` float32 ``[row, column, intensity]``.
        """
        pattern = np.asarray(pattern, dtype=np.float32)
        confident = self.detect_stack(pattern[None], mask)[0]
        response = np.zeros(pattern.shape, dtype=np.float32)
        if len(confident):
            rows, columns = _nearest_pixels(confident, pattern.shape)
            response[rows, columns] = confident[:, 2]
        return response, self.with_intensity(pattern, confident)

    @staticmethod
    def with_intensity(pattern, peaks):
        """Replace the confidence column by the disk-mean intensity."""
        if len(peaks) == 0:
            return empty_peaks()
        intensity = disk_mean_intensity(
            np.asarray(pattern, dtype=np.float32),
            peaks[:, 0],
            peaks[:, 1],
            INTENSITY_RADIUS,
        )
        return np.column_stack([peaks[:, :2], intensity]).astype(np.float32)


def navigation_depth(navigation_sigma, neighbour_refine=False):
    depth = int(math.ceil(3 * navigation_sigma)) if navigation_sigma > 0 else 0
    return max(depth, 1) if neighbour_refine else depth


def _find_vectors_block(
    block,
    finder,
    navigation_dimension,
    navigation_sigma,
    depth,
    mask,
    neighbour_refine,
    get_intensity,
):
    """Detect every pattern of one navigation block padded by ``depth``.

    Returns an object array over the trimmed navigation shape whose elements
    are ``(N, 3)`` float32 ``[row, column, intensity]``, or ``(N, 2)``
    without ``get_intensity``.
    """
    block = np.asarray(block, dtype=np.float32)
    if navigation_sigma > 0:
        sigma = (navigation_sigma,) * navigation_dimension + (0, 0)
        # truncate=3 keeps the kernel inside the 3-sigma padding, so the
        # result does not depend on where the chunk boundaries fall.
        block = gaussian_filter(block, sigma=sigma, mode="reflect", truncate=3.0)
    navigation_shape = block.shape[:navigation_dimension]
    patterns = block.reshape((-1,) + block.shape[navigation_dimension:])
    found = finder.detect_stack(patterns, mask)
    trimmed = tuple(slice(depth, size - depth) for size in navigation_shape)

    if neighbour_refine:
        found = _refine_with_neighbours(found, navigation_shape, patterns.shape[1:])

    indices = np.arange(len(patterns)).reshape(navigation_shape)[trimmed]
    output = np.empty(indices.shape, dtype=object)
    columns = slice(None) if get_intensity else slice(0, 2)
    for position, index in np.ndenumerate(indices):
        peaks = finder.with_intensity(patterns[index], found[index])
        output[position] = peaks[:, columns]
    return output


def _refine_with_neighbours(found, navigation_shape, pattern_shape):
    """Keep the candidates that recur in neighbouring scan positions.

    Only the trimmed interior is refined, so every refined position has all
    four neighbours in the padded block.
    """
    from pyxem.utils._neural_spots.refine import refine

    rows, columns = navigation_shape
    grid = np.arange(rows * columns).reshape(navigation_shape)
    refined = list(found)
    for row in range(1, rows - 1):
        for column in range(1, columns - 1):
            neighbours = [
                found[grid[row + dr, column + dc]]
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            ]
            refined[grid[row, column]] = refine(
                found[grid[row, column]],
                neighbours,
                frame_shape=pattern_shape,
                w_conf=0.2,
                w_persist=1.0,
                w_friedel=0.2,
                keep_score=0.55,
                min_persist=0.5,
            )
    return refined


def find_vectors_array(
    data,
    finder,
    navigation_dimension,
    navigation_sigma=0.0,
    mask=None,
    neighbour_refine=False,
    get_intensity=True,
):
    """Run ``finder`` over every pattern of ``data``.

    Returns an object array with the navigation shape of ``data``: a numpy
    array for numpy input, a lazy dask array for dask input.
    """
    lazy = isinstance(data, da.Array)
    if not lazy:
        data = da.from_array(
            data,
            chunks=(EAGER_NAVIGATION_CHUNK,) * navigation_dimension + (-1, -1),
        )
    elif any(len(chunks) > 1 for chunks in data.chunks[navigation_dimension:]):
        # A detector needs whole patterns; only the two signal axes are merged.
        data = data.rechunk({navigation_dimension: -1, navigation_dimension + 1: -1})

    depth = navigation_depth(navigation_sigma, neighbour_refine)
    navigation_chunks = data.chunks[:navigation_dimension]
    if depth:
        # Every chunk must be at least as deep as the padding it lends.
        data = data.rechunk(
            {
                axis: ensure_minimum_chunksize(depth, data.chunks[axis])
                for axis in range(navigation_dimension)
            }
        )
        navigation_chunks = data.chunks[:navigation_dimension]
        data = da.overlap.overlap(
            data,
            depth={axis: depth for axis in range(navigation_dimension)},
            boundary="reflect",
        )

    result = data.map_blocks(
        _find_vectors_block,
        finder=finder,
        navigation_dimension=navigation_dimension,
        navigation_sigma=float(navigation_sigma),
        depth=depth,
        mask=mask,
        neighbour_refine=neighbour_refine,
        get_intensity=get_intensity,
        drop_axis=(navigation_dimension, navigation_dimension + 1),
        chunks=navigation_chunks,
        dtype=object,
        meta=np.empty((0,) * navigation_dimension, dtype=object),
    )
    if lazy:
        return result
    return result.compute(scheduler="threads")


def sample_patterns(data, navigation_dimension, fractions=(0.25, 0.5, 0.75)):
    """A few patterns spread across the scan, each read on its own."""
    navigation_shape = data.shape[:navigation_dimension]
    total = int(np.prod(navigation_shape))
    patterns = []
    for fraction in fractions:
        index = np.unravel_index(
            min(total - 1, int(fraction * total)), navigation_shape
        )
        pattern = data[index]
        if hasattr(pattern, "compute"):
            pattern = pattern.compute()
        patterns.append(np.asarray(pattern, dtype=np.float32))
    return patterns


def automatic_mask(data, navigation_dimension, dilation=DEFAULT_MASK_DILATION):
    """Detect a beam stop from the mean of one block of up to 64 patterns.

    A beam stop does not move across a scan, so one contiguous block (one
    storage chunk on lazy data) reproduces it without reading the dataset.
    """
    side = int(math.ceil(64 ** (1 / navigation_dimension)))
    if isinstance(data, da.Array):
        corner = tuple(
            slice(0, min(side, chunks[0]))
            for chunks in data.chunks[:navigation_dimension]
        )
        block = data[corner].compute()
    else:
        block = data[(slice(0, side),) * navigation_dimension]
    block = np.asarray(block, dtype=np.float64)
    return detect_beam_stop(
        block.reshape((-1,) + block.shape[navigation_dimension:]).mean(axis=0),
        dilation=dilation,
    )


def resolve_finder(signal, method, **kwargs):
    """Build the :class:`SpotFinder` and run settings ``get_diffraction_vectors``
    asked for, resolving the automatic choices (beam-stop mask, background
    sigma) on a few patterns of ``signal``.

    Returns ``(finder, settings, mask_options)``: ``settings`` holds
    ``mask``, ``navigation_sigma`` and ``neighbour_refine``; ``mask_options``
    says whether the mask was detected automatically and with what dilation,
    so an interactive tool can re-dilate it.
    """
    unknown = set(kwargs) - set(FINDER_ARGUMENTS) - set(RUN_ARGUMENTS)
    if unknown:
        raise TypeError(
            f"get_diffraction_vectors(method={method!r}) got unexpected keyword "
            f"arguments {sorted(unknown)}"
        )
    finder_arguments = {k: kwargs[k] for k in FINDER_ARGUMENTS if k in kwargs}
    background_sigma = finder_arguments.pop("background_sigma", "auto")
    finder = SpotFinder(
        background_sigma=(
            None if isinstance(background_sigma, str) else background_sigma
        ),
        **finder_arguments,
    )
    navigation_dimension = signal.axes_manager.navigation_dimension
    neighbour_refine = bool(kwargs.get("neighbour_refine", False))
    if neighbour_refine and navigation_dimension != 2:
        raise ValueError(
            "neighbour_refine needs a signal with two navigation dimensions."
        )
    data = signal.data

    mask = kwargs.get("mask")
    mask_options = dict(
        automatic=isinstance(mask, str),
        dilation=int(kwargs.get("mask_dilation", DEFAULT_MASK_DILATION)),
    )
    if isinstance(mask, str):
        if mask != "auto":
            raise ValueError(f"mask must be an array, 'auto' or None, not {mask!r}")
        mask = automatic_mask(
            data, navigation_dimension, dilation=mask_options["dilation"]
        )
    elif mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != data.shape[navigation_dimension:]:
            raise ValueError(
                f"mask shape {mask.shape} does not match the pattern shape "
                f"{data.shape[navigation_dimension:]}"
            )

    if background_sigma == "auto":
        calibrated = finder.calibrate(sample_patterns(data, navigation_dimension))
        if "threshold" not in finder_arguments:
            finder.threshold = calibrated["threshold"]

    settings = dict(
        mask=mask,
        navigation_sigma=float(kwargs.get("navigation_sigma", 0.0)),
        neighbour_refine=neighbour_refine,
    )
    return finder, settings, mask_options
