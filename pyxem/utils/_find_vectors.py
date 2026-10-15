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

"""Running a spot detector over every pattern of a dataset.

The dataset is processed one navigation chunk at a time. Each chunk is padded
in navigation space (so the optional navigation blur and the neighbour check
see across chunk seams), blurred, detected, and trimmed back. The full dataset
is never materialised: a lazy signal only ever computes one padded chunk.
"""

import math

import dask.array as da
from dask.array.overlap import ensure_minimum_chunksize
import numpy as np
from scipy.ndimage import gaussian_filter

from pyxem.utils import _spot_finding as spots

#: Navigation chunk side used when an in-memory signal is processed.
EAGER_NAVIGATION_CHUNK = 32
#: Radius over which a neural detection's intensity is averaged.
NEURAL_INTENSITY_RADIUS = 5.0


class SpotFinder:
    """One detection method with its parameters resolved.

    Parameters are the keyword arguments of
    :meth:`pyxem.signals.Diffraction2D.find_vectors`; ``None`` means the
    method's default.
    """

    def __init__(
        self,
        method="dog",
        threshold=None,
        min_distance=None,
        disk_radius=None,
        sigma1=spots.DEFAULT_SIGMA1,
        sigma2=spots.DEFAULT_SIGMA2,
        subpixel=True,
        spot_radius=None,
        background_sigma=None,
        model_path=None,
        model_version=None,
        device=None,
    ):
        method = str(method).lower()
        if method not in spots.METHODS:
            raise ValueError(f"method must be one of {spots.METHODS}, not {method!r}")
        if method == spots.METHOD_NEURAL:
            from pyxem import TORCH_INSTALLED

            if not TORCH_INSTALLED:
                raise ImportError(
                    "method='neural' requires torch: `pip install pyxem[neural]`."
                )
        self.method = method
        self.threshold = float(
            spots.DEFAULT_THRESHOLD[method] if threshold is None else threshold
        )
        self.min_distance = int(
            spots.DEFAULT_MIN_DISTANCE[method] if min_distance is None else min_distance
        )
        self.disk_radius = int(
            spots.DEFAULT_DISK_RADIUS if disk_radius is None else disk_radius
        )
        self.sigma1 = float(sigma1)
        self.sigma2 = float(sigma2)
        self.subpixel = bool(subpixel)
        self.spot_radius = None if not spot_radius else float(spot_radius)
        self.background_sigma = (
            None if background_sigma is None else float(background_sigma)
        )
        self.model_path = model_path
        self.model_version = model_version
        self.device = device

    @property
    def parameters(self):
        return {
            "method": self.method,
            "threshold": self.threshold,
            "min_distance": self.min_distance,
            "disk_radius": self.disk_radius,
            "sigma1": self.sigma1,
            "sigma2": self.sigma2,
            "subpixel": self.subpixel,
            "spot_radius": self.spot_radius,
            "background_sigma": self.background_sigma,
        }

    def marker_radius(self):
        """Radius in pixels at which a detection is drawn."""
        if self.method == spots.METHOD_NXCORR:
            return float(self.disk_radius)
        if self.method == spots.METHOD_DOG:
            return math.sqrt(2) * self.sigma2
        return self.spot_radius or NEURAL_INTENSITY_RADIUS

    def model(self):
        from pyxem.utils._neural_spots import get_model

        return get_model(self.model_path, self.model_version, self.device)

    def calibrate(self, sample_patterns):
        """Tune the neural background sigma (and a weak-data threshold) in place."""
        from pyxem.utils._neural_spots.inference import calibrate

        model, device = self.model()
        result = calibrate(
            model, sample_patterns, device, spot_diameter=self._spot_diameter()
        )
        self.background_sigma = result["background_sigma"]
        return result

    def _spot_diameter(self):
        return 2.0 * self.spot_radius if self.spot_radius else None

    def detect(self, pattern, mask=None):
        """Detect spots in one pattern.

        Returns ``(response, peaks)``: the detector's transformed image and
        ``(N, 3)`` float32 ``[row, column, intensity]``.
        """
        pattern = np.asarray(pattern, dtype=np.float32)
        if self.method == spots.METHOD_NXCORR:
            return spots.find_spots_nxcorr(
                pattern,
                self.disk_radius,
                self.threshold,
                self.min_distance,
                subpixel=self.subpixel,
                mask=mask,
            )
        if self.method == spots.METHOD_DOG:
            return spots.find_spots_dog(
                pattern,
                self.sigma1,
                self.sigma2,
                self.threshold,
                self.min_distance,
                subpixel=self.subpixel,
                mask=mask,
            )
        confident = self.detect_stack(pattern[None], mask)[0]
        # The network returns no dense map; paint each detection's confidence
        # at its pixel so the response still shows where it fired.
        response = np.zeros(pattern.shape, dtype=np.float32)
        if len(confident):
            rows, columns = _nearest_pixels(confident, pattern.shape)
            response[rows, columns] = confident[:, 2]
        return response, self.with_intensity(pattern, confident)

    def detect_stack(self, patterns, mask=None):
        """Detect spots in a ``(N, rows, columns)`` stack.

        For the neural method the third column is the network's confidence;
        pass the result through :meth:`with_intensity` to get intensities.
        """
        if self.method != spots.METHOD_NEURAL:
            return [self.detect(pattern, mask)[1] for pattern in patterns]
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

    def with_intensity(self, pattern, peaks):
        """Replace a neural confidence column by the disk-mean intensity."""
        if self.method != spots.METHOD_NEURAL:
            return peaks
        if len(peaks) == 0:
            return spots.empty_peaks()
        intensity = spots.disk_mean_intensity(
            np.asarray(pattern, dtype=np.float32),
            peaks[:, 0],
            peaks[:, 1],
            NEURAL_INTENSITY_RADIUS,
        )
        return np.column_stack([peaks[:, :2], intensity]).astype(np.float32)


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
):
    """Detect every pattern of one navigation block padded by ``depth``.

    Returns an object array over the trimmed navigation shape whose elements
    are ``(N, 3)`` float32 ``[row, column, intensity]``.
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
    for position, index in np.ndenumerate(indices):
        output[position] = finder.with_intensity(patterns[index], found[index])
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


def automatic_mask(data, navigation_dimension, dilation=spots.DEFAULT_MASK_DILATION):
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
    return spots.detect_beam_stop(
        block.reshape((-1,) + block.shape[navigation_dimension:]).mean(axis=0),
        dilation=dilation,
    )
