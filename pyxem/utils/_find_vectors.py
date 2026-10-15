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
DEFAULT_MIN_DISTANCE = 4
DEFAULT_MASK_DILATION = 5
#: Navigation chunk side used when an in-memory signal is processed.
EAGER_NAVIGATION_CHUNK = 32
#: Radius over which a detection's intensity is averaged.
INTENSITY_RADIUS = 5.0

#: Column indices kept with and without ``get_intensity``.
OUTPUT_COLUMNS = {True: [0, 1, 2, 3, 4], False: [0, 1, 3, 4]}

#: Keyword arguments of ``get_diffraction_vectors`` that configure the finder.
FINDER_ARGUMENTS = (
    "threshold",
    "min_distance",
    "spot_radius",
    "background_sigma",
    "model",
    "model_path",
    "model_version",
    "device",
    "refine_centres",
    "friedel_partner",
)
#: Keyword arguments that configure how the finder runs over the dataset.
RUN_ARGUMENTS = ("navigation_sigma", "mask", "mask_dilation", "neighbour_refine")


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
    if mask is None or len(peaks) == 0 or not mask.any():
        return peaks
    rows, columns = _nearest_pixels(peaks, mask.shape)
    return peaks[~mask[rows, columns]]


#: Columns of every detection: position, the raw disk-mean intensity, the
#: network's confidence and the centre's positional uncertainty (pixels).
PEAK_COLUMNS = ("y", "x", "intensity", "confidence", "sigma")


def empty_peaks():
    return np.zeros((0, len(PEAK_COLUMNS)), dtype=np.float32)


def _masked_median(values, mask):
    from pyxem.utils._neural_spots.centre_refine import masked_median

    return masked_median(values, mask)


def _counts_radius(reference, spot_radius):
    """Radius (px) of the aperture a disk's counts are read in for its sigma:
    the given spot radius, else the autocorrelation disk-size estimate the
    network scales by."""
    if spot_radius:
        return float(spot_radius)
    from pyxem.utils._neural_spots.preprocess import estimate_disk_diameter

    diameter = estimate_disk_diameter(np.asarray(reference, np.float32))
    if not np.isfinite(diameter):
        return INTENSITY_RADIUS
    return float(np.clip(0.5 * diameter, 2.0, 64.0))


def _disk_counts(pattern, positions, radius):
    """Background-subtracted counts inside ``radius`` of each ``(y, x)``, and
    the background per pixel (median of the ring 1-4 px outside the disk)."""
    outer = radius + 4.0
    half = int(np.ceil(outer)) + 1
    padded = np.pad(np.asarray(pattern, np.float32), half, mode="edge")
    rows = np.rint(positions[:, 0]).astype(np.intp)
    columns = np.rint(positions[:, 1]).astype(np.intp)
    steps = np.arange(-half, half + 1)
    grid_rows = np.clip(
        (rows[:, None] + half + steps[None, :])[:, :, None], 0, padded.shape[0] - 1
    )
    grid_columns = np.clip(
        (columns[:, None] + half + steps[None, :])[:, None, :], 0, padded.shape[1] - 1
    )
    windows = padded[grid_rows, grid_columns]
    offset_rows = (rows[:, None] + steps[None, :])[:, :, None] - positions[
        :, 0, None, None
    ]
    offset_columns = (columns[:, None] + steps[None, :])[:, None, :] - positions[
        :, 1, None, None
    ]
    distance = np.sqrt(offset_rows**2 + offset_columns**2)
    ring = (distance > radius + 1.0) & (distance <= outer)
    background = _masked_median(
        windows.reshape(len(positions), -1), ring.reshape(len(positions), -1)
    )
    signal = ((windows - background[:, None, None]) * (distance <= radius)).sum((1, 2))
    return signal, background


def _peak_records(pattern, predictions, counts_radius, refined_sigma=None):
    """The network's ``(N, 4)`` ``[y, x, confidence, width]`` -> ``(N, 5)``
    ``[y, x, intensity, confidence, sigma]``.

    ``refined_sigma`` is the centre steps' own uncertainty (NaN where they
    kept a disk); when they ran it is the sigma column, else the counting
    noise limit is. The two are never mixed in one pattern: the counting
    limit is tens of times smaller, so a disk the steps declined would
    outweigh every disk they placed.
    """
    from pyxem.utils._neural_spots.centre import positional_sigma

    if len(predictions) == 0:
        return empty_peaks()
    pattern = np.asarray(pattern, np.float32)
    positions = predictions[:, :2]
    intensity = disk_mean_intensity(
        pattern, positions[:, 0], positions[:, 1], INTENSITY_RADIUS
    )
    if refined_sigma is not None:
        sigma = refined_sigma
    else:
        sigma = positional_sigma(
            predictions[:, 3], *_disk_counts(pattern, positions, counts_radius)
        )
    return np.column_stack([positions, intensity, predictions[:, 2], sigma]).astype(
        np.float32
    )


class SpotFinder:
    """The neural spot detector with its settings resolved.

    Parameters are the ``method="neural"`` keyword arguments of
    :meth:`pyxem.signals.Diffraction2D.get_diffraction_vectors`.

    Every pattern of a dataset is rescaled by the disk size of one
    :attr:`reference` pattern (as SpyDE's batch resolves it from a chunk's
    first pattern), so the result of a pattern does not depend on which
    chunk, or which preview, it is computed in.
    """

    def __init__(
        self,
        threshold=None,
        min_distance=DEFAULT_MIN_DISTANCE,
        spot_radius=None,
        background_sigma=None,
        model=None,
        model_path=None,
        model_version=None,
        device=None,
        refine_centres=True,
        friedel_partner=True,
    ):
        from pyxem import TORCH_INSTALLED

        if not TORCH_INSTALLED:
            raise ImportError(
                "method='neural' requires torch: `pip install pyxem[neural]`."
            )
        self.threshold = None if threshold is None else float(threshold)
        self.min_distance = int(min_distance)
        self.spot_radius = float(spot_radius) if spot_radius else None
        self.background_sigma = (
            None if background_sigma is None else float(background_sigma)
        )
        self.model_name = model
        self.model_path = model_path
        self.model_version = model_version
        self.device = device
        self.refine_centres = bool(refine_centres)
        self.friedel_partner = bool(friedel_partner)
        self.reference = None
        if self.threshold is None:
            self.threshold = float(getattr(self.model()[0], "default_threshold", 0.3))

    @property
    def parameters(self):
        return {
            "method": METHOD_NEURAL,
            "model": self.model_name or _default_detector(),
            "threshold": self.threshold,
            "min_distance": self.min_distance,
            "spot_radius": self.spot_radius,
            "background_sigma": self.background_sigma,
            "refine_centres": self.refine_centres,
            "friedel_partner": self.friedel_partner,
        }

    def marker_radius(self):
        """Radius in pixels at which a detection is drawn."""
        if self.spot_radius:
            return self.spot_radius
        if self.reference is not None:
            return _counts_radius(self.reference, None)
        return INTENSITY_RADIUS

    def model(self):
        from pyxem.utils._neural_spots import get_model

        return get_model(
            self.model_name, self.model_path, self.model_version, self.device
        )

    def _spot_diameter(self):
        return 2.0 * self.spot_radius if self.spot_radius else None

    def calibrate(self, sample_patterns):
        """Tune the background sigma (and a weak-data threshold) in place."""
        from pyxem.utils._neural_spots.device_lock import accelerator_lock
        from pyxem.utils._neural_spots.infer import calibrate

        model, device = self.model()
        with accelerator_lock(device):
            result = calibrate(
                model, sample_patterns, device, spot_diameter=self._spot_diameter()
            )
        self.background_sigma = result["bg_sigma"]
        return result

    def detect_stack(self, patterns, mask=None):
        """Detect spots in a ``(N, rows, columns)`` stack.

        Returns a list of ``(N_i, 5)`` float32
        ``[y, x, intensity, confidence, sigma]``, the same as SpyDE's batch.
        """
        from pyxem.utils._neural_spots.device_lock import accelerator_lock
        from pyxem.utils._neural_spots.infer import detect_batch

        patterns = np.asarray(patterns, dtype=np.float32)
        if len(patterns) == 0:
            return []
        reference = patterns[0] if self.reference is None else self.reference
        model, device = self.model()
        with accelerator_lock(device):
            found = detect_batch(
                model,
                np.concatenate([np.asarray(reference, np.float32)[None], patterns]),
                device,
                thresh=self.threshold,
                min_distance=self.min_distance,
                bg_sigma=self.background_sigma,
                spot_diameter=self._spot_diameter(),
                with_width=True,
            )[1:]
        predictions = [
            _drop_masked(np.asarray(p, np.float32).reshape(-1, 4), mask) for p in found
        ]
        counts_radius = _counts_radius(reference, self.spot_radius)
        predictions, refined_sigmas = self._refined(
            patterns, predictions, counts_radius, device
        )
        return [
            _peak_records(pattern, prediction, counts_radius, sigma)
            for pattern, prediction, sigma in zip(patterns, predictions, refined_sigmas)
        ]

    def _refined(self, patterns, predictions, spot_radius, device):
        """The centre steps over a stack's predictions, one batch per step:
        the refine network, then the Friedel-partner network on its centres.
        Returns the predictions with the refined ``[y, x]`` and, per pattern,
        the uncertainty of the last step that moved each disk (None with no
        step)."""
        from pyxem.utils._neural_spots.centre_refine import (
            CENTRE_MODEL,
            FRIEDEL_MODEL,
            load_step,
            refine_centres,
        )

        if not self.refine_centres:
            return predictions, [None] * len(predictions)
        positions = [p[:, :2] for p in predictions]
        sigmas = None
        steps = [CENTRE_MODEL] + ([FRIEDEL_MODEL] if self.friedel_partner else [])
        for step in steps:
            refiner = load_step(step, device)
            if refiner is None:
                continue
            stepped, step_sigmas = refine_centres(
                patterns, positions, spot_radius, refiner
            )
            if step_sigmas is not None:
                if sigmas is None:
                    sigmas = step_sigmas
                else:
                    # A disk this step kept (NaN) keeps the earlier uncertainty.
                    sigmas = [
                        np.where(np.isnan(current), previous, current)
                        for previous, current in zip(sigmas, step_sigmas)
                    ]
            positions = stepped
        if sigmas is None:
            sigmas = [None] * len(predictions)
        refined = []
        for prediction, position in zip(predictions, positions):
            prediction = prediction.copy()
            prediction[:, :2] = position
            refined.append(prediction)
        return refined, sigmas

    def detect(self, pattern, mask=None):
        """Detect spots in one pattern.

        Returns ``(response, peaks)``: an image with each detection's
        confidence painted at its pixel (the network returns no dense map),
        and ``(N, 5)`` float32 ``[y, x, intensity, confidence, sigma]``.
        """
        pattern = np.asarray(pattern, dtype=np.float32)
        peaks = self.detect_stack(pattern[None], mask)[0]
        response = np.zeros(pattern.shape, dtype=np.float32)
        if len(peaks):
            rows, columns = _nearest_pixels(peaks, pattern.shape)
            response[rows, columns] = peaks[:, 3]
        return response, peaks


def _default_detector():
    from pyxem.utils._neural_spots import DEFAULT_DETECTOR

    return DEFAULT_DETECTOR


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
    are float32 ``[y, x, intensity, confidence, sigma]`` rows (without the
    intensity when not ``get_intensity``).
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
    columns = OUTPUT_COLUMNS[get_intensity]
    for position, index in np.ndenumerate(indices):
        output[position] = found[index][:, columns]
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
            peaks = found[grid[row, column]]
            if len(peaks) == 0:
                continue
            kept = refine(
                peaks[:, :3],
                [neighbour[:, :3] for neighbour in neighbours],
                frame_shape=pattern_shape,
                w_conf=0.2,
                w_persist=1.0,
                w_friedel=0.2,
                keep_score=0.55,
                min_persist=0.5,
            )
            refined[grid[row, column]] = _with_source_columns(kept, peaks)
    return refined


def _with_source_columns(kept, peaks):
    """``refine`` works on ``[y, x, score]`` and may merge near-duplicates, so
    its rows are not a subset of ``peaks``: carry the confidence and sigma over
    from the nearest original detection."""
    kept = np.asarray(kept, np.float32).reshape(-1, 3)
    if len(kept) == 0:
        return np.zeros((0, peaks.shape[1]), np.float32)
    distance = np.hypot(
        kept[:, None, 0] - peaks[None, :, 0], kept[:, None, 1] - peaks[None, :, 1]
    )
    source = peaks[distance.argmin(1)]
    return np.column_stack([kept, source[:, 3:]]).astype(np.float32)


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
    asked for, resolving the automatic choices (the reference pattern that
    fixes the rescale, beam-stop mask, background sigma) on a few patterns of
    ``signal``.

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

    samples = sample_patterns(data, navigation_dimension)
    # The middle sample fixes the disk-size rescale for every pattern.
    finder.reference = samples[1] if len(samples) > 1 else samples[0]
    if background_sigma == "auto":
        calibrated = finder.calibrate(samples)
        if "threshold" not in finder_arguments:
            finder.threshold = calibrated["thresh"]

    settings = dict(
        mask=mask,
        navigation_sigma=float(kwargs.get("navigation_sigma", 0.0)),
        neighbour_refine=neighbour_refine,
    )
    return finder, settings, mask_options
