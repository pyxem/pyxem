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

"""Single-pattern spot detectors used by :meth:`Diffraction2D.find_vectors`.

Every detector takes one diffraction pattern and returns
``(response, peaks)``: the detector's transformed image (for display) and an
``(N, 3)`` float32 array of ``[row, column, intensity]`` in pixels. The
intensity column is always the mean of the raw pattern over a small disk
around each peak, never the detector score, so every method reports the same
physical quantity.

A beam stop is a peak-rejection region, not an image edit: the pattern is
never filled or zeroed (a fill makes a sharp step at the mask edge that every
detector fires on), detections inside the mask are simply dropped.

Ported from SpyDE (https://github.com/directelectron/spyde).
"""

import functools

import numpy as np
from scipy.fft import irfft2, next_fast_len, rfft2
from scipy.ndimage import correlate1d, gaussian_filter, maximum_filter

METHOD_NXCORR = "nxcorr"
METHOD_DOG = "dog"
METHOD_NEURAL = "neural"
METHODS = (METHOD_DOG, METHOD_NXCORR, METHOD_NEURAL)

#: Per-method defaults. The thresholds live on different scales: NXCORR is a
#: correlation in [-1, 1], DoG an absolute band-pass signal-to-noise ratio and
#: the neural detector a heat-map confidence in [0, 1].
DEFAULT_THRESHOLD = {METHOD_NXCORR: 0.5, METHOD_DOG: 10.0, METHOD_NEURAL: 0.3}
DEFAULT_MIN_DISTANCE = {METHOD_NXCORR: 5, METHOD_DOG: 3, METHOD_NEURAL: 4}
DEFAULT_DISK_RADIUS = 5
DEFAULT_SIGMA1 = 0.8
DEFAULT_SIGMA2 = 2.0
DEFAULT_MASK_DILATION = 5


def empty_peaks():
    return np.zeros((0, 3), dtype=np.float32)


@functools.lru_cache(maxsize=32)
def _disk_kernel(radius):
    """Flat disk of ``radius`` px, normalised to unit sum."""
    radius = int(radius)
    rows, columns = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    disk = (rows**2 + columns**2 <= radius**2).astype(np.float32)
    return disk / disk.sum()


@functools.lru_cache(maxsize=32)
def _disk_kernel_fft(radius, height, width):
    disk = _disk_kernel(radius)
    embedded = np.zeros((height, width), dtype=np.float32)
    embedded[: disk.shape[0], : disk.shape[1]] = disk
    return rfft2(embedded)


@functools.lru_cache(maxsize=64)
def _gaussian_kernel_1d(sigma):
    """Normalised 1-D Gaussian truncated at 3 sigma."""
    sigma = float(sigma)
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    half_width = max(1, int(round(3.0 * sigma)))
    x = np.arange(-half_width, half_width + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return (kernel / kernel.sum()).astype(np.float32)


def subpixel_parabola(surface, peaks_px):
    """Refine integer peaks to the vertex of a 3-point parabola along each axis.

    Returns ``(N, 2)`` float32 ``[row, column]``. The shift is clamped to one
    pixel so a flat neighbourhood cannot throw a peak away.
    """
    height, width = surface.shape
    refined = peaks_px.astype(np.float32).copy()
    for i, (row, column) in enumerate(np.asarray(peaks_px, dtype=int)):
        if 0 < row < height - 1:
            a, b, c = (
                surface[row - 1, column],
                surface[row, column],
                surface[row + 1, column],
            )
            denominator = float(a - 2.0 * b + c)
            if denominator != 0.0:
                refined[i, 0] = row + np.clip(
                    0.5 * float(a - c) / denominator, -1.0, 1.0
                )
        if 0 < column < width - 1:
            a, b, c = (
                surface[row, column - 1],
                surface[row, column],
                surface[row, column + 1],
            )
            denominator = float(a - 2.0 * b + c)
            if denominator != 0.0:
                refined[i, 1] = column + np.clip(
                    0.5 * float(a - c) / denominator, -1.0, 1.0
                )
    return refined


def _sample_bilinear(pattern, rows, columns):
    height, width = pattern.shape
    rows = np.clip(np.asarray(rows, dtype=np.float64), 0.0, height - 1.0001)
    columns = np.clip(np.asarray(columns, dtype=np.float64), 0.0, width - 1.0001)
    row0 = np.floor(rows).astype(np.intp)
    column0 = np.floor(columns).astype(np.intp)
    row_fraction = (rows - row0).astype(np.float32)
    column_fraction = (columns - column0).astype(np.float32)
    p = pattern.astype(np.float32, copy=False)
    top = (1 - column_fraction) * p[row0, column0] + column_fraction * p[
        row0, column0 + 1
    ]
    bottom = (1 - column_fraction) * p[row0 + 1, column0] + column_fraction * p[
        row0 + 1, column0 + 1
    ]
    return ((1 - row_fraction) * top + row_fraction * bottom).astype(np.float32)


def disk_mean_intensity(pattern, rows, columns, radius):
    """Mean of ``pattern`` over a disk of ``radius`` px around each position.

    A single-pixel sample is dominated by shot noise and by where the subpixel
    centre happened to land, which bands an intensity-weighted virtual image;
    the disk mean tracks the true spot brightness. Falls back to a bilinear
    point sample for ``radius < 1``.
    """
    rows = np.asarray(rows, dtype=np.float64)
    columns = np.asarray(columns, dtype=np.float64)
    radius = int(max(0, round(float(radius))))
    if radius < 1 or rows.size == 0:
        return _sample_bilinear(pattern, rows, columns)
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


def _local_maxima(score, threshold, min_distance, excluded=None):
    """Integer peaks of ``score`` above ``threshold``, at least ``min_distance``
    apart, strongest first (greedy non-maximum suppression)."""
    min_distance = int(min_distance)
    is_peak = (score == maximum_filter(score, size=2 * min_distance + 1)) & (
        score >= threshold
    )
    if excluded is not None:
        is_peak &= ~excluded
    peaks_px = np.argwhere(is_peak)
    if len(peaks_px) <= 1:
        return peaks_px
    peaks_px = peaks_px[np.argsort(-score[peaks_px[:, 0], peaks_px[:, 1]])]
    kept = np.ones(len(peaks_px), dtype=bool)
    min_distance_squared = min_distance * min_distance
    for i in range(len(peaks_px)):
        if not kept[i]:
            continue
        delta = peaks_px[i + 1 :] - peaks_px[i]
        kept[i + 1 :][(delta**2).sum(axis=1) <= min_distance_squared] = False
    return peaks_px[kept]


def _peaks_with_intensity(pattern, positions, intensity_radius):
    intensity = disk_mean_intensity(
        pattern, positions[:, 0], positions[:, 1], intensity_radius
    )
    return np.column_stack([positions, intensity]).astype(np.float32)


def find_spots_nxcorr(
    pattern,
    disk_radius=DEFAULT_DISK_RADIUS,
    threshold=DEFAULT_THRESHOLD[METHOD_NXCORR],
    min_distance=DEFAULT_MIN_DISTANCE[METHOD_NXCORR],
    subpixel=True,
    mask=None,
    window_pad=1,
):
    """Window-normalised cross-correlation against a flat disk.

    Lewis's fast normalised cross-correlation: the numerator is one FFT
    correlation, the window mean and variance come from integral images. The
    score is intensity independent, so ``threshold`` (typically 0.3-0.6) means
    the same thing on every pattern.

    Parameters
    ----------
    pattern : numpy.ndarray
        One 2-D diffraction pattern.
    disk_radius : int
        Radius of the disk template in pixels.
    threshold : float
        Minimum correlation score.
    min_distance : int
        Minimum separation between peaks in pixels.
    subpixel : bool
        Refine each peak to the parabolic vertex of the score surface.
    mask : numpy.ndarray of bool, optional
        Pixels where no peak may be reported (a beam stop).
    window_pad : int
        Extra pixels on the window used for the local statistics, so a ring of
        background is sampled and a single hot pixel cannot inflate the score.

    Returns
    -------
    response : numpy.ndarray
        The correlation score clipped to [-1, 1].
    peaks : numpy.ndarray
        ``(N, 3)`` float32 ``[row, column, intensity]``.
    """
    radius = int(disk_radius)
    pattern = np.asarray(pattern, dtype=np.float32)
    height, width = pattern.shape

    padded_height = next_fast_len(height + 2 * radius)
    padded_width = next_fast_len(width + 2 * radius)
    buffer = np.zeros((padded_height, padded_width), dtype=np.float32)
    buffer[: height + 2 * radius, : width + 2 * radius] = np.pad(
        pattern, radius, mode="reflect"
    )
    # ``s=`` is required: without it irfft2 infers an even width, which is
    # wrong whenever next_fast_len returns an odd one and shifts every peak.
    cross_correlation = irfft2(
        rfft2(buffer) * _disk_kernel_fft(radius, padded_height, padded_width).conj(),
        s=(padded_height, padded_width),
    )[:height, :width].astype(np.float32)

    disk = _disk_kernel(radius)
    template_size = disk.size
    template_mean = float(disk.mean())
    template_std = float(np.sqrt(np.sum((disk - template_mean) ** 2) / template_size))

    window_radius = radius + int(window_pad)
    window_width = 2 * window_radius + 1
    window_size = window_width**2
    statistics_source = np.pad(pattern, window_radius, mode="reflect")

    def window_sum(values):
        integral = np.zeros(
            (values.shape[0] + 1, values.shape[1] + 1), dtype=np.float64
        )
        integral[1:, 1:] = values.cumsum(axis=0).cumsum(axis=1)
        w = window_width
        return (
            integral[w : height + w, w : width + w]
            - integral[:height, w : width + w]
            - integral[w : height + w, :width]
            + integral[:height, :width]
        )

    window_mean = window_sum(statistics_source) / window_size
    window_variance = window_sum(statistics_source**2) / window_size - window_mean**2
    window_std = np.sqrt(np.maximum(window_variance, 0.0))

    # Floor the denominator at 1 % of the global spread: in an almost empty
    # region the window std tends to zero and the score would blow up.
    denominator_floor = 0.01 * (float(pattern.std()) or 1.0) * template_std
    numerator = cross_correlation / template_size - window_mean * template_mean
    # Peaks are located on the UNCLIPPED score. Because the statistics window
    # is larger than the template the score exceeds 1 near a disk centre, and
    # clipping it there flattens the peak into a plateau the parabola cannot
    # refine.
    score = (
        numerator / np.maximum(window_std * template_std, denominator_floor)
    ).astype(np.float32)
    has_mask = mask is not None and mask.any()
    if has_mask:
        score[mask] = -1.0
    response = np.clip(score, -1.0, 1.0)

    peaks_px = _local_maxima(score, threshold, min_distance, mask if has_mask else None)
    if len(peaks_px) == 0:
        return response, empty_peaks()
    positions = (
        subpixel_parabola(score, peaks_px) if subpixel else peaks_px.astype(np.float32)
    )
    return response, _peaks_with_intensity(pattern, positions, radius)


def find_spots_dog(
    pattern,
    sigma1=DEFAULT_SIGMA1,
    sigma2=DEFAULT_SIGMA2,
    threshold=DEFAULT_THRESHOLD[METHOD_DOG],
    min_distance=DEFAULT_MIN_DISTANCE[METHOD_DOG],
    subpixel=True,
    mask=None,
):
    """Difference-of-Gaussians band-pass detector for small spots.

    ``G(sigma1) * I - G(sigma2) * I`` removes pixel noise below ``sigma1`` and
    the smooth diffuse background above ``sigma2`` in one separable pass, so a
    faint spot stands out without a radius-matched template.

    The threshold is an absolute signal-to-noise ratio: the response divided
    by the robust spread (1.4826 x median absolute deviation) of the unmasked
    pixels. Typical values are 6-15.

    Returns
    -------
    response : numpy.ndarray
        The band-pass signal-to-noise ratio.
    peaks : numpy.ndarray
        ``(N, 3)`` float32 ``[row, column, intensity]``.
    """
    pattern = np.asarray(pattern, dtype=np.float32)
    has_mask = mask is not None and mask.any()

    def blur(sigma):
        kernel = _gaussian_kernel_1d(sigma)
        return correlate1d(
            correlate1d(pattern, kernel, axis=0, mode="reflect"),
            kernel,
            axis=1,
            mode="reflect",
        )

    band_pass = blur(sigma1) - blur(sigma2)
    # The bright rim around a beam stop leaves a strong response; keep it out
    # of the noise estimate or it buries faint real spots.
    statistics_source = band_pass[~mask] if has_mask else band_pass.ravel()
    median = float(np.median(statistics_source))
    spread = 1.4826 * float(np.median(np.abs(statistics_source - median)))
    if spread <= 1e-12:
        spread = float(statistics_source.std())
    if spread <= 1e-12:
        spread = float(np.abs(statistics_source).max()) or 1.0
    response = ((band_pass - median) / spread).astype(np.float32)
    if has_mask:
        response[mask] = 0.0

    peaks_px = _local_maxima(
        response, threshold, min_distance, mask if has_mask else None
    )
    if len(peaks_px) == 0:
        return response, empty_peaks()
    positions = (
        subpixel_parabola(response, peaks_px)
        if subpixel
        else peaks_px.astype(np.float32)
    )
    return response, _peaks_with_intensity(
        pattern, positions, max(1.0, np.ceil(sigma2))
    )


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


def estimate_disk_radius(pattern):
    """Estimate the diffraction disk radius in pixels from one pattern.

    Sweeps a scale-normalised Laplacian of Gaussian over the background
    subtracted pattern; its response peaks at ``sigma = radius / sqrt(2)``.
    """
    from skimage.filters import laplace

    pattern = np.asarray(pattern, dtype=np.float64)
    smallest_side = min(pattern.shape)
    fallback = max(3, int(smallest_side * 0.05))
    foreground = np.clip(
        pattern - gaussian_filter(pattern, sigma=smallest_side * 0.15), 0, None
    )
    if foreground.max() == 0:
        return fallback
    foreground /= foreground.max()
    sigmas = np.geomspace(1.0, max(2.0, smallest_side * 0.15), 20)
    responses = [
        sigma**2 * np.abs(laplace(gaussian_filter(foreground, sigma=sigma))).max()
        for sigma in sigmas
    ]
    radius = int(round(sigmas[int(np.argmax(responses))] * np.sqrt(2)))
    return max(3, min(radius, int(smallest_side * 0.25)))
