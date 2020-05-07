# -*- coding: utf-8 -*-
# Copyright 2016-2020 The pyXem developers
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

import numpy as np
import scipy.ndimage as ndi
from numba import njit
from skimage.feature import blob_dog, blob_log, corner_peaks, match_template
from sklearn.cluster import DBSCAN


NO_PEAKS = np.array([[np.nan, np.nan]])


@njit(cache=True)
def _fast_mean(X):
    """JIT-compiled mean of array.

    Parameters
    ----------
    X : numpy.ndarray
        Input array.

    Returns
    -------
    float
        Mean of X.

    Notes
    -----
    Used by scipy.ndimage.generic_filter in the find_peaks_stat
    method to reduce overhead of repeated Python function calls.
    See https://github.com/scipy/scipy/issues/8916 for more details.
    """
    return np.mean(X)


@njit(cache=True)
def _fast_std(X):
    """JIT-compiled standard deviation of array.

    Parameters
    ----------
    X : numpy.ndarray
        Input array.

    Returns
    -------
    float
        Standard deviation of X.

    Notes
    -----
    Used by scipy.ndimage.generic_filter in the find_peaks_stat
    method to reduce overhead of repeated Python function calls.
    See https://github.com/scipy/scipy/issues/8916 for more details.
    """
    return np.std(X)


def clean_peaks(peaks):
    """Utility function to deal with no peaks being found.

    Parameters
    ----------
    peaks : numpy.ndarray
        Array of peak pixel coordinates with shape (n_peaks, 2).

    Returns
    -------
    peaks : numpy.ndarray
        Array of peak pixel coordinates with shape (n_peaks, 2).

    NO_PEAKS : str
        Flag indicating no peaks found.

    """
    if len(peaks) == 0:
        return NO_PEAKS
    else:
        return peaks


def find_peaks_zaefferer(z, grad_threshold=0.1, window_size=40, distance_cutoff=50.0):
    """Method to locate positive peaks in an image based on gradient
    thresholding and subsequent refinement within masked regions.

    Parameters
    ----------
    z : numpy.ndarray
        Matrix of image intensities.
    grad_threshold : float
        The minimum gradient required to begin a peak search.
    window_size : int
        The size of the square window within which a peak search is
        conducted. If odd, will round down to even.
    distance_cutoff : float
        The maximum distance a peak may be from the initial
        high-gradient point.

    Returns
    -------
    peaks : numpy.ndarray
        Array of peak pixel coordinates with shape (n_peaks, 2).

    Notes
    -----
    Implemented as described in Zaefferer "New developments of computer-aided
    crystallographic analysis in transmission electron microscopy" J. Ap. Cryst.
    This version by Ben Martineau (2016)
    """

    def box(x, y, window_size, x_max, y_max):
        """Produces a list of coordinates in the box about (x, y)."""
        a = int(window_size / 2)
        x_min = max(0, x - a)
        x_max = min(x_max, x + a)
        y_min = max(0, y - a)
        y_max = min(y_max, y + a)
        return np.mgrid[x_min:x_max, y_min:y_max].reshape(2, -1, order="F")

    def get_max(image, box):
        """Finds the coordinates of the maximum of 'image' in 'box'."""
        vals = image[tuple(box)]
        ind = np.argmax(vals)
        return tuple(box[:, ind])

    def squared_distance(x, y):
        """Calculates the squared distance between two points."""
        return (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2

    def gradient(image):
        """Calculates the square of the 2D partial gradient.

        Parameters
        ----------
        image : numpy.ndarray
            Array of image intensities.

        Returns
        -------
        image_gradient : numpy.ndarray
            Array of image gradient intensities.
        """
        gradient_of_image = np.gradient(image)
        gradient_of_image = gradient_of_image[0] ** 2 + gradient_of_image[1] ** 2
        return gradient_of_image

    # Generate an ordered list of matrix coordinates.
    z = z / np.max(z)
    coordinates = np.indices(z.data.shape).reshape(2, -1).T

    # Calculate the gradient at every point.
    image_gradient = gradient(z)

    # Boolean matrix of high-gradient points.
    coordinates = coordinates[(image_gradient >= grad_threshold).flatten()]

    # Compare against squared distance (avoids repeated sqrt calls)
    distance_cutoff_sq = distance_cutoff ** 2

    peaks = []

    for coordinate in coordinates:
        # Iterate over coordinates where the gradient is high enough.
        b = box(coordinate[0], coordinate[1], window_size, z.shape[0], z.shape[1])
        p_old = (0, 0)
        p_new = get_max(z, b)

        while p_old[0] != p_new[0] and p_old[1] != p_new[1]:
            p_old = p_new
            b = box(p_old[0], p_old[1], window_size, z.shape[0], z.shape[1])
            p_new = get_max(z, b)
            if squared_distance(coordinate, p_new) > distance_cutoff_sq:
                break
            peaks.append(p_new)

    peaks = np.array([p for p in set(peaks)])
    return clean_peaks(peaks)


def find_peaks_stat(z, alpha=1.0, window_radius=10, convergence_ratio=0.05):
    """Locate positive peaks in an image based on statistical refinement and
    difference with respect to mean intensity.

    Parameters
    ----------
    z : numpy.ndarray
        Array of image intensities.
    alpha : float
        Only maxima above `alpha * sigma` are found, where `sigma` is the
        local, rolling standard deviation of the image.
    window_radius : int
        The pixel radius of the circular window for the calculation of the
        rolling mean and standard deviation.
    convergence_ratio : float
        The algorithm will stop finding peaks when the proportion of new peaks
        being found is less than `convergence_ratio`.

    Returns
    -------
    peaks : numpy.ndarray
        Array of peak pixel coordinates with shape (n_peaks, 2).

    Notes
    -----
    Implemented as described in the PhD thesis of Thomas White, University of
    Cambridge, 2009, with minor modifications to resolve ambiguities.

    The algorithm is as follows:

    1. Adjust the contrast and intensity bias of the image so that all pixels
       have values between 0 and 1.
    2. For each pixel, determine the mean and standard deviation of all pixels
       inside a circle of radius 10 pixels centered on that pixel.
    3. If the value of the pixel is greater than the mean of the pixels in the
       circle by more than one standard deviation, set that pixel to have an
       intensity of 1. Otherwise, set the intensity to 0.
    4. Smooth the image by convovling it twice with a flat 3x3 kernel.
    5. Let k = (1/2 - mu)/sigma where mu and sigma are the mean and standard
       deviations of all the pixel intensities in the image.
    6. For each pixel in the image, if the value of the pixel is greater than
       mu + k*sigma set that pixel to have an intensity of 1. Otherwise, set the
       intensity to 0.
    7. Detect peaks in the image by locating the centers of gravity of regions
       of adjacent pixels with a value of 1.
    8. Repeat #4-7 until the number of peaks found in the previous step
       converges to within 5%.
    """

    def normalize(image):
        """Scales the image to intensities between 0 and 1."""
        return image / np.max(image)

    def _local_stat(image, radius, func):
        """Calculates rolling method 'func' over a circular kernel."""
        x, y = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        kernel = np.hypot(x, y) < radius
        stat = ndi.filters.generic_filter(image, func, footprint=kernel)
        return stat

    def local_mean(image, radius):
        """Calculates rolling mean over a circular kernel."""
        return _local_stat(image, radius, _fast_mean)

    def local_std(image, radius):
        """Calculates rolling standard deviation over a circular kernel."""
        return _local_stat(image, radius, _fast_std)

    def single_pixel_desensitize(image):
        """Reduces single-pixel anomalies by nearest-neighbor smoothing."""
        kernel = np.array([[0.5, 1, 0.5], [1, 1, 1], [0.5, 1, 0.5]])
        smoothed_image = ndi.filters.generic_filter(image, _fast_mean, footprint=kernel)
        return smoothed_image

    def stat_binarise(image):
        """Peaks more than one standard deviation from the mean set to one."""
        image_rolling_mean = local_mean(image, window_radius)
        image_rolling_std = local_std(image, window_radius)
        image = single_pixel_desensitize(image)
        binarised_image = np.zeros(image.shape)
        stat_mask = image > (image_rolling_mean + alpha * image_rolling_std)
        binarised_image[stat_mask] = 1
        return binarised_image

    def smooth(image):
        """Image convolved twice using a uniform 3x3 kernel."""
        image = ndi.filters.uniform_filter(image, size=3)
        image = ndi.filters.uniform_filter(image, size=3)
        return image

    def half_binarise(image):
        """Image binarised about values of one-half intensity."""
        binarised_image = np.where(image > 0.5, 1, 0)
        return binarised_image

    def separate_peaks(binarised_image):
        """Identify adjacent 'on' coordinates via DBSCAN."""
        bi = binarised_image.astype("bool")
        coordinates = np.indices(bi.shape).reshape(2, -1).T[bi.flatten()]
        db = DBSCAN(2, 3)
        peaks = []
        if coordinates.shape[0] > 0:  # we have at least some peaks
            labeled_points = db.fit_predict(coordinates)
            for peak_label in list(set(labeled_points)):
                peaks.append(coordinates[labeled_points == peak_label])
        return peaks

    def _peak_find_once(image):
        """Smooth, binarise, and find peaks according to main algorithm."""
        image = smooth(image)  # 4
        image = half_binarise(image)  # 5
        peaks = separate_peaks(image)  # 6
        centers = np.array([np.mean(peak, axis=0) for peak in peaks])  # 7
        return image, centers

    def stat_peak_finder(image, convergence_ratio):
        """Find peaks in image. Algorithm stages in comments."""
        # Image preparation
        image = normalize(image)  # 1
        image = stat_binarise(image)  # 2, 3
        # Perform first iteration of peak finding
        image, peaks_curr = _peak_find_once(image)  # 4-7
        n_peaks = len(peaks_curr)
        if n_peaks == 0:  # pragma: no cover
            return peaks_curr

        m_peaks = 0
        # Repeat peak finding with more blurring to convergence
        while (n_peaks - m_peaks) / n_peaks > convergence_ratio:  # 8
            m_peaks = n_peaks
            image, peaks_curr = _peak_find_once(image)
            n_peaks = len(peaks_curr)
            if n_peaks == 0:
                return peaks_curr

        return peaks_curr

    return clean_peaks(stat_peak_finder(z, convergence_ratio))


def find_peaks_dog(
    z,
    min_sigma=1.0,
    max_sigma=50.0,
    sigma_ratio=1.6,
    threshold=0.2,
    overlap=0.5,
    exclude_border=False,
):
    """
    Finds peaks via the difference of Gaussian Matrices method from
    `scikit-image`.

    Parameters
    ----------
    z : numpy.ndarray
        2-d array of intensities
    float min_sigma, max_sigma, sigma_ratio, threshold, overlap
        Additional parameters to be passed to the algorithm. See `blob_dog`
        documentation for details:
        http://scikit-image.org/docs/dev/api/skimage.feature.html#blob-dog

    Returns
    -------
    peaks : numpy.ndarray
        Array of peak pixel coordinates with shape (n_peaks, 2).

    Notes
    -----
    While highly effective at finding even very faint peaks, this method is
    sensitive to fluctuations in intensity near the edges of the image.

    """
    z = z / np.max(z)
    blobs = blob_dog(
        z,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        sigma_ratio=sigma_ratio,
        threshold=threshold,
        overlap=overlap,
    )

    centers = blobs[:, :2]
    return centers


def find_peaks_log(
    z,
    min_sigma=1.0,
    max_sigma=50.0,
    num_sigma=int(10),
    threshold=0.2,
    overlap=0.5,
    log_scale=False,
    exclude_border=False,
):
    """
    Finds peaks via the Laplacian of Gaussian Matrices method from
    `scikit-image`.

    Parameters
    ----------
    z : numpy.ndarray
        Array of image intensities.
    float min_sigma, max_sigma, num_sigma, threshold, overlap, log_scale
        Additional parameters to be passed to the algorithm. See
        `blob_log` documentation for details:
        http://scikit-image.org/docs/dev/api/skimage.feature.html#blob-log

    Returns
    -------
    peaks : numpy.ndarray
        Array of peak pixel coordinates with shape (n_peaks, 2).

    """
    z = z / np.max(z)
    blobs = blob_log(
        z,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        overlap=overlap,
        log_scale=log_scale,
    )

    centers = blobs[:, :2]
    return centers


def find_peaks_xc(z, disc_image, min_distance=5, peak_threshold=0.2):
    """
    Find peaks using the the correlation between the image and a reference peaks

    Parameters
    ----------

    z: numpy.ndarray
        Array of image intensities.
    disc_image: numpy.ndarray (square)
        Array containing a single bright disc, similar to those to detect.
    min_distance: int
        The minimum expected distance between peaks (in pixels)
    peak_threshold: float between 0 and 1
        Larger values will lead to fewer peaks in the output.

    Returns
    -------
    peaks : numpy.ndarray
        Array of peak pixel coordinates with shape (n_peaks, 2).

    """
    response_image = match_template(z, disc_image, pad_input=True)
    peaks = corner_peaks(
        response_image, min_distance=min_distance, threshold_rel=peak_threshold
    )
    # make return format the same as the other peak finders
    peaks -= 1

    return clean_peaks(peaks)
