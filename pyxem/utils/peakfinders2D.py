# -*- coding: utf-8 -*-
# Copyright 2017-2018 The pyXem developers
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
from pyxem.utils.peakfinder_utils import *



NO_PEAKS = np.array([[[np.nan, np.nan]]])


def clean_peaks(peaks):
    if len(peaks) == 0:
        return NO_PEAKS
    else:
        return peaks


def find_peaks_zaefferer(z, grad_threshold=0.1, window_size=40,
                         distance_cutoff=50.):
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
        (n_peaks, 2)
        Peak pixel coordinates.

    Notes
    -----
    Implemented as described in Zaefferer "New developments of computer-aided
    crystallographic analysis in transmission electron microscopy" J. Ap. Cryst.
    This version by Ben Martineau (2016)
    """

    # Generate an ordered list of matrix coordinates.
    if len(z.shape) != 2:
        raise ValueError("'z' should be a 2-d image matrix.")
    z = z / np.max(z)
    coordinates = np.indices(z.data.shape).reshape(2, -1).T
    # Calculate the gradient at every point.
    image_gradient = gradient(z)
    # Boolean matrix of high-gradient points.
    gradient_is_above_threshold = image_gradient >= grad_threshold
    peaks = []
    for coordinate in coordinates[gradient_is_above_threshold.flatten()]:
        # Iterate over coordinates where the gradient is high enough.
        b = box(coordinate[0], coordinate[1], window_size, z.shape[0],
                z.shape[1])
        p_old = np.array([0, 0])
        p_new = get_max(z, b)
        while np.all(p_old != p_new):
            p_old = p_new
            b = box(p_old[0], p_old[1], window_size, z.shape[0], z.shape[1])
            p_new = get_max(z, b)
            if distance(coordinate, p_new) > distance_cutoff:
                break
            peaks.append(tuple(p_new))
    peaks = np.array([np.array(p) for p in set(peaks)])
    return clean_peaks(peaks)


def find_peaks_stat(z, alpha=1., window_radius=10, convergence_ratio=0.05):
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
    numpy.ndarray
        (n_peaks, 2)
        Array of peak coordinates.

    Notes
    -----
    Implemented as described in the PhD thesis of Thomas White (2009) the
    algorithm was developed by Gordon Ball during a summer project in
    Cambridge.
    This version by Ben Martineau (2016), with minor modifications to the
    original where methods were ambiguous or unclear.

    Algorithm stags in the comments
    """
    image = normalize(z)  # 1
    image = stat_binarise(image,window_radius,alpha)  # 2, 3
    n_peaks = np.infty  # Initial number of peaks
    image, peaks = peak_find_once(image)  # 4-6
    m_peaks = len(peaks)  # Actual number of peaks
    while (n_peaks - m_peaks) / n_peaks > convergence_ratio:  # 8
        n_peaks = m_peaks
        image, peaks = peak_find_once(image)
        m_peaks = len(peaks)
    peak_centers = np.array([np.mean(peak, axis=0) for peak in peaks])  # 7

    return clean_peaks(peak_centers)


def find_peaks_dog(z, min_sigma=1., max_sigma=50., sigma_ratio=1.6,
                   threshold=0.2, overlap=0.5):
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
    numpy.ndarray
        Array of peak coordinates of shape `(n_peaks, 2)`

    Notes
    -----
    While highly effective at finding even very faint peaks, this method is
    sensitive to fluctuations in intensity near the edges of the image.

    """
    from skimage.feature import blob_dog
    z = z / np.max(z)
    blobs = blob_dog(z, min_sigma=min_sigma, max_sigma=max_sigma,
                     sigma_ratio=sigma_ratio, threshold=threshold,
                     overlap=overlap)
    try:
        centers = blobs[:, :2]
    except IndexError:
        return NO_PEAKS
    clean_centers = []
    for center in centers:
        if len(np.intersect1d(center, (0, 1) + z.shape + tuple(
                        c - 1 for c in z.shape))) > 0:
            continue
        clean_centers.append(center)
    return np.array(clean_centers)


def find_peaks_log(z, min_sigma=1., max_sigma=50., num_sigma=10.,
                   threshold=0.2, overlap=0.5, log_scale=False):
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
    numpy.ndarray
        (n_peaks, 2)
        Array of peak coordinates.

    """
    from skimage.feature import blob_log
    z = z / np.max(z)
    blobs = blob_log(z, min_sigma=min_sigma, max_sigma=max_sigma,
                     num_sigma=num_sigma, threshold=threshold, overlap=overlap,
                     log_scale=log_scale)
    # Attempt to return only peak positions. If no peaks exist, return an
    # empty array.
    try:
        centers = blobs[:, :2]
    except IndexError:
        return NO_PEAKS
    return centers

def find_peaks_regionprops(z, min_sigma=4, max_sigma=5, threshold=1,
                           min_size=50, return_props=False):
    """
    Finds peaks using regionprops.
    Uses the difference of two gaussian convolutions to separate signal from
    background, and then uses the skimage.measure.regionprops function to find
    connected islands (peaks). Small blobs can be rejected using `min_size`.

    Parameters
    ----------
    z : numpy.ndarray
        Array of image intensities.
    min_sigma : int, float
        Standard deviation for the minimum gaussian convolution
    max_sigma : int, float
        Standard deviation for the maximum gaussian convolution
    threshold : int, float
        Minimum difference in intensity
    min_size : int
        Minimum size in pixels of blob
    return_props : bool
        Return skimage.measure.regionprops

    Returns
    -------
    numpy.ndarray
        (n_peaks, 2)
        Array of peak coordinates.

    """
    from skimage import morphology, measure

    difference = ndi.gaussian_filter(z, min_sigma) - ndi.gaussian_filter(z, max_sigma)

    labels, numlabels = ndi.label(difference > threshold)
    labels = morphology.remove_small_objects(labels, min_size)

    props = measure.regionprops(labels, z)

    if return_props:
        return props
    else:
        peaks = np.array([prop.centroid for prop in props])
        return clean_peaks(peaks)
