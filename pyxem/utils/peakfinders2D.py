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
from skimage.feature import match_template,peak_local_max

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

    def box(x, y, window_size, x_max, y_max):
        """Produces a list of coordinates in the box about (x, y)."""
        a = int(window_size / 2)
        x_min = max(0, x - a)
        x_max = min(x_max, x + a)
        y_min = max(0, y - a)
        y_max = min(y_max, y + a)
        return np.array(
            np.meshgrid(range(x_min, x_max), range(y_min, y_max))).reshape(
            2, -1).T

    def get_max(image, box):
        """Finds the coordinates of the maximum of 'image' in 'box'."""
        vals = image[box[:, 0], box[:, 1]]
        max_position = box[np.argmax(vals)]
        return max_position

    def distance(x, y):
        """Calculates the distance between two points."""
        v = x - y
        return np.sqrt(np.sum(np.square(v)))

    def gradient(image):
        """Calculates the square of the 2-d partial gradient.

        Parameters
        ----------
        image : numpy.ndarray

        Returns
        -------
        numpy.ndarray

        """
        gradient_of_image = np.gradient(image)
        gradient_of_image = gradient_of_image[0] ** 2 + gradient_of_image[
                                                            1] ** 2
        return gradient_of_image

    # Generate an ordered list of matrix coordinates.
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
    """
    from scipy.ndimage.filters import generic_filter
    from scipy.ndimage.filters import uniform_filter
    from sklearn.cluster import DBSCAN

    def normalize(image):
        """Scales the image to intensities between 0 and 1."""
        return image / np.max(image)

    def _local_stat(image, radius, func):
        """Calculates rolling method 'func' over a circular kernel."""
        x, y = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        kernel = np.hypot(x, y) < radius
        stat = generic_filter(image, func, footprint=kernel)
        return stat

    def local_mean(image, radius):
        """Calculates rolling mean over a circular kernel."""
        return _local_stat(image, radius, np.mean)

    def local_std(image, radius):
        """Calculates rolling standard deviation over a circular kernel."""
        return _local_stat(image, radius, np.std)

    def single_pixel_desensitize(image):
        """Reduces single-pixel anomalies by nearest-neighbor smoothing."""
        kernel = np.array([[0.5, 1, 0.5], [1, 1, 1], [0.5, 1, 0.5]])
        smoothed_image = generic_filter(image, np.mean, footprint=kernel)
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
        image = uniform_filter(image, size=3)
        image = uniform_filter(image, size=3)
        return image

    def half_binarise(image):
        """Image binarised about values of one-half intensity."""
        binarised_image = np.where(image > 0.5, 1, 0)
        return binarised_image

    def separate_peaks(binarised_image):
        """Identify adjacent 'on' coordinates via DBSCAN."""
        bi = binarised_image.astype('bool')
        coordinates = np.indices(bi.shape).reshape(2, -1).T[
            bi.flatten()]
        db = DBSCAN(2, 3)
        peaks = []
        labeled_points = db.fit_predict(coordinates)
        for peak_label in list(set(labeled_points)):
            peaks.append(coordinates[labeled_points == peak_label])
        return peaks

    def _peak_find_once(image):
        """Smooth, binarise, and find peaks according to main algorithm."""
        image = smooth(image)
        image = half_binarise(image)
        peaks = separate_peaks(image)
        return image, peaks

    def stat_peak_finder(image):
        """Find peaks in image. Algorithm stages in comments."""
        image = normalize(image)  # 1
        image = stat_binarise(image)  # 2, 3
        n_peaks = np.infty  # Initial number of peaks
        image, peaks = _peak_find_once(image)  # 4-6
        m_peaks = len(peaks)  # Actual number of peaks
        """
        #XXX
        while (n_peaks - m_peaks) / n_peaks > convergence_ratio:  # 8
            n_peaks = m_peaks
            image, peaks = _peak_find_once(image)
            m_peaks = len(peaks)
        """
        peak_centers = np.array(
            [np.mean(peak, axis=0) for peak in peaks])  # 7
        return peak_centers

    return clean_peaks(stat_peak_finder(z))


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

    centers = blobs[:, :2]
    return centers


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

    centers = blobs[:, :2]
    return centers

def find_peaks_xc(z,disc_image,min_distance=5,peak_threshold=0.2):
    """
    Find peaks using the the correlation between the image and a reference peaks

    Parameters
    ----------

    z: numpy.ndarray
        Array of image intensities.
    disc_image: numpy.ndarray (square)
        Array containing a single bright disc, similar to those you seek to detect
    min_distance: int
        The minimum expected distance between peaks (in pixels)
    peak_threshold: float between 0 and 1
        Internally passed argument, larger values will lead to fewer peaks in the output

    Returns
    -------
    numpy.ndarray
        (n_peaks, 2)
        Array of peak coordinates.

    """
    response_image = match_template(z,disc_image,pad_input=True)
    peaks = peak_local_max(response_image,min_distance=min_distance,threshold_rel=peak_threshold)
    peaks -= 1 #this means the return format is the same as the other peak finders

    return peaks
