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
from scipy.ndimage.filters import generic_filter
from scipy.ndimage.filters import uniform_filter
from sklearn.cluster import DBSCAN


"""
zaefferer
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

"""
Stat
"""

def normalize(image):
    """Scales the image to intensities between 0 and 1."""
    return image / np.max(image)

def local_stat(image, radius, func):
    """Calculates rolling method 'func' over a circular kernel."""
    x, y = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    kernel = np.hypot(x, y) < radius
    stat = generic_filter(image, func, footprint=kernel)
    return stat

def local_mean(image, radius):
    """Calculates rolling mean over a circular kernel."""
    return local_stat(image, radius, np.mean)

def local_std(image, radius):
    """Calculates rolling standard deviation over a circular kernel."""
    return local_stat(image, radius, np.std)

def single_pixel_desensitize(image):
    """Reduces single-pixel anomalies by nearest-neighbor smoothing."""
    kernel = np.array([[0.5, 1, 0.5], [1, 1, 1], [0.5, 1, 0.5]])
    smoothed_image = generic_filter(image, np.mean, footprint=kernel)
    return smoothed_image

def stat_binarise(image,window_radius,alpha):
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

def peak_find_once(image):
    """Smooth, binarise, and find peaks according to main algorithm."""
    image = smooth(image)
    image = half_binarise(image)
    peaks = separate_peaks(image)
    return image, peaks
