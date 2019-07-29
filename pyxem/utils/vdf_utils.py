# -*- coding: utf-8 -*-
# Copyright 2017-2019 The pyXem developers
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
import matplotlib.pyplot as plt

from scipy.ndimage import distance_transform_edt, label, center_of_mass
from scipy.spatial import distance_matrix

from skimage.feature import peak_local_max
from skimage.filters import sobel, threshold_li
from skimage.morphology import watershed

from sklearn.cluster import DBSCAN


def normalize_vdf(im):
    """Normalizes image intensity by dividing by maximum value.

    Parameters
    ----------
    im : np.array()
        Array of image intensities

    Returns
    -------
    imn : np.array()
        Array of normalized image intensities

    """
    imn = im / im.max()
    return imn


def norm_cross_corr(image, template):
    """Calculates the normalised cross-correlation between an image
    and a template at zero displacement.

    Parameters
    ----------
    image: np.array
        Image
    template: np.array
        Reference image

    Returns
    -------
    corr : float
        Normalised cross-correlation between image and template.
    """
    f, t = image - np.average(image), template - np.average(template)
    corr = np.sum(f * t) / np.sqrt(np.sum(f**2) * np.sum(t**2))

    return corr


def separate(vdf_temp, min_distance, min_size, max_size,
             max_number_of_grains, threshold=False,
             exclude_border=False, plot_on=False):
    """Separate segments from one VDF image using edge-detection by the
    sobel transform and the watershed segmentation implemented in
    scikit-image. See [1,2] for examples from scikit-image.

    Parameters
    ----------
    vdf_temp : np.array
        One VDF image.
    min_distance: int
        Minimum distance (in pixels) between markers for them to be
        considered separate markers for the watershed segmentation.
    min_size : float
        Grains with size (i.e. total number of pixels) below min_size
        are discarded.
    max_size : float
        Grains with size (i.e. total number of pixels) above max_size
        are discarded.
    max_number_of_grains : int
        Maximum number of grains included in the returned separated
        grains. If it is exceeded, those with highest peak intensities
        will be returned.
    threshold : bool
        If True, a mask is calculated by thresholding the VDF image by
        the Li threshold method in scikit-image. If False (default), the
        mask is the boolean VDF image. 
    exclude_border : int or True, optional
        If non-zero integer, peaks within a distance of exclude_border
        from the boarder will be discarded. If True, peaks at or closer
        than min_distance of the boarder, will be discarded.
    plot_on : bool
        If True, the VDF, the mask, the distance transform
        and the separated grains will be plotted in one figure window.

    Returns
    -------
    sep : np.array
        Array containing segments from VDF images (i.e. separated
        grains). Shape: (image size x, image size y, number of grains)

    References
    ----------
    [1] http://scikit-image.org/docs/dev/auto_examples/segmentation/
        plot_watershed.html
    [2] http://scikit-image.org/docs/dev/auto_examples/xx_applications/
        plot_coins_segmentation.html#sphx-glr-auto-examples-xx-
        applications-plot-coins-segmentation-py
    """

    # Create a mask from the input VDF image.
    if threshold:
        th = threshold_li(vdf_temp)
        mask = np.zeros_like(vdf_temp)
        mask[vdf_temp > th] = True
    else:
        mask = vdf_temp.astype('bool')
    if np.any(np.nonzero(mask)) is False:
        return None

    # Calculate the eucledian distance from each point in the mask to the
    # nearest background point of value 0.
    distance = distance_transform_edt(mask)

    # Find the coordinates of the local maxima of the distance transform that
    # lie inside a region defined by (2*min_distance+1).
    local_maxi = peak_local_max(distance, indices=False,
                                min_distance=min_distance,
                                num_peaks=max_number_of_grains,
                                exclude_border=exclude_border,
                                threshold_rel=None)
    maxi_coord1 = np.where(local_maxi)

    # If there are any local maxima positioned at equal values of
    # distance, the peak_local_max function will return all those
    # maxima, irrespectively of the distances between them. Thus, below
    # we check if such maxima are closer than min_distance. If so,
    # we replace the maxima lying inside a cluster (found by DBSCAN by
    # using min_distance) by the one maximum closest to their center of
    # mass.
    max_values, equal_max_count = np.unique(
        distance[np.where(local_maxi)], return_counts=True)
    if np.any(equal_max_count > 1):
        for max_value in max_values[equal_max_count > 1]:
            equal_maxi = np.zeros_like(distance).astype('bool')
            equal_maxi[np.where((distance == max_value) &
                                (local_maxi == True))] = True
            equal_maxi_coord = np.reshape(np.transpose(
                np.where(equal_maxi)), (-1, 2)).astype('int')
            clusters = DBSCAN(eps=min_distance,
                              min_samples=1).fit(equal_maxi_coord)
            unique_labels, unique_labels_count = np.unique(
                clusters.labels_, return_counts=True)
            for n in np.arange(unique_labels.max()+1):
                if unique_labels_count[n] > 1:
                    equal_maxi_coord_temp = clusters.components_[
                        clusters.labels_ == n]
                    equal_maxi_temp = np.zeros_like(
                        distance).astype('bool')
                    equal_maxi_temp[
                        np.transpose(equal_maxi_coord_temp)[0],
                        np.transpose(equal_maxi_coord_temp)[1]] = True
                    com = np.array(center_of_mass(equal_maxi_temp))
                    index = distance_matrix(
                        [com], equal_maxi_coord_temp).argmin()
                    local_maxi[np.where(equal_maxi_temp)] = False
                    local_maxi[equal_maxi_coord_temp[index][0],
                               equal_maxi_coord_temp[index][1]] = True

    # Find the edges of the VDF image using the sobel transform.
    elevation = sobel(vdf_temp)

    # 'Flood' the elevation (i.e. edge) image from basins at the marker
    # positions. The marker positions are the local maxima of the
    # distance. Find the locations where different basins meet, i.e. the
    # watershed lines (segment boundaries). Only search for segments
    # (labels) in the area defined by mask.
    labels = watershed(elevation, markers=label(local_maxi)[0], mask=mask)

    sep = np.zeros((np.shape(vdf_temp)[0], np.shape(vdf_temp)[1],
                    (np.max(labels))), dtype='int32')
    n, i = 1, 0
    while (np.max(labels)) > n - 1:
        sep_temp = labels * (labels == n) / n
        sep_temp = np.nan_to_num(sep_temp)
        # Discard segment if it is too small, or add it to separated
        # segments.
        if ((np.sum(sep_temp, axis=(0, 1)) < min_size)
                or (max_size is not None
                    and np.sum(sep_temp, axis=(0, 1)) > max_size)):
            sep = np.delete(sep, ((n - i) - 1), axis=2)
            i = i + 1
        else:
            sep[:, :, (n - i) - 1] = sep_temp
        n = n + 1
    # Put the intensity from the input VDF image into each segment area.
    vdf_sep = np.broadcast_to(vdf_temp.T, np.shape(sep.T)) * (sep.T == 1)

    if plot_on:
        # If segments have been discarded, make new labels that do not
        # include the discarded segments.
        if np.max(labels) != (np.shape(sep)[2]) and (np.shape(sep)[2] != 0):
            labels = sep[:, :, 0]
            for i in range(1, np.shape(sep)[2]):
                labels = labels + sep[..., i] * (i + 1)
        # If no separated particles were found, set all elements in
        # labels to 0.
        elif np.shape(sep)[2] == 0:
            labels = np.zeros(np.shape(labels))

        seps_img_sum = np.zeros_like(vdf_temp).astype('float64')
        for l, vdf in zip(np.arange(1, np.max(labels)+1), vdf_sep):
            mask_l = np.zeros_like(labels).astype('bool')
            mask_l[np.where(labels == l)] = 1
            seps_img_sum += vdf_temp * mask_l /\
                            np.max(vdf_temp[np.where(labels == l)])
            seps_img_sum[np.where(labels == l)] += l

        maxi_coord = np.where(local_maxi)

        fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(vdf_temp, cmap=plt.cm.magma)
        ax[0].axis('off')
        ax[0].set_title('VDF')

        ax[1].imshow(mask, cmap=plt.cm.gray)
        ax[1].axis('off')
        ax[1].set_title('Mask')

        ax[2].imshow(distance, cmap=plt.cm.magma)
        ax[2].axis('off')
        ax[2].set_title('Distance and maxima')
        ax[2].plot(maxi_coord1[1], maxi_coord1[0], 'r+')
        ax[2].plot(maxi_coord[1], maxi_coord[0], 'gx')

        ax[3].imshow(elevation, cmap=plt.cm.magma)
        ax[3].axis('off')
        ax[3].set_title('Elevation')

        ax[4].imshow(labels, cmap=plt.cm.gnuplot2)
        ax[4].axis('off')
        ax[4].set_title('Labels')

        ax[5].imshow(seps_img_sum, cmap=plt.cm.magma)
        ax[5].axis('off')
        ax[5].set_title('Segments')

    return vdf_sep


def get_gaussian2d(a, xo, yo, x, y, sigma):
    """ Obtain a 2D Gaussian of amplitude a and standard deviation
    sigma, centred at (xo, yo), on a grid given by x and y.

    Parameters
    ----------
    a : float
        Amplitude. The Gaussian is simply multiplied by a.
    xo : float
        Center of Gaussian on x-axis.
    yo : float
        Center of Gaussian on y-axis.
    x : array
        Array representing the row indices of the grid the Gaussian
        should be placed on (x-axis).
    y : array
        Array representing the column indices of the grid the Gaussian
        should be placed on (y-axis).
    sigma : float
        Standard deviation of Gaussian.

    Returns
    -------
    gaussian : array
        Array with the 2D Gaussian.
    """

    gaussian = a / (2 * np.pi * sigma ** 2) * np.exp(
        -((x - xo) ** 2 + (y - yo) ** 2) / (2 * sigma ** 2))

    return gaussian

