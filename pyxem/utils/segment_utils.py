# -*- coding: utf-8 -*-
# Copyright 2016-2021 The pyXem developers
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
from numpy.ma import masked_where

import matplotlib.pyplot as plt

from scipy.ndimage import distance_transform_edt, label, binary_erosion
from scipy.spatial import distance_matrix
from scipy.signal import convolve2d

from skimage.feature import peak_local_max
from skimage.filters import sobel, threshold_li
from skimage.morphology import watershed, disk

from sklearn.cluster import DBSCAN


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
    if np.all(f == 0) and np.all(t == 0):
        # both arrays contains only 0's
        corr = 1
    elif np.all(f == 0) or np.all(t == 0):
        # one arrays contains only 0's
        corr = 0
    else:
        # no divide by zero to worry about now, normal corr definition in use
        corr = np.sum(f * t) / np.sqrt(np.sum(f ** 2) * np.sum(t ** 2))

    return corr


def separate_watershed(
    vdf_temp,
    min_distance=1,
    min_size=1,
    max_size=np.inf,
    max_number_of_grains=np.inf,
    marker_radius=1,
    threshold=False,
    exclude_border=False,
    plot_on=False,
):
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
    marker_radius : float
        If 1 or larger, each marker for watershed is expanded to a disk
        of radius marker_radius. marker_radius should not exceed
        2*min_distance.
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
        mask = vdf_temp.astype("bool")

    # Calculate the Eucledian distance from each point in the mask to the
    # nearest background point of value 0.
    distance = distance_transform_edt(mask)

    # If exclude_boarder is given, the edge of the distance is removed
    # by erosion. The distance image is used to find markers, and so the
    # erosion is done to avoid that markers are located at the edge
    # of the mask.
    if exclude_border > 0:
        distance_mask = binary_erosion(distance, structure=disk(exclude_border))
        distance = distance * distance_mask.astype("bool")

    # Find the coordinates of the local maxima of the distance transform.
    local_maxi = peak_local_max(
        distance,
        indices=False,
        min_distance=1,
        num_peaks=max_number_of_grains,
        exclude_border=exclude_border,
        threshold_rel=None,
    )
    maxi_coord1 = np.where(local_maxi)

    # Discard maxima that are found at pixels that are connected to a
    # smaller number of pixels than min_size. Used as markers, these would lead
    # to segments smaller than min_size and should therefore not be
    # considered when deciding which maxima to use as markers.
    if min_size > 1:
        labels_check = label(mask)[0]
        delete_indices = []
        for i in np.arange(np.shape(maxi_coord1)[1]):
            index = np.transpose(maxi_coord1)[i]
            label_value = labels_check[index[0], index[1]]
            if len(labels_check[labels_check == label_value]) < min_size:
                delete_indices.append(i)
                local_maxi[index[0], index[1]] = False
        maxi_coord1 = np.delete(maxi_coord1, delete_indices, axis=1)

    # Cluster the maxima by DBSCAN based on min_distance. For each
    # cluster, only the maximum closest to the average maxima position is
    # used as a marker.
    if min_distance > 1 and np.shape(maxi_coord1)[1] > 1:
        clusters = DBSCAN(
            eps=min_distance,
            metric="euclidean",
            min_samples=1,
        ).fit(np.transpose(maxi_coord1))
        local_maxi = np.zeros_like(local_maxi)
        for n in np.arange(clusters.labels_.max() + 1):
            maxi_coord1_n = np.transpose(maxi_coord1)[clusters.labels_ == n]
            com = np.average(maxi_coord1_n, axis=0).astype("int")
            index = distance_matrix([com], maxi_coord1_n).argmin()
            index = maxi_coord1_n[index]
            local_maxi[index[0], index[1]] = True

    # Use the resulting maxima as markers. Each marker should have a
    # unique label value. For each maximum, generate markers with the same
    # label value in a radius given by marker_radius centered at the
    # maximum position. This is done to make the segmentation more robust
    # to local changes in pixel values around the marker.
    markers = label(local_maxi)[0]
    if marker_radius >= 1:
        disk_mask = disk(marker_radius)
        for mm in np.arange(1, np.max(markers) + 1):
            im = np.zeros_like(markers)
            im[np.where(markers == mm)] = markers[np.where(markers == mm)]
            markers_temp = convolve2d(
                im, disk_mask, boundary="fill", mode="same", fillvalue=0
            )
            markers[np.where(markers_temp)] = mm
    markers = markers * mask

    # Find the edges of the VDF image using the Sobel transform.
    elevation = sobel(vdf_temp)

    # 'Flood' the elevation (i.e. edge) image from basins at the marker
    # positions. Find the locations where different basins meet, i.e.
    # the watershed lines (segment boundaries). Only search for segments
    # (labels) in the area defined by mask.
    labels = watershed(elevation, markers=markers, mask=mask)

    sep = np.zeros(
        (np.shape(vdf_temp)[0], np.shape(vdf_temp)[1], (np.max(labels))), dtype="int32"
    )
    n, i = 1, 0
    while (np.max(labels)) > n - 1:
        sep_temp = labels * (labels == n) / n
        sep_temp = np.nan_to_num(sep_temp)
        # Discard a segment if it is too small or too large, or else add
        # it to the list of separated segments.
        if (np.sum(sep_temp, axis=(0, 1)) < min_size) or np.sum(
            sep_temp, axis=(0, 1)
        ) > max_size:
            sep = np.delete(sep, ((n - i) - 1), axis=2)
            i = i + 1
        else:
            sep[:, :, (n - i) - 1] = sep_temp
        n = n + 1
    # Put the intensity from the input VDF image into each segmented area.
    vdf_sep = np.broadcast_to(vdf_temp.T, np.shape(sep.T)) * (sep.T == 1)

    if plot_on:  # pragma: no cover
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

        seps_img_sum = np.zeros_like(vdf_temp).astype("float64")
        for lbl, vdf in zip(np.arange(1, np.max(labels) + 1), vdf_sep):
            mask_l = np.zeros_like(labels).astype("bool")
            _idx = np.where(labels == lbl)
            mask_l[_idx] = 1
            seps_img_sum += vdf_temp * mask_l / np.max(vdf_temp[_idx])
            seps_img_sum[_idx] += lbl

        maxi_coord = np.where(local_maxi)

        fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(vdf_temp, cmap=plt.cm.magma_r)
        ax[0].axis("off")
        ax[0].set_title("VDF")

        ax[1].imshow(mask, cmap=plt.cm.gray_r)
        ax[1].axis("off")
        ax[1].set_title("Mask")

        ax[2].imshow(distance, cmap=plt.cm.gray_r)
        ax[2].axis("off")
        ax[2].set_title("Distance and markers")
        ax[2].imshow(masked_where(markers == 0, markers), cmap=plt.cm.gist_rainbow)
        ax[2].plot(maxi_coord1[1], maxi_coord1[0], "k+")
        ax[2].plot(maxi_coord[1], maxi_coord[0], "gx")

        ax[3].imshow(elevation, cmap=plt.cm.magma_r)
        ax[3].axis("off")
        ax[3].set_title("Elevation")

        ax[4].imshow(labels, cmap=plt.cm.gnuplot2_r)
        ax[4].axis("off")
        ax[4].set_title("Labels")

        ax[5].imshow(seps_img_sum, cmap=plt.cm.magma_r)
        ax[5].axis("off")
        ax[5].set_title("Segments")

    return vdf_sep


def get_gaussian2d(a, xo, yo, x, y, sigma):
    """Obtain a 2D Gaussian of amplitude a and standard deviation
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

    # TODO This function should be removed in view of its duplication within diffsims
    gaussian = (
        a
        / (2 * np.pi * sigma ** 2)
        * np.exp(-((x - xo) ** 2 + (y - yo) ** 2) / (2 * sigma ** 2))
    )

    return gaussian
