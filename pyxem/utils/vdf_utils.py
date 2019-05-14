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

from skimage.feature import peak_local_max, match_template
from skimage.filters import sobel
from skimage.morphology import watershed

from sklearn.cluster import DBSCAN

from hyperspy.signals import BaseSignal, Signal2D


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
    and a template using match_template from skimage.feature.

    Parameters
    ----------
    image: np.array
        Image
    template: np.array
        Reference image
        
    Returns
    -------
    corr : float
        Normalised cross-correlation between image and template at zero
        displacement.
    """
    # If image and template are alike, 1 will be returned directly.
    if np.array_equal(image, template):
        corr = 1.
    else: 
        # Return only the value in the middle, i.e. at zero displacement
        corr = match_template(
            image=image, template=template, pad_input=True, mode='constant',
            constant_values=0)[int(np.shape(image)[0]/2),
                               int(np.shape(image)[1]/2)]

    return corr


def get_vectors_and_indices_i(vectors_add, vectors_i, indices_add, indices_i):
    """ Obtain an array of vectors resulting from merging the vectors in
    vectors_i and vectors_add and then removing duplicates. Also, an
    array of indices corresponding to the vectors will be obtained.

    Parameters
    ----------
    vectors_add : ndarray
        Array of vectors.
    vectors_i : ndarray
        Array of vectors.
    indices_add : ndarray
        Array of arrays of integers corresponding to the indices of the
        vectors in vectors_add, referring to the indices in the original
        VDFSegment.
    indices_i : ndarray
        Indices corresponding to the vectors in vectors_i.
    Returns
    -------
    g : ndarray
        Vectors in the form of an object of shape (n, 2) with all the
        unique n vectors found in vectors_i and vectors_add. '
    indices : ndarray
        Indices corresponding to the vectors in g.
    """
    # TODO Is it possible to simplify further?

    if len(np.shape(vectors_i)) == 1:
        g = np.array([vectors_i])
    else:
        g = vectors_i.copy()

    if len(np.shape(indices_i)) > 1:
        indices = np.array([], dtype=object)
        for index in indices_i:
            indices = np.append(indices, index).astype(object)
    else:
        indices = indices_i.copy().astype(object)

    for i, index in zip(range(np.shape(vectors_add)[0]), indices_add):
        indices = np.append(indices, index)
        if len(np.shape(vectors_add[i])) == 1:
            g = np.append(g, np.array([vectors_add[i]]), axis=0)
        else:
            g = np.append(g, vectors_add[i], axis=0)

    # Check if there are any duplicates.
    g_delete = np.array([]).astype(int)

    for i in range(np.shape(g)[0]):
        g_is_equal = list(map(lambda x: np.array_equal(g[i], x), g[i + 1:]))

        if sum(g_is_equal):
            g_delete = np.append(g_delete, i)

            # If the equal vectors do not have the same indices, make sure that
            # all required indices are added to the correct element.
            if np.shape(np.where(g_is_equal))[1]>1:
                index = int(np.where(g_is_equal)[0][0]+i+1)
            else:
                index = int(np.where(g_is_equal)[0]+i+1)

            if not np.all(np.isin(indices[i], indices[index])):
                indices[index] = np.append(indices[index],
                                           indices[i]).astype(int)

    # Delete duplicates.
    g = np.delete(g, g_delete, axis=0)
    indices = np.delete(indices, g_delete, axis=0)

    return g, indices


def separate(vdf_temp, background_value, min_distance, min_size,
             max_size, max_number_of_grains, exclude_border=False,
             plot_on=False):
    """Separate segments from one VDF image using edge-detection by the
    sobel transform and the watershed segmentation implemented in
    scikit-image. See [1,2] for examples from scikit-image.

    Parameters
    ----------
    vdf_temp : np.array
        One VDF image.
    background_value : int
        The value of the background intensity for vdf_temp. Can be found
        by using get_vdf_background_intensities. Only VDF intensities
        above background_value are considered signal.
    min_distance: int
        Minimum distance (in pixels) between grains required for them to
        be considered as separate grains.
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
    exclude_border : int or True, optional
        If non-zero integer, peaks within a distance of exclude_border
        from the boarder will be discarded. If True, peaks at or closer
        than min_distance of the boarder, will be discarded.
    plot_on : bool
        If True, the VDF, the thresholded VDF, the distance transform
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

    # Create a mask that is True where the VDF intensity is larger than a
    # background value.
    mask = (vdf_temp > background_value).astype('bool')
    if np.any(np.nonzero(mask)) is False:
        print('All VDF intensities are below the background value of ' +
              str(background_value) + ', so no segments were found.\n')
        return None

    # Calculate the eucledian distance from each point in a binary image to the
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
    # (labels) in the area defined by mask (thresholded input VDF).
    labels = watershed(elevation, markers=label(local_maxi), mask=mask)

    if not np.max(labels):
        print('No segments were found. Check input parameters.\n')

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


def get_circular_mask(vec, radius, cx, cy, x, y):
    """Obtain a boolean mask that is only True within a circle centred
    at vector vec and that has the given radius.

    Parameters
    ----------
    vec : np.array
        2-dimensional array that gives the centre values of the circle;
        [vec x, vec y].
    radius : float
        Radius of the circle.
    cx : float
        Offset in the x-direction.
    cy : float
        Offset in the y-direction.
    x : np.array
        Indies for the x-axis of the mask. For the intended use;
        y, x = np.indices((shape_x, shape_y)).
    y : np.array
        Indies for the y-axis of the mask.

    Returns
    -------
    mask_temp : bool
        Mask that is only True within a circle given by the input
        parameters.
    """
    # TODO Check if centered correctly!!
    radial_grid_temp = np.sqrt((x - (vec[0] - cx)) ** 2 +
                               (y - (vec[1] - cy)) ** 2)
    mask_temp = (radial_grid_temp > radius).choose(radius, 0)
    mask_temp = (radial_grid_temp <= radius).choose(mask_temp, 1)
    return mask_temp.astype('bool')


def get_vdf_background_from_sum_signal(unique_vectors, radius,
                                       sum_signal, navigation_size,
                                       plot_background):
    """ Obtain an array of the background intensities for VDFs resulting
    from the vectors in unique_vectors. The background intensities are
    calculated by radially integrating a sum_signal where all the
    vectors in unique_vectors are masked out by virtual apertures of
    the given radius. The integrated intensity is then divided by
    navigation_size to give an average value, and then it is convolved
    with a virtual aperture function in 1D to give the average background
    intensity for each VDF image.

    Parameters
    ----------
    unique_vectors : DiffractionVector
        A DiffractionVector with shape (n, 2) with n unique vectors
        corresponding to n VDFs.
    radius : float
        Radius of the virtual aperture used to mask away all the unique
        vectors. Given in reciprocal Angstroms.
    sum_signal : Signal2D
        The image to calculate the background intensities from.
        To obtain the average background intensities, this should be the
        sum of all signals. For VDFs resulting from an
        ElectronDiffraction signal s, this is given by s.sum().
    navigation_size : int
        The total number of pixels in each VDF. For VDFs resulting from
        an ElectronDiffraction signal s, this is given by
        s.axes_manager.navigation_size.
    plot_background : bool
        If True, the masked sum_signal, integrated masked sum_signal in
        1D and the background intensities in 1D are plotted.

    Returns
    -------
    bkg_values : np.array
        The (average) background intensities for VDFs created with a
        virtual aperture of the given radius, corresponding to each
        vector in unique_vector.
    """
    gmags = unique_vectors.get_magnitudes().data
    scale = sum_signal.axes_manager.signal_axes[0].scale
    dp_shape_y, dp_shape_x = sum_signal.axes_manager.signal_shape
    cy = sum_signal.axes_manager.signal_axes[0].offset
    cx = sum_signal.axes_manager.signal_axes[1].offset
    y, x = np.indices((dp_shape_x, dp_shape_y))
    x, y = x * scale, y * scale

    # Create a mask for the sum_signal where all the diffraction vectors
    # have been masked out by creating circular apertures of the given
    # radius for all vectors. For sum_signal, the region within each
    # aperture is set to nan, so that the diffraction vector intensities
    # do not contribute to the calculated average background.
    vector_mask = np.sum(list(map(
        lambda b: get_circular_mask(b, radius, cx, cy, x, y),
        unique_vectors.data)), axis=0).astype('bool')
    sum_signal_masked = sum_signal.data.copy().astype('float32')
    sum_signal_masked[np.where(vector_mask == 1)] = np.nan

    # The masked sum_signal is integrated radially to give a average 1D
    # background.
    mask = ~np.isnan(sum_signal_masked)
    radial_grid = (np.sqrt((x / scale + cx / scale + 0.5) ** 2 + (
                y / scale + cy / scale + 0.5) ** 2) - 0.5).astype('int')
    sum_masked_1d = np.bincount(radial_grid[mask].ravel(),
                                weights=sum_signal_masked[mask].ravel()) \
                    / np.bincount(radial_grid[mask].ravel())

    # A circular aperture of the given radius is integrated in 1D.
    aperture = get_circular_mask([0, 0], radius, cx, cy, x, y)
    aperture_1d = np.sum(aperture, axis=0)
    aperture_1d = aperture_1d[aperture_1d > 0]

    # The average background for VDFs calculated with aprtures of the
    # given radius can then be found by convoluting the aperture in 1D
    # with the average 1D background.
    bkg_1d = np.convolve(aperture_1d, sum_masked_1d, mode='same') \
             / navigation_size

    # The background value for each VDF is then given by the magnitude
    # of the corresponding diffraction vector.
    axis = np.arange(np.max(radial_grid) + 1) * scale
    bkg_values = np.array(
        list(map(lambda a: bkg_1d[(np.abs(axis - a)).argmin()], gmags)))
    bkg_values = bkg_values.astype('int')

    # Optionally plot the masked sum_signal, in 2D and 1D, and the
    # average background in 1D.
    if plot_background:
        BaseSignal(sum_signal_masked).plot(cmap='magma_r', vmax=30000)
        BaseSignal(sum_masked_1d).plot()
        BaseSignal(bkg_1d).plot()

    return bkg_values


def get_single_pattern_background(pattern, peak_positions, radius, cx,
                                  cy, x, y, radial_grid):
    """ Get the average radial integral of the pattern where all peaks
    have been masked out by an aperture of a given radius.

    Parameters
    ----------
    pattern : np.array
        The image to calculate the background intensities from.
    peak_positions : DiffractionVector
        A DiffractionVector signal with shape (n, 2) that holds the
        positions of n unique vectors.
    radius : float
        Radius of the virtual aperture used to mask away all the unique
        vectors. Given in reciprocal Angstroms.
        cx : float
        Offset in the x-direction.
    cx : float
        Offset in the x-direction for the centre of pattern.
    cy : float
        Offset in the y-direction for the centre of pattern.
    x : np.array
        Indies for the x-axis of the pattern. For the intended use;
        y, x = np.indices((shape_x, shape_y)).
    y : np.array
        Indies for the y-axis of the pattern.
    radial_grid : np.array
        The radial integral is calculated over raidal_grid. Typically,
        this is given by:
        radial_grid = (np.sqrt((x / scale + cx / scale + 0.5) ** 2 + (
            y / scale + cy / scale + 0.5) ** 2) - 0.5).astype('int')

    Returns
    -------
    pattern_masked_1d : np.array
        The background intensities found by averaging the radial
        integral of the pattern with all peak_positions masked away by
        circles of the given radius.
    """
    vector_mask = np.sum(list(map(
        lambda b: get_circular_mask(b, radius, cx, cy, x, y),
        peak_positions.data)), axis=0).astype('bool')
    pattern_masked = pattern.data.copy().astype('float32')
    pattern_masked[np.where(vector_mask == 1)] = np.nan

    # The masked sum_signal is integrated radially to give an average 1D
    # background.
    mask = ~np.isnan(pattern_masked)
    pattern_masked_1d = np.bincount(
        radial_grid[mask].ravel(),
        weights=pattern_masked[mask].ravel()) \
        / np.bincount(radial_grid[mask].ravel())

    return pattern_masked_1d


def get_background(signal, peak_positions, radius,
                   pattern_fraction=0.1, return_std=False,
                   return_max=False):
    """ Obtain an array of the background intensities by radially
    integrating a random fraction of all patterns in signal, where peaks
    are masked out by circular apertures of the given radius.

    Parameters
    ----------
    signal : ElectronDiffraction
        ElectronDiffraction signal to calculate the background from.
    peak_positions : DiffractionVector
        A DiffractionVector with shape equal to the shape of signal,
        where each navigation position holds the positions of the unique
        vectors found at the corresponding navigation positions in
        signal. Typically obtained from signal.find_peaks().
    radius : float
        Radius of the virtual aperture used to mask away all the peaks.
        Given in reciprocal Angstroms.
    pattern_fraction : float
        Float (0., 1.) that determines how many patterns from signal (of
        the total number of patterns) that will be randomly selected and
        used for the background calculation.
    return_std : bool
        If True, the standard deviations of the masked radial integrals
        are returned.
    return_max : bool
        If True, the maximum values of the radial integrals of all
        selected patterns are returned.

    Returns
    -------
    bg_1d : np.array
        The (average) radial integral of a fraction of randomly selected
        patterns, with the peak positions masked out.
    std : np.array, optional
        The standard deviation of the masked radial integrals. Returned
        if return_std=True.
    bg_1d_max : np.array, optional
        The maximum values of the masked radial integrals. Returned if
        return_max=True.

    """
    scale = signal.axes_manager.signal_axes[0].scale
    dp_shape_y, dp_shape_x = signal.axes_manager.signal_shape
    nav_size_x, nav_size_y = signal.axes_manager.navigation_shape
    number_of_patterns = int(signal.axes_manager.navigation_size*pattern_fraction)
    cy = signal.axes_manager.signal_axes[0].offset
    cx = signal.axes_manager.signal_axes[1].offset
    y, x = np.indices((dp_shape_x, dp_shape_y))
    x, y = x * scale, y * scale
    radial_grid = (np.sqrt((x / scale + cx / scale + 0.5) ** 2 + (
            y / scale + cy / scale + 0.5) ** 2) - 0.5).astype('int')
    # Create a mask for the sum_signal where all the diffraction vectors
    # have been masked out by creating circular apertures of the given
    # radius for all vectors. For sum_signal, the region within each
    # aperture is set to nan, so that the diffraction vector intensities
    # do not contribute to the calculated average background.
    random_indices_x = np.random.randint(low=0, high=nav_size_x-1, size=number_of_patterns, dtype='int')
    random_indices_y = np.random.randint(low=0, high=nav_size_y-1, size=number_of_patterns, dtype='int')
    bgs = []
    for i in range(number_of_patterns):
        bgs.append(get_single_pattern_background(
            signal.inav[random_indices_x[i], random_indices_y[i]],
            peak_positions.inav[random_indices_x[i], random_indices_y[i]],
            radius, cx, cy, x, y, radial_grid))
    bg_1d = np.mean(bgs, axis=0)
    bg_1d_max = np.max(bgs, axis=0)
    if return_std and return_max:
        std = np.std(bgs, axis=0)
        return bg_1d, std, bg_1d_max
    elif not return_std and return_max:
        return bg_1d, bg_1d_max
    elif return_std and not return_max:
        std = np.std(bgs, axis=0)
        return bg_1d, std
    else:
        return bg_1d


def get_vdf_background(signal, unique_vectors, radius, bg, std=None,
                       maxs=None, return_radials=False):
    """ Obtain the background intensities for VDFs corresponding to the
    vectors in unique_vectors, by convoluting the given 1d background
    with a 1d virtual aperture function of size given by radius.

    Parameters
    ----------
    signal : ElectronDiffraction
        ElectronDiffraction signal that unique_vectors and bg originate
        from.
    unique_vectors : DiffractionVector
        A DiffractionVector with shape (n, 2) with n unique vectors
        corresponding to n VDFs.
    radius : float
        Radius of the virtual aperture used to mask away all the unique
        vectors. Given in reciprocal Angstroms.
    bg : np.array
        The 1d background of signal, typically given as the (average)
        radial integral of patterns, with the peak positions masked out.
        See get_background().
    std : np.array
        The standard deviation corresponding to the background values
        given in bg. Default is None, and if given, the std values are
        convoluted with a 1d circular aperture function, and returned.
    maxs : np.array
        The maximum values of the 1d background. Default is None, and if
        given, the maxs values are convoluted with a 1d circular
        aperture function, and returned.
    return_radials : bool
        If True (default is False), the full 1d arrays are returned, in
        addition to arrays only holding the values corresponding to the
        given unique_vectors.

    Returns
    -------
    arrays_in_return : array
        Array that holds the VDF background values corresponding to each
        vector in unique_vectors. If std and/or maxs is given, the VDF
        standard deviation and maxima arrays are also returned. Finally,
        if return_radials is True (default is False), the full 1d
        arrays are also returned. The returned arrays are given as:
        [bkg_values, std_values, max_values,
        bkg_1d, vdf_std_1d, vdf_max_1d].

    """
    gmags = unique_vectors.get_magnitudes().data
    dp_shape_y, dp_shape_x = signal.axes_manager.signal_shape
    cy = signal.axes_manager.signal_axes[0].offset
    cx = signal.axes_manager.signal_axes[1].offset
    scale = signal.axes_manager.signal_axes[0].scale
    y, x = np.indices((dp_shape_x, dp_shape_y))
    x, y = x * scale, y * scale

    # A circular aperture of the given radius is integrated in 1D.
    aperture = get_circular_mask([0, 0], radius, cx, cy, x, y)
    aperture_1d = np.sum(aperture, axis=0)
    aperture_1d = aperture_1d[aperture_1d > 0]

    # The average background for VDFs calculated with apertures of the
    # given radius can then be found by convoluting the aperture in 1D
    # with the average 1D background.
    bkg_1d = np.convolve(aperture_1d, bg, mode='same')

    # The background value for each VDF is then given by the magnitude
    # of the corresponding diffraction vector.
    axis = np.arange(len(bg) + 1) * scale
    bkg_values = np.array(
        list(map(lambda a: bkg_1d[(np.abs(axis - a)).argmin()], gmags)))
    bkg_values = bkg_values.astype('int')

    if std is not None:
        vdf_std_1d = np.convolve(aperture_1d, std, mode='same')
        std_values = np.array(list(map(lambda a: vdf_std_1d[
            (np.abs(axis - a)).argmin()], gmags)))

    if maxs is not None:
        vdf_max_1d = np.convolve(aperture_1d, maxs, mode='same')
        max_values = np.array(list(map(lambda a: vdf_max_1d[
            (np.abs(axis - a)).argmin()], gmags)))

    if return_radials:
        vdf_std_1d, vdf_max_1d, bkg_1d = None, None, None

    arrays_in_return = [a for a in [bkg_values, std_values, max_values,
                                    bkg_1d, vdf_std_1d, vdf_max_1d]
                        if a is not None]
    return arrays_in_return
