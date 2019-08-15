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
import scipy.ndimage as ndi
import pyxem as pxm  # for ElectronDiffraction2D

from scipy.ndimage.interpolation import shift
from scipy.optimize import curve_fit, minimize
from skimage import transform as tf
from skimage import morphology, filters
from skimage.morphology import square, opening
from skimage.filters import (threshold_sauvola, threshold_otsu)
from skimage.draw import ellipse_perimeter
from skimage.feature import register_translation
from scipy.optimize import curve_fit
from tqdm import tqdm


"""
This module contains utility functions for processing electron diffraction
patterns.
"""


def _index_coords(z, origin=None):
    """Creates x & y coords for the indicies in a numpy array.

    Parameters
    ----------
    z : np.array()
        Two-dimensional data array containing signal.
    origin : tuple
        (x,y) defaults to the center of the image. Specify origin=(0,0) to set
        the origin to the *top-left* corner of the image.

    Returns
    -------
    x, y : arrays
        Corrdinates for the indices of a numpy array.
    """
    ny, nx = z.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_x, origin_y = origin

    x, y = np.meshgrid(np.arange(float(nx)), np.arange(float(ny)))

    x -= origin_x
    y -= origin_y
    return x, y


def _cart2polar(x, y):
    """Transform Cartesian coordinates to polar coordinates.

    Parameters
    ----------
    x, y : floats or arrays
        Cartesian coordinates

    Returns
    -------
    r, theta : floats or arrays
        Polar coordinates

    """
    r = np.sqrt(x**2 + y**2)
    theta = -np.arctan2(y, x)  # θ = 0 horizontal, +ve = anticlockwise
    return r, theta


def _polar2cart(r, theta):
    """Transform polar coordinates to Cartesian coordinates.

    Parameters
    ----------
    r, theta : floats or arrays
        Polar coordinates

    Returns
    -------
    x, y : floats or arrays
        Cartesian coordinates
    """
    # +ve quadrant in bottom right corner when plotted
    x = r * np.cos(theta)
    y = -r * np.sin(theta)
    return x, y


def radial_average(z, mask=None):
    """Calculate the radial profile by azimuthal averaging about the center.

    Parameters
    ----------
    z : np.array()
        Two-dimensional data array containing signal.
    mask : np.array()
        Array with the same dimensions as z comprizing 0s for excluded pixels
        and 1s for non-excluded pixels.

    Returns
    -------
    radial_profile : np.array()
        One-dimensional radial profile of z.
    """
    # geometric shape work, not 0 indexing
    center = ((z.shape[0] / 2) - 0.5, (z.shape[1] / 2) - 0.5)

    y, x = np.indices(z.shape)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = np.rint(r - 0.5).astype(np.int)
    # the subtraction of 0.5 gets the 0 in the correct place

    if mask is None:
        tbin = np.bincount(r.ravel(), z.ravel())
        nr = np.bincount(r.ravel())
    else:
        # the mask is applied on the z array.
        masked_array = z * mask
        tbin = np.bincount(r.ravel(), masked_array.ravel())
        nr = np.bincount(r.ravel(), mask.ravel())

    averaged = np.nan_to_num(tbin / nr)

    return averaged


def gain_normalise(z, dref, bref):
    """Apply gain normalization to experimentally acquired electron
    diffraction pattern.

    Parameters
    ----------
    z : np.array()
        Two-dimensional data array containing signal.
    dref : ElectronDiffraction2D
        Two-dimensional data array containing dark reference.
    bref : ElectronDiffraction2D
        Two-dimensional data array containing bright reference.

    Returns
    -------
    z1 : np.array()
        Two dimensional data array of gain normalized z.
    """
    return ((z - dref) / (bref - dref)) * np.mean((bref - dref))


def remove_dead(z, deadpixels, deadvalue="average", d=1):
    """Remove dead pixels from experimental electron diffraction patterns.

    Parameters
    ----------
    z : np.array()
        Two-dimensional data array containing signal.
    deadpixels : np.array()
        Array containing the array indices of dead pixels in the diffraction
        pattern.
    deadvalue : string
        Specify how deadpixels should be treated, options are;
            'average': takes the average of adjacent pixels
            'nan':  sets the dead pixel to nan

    Returns
    -------
    img : array
        Two-dimensional data array containing z with dead pixels removed.
    """
    z_bar = np.copy(z)
    if deadvalue == 'average':
        for (i, j) in deadpixels:
            neighbours = z[i - d:i + d + 1, j - d:j + d + 1].flatten()
            z_bar[i, j] = np.mean(neighbours)

    elif deadvalue == 'nan':
        for (i, j) in deadpixels:
            z_bar[i, j] = np.nan
    else:
        raise NotImplementedError("The method specified is not implemented. "
                                  "See documentation for available "
                                  "implementations.")

    return z_bar


def convert_affine_to_transform(D, shape):
    """ Converts an affine transform on a diffraction pattern to a suitable
    form for skimage.transform.warp()

    Parameters
    ----------
    D : np.array
        Affine transform to be applied
    shape : tuple
        Shape tuple in form (y,x) for the diffraction pattern

    Returns
    -------
    transformation : np.array
        3x3 numpy array of the transformation to be applied.

    """

    shift_x = (shape[1] - 1) / 2
    shift_y = (shape[0] - 1) / 2

    tf_shift = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = tf.SimilarityTransform(translation=[shift_x, shift_y])

    # This defines the transform you want to perform
    distortion = tf.AffineTransform(matrix=D)

    # skimage transforms can be added like this, does matrix multiplication,
    # hence the need for the brackets. (Note tf.warp takes the inverse)
    transformation = (tf_shift + (distortion + tf_shift_inv)).inverse

    return transformation


def apply_transformation(z, transformation, keep_dtype, order=1, *args, **kwargs):
    """Apply a transformation to a 2-dimensional array.

    Parameters
    ----------
    z : np.array
        Array to be transformed
    transformation : np.array
        3x3 numpy array specifying the transformation to be applied.
    order : int
        Interpolation order.
    keep_dtype : bool
        If True dtype of returned object is that of z
    *args :
        To be passed to skimage.warp
    **kwargs :
        To be passed to skimage.warp

    Returns
    -------
    trans : array
        Affine transformed diffraction pattern.

    Notes
    -----
    Generally used in combination with pyxem.expt_utils.convert_affine_to_transform
    """
    if keep_dtype == False:
        trans = tf.warp(z, transformation,
                        order=order, *args, **kwargs)
    if keep_dtype == True:
        trans = tf.warp(z, transformation,
                        order=order, preserve_range=True, *args, **kwargs)
        trans = trans.astype(z.dtype)

    return trans


def regional_filter(z, h):
    """Perform a h-dome regional filtering of the an image for background
    subtraction.

    Parameters
    ----------
    h : float
        h-dome cutoff value.

    Returns
    -------
        h-dome subtracted image as np.array
    """
    seed = np.copy(z)
    seed = z - h
    mask = z
    dilated = morphology.reconstruction(seed, mask, method='dilation')

    return z - dilated


def subtract_background_dog(z, sigma_min, sigma_max):
    """Difference of gaussians method for background removal.

    Parameters
    ----------
    sigma_max : float
        Large gaussian blur sigma.
    sigma_min : float
        Small gaussian blur sigma.

    Returns
    -------
        Denoised diffraction pattern as np.array
    """
    blur_max = ndi.gaussian_filter(z, sigma_max)
    blur_min = ndi.gaussian_filter(z, sigma_min)

    return np.maximum(np.where(blur_min > blur_max, z, 0) - blur_max, 0)


def subtract_background_median(z, footprint=19, implementation='scipy'):
    """Remove background using a median filter.

    Parameters
    ----------
    footprint : int
        size of the window that is convoluted with the array to determine
        the median. Should be large enough that it is about 3x as big as the
        size of the peaks.
    implementation: str
        One of 'scipy', 'skimage'. Skimage is much faster, but it messes with
        the data format. The scipy implementation is safer, but slower.

    Returns
    -------
        Pattern with background subtracted as np.array
    """

    if implementation == 'scipy':
        bg_subtracted = z - ndi.median_filter(z, size=footprint)
    elif implementation == 'skimage':
        selem = morphology.square(footprint)
        # skimage only accepts input image as uint16
        bg_subtracted = z - filters.median(z.astype(np.uint16), selem).astype(z.dtype)

    return np.maximum(bg_subtracted, 0)


def subtract_reference(z, bg):
    """Subtracts background using a user-defined background pattern.

    Parameters
    ----------
    z : np.array()
        Two-dimensional data array containing signal.
    bg: array()
        User-defined diffraction pattern to be subtracted as background.

    Returns
    -------
    im : np.array()
        Two-dimensional data array containing signal with background removed.
    """
    im = z.astype(np.float64) - bg
    for i in range(0, z.shape[0]):
        for j in range(0, z.shape[1]):
            if im[i, j] < 0:
                im[i, j] = 0
    return im


def circular_mask(shape, radius, center=None):
    """Produces a mask of radius 'r' centered on 'center' of shape 'shape'.

    Parameters
    ----------
    shape : tuple
        The shape of the signal to be masked.
    radius : int
        The radius of the circular mask.
    center : tuple (optional)
        The center of the circular mask. Default: (0, 0)

    Returns
    -------
    mask : np.array()
        The circular mask.

    """
    l_x, l_y = shape
    x, y = center if center else (l_x / 2, l_y / 2)
    X, Y = np.ogrid[:l_x, :l_y]
    mask = (X - x) ** 2 + (Y - y) ** 2 < radius ** 2
    return mask


def reference_circle(coords, dimX, dimY, radius):
    """Draw the perimeter of an circle at a given position in the diffraction
    pattern (e.g. to provide a reference for finding the direct beam center).

    Parameters
    ----------
    coords : np.array size n,2
        size n,2 array of coordinates to draw the circle.
    dimX : int
        first dimension of the diffraction pattern (size)
    dimY : int
        second dimension of the diffraction pattern (size)
    radius : int
        radius of the circle to be drawn

    Returns
    -------
    img: np.array
        Array containing the circle at the position given in the coordinates.
    """
    img = np.zeros((dimX, dimY))

    for n in range(np.size(coords, 0)):
        rr, cc = ellipse_perimeter(coords[n, 0], coords[n, 1], radius, radius)
        img[rr, cc] = 1

    return img


def find_beam_offset_cross_correlation(z, radius_start=4, radius_finish=8):
    """Method to center the direct beam center by a cross-correlation algorithm.
    The shift is calculated relative to an circle perimeter. The circle can be
    refined across a range of radii during the centring procedure to improve
    performance in regions where the direct beam size changes,
    e.g. during sample thickness variation.

    Parameters
    ----------
    radius_start : int
        The lower bound for the radius of the central disc to be used in the
        alignment.
    radius_finish : int
        The upper bounds for the radius of the central disc to be used in the
        alignment.

    Returns
    -------
    shift: np.array
        np.array containing offset (from center) of the direct beam positon.
    """
    radiusList = np.arange(radius_start, radius_finish)
    errRecord = np.zeros_like(radiusList, dtype='single')
    origin = np.array([[round(np.size(z, axis=-2) / 2), round(np.size(z, axis=-1) / 2)]])

    for ind in np.arange(0, np.size(radiusList)):
        radius = radiusList[ind]
        ref = reference_circle(origin, np.size(z, axis=-2), np.size(z, axis=-1), radius)
        h0 = np.hanning(np.size(ref, 0))
        h1 = np.hanning(np.size(ref, 1))
        hann2d = np.sqrt(np.outer(h0, h1))
        ref = hann2d * ref
        im = hann2d * z
        shift, error, diffphase = register_translation(ref, im, 10)
        errRecord[ind] = error
        index_min = np.argmin(errRecord)

    ref = reference_circle(origin, np.size(z, axis=-2), np.size(z, axis=-1), radiusList[index_min])
    h0 = np.hanning(np.size(ref, 0))
    h1 = np.hanning(np.size(ref, 1))
    hann2d = np.sqrt(np.outer(h0, h1))
    ref = hann2d * ref
    im = hann2d * z
    shift, error, diffphase = register_translation(ref, im, 100)

    return (shift - 0.5)


def calc_radius_with_distortion(x, y, xc, yc, asym, rot):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    xp = x * np.cos(rot) - y * np.sin(rot)
    yp = x * np.sin(rot) + y * np.cos(rot)
    xcp = xc * np.cos(rot) - yc * np.sin(rot)
    ycp = xc * np.sin(rot) + yc * np.cos(rot)

    return np.sqrt((xp - xcp)**2 + asym * (yp - ycp)**2)


def call_ring_pattern(xcentre, ycentre):
    """
    Function to make a call to the function ring_pattern without passing the
    variables directly (necessary for using scipy.optimize.curve_fit).

    Parameters
    ----------
    xcentre : float
        The coordinate (fractional pixel units) of the diffraction
        pattern centre in the first dimension
    ycentre : float
        The coordinate (fractional pixel units) of the diffraction
        pattern centre in the second dimension

    Returns
    -------
    ring_pattern : function
        A function that calculates a ring pattern given a set of points and
        parameters.

    """
    def ring_pattern(pts, scale, amplitude, spread, direct_beam_amplitude,
                     asymmetry, rotation):
        """
        Calculats a polycrystalline gold diffraction pattern given a set of
        pixel coordinates (points).
        It uses tabulated values of the spacings (in reciprocal Angstroms)
        and relative intensities of rings derived from X-ray scattering factors.

        Parameters
        -----------
        pts : 1D array
            One-dimensional array of points (first half as first-dimension
            coordinates, second half as second-dimension coordinates)
        scale : float
            An initial guess for the diffraction calibration
            in 1/Angstrom units
        amplitude : float
            An initial guess for the amplitude of the polycrystalline rings
            in arbitrary units
        spread : float
            An initial guess for the spread within each ring (Gaussian width)
        direct_beam_amplitude : float
            An initial guess for the background intensity from
            the direct beam disc in arbitrary units
        asymmetry : float
            An initial guess for any elliptical asymmetry in the pattern
            (for a perfectly circular pattern asymmetry=1)
        rotation : float
            An initial guess for the rotation of the (elliptical) pattern
            in radians.

        Returns
        -------
        ring_pattern : np.array()
            A one-dimensional array of the intensities of the ring pattern
            at the supplied points.

        """
        ring1, ring2, ring3, ring4, ring5, ring6, ring7, ring8 = 0.4247, \
            0.4904, 0.6935, 0.8132, 0.8494, 0.9808, 1.0688, 1.0966
        ring1, ring2, ring3, ring4, ring5, ring6, ring7, ring8 = ring1 * scale, \
            ring2 * scale, ring3 * scale, ring4 * scale, ring5 * scale, \
            ring6 * scale, ring7 * scale, ring8 * scale
        amp1, amp2, amp3, amp4, amp5, amp6, amp7, amp8 = 1, 0.44, 0.19, \
            0.16, 0.04, 0.014, 0.038, 0.036

        x = pts[:round(np.size(pts, 0) / 2)]
        y = pts[round(np.size(pts, 0) / 2):]
        Ri = calc_radius_with_distortion(x, y, xcentre, ycentre,
                                         asymmetry, rotation)

        denom = 2 * spread**2
        v0 = direct_beam_amplitude * Ri**-2  # np.exp((-1*(Ri)*(Ri))/d0)
        v1 = amp1 * np.exp((-1 * (Ri - ring1) * (Ri - ring1)) / denom)
        v2 = amp2 * np.exp((-1 * (Ri - ring2) * (Ri - ring2)) / denom)
        v3 = amp3 * np.exp((-1 * (Ri - ring3) * (Ri - ring3)) / denom)
        v4 = amp4 * np.exp((-1 * (Ri - ring4) * (Ri - ring4)) / denom)
        v5 = amp5 * np.exp((-1 * (Ri - ring5) * (Ri - ring5)) / denom)
        v6 = amp6 * np.exp((-1 * (Ri - ring6) * (Ri - ring6)) / denom)
        v7 = amp7 * np.exp((-1 * (Ri - ring7) * (Ri - ring7)) / denom)
        v8 = amp8 * np.exp((-1 * (Ri - ring8) * (Ri - ring8)) / denom)

        return amplitude * (v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8).ravel()
    return ring_pattern


def peaks_as_gvectors(z, center, calibration):
    """Converts peaks found as array indices to calibrated units, for use in a
    hyperspy map function.

    Parameters
    ----------
    z : numpy array
        peak postitions as array indices.
    center : numpy array
        diffraction pattern center in array indices.
    calibration : float
        calibration in reciprocal Angstroms per pixels.

    Returns
    -------
    g : numpy array
        peak positions in calibrated units.

    """
    g = (z - center) * calibration
    return np.array([g[0].T[1], g[0].T[0]]).T


def investigate_dog_background_removal_interactive(sample_dp,
                                                   std_dev_maxs,
                                                   std_dev_mins):
    """Utility function to help the parameter selection for the difference of
    gaussians (dog) background subtraction method

    Parameters
    ----------
    sample_dp : ElectronDiffraction2D
        A single diffraction pattern
    std_dev_maxs : iterable
        Linearly spaced maximum standard deviations to be tried, ascending
    std_dev_mins : iterable
        Linearly spaced minimum standard deviations to be tried, ascending

    Returns
    -------
    A hyperspy like navigation (sigma parameters), signal (proccessed patterns)
    plot

    See Also
    --------
    subtract_background_dog : The background subtraction method used.
    np.arange : Produces suitable objects for std_dev_maxs

    """
    gauss_processed = np.empty((
        len(std_dev_maxs),
        len(std_dev_mins),
        *sample_dp.axes_manager.signal_shape))

    for i, std_dev_max in enumerate(tqdm(std_dev_maxs, leave=False)):
        for j, std_dev_min in enumerate(std_dev_mins):
            gauss_processed[i, j] = sample_dp.remove_background('gaussian_difference',
                                                                sigma_min=std_dev_min, sigma_max=std_dev_max,
                                                                show_progressbar=False)
    dp_gaussian = pxm.ElectronDiffraction2D(gauss_processed)
    dp_gaussian.metadata.General.title = 'Gaussian preprocessed'
    dp_gaussian.axes_manager.navigation_axes[0].name = r'$\sigma_{\mathrm{min}}$'
    dp_gaussian.axes_manager.navigation_axes[1].name = r'$\sigma_{\mathrm{max}}$'
    for axes_number, axes_value_list in [(0, std_dev_mins), (1, std_dev_maxs)]:
        dp_gaussian.axes_manager.navigation_axes[axes_number].offset = axes_value_list[0]
        dp_gaussian.axes_manager.navigation_axes[axes_number].scale = axes_value_list[1] - axes_value_list[0]
        dp_gaussian.axes_manager.navigation_axes[axes_number].units = ''

    dp_gaussian.plot(cmap='viridis')
    return None
