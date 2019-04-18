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
from scipy.ndimage.interpolation import shift
from scipy.optimize import curve_fit, minimize
from skimage import transform as tf
from skimage import morphology, filters
from skimage.morphology import square, opening
from skimage.filters import (threshold_sauvola, threshold_otsu)
from skimage.draw import ellipse_perimeter
from skimage.feature import register_translation
from scipy.optimize import curve_fit


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
    """Calculate the radial profile by azimuthal averaging about a specified
    center.

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
    dref : ElectronDiffraction
        Two-dimensional data array containing dark reference.
    bref : ElectronDiffraction
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


def affine_transformation(z, transformation, order, *args, **kwargs):
    """Apply an affine transformation to a 2-dimensional array.

    Parameters
    ----------
    z : np.array
        Array to be transformed
    matrix : np.array
        3x3 numpy array specifying the affine transformation to be applied.
    order : int
        Interpolation order.
    *args :
        To be passed to skimage.warp
    **kwargs :
        To be passed to skimage.warp

    Returns
    -------
    trans : array
        Affine transformed diffraction pattern.
    """
    trans = tf.warp(z, transformation,
                    order=order, *args, **kwargs)
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
