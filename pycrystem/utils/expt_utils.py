# -*- coding: utf-8 -*-
# Copyright 2017 The PyCrystEM developers
#
# This file is part of PyCrystEM.
#
# PyCrystEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyCrystEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyCrystEM.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division

import math
import numpy as np
import scipy.ndimage as ndi
from skimage import transform as tf
from skimage import morphology, filters
from skimage.morphology import square

"""
This module contains utility functions for treating experimental (scanning)
electron diffraction data.
"""

def radial_average(z, center):
    """Calculate the radial average profile about a defined center.

    Parameters
    ----------
    center : array_like
        The center about which the radial integration is performed.

    Returns
    -------

    radial_profile :

    """
    y, x = np.indices(z.shape)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), z.ravel())
    nr = np.bincount(r.ravel())
    radial_average = tbin / nr

    return radial_average

def gain_normalise(z, dref, bref):
    """Apply gain normalization to experimentally acquired electron
    diffraction pattern.

    Parameters
    ----------
    dref : ElectronDiffraction
        Dark reference image.

    bref : ElectronDiffraction
        Bright reference image.
    """
    return ((z- dref) / (bref - dref)) * np.mean((bref - dref))

def remove_dead(z, deadpixels, deadvalue):
    """Remove dead pixels from experimental electron diffraction patterns.

    Parameters
    ----------
    deadpixels : ElectronDiffraction
        List
    deadvalue : string
        Specify how deadpixels should be treated. 'average' sets the dead
        pixel value to the average of adjacent pixels. 'nan' sets the dead
        pixel to nan

    """
    img = z
    if deadvalue=='average':
        for (i,j) in deadpixels:
            neighbours = z[i-d:i+d+1, j-d:j+d+1].flatten()
            img[i,j] = np.mean(neighbours)

    elif deadvalue=='nan':
        for (i,j) in deadpixels:
            img[i,j] = nan
    else:
        raise NotImplementedError("The method specified is not implemented. "
                                  "See documentation for available "
                                  "implementations.")

    return img

def affine_transformation(z, order=3, **kwargs):
    """Apply an affine transform to a 2-dimensional array.

    Parameters
    ----------
    matrix : 3 x 3

    Returns
    -------
    trans : array
        Transformed 2-dimensional array
    """
    shift_y, shift_x = np.array(z.shape[:2]) / 2.
    tf_shift = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = tf.SimilarityTransform(translation=[shift_x, shift_y])

    transformation = tf.AffineTransform(**kwargs)
    trans = tf.warp(z, (tf_shift + (transformation + tf_shift_inv)).inverse,
                    order=order)

    return trans

def regional_filter(z, h):
    """Perform a h-dome regional filtering of the an image for background
    subtraction.

    Parameters
    ----------

    z : image as numpy array

    h :

    Returns
    -------

    h-dome subtracted image.
    """
    seed = np.copy(z)
    seed = z - h
    mask = z
    dilated = morphology.reconstruction(seed, mask, method='dilation')

    return z - dilated

def regional_flattener(z, h):
    """Localised erosion of the image 'z' for features below a value 'h'"""
    seed = np.copy(z) + h
    mask = z
    eroded = morphology.reconstruction(seed, mask, method='erosion')
    return eroded - h

def circular_mask(shape, radius, center):
    """

    Parameters
    ----------

    Returns
    -------

    """
    r = radius
    nx, ny = shape[0], shape[1]
    a, b = center

    y, x = np.ogrid[-b:ny-b, -a:nx-a]
    mask = x*x + y*y <= r*r

    return mask

def gaussian_difference_bkg(z, sigma_min, sigma_max):
    """Difference of gaussians method for background removal.

    Parameters
    ----------
    sigma_max : float
        Large gaussian blur sigma.

    sigma_min : float
        Small gaussian blur sigma.

    Returns
    -------
    Denoised diffraction pattern.
    """
    blur_max = ndi.gaussian_filter(z, sigma_max)
    blur_min = ndi.gaussian_filter(z, sigma_min)

    return np.maximum(np.where(blur_min > blur_max, z, 0) - blur_max, 0)

def blur_center(z, sigma):
    """Estimate direct beam position by blurring the image with a large
    Gaussian kernel and finding the maximum.

    Parameters
    ----------

    sigma : float
        Sigma value for Gaussian blurring kernel.

    Returns
    -------

    center : np.array
        np.array containing indices of estimated direct beam positon.
    """
    blurred = ndi.gaussian_filter(z, sigma)
    center = np.unravel_index(blurred.argmax(), blurred.shape)

    return np.array(center)

def refine_beam_position(z, start, radius):
    """Refine the position of the direct beam and hence an estimate for the
    position of the pattern center in each SED pattern.

    Parameters
    ----------
    radius : int
        Defines the size of the circular region within which the direct beam
        position is refined.
    center : bool
        If True the direct beam position is refined to sub-pixel precision
        via calculation of the intensity center of mass.

    Return
    ------
    center: array
        Refined position (x, y) of the direct beam.
    Notes
    -----
    This method is based on work presented by Thomas White in his PhD (2009)
    which itself built on Zaefferer (2000).
    """
    # initialise problem with initial center estimate
    c_int = z[start[0], start[1]]
    mask = circular_mask(shape=z.shape, radius=radius, center=start)
    z_tmp = z * mask
    # refine center position with shifting ROI
    if c_int == z_tmp.max():
        maxes = np.asarray(np.where(z_tmp == z_tmp.max()))
        c = np.rint([np.average(maxes[0]), np.average(maxes[1])])
        c = c.astype(int)
        c_int = z[c[0], c[1]]
        mask = circular_mask(shape=z.shape, radius=radius, center=c)
        ztmp = z * mask
    while c_int < z_tmp.max():
        maxes = np.asarray(np.where(z_tmp == z_tmp.max()))
        c = np.rint([np.average(maxes[0]),
                            np.average(maxes[1])])
        c = c.astype(int)
        c_int = z[c[0], c[1]]
        mask = circular_mask(shape=z.shape, radius=radius, center=c)
        ztmp = z * mask

    # For some reason the dask array is behaving badly in this function
    # so convert it to an array before computation
    c = np.asarray(ndi.measurements.center_of_mass(np.array(ztmp)))

    return c
