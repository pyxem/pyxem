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
from scipy.ndimage.interpolation import shift
from scipy.optimize import curve_fit, minimize
from skimage import transform as tf
from skimage import morphology, filters
from skimage.morphology import square, opening
from skimage.filters import (threshold_sauvola, threshold_otsu)

try:
    from .radialprofile import radialprofile as radialprofile_cy
except ImportError:
    _USE_CY_RADIAL_PROFILE = False
else:
    _USE_CY_RADIAL_PROFILE = True

"""
This module contains utility functions for processing electron diffraction
patterns.
"""

def _index_coords(z, origin=None):
    """
    Creates x & y coords for the indicies in a numpy array

    Parameters
    ----------
    data : numpy array
        2D data
    origin : (x,y) tuple
        defaults to the center of the image. Specify origin=(0,0)
        to set the origin to the *bottom-left* corner of the image.

    Returns
    -------
        x, y : arrays
    """
    ny, nx = z.shape[:2]
    if origin is None:
        origin_x, origin_y = nx//2, ny//2
    else:
        origin_x, origin_y = origin

    x, y = np.meshgrid(np.arange(float(nx)), np.arange(float(ny)))

    x -= origin_x
    y -= origin_y
    return x, y

def _cart2polar(x, y):
    """
    Transform Cartesian coordinates to polar

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
    theta = np.arctan2(x, y)  # θ referenced to vertical
    return r, theta

def _polar2cart(r, theta):
    """
    Transform polar coordinates to Cartesian

    Parameters
    -------
    r, theta : floats or arrays
        Polar coordinates

    Returns
    ----------
    x, y : floats or arrays
        Cartesian coordinates
    """
    y = r * np.cos(theta)   # θ referenced to vertical
    x = r * np.sin(theta)
    return x, y

def radial_average(z,cython=False):
    """Calculate the radial profile by azimuthal averaging about a specified
    center.

    Parameters
    ----------

    cython=False
        Set to False if cython needs to be avoided. If cythonized option is not
        not avaliable the behaviour is equivilant to cython == False
    Returns
    -------
    radial_profile : array
        Radial profile of the diffraction pattern.
    """
    
    center = ((z.shape[0]/2)-0.5,(z.shape[1]/2)-0.5) #geometric shape work, not 0 indexing
    
    if _USE_CY_RADIAL_PROFILE and cython:
        averaged = radialprofile_cy(z, center)
    else:
        y, x = np.indices(z.shape)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = np.rint(r-0.5).astype(np.int) 
        #the subtraction of 0.5 gets the 0 in the correct place

        tbin = np.bincount(r.ravel(), z.ravel())
        nr = np.bincount(r.ravel())
        averaged = np.nan_to_num(tbin / nr)

    return averaged

def reproject_polar(z, origin=None, jacobian=False, dr=1, dt=None):
    """
    Reprojects a 2D diffraction pattern into a polar coordinate system.

    Parameters
    ----------
    origin : tuple
        The coordinate (x0, y0) of the image center, relative to bottom-left. If
        'None'defaults to
    Jacobian : boolean
        Include ``r`` intensity scaling in the coordinate transform.
        This should be included to account for the changing pixel size that
        occurs during the transform.
    dr : float
        Radial coordinate spacing for the grid interpolation
        tests show that there is not much point in going below 0.5
    dt : float
        Angular coordinate spacing (in radians)
        if ``dt=None``, dt will be set such that the number of theta values
        is equal to the maximum value between the height or the width of
        the image.

    Returns
    -------
    output : 2D np.array
        The polar image (r, theta)

    Notes
    -----
    Adapted from: PyAbel, www.github.com/PyAbel/PyAbel

    """
    # bottom-left coordinate system requires numpy image to be np.flipud
    data = np.flipud(z)

    ny, nx = data.shape[:2]
    if origin is None:
        origin = (nx//2, ny//2)

    # Determine that the min and max r and theta coords will be...
    x, y = _index_coords(z, origin=origin)  # (x,y) coordinates of each pixel
    r, theta = _cart2polar(x, y)  # convert (x,y) -> (r,θ), note θ=0 is vertical

    nr = np.int(np.ceil((r.max()-r.min())/dr))

    if dt is None:
        nt = max(nx, ny)
    else:
        # dt in radians
        nt = np.int(np.ceil((theta.max()-theta.min())/dt))

    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i = np.linspace(r.min(), r.max(), nr, endpoint=False)
    theta_i = np.linspace(theta.min(), theta.max(), nt, endpoint=False)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    X, Y = _polar2cart(r_grid, theta_grid)

    X += origin[0]  # We need to shift the origin
    Y += origin[1]  # back to the bottom-left corner...
    xi, yi = X.flatten(), Y.flatten()
    coords = np.vstack((yi, xi))  # (map_coordinates requires a 2xn array)

    zi = ndi.map_coordinates(z, coords)
    output = zi.reshape((nr, nt))

    if jacobian:
        output = output*r_i[:, np.newaxis]

    return output

def gain_normalise(z, dref, bref):
    """Apply gain normalization to experimentally acquired electron
    diffraction pattern.

    Parameters
    ----------
    dref : ElectronDiffraction
        Dark reference image.
    bref : ElectronDiffraction
        Flat-field bright reference image.

    Returns
    -------
        Gain normalized diffraction pattern
    """
    return ((z- dref) / (bref - dref)) * np.mean((bref - dref))

def remove_dead(z, deadpixels, deadvalue="average", d=1):
    """Remove dead pixels from experimental electron diffraction patterns.

    Parameters
    ----------
    deadpixels : array
        Array containing the array indices of dead pixels in the diffraction
        pattern.
    deadvalue : string
        Specify how deadpixels should be treated, options are;
            'average': takes the average of adjacent pixels
            'nan':  sets the dead pixel to nan

    Returns
    -------
    img : array
        Array containing the diffraction pattern with dead pixels removed.
    """
    if deadvalue == 'average':
        for (i,j) in deadpixels:
            neighbours = z[i-d:i+d+1, j-d:j+d+1].flatten()
            z[i,j] = np.mean(neighbours)

    elif deadvalue == 'nan':
        for (i,j) in deadpixels:
            z[i,j] = np.nan
    else:
        raise NotImplementedError("The method specified is not implemented. "
                                  "See documentation for available "
                                  "implementations.")

    return z

def affine_transformation(z, order, **kwargs):
    """Apply an affine transformation to a 2-dimensional array.

    Parameters
    ----------
    matrix : np.array
        3x3 numpy array specifying the affine transformation to be applied.
    order : int
        Interpolation order.

    Returns
    -------
    trans : array
        Affine transformed diffraction pattern.
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

def regional_flattener(z, h):
    """Localised erosion of the image 'z' for features below a value 'h'"""
    seed = np.copy(z) + h
    mask = z
    eroded = morphology.reconstruction(seed, mask, method='erosion')
    return eroded - h

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
    else:
        raise ValueError("Unknown implementation `{}`".format(implementation))

    return np.maximum(bg_subtracted, 0)

def subtract_reference(z, bg):
    """Subtracts background using a user-defined background pattern.

    Parameters
    ----------
    bg: array
        User-defined diffraction pattern to be subtracted as background.
    """
    im = z.astype(np.float64)-bg
    for i in range (0,z.shape[0]):
        for j in range (0,z.shape[1]):
            if im[i,j]<0:
                im[i,j]=0
    return im

def circular_mask(shape, radius, center=None):
    """Produces a mask of radius 'r' centered on 'center' of shape 'shape'.

    Parameters
    ----------
    shape : tuple
    radius : int
    center : tuple (optional)
        Default: (0, 0)

    Returns
    -------
    np.ndarray

    """
    l_x, l_y = shape
    x, y = center if center else (l_x / 2, l_y / 2)
    X, Y = np.ogrid[:l_x, :l_y]
    mask = (X - x) ** 2 + (Y - y) ** 2 < radius ** 2
    return mask

def find_beam_position_blur(z, sigma=30):
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
    blurred = ndi.gaussian_filter(z, sigma, mode='wrap')
    center = np.unravel_index(blurred.argmax(), blurred.shape)
    return np.array(center)

def enhance_gauss_sauvola(z, sigma_blur, sigma_enhance, k, window_size, threshold, morph_opening=True):
    z = z.astype(np.float64)
    im1 = ndi.gaussian_filter(z, sigma=sigma_blur, mode='mirror')
    im2 = ndi.gaussian_filter(im1, sigma=sigma_enhance, mode='constant', cval=255)
    im3= im1 - im2
    im3[im3<0] = 0

    mask = im3 > threshold
    im4 = im3*mask

    thresh_sauvola = threshold_sauvola(im4, window_size, k)
    binary_sauvola = im4 > thresh_sauvola
    if morph_opening:
        final = opening(binary_sauvola)
    else:
        final=binary_sauvola

    return final

def peaks_as_gvectors(z, center, calibration):
    g = (z - center) * calibration
    return g[0]
