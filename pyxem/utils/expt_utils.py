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
import scipy.ndimage as ndi
import pyxem as pxm  # for ElectronDiffraction2D

from scipy.interpolate import interp1d
from skimage import transform as tf
from skimage import morphology, filters
from skimage.draw import ellipse_perimeter
from skimage.registration import phase_cross_correlation
from tqdm import tqdm

from pyxem.utils.pyfai_utils import get_azimuthal_integrator


"""
This module contains utility functions for processing electron diffraction
patterns.
"""


def _index_coords(z, origin=None):
    """Creates x & y coords for the indices in a numpy array.

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
    r = np.sqrt(x ** 2 + y ** 2)
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


def azimuthal_integrate1d(
    z, azimuthal_integrator, npt_rad, mask=None, sum=False, **kwargs
):
    """Calculate the azimuthal integral of z around a determined origin.

    This method is used for signals where the origin is constant, compared to
    azimuthal_integrate which is used when the origin in the data changes and
    is iterated over.

    Parameters
    ----------
    z : np.array()
        Two-dimensional data array containing the signal.
    azimuthal_integrator : pyFAI.azimuthal_integrator.AzimuthalIntegrator object
        An AzimuthalIntegrator that is already initialised and used to calculate
        the integral.
    npt_rad:
        The number of radial points to integrate
    mask: Boolean Array
        A boolean array with pixels to ignore
    sum: bool
        Returns the integrated intensity rather than the mean.
    **kwargs :
        Keyword arguments to be passed to ai.integrate2d

    Returns
    -------
    tth : np.array()
        One-dimensional scattering vector axis of z.
    I : np.array()
        One-dimensional azimuthal integral of z.
    """
    output = azimuthal_integrator.integrate1d(z, npt=npt_rad, mask=mask, **kwargs)
    if sum:
        return np.transpose(output._sum_signal)
    else:
        return output[1]


def azimuthal_integrate2d(
    z, azimuthal_integrator, npt_rad, npt_azim=None, mask=None, sum=False, **kwargs
):
    """Calculate the azimuthal integral of z around a determined origin.

    This method is used for signals where the origin is constant, compared to
    azimuthal_integrate which is used when the origin in the data changes and
    is iterated over.

    Parameters
    ----------
    z : np.array()
        Two-dimensional data array containing the signal.
    azimuthal_integrator : pyFAI.azimuthal_integrator.AzimuthalIntegrator object
        An AzimuthalIntegrator that is already initialised and used to calculate
        the integral.
    npt_rad: int
        The number of radial points to integrate
    npt_azim: int
        The number of azimuthal points to integrate
    mask: Boolean Array
        The mask used to ignore points.
    sum: bool
        If True the sum is returned, otherwise the average is returned.
    **kwargs :
        Keyword arguments to be passed to ai.integrate2d

    Returns
    -------
    I : np.array()
        Two-dimensional azimuthal integral of z.
    """
    output = azimuthal_integrator.integrate2d(
        z, npt_rad=npt_rad, npt_azim=npt_azim, mask=mask, **kwargs
    )
    if sum:
        return np.transpose(output._sum_signal)
    else:
        return np.transpose(output[0])


def integrate_radially(
    z, azimuthal_integrator, npt, npt_rad, mask=None, sum=False, **kwargs
):
    """Calculate the radial integrated profile curve as I = f(chi)

    Parameters
    ----------
    z : np.array()
        Two-dimensional data array containing the signal.
    azimuthal_integrator : pyFAI.azimuthal_integrator.AzimuthalIntegrator object
        An AzimuthalIntegrator that is already initialised and used to calculate
        the integral.
    npt: int
         The number of points in the output pattern
    npt_rad: int
        The number of points in the radial space. Too few points may lead to huge rounding errors.
    mask: Boolean Array
        A boolean array with pixels to ignore
    sum: bool
        Returns the integrated intensity rather than the mean.
    **kwargs :
        Keyword arguments to be passed to ai.integrate2d

    Returns
    -------
    tth : np.array()
        One-dimensional scattering vector axis of z.
    I : np.array()
        One-dimensional azimuthal integral of z.
    """
    output = azimuthal_integrator.integrate_radial(
        z, npt=npt, npt_rad=npt_rad, mask=mask, **kwargs
    )
    if sum:
        return np.transpose(output._sum_signal)
    else:
        return output[1]


def medfilt_1d(z, azimuthal_integrator, npt_rad, npt_azim, mask=None, **kwargs):
    """Perform the 2D integration and filter along each row using a median filter

    Parameters
    ----------
    z : np.array()
        Two-dimensional data array containing the signal.
    azimuthal_integrator : pyFAI.azimuthal_integrator.AzimuthalIntegrator object
        An AzimuthalIntegrator that is already initialised and used to calculate
        the integral.
    npt: int
         The number of points in the output pattern
    npt_rad: int
        The number of points in the radial space. Too few points may lead to huge rounding errors.
    mask: Boolean Array
        A boolean array with pixels to ignore
    sum: bool
        Returns the integrated intensity rather than the mean.
    **kwargs :
        Keyword arguments to be passed to ai.integrate2d

    Returns
    -------
    tth : np.array()
        One-dimensional scattering vector axis of z.
    I : np.array()
        One-dimensional azimuthal integral of z.
    """
    output = azimuthal_integrator.medfilt1d(
        z, npt_rad=npt_rad, npt_azim=npt_azim, mask=mask, **kwargs
    )
    return output[1]


def sigma_clip(z, azimuthal_integrator, npt_rad, npt_azim, mask=None, **kwargs):
    """Perform the 2D integration and perform a sigm-clipping iterative
     filter along each row. see the doc of scipy.stats.sigmaclip for the options.


    Parameters
    ----------
    z : np.array()
        Two-dimensional data array containing the signal.
    azimuthal_integrator : pyFAI.azimuthal_integrator.AzimuthalIntegrator object
        An AzimuthalIntegrator that is already initialised and used to calculate
        the integral.
    npt_rad: int
         The number of points in the output pattern
    npt_azim: int
        The number of points in the radial space. Too few points may lead to huge rounding errors.
    mask: Boolean Array
        A boolean array with pixels to ignore
    sum: bool
        Returns the integrated intensity rather than the mean.
    **kwargs :
        Keyword arguments to be passed to ai.integrate2d

    Returns
    -------
    tth : np.array()
        One-dimensional scattering vector axis of z.
    I : np.array()
        One-dimensional azimuthal integral of z.
    """
    output = azimuthal_integrator.sigma_clip(
        z, npt_rad=npt_rad, npt_azim=npt_azim, mask=mask, **kwargs
    )
    return output[1]


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


def remove_dead(z, deadpixels):
    """Remove dead pixels from experimental electron diffraction patterns.

    Parameters
    ----------
    z : np.array()
        Two-dimensional data array containing signal.
    deadpixels : np.array()
        Array containing the array indices of dead pixels in the diffraction
        pattern.

    Returns
    -------
    img : array
        Two-dimensional data array containing z with dead pixels removed.
    """
    z_bar = np.copy(z)
    for (i, j) in deadpixels:
        z_bar[i, j] = (z[i - 1, j] + z[i + 1, j] + z[i, j - 1] + z[i, j + 1]) / 4

    return z_bar


def convert_affine_to_transform(D, shape):
    """Converts an affine transform on a diffraction pattern to a suitable
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
    if keep_dtype is False:
        trans = tf.warp(z, transformation, order=order, *args, **kwargs)
    if keep_dtype is True:
        trans = tf.warp(
            z, transformation, order=order, preserve_range=True, *args, **kwargs
        )
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
    dilated = morphology.reconstruction(seed, mask, method="dilation")

    return z - dilated


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


def _find_peak_max(arr, sigma, upsample_factor, kind):
    """Find the index of the pixel corresponding to peak maximum in 1D pattern

    Parameters
    ----------
    sigma : int
        Sigma value for Gaussian blurring kernel for initial beam center estimation.
    upsample_factor : int
        Upsample factor for subpixel maximum finding, i.e. the maximum will
        be found with a precision of 1 / upsample_factor of a pixel.
    kind : str or int, optional
        Specifies the kind of interpolation as a string (‘linear’, ‘nearest’,
        ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, ‘next’, where
        ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline
        interpolation of zeroth, first, second or third order; ‘previous’
        and ‘next’ simply return the previous or next value of the point) or as
        an integer specifying the order of the spline interpolator to use.

    Returns
    -------
    center: float
        Pixel position of the maximum
    """
    y1 = ndi.filters.gaussian_filter1d(arr, sigma)
    c1 = np.argmax(y1)  # initial guess for beam center

    m = upsample_factor
    window = 10
    win_len = 2 * window + 1

    try:
        r1 = np.linspace(c1 - window, c1 + window, win_len)
        f = interp1d(r1, y1[c1 - window : c1 + window + 1], kind=kind)
        r2 = np.linspace(
            c1 - window, c1 + window, win_len * m
        )  # extrapolate for subpixel accuracy
        y2 = f(r2)
        c2 = np.argmax(y2) / m  # find beam center with `m` precision
    except ValueError:  # if c1 is too close to the edges, return initial guess
        center = c1
    else:
        center = c2 + c1 - window

    return center


def find_beam_center_interpolate(z, sigma, upsample_factor, kind):
    """Find the center of the primary beam in the image `img` by summing along
    X/Y directions and finding the position along the two directions independently.

    Parameters
    ----------
    sigma : int
        Sigma value for Gaussian blurring kernel for initial beam center estimation.
    upsample_factor : int
        Upsample factor for subpixel beam center finding, i.e. the center will
        be found with a precision of 1 / upsample_factor of a pixel.
    kind : str or int, optional
        Specifies the kind of interpolation as a string (‘linear’, ‘nearest’,
        ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, ‘next’, where
        ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline
        interpolation of zeroth, first, second or third order; ‘previous’
        and ‘next’ simply return the previous or next value of the point) or as
        an integer specifying the order of the spline interpolator to use.

    Returns
    -------
    center : np.array
        np.array, [y, x] containing indices of estimated direct beam positon
    """
    xx = np.sum(z, axis=1)
    yy = np.sum(z, axis=0)

    cx = _find_peak_max(xx, sigma, upsample_factor=upsample_factor, kind=kind)
    cy = _find_peak_max(yy, sigma, upsample_factor=upsample_factor, kind=kind)

    center = np.array([cy, cx])
    return center


def find_beam_center_blur(z, sigma):
    """Estimate direct beam position by blurring the image with a large
    Gaussian kernel and finding the maximum.

    Parameters
    ----------
    sigma : float
        Sigma value for Gaussian blurring kernel.

    Returns
    -------
    center : np.array
        np.array [y, x] containing indices of estimated direct beam positon.
    """
    blurred = ndi.gaussian_filter(z, sigma, mode="wrap")
    center = np.unravel_index(blurred.argmax(), blurred.shape)[::-1]
    return np.array(center)


def find_beam_offset_cross_correlation(z, radius_start, radius_finish):
    """Find the offset of the direct beam from the image center by a cross-correlation algorithm.
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
        np.array [y, x] containing offset (from center) of the direct beam positon.
    """
    radiusList = np.arange(radius_start, radius_finish)
    errRecord = np.zeros_like(radiusList, dtype="single")
    origin = np.array(
        [[round(np.size(z, axis=-2) / 2), round(np.size(z, axis=-1) / 2)]]
    )

    for ind in np.arange(0, np.size(radiusList)):
        radius = radiusList[ind]
        ref = reference_circle(origin, np.size(z, axis=-2), np.size(z, axis=-1), radius)
        h0 = np.hanning(np.size(ref, 0))
        h1 = np.hanning(np.size(ref, 1))
        hann2d = np.sqrt(np.outer(h0, h1))
        ref = hann2d * ref
        im = hann2d * z
        shift, error, diffphase = phase_cross_correlation(ref, im, upsample_factor=10)
        errRecord[ind] = error
        index_min = np.argmin(errRecord)

    ref = reference_circle(
        origin, np.size(z, axis=-2), np.size(z, axis=-1), radiusList[index_min]
    )
    h0 = np.hanning(np.size(ref, 0))
    h1 = np.hanning(np.size(ref, 1))
    hann2d = np.sqrt(np.outer(h0, h1))
    ref = hann2d * ref
    im = hann2d * z
    shift, error, diffphase = phase_cross_correlation(ref, im, upsample_factor=100)

    shift = shift[::-1]
    return shift - 0.5


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


def investigate_dog_background_removal_interactive(
    sample_dp, std_dev_maxs, std_dev_mins
):
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
    gauss_processed = np.empty(
        (len(std_dev_maxs), len(std_dev_mins), *sample_dp.axes_manager.signal_shape)
    )

    for i, std_dev_max in enumerate(tqdm(std_dev_maxs, leave=False)):
        for j, std_dev_min in enumerate(std_dev_mins):
            gauss_processed[i, j] = sample_dp.subtract_diffraction_background(
                "difference of gaussians",
                lazy_result=False,
                min_sigma=std_dev_min,
                max_sigma=std_dev_max,
                show_progressbar=False,
            )
    dp_gaussian = pxm.signals.ElectronDiffraction2D(gauss_processed)
    dp_gaussian.metadata.General.title = "Gaussian preprocessed"
    dp_gaussian.axes_manager.navigation_axes[0].name = r"$\sigma_{\mathrm{min}}$"
    dp_gaussian.axes_manager.navigation_axes[1].name = r"$\sigma_{\mathrm{max}}$"
    for axes_number, axes_value_list in [(0, std_dev_mins), (1, std_dev_maxs)]:
        dp_gaussian.axes_manager.navigation_axes[axes_number].offset = axes_value_list[
            0
        ]
        dp_gaussian.axes_manager.navigation_axes[axes_number].scale = (
            axes_value_list[1] - axes_value_list[0]
        )
        dp_gaussian.axes_manager.navigation_axes[axes_number].units = ""

    dp_gaussian.plot(cmap="viridis")
    return None
