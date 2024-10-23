# -*- coding: utf-8 -*-
# Copyright 2016-2024 The pyXem developers
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

"""
This module contains utility functions for processing electron diffraction
patterns.
"""

import numpy as np
import scipy.ndimage as ndi
import pyxem as pxm  # for ElectronDiffraction2D

from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from skimage import transform as tf
from skimage.feature import match_template
from skimage import morphology, filters
from skimage.draw import ellipse_perimeter
from skimage.registration import phase_cross_correlation
from tqdm import tqdm
from packaging.version import Version

from pyxem.utils.cuda_utils import is_cupy_array
from pyxem.utils._deprecated import deprecated
import pyxem.utils._pixelated_stem_tools as pst

try:
    import cupy as cp
    import cupyx.scipy.ndimage as ndigpu

    CUPY_INSTALLED = True
except ImportError:
    CUPY_INSTALLED = False
    cp = None
    ndigpu = None

new_float_type = {
    # preserved types
    np.float32().dtype.char: np.float32,
    np.float64().dtype.char: np.float64,
    np.complex64().dtype.char: np.complex64,
    np.complex128().dtype.char: np.complex128,
    # altered types
    np.float16().dtype.char: np.float32,
    "g": np.float64,  # np.float128 ; doesn't exist on windows
    "G": np.complex128,  # np.complex256 ; doesn't exist on windows
}


def _index_coords(z, origin=None):
    """Creates x & y coords for the indices in a numpy array.

    Parameters
    ----------
    z : numpy.ndarray
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


def gain_normalise(z, dref, bref):
    """Apply gain normalization to experimentally acquired electron
    diffraction pattern.

    Parameters
    ----------
    z : numpy.ndarray
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
    for i, j in deadpixels:
        z_bar[i, j] = (z[i - 1, j] + z[i + 1, j] + z[i, j - 1] + z[i, j + 1]) / 4

    return z_bar


def convert_affine_to_transform(D, shape):
    """Converts an affine transform on a diffraction pattern to a suitable
    form for :func:`skimage.transform.warp`

    Parameters
    ----------
    D : numpy.ndarray
        Affine transform to be applied
    shape : tuple
        Shape tuple in form (y,x) for the diffraction pattern

    Returns
    -------
    transformation : numpy.ndarray
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
    z : numpy.ndarray
        Array to be transformed
    transformation : numpy.ndarray
        3x3 numpy array specifying the transformation to be applied.
    order : int
        Interpolation order.
    keep_dtype : bool
        If True dtype of returned object is that of z
    *args :
        To be passed to :func:`skimage.transform.warp`
    **kwargs :
        To be passed to :func:`skimage.transform.warp`

    Returns
    -------
    trans : array
        Affine transformed diffraction pattern.

    Notes
    -----
    Generally used in combination with :func:`pyxem.expt_utils.convert_affine_to_transform`

    See Also
    --------
    pyxem.expt_utils.convert_affine_to_transform
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
        h-dome subtracted image as numpy.ndarray
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
    mask : numpy.ndarray
        The circular mask.

    """
    l_x, l_y = shape
    x, y = center if center else (l_x / 2, l_y / 2)
    X, Y = np.ogrid[:l_x, :l_y]
    mask = (X - x) ** 2 + (Y - y) ** 2 < radius**2
    return mask


def reference_circle(coords, dimX, dimY, radius):
    """Draw the perimeter of an circle at a given position in the diffraction
    pattern (e.g. to provide a reference for finding the direct beam center).

    Parameters
    ----------
    coords : numpy.ndarray size n,2
        size n,2 array of coordinates to draw the circle.
    dimX : int
        first dimension of the diffraction pattern (size)
    dimY : int
        second dimension of the diffraction pattern (size)
    radius : int
        radius of the circle to be drawn

    Returns
    -------
    img: numpy.ndarray
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
    y1 = ndi.gaussian_filter1d(arr, sigma)
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
    center : numpy.ndarray
        numpy.ndarray, [y, x] containing indices of estimated direct beam positon
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
    center : numpy.ndarray
        numpy.ndarray [x, y] containing indices of estimated direct beam positon.
    """
    if is_cupy_array(z):
        gaus = ndigpu.gaussian_filter
        dispatcher = cp
    else:
        gaus = ndi.gaussian_filter
        dispatcher = np
    blurred = gaus(z, sigma, mode="wrap")
    center = dispatcher.unravel_index(blurred.argmax(), blurred.shape)[::-1]
    return dispatcher.array(center)


def find_center_of_mass(
    signal,
    threshold=None,
    mask=None,
    **kwargs,
):
    if "inplace" in kwargs and kwargs["inplace"]:
        raise ValueError("Inplace is not allowed for center_of_mass")
    else:
        kwargs["inplace"] = False

    det_shape = signal.axes_manager.signal_shape
    if mask is not None:
        x, y, r = mask
        mask = pst._make_circular_mask(x, y, det_shape[0], det_shape[1], r)

    ans = signal.map(
        center_of_mass_from_image,
        threshold=threshold,
        mask=mask,
        **kwargs,
    )
    ans.set_signal_type("beam_shift")
    ans.axes_manager.signal_axes[0].name = "Beam position"
    return ans


def center_of_mass_from_image(z, mask=None, threshold=None):
    """Estimate direct beam position by calculating the center of mass of the
    image.

    Parameters
    ----------
    z : numpy.ndarray
        Two-dimensional data array containing signal.
    mask : numpy.ndarray
        Two-dimensional data array containing mask.
    threshold : float
        Threshold value for center of mass calculation.

    Returns
    -------
    center : numpy.ndarray
        numpy.ndarray [x, y] containing indices of estimated direct beam positon.
    """
    if mask is not None:
        z = z * mask
    if threshold is not None:
        z[z < (np.mean(z) * threshold)] = 0
    center = np.array(ndi.center_of_mass(z))[::-1]
    return center


def find_beam_offset_cross_correlation(z, radius_start, radius_finish, **kwargs):
    """Find the offset of the direct beam from the image center by a cross-correlation algorithm.
    The shift is calculated relative to an circle perimeter. The circle can be
    refined across a range of radii during the centring procedure to improve
    performance in regions where the direct beam size changes,
    e.g. during sample thickness variation.

    Parameters
    ----------
    z: numpy.ndarray
        The two dimensional array/image that is operated on
    radius_start : int
        The lower bound for the radius of the central disc to be used in the
        alignment.
    radius_finish : int
        The upper bounds for the radius of the central disc to be used in the
        alignment.
    **kwargs:
        Any additional keyword arguments defined by :func:`skimage.registration.phase_cross_correlation`

    Returns
    -------
    shift: numpy.ndarray
        numpy.ndarray [y, x] containing offset (from center) of the direct beam positon.
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
        shift, error, diffphase = phase_cross_correlation(
            ref, im, upsample_factor=10, **kwargs
        )
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
    shift, error, diffphase = phase_cross_correlation(
        ref, im, upsample_factor=100, **kwargs
    )

    shift = shift[::-1]
    return shift - 0.5


def peaks_as_gvectors(z, center, calibration):
    """Converts peaks found as array indices to calibrated units, for use in a
    hyperspy map function.

    Parameters
    ----------
    z : numpy.ndarray
        peak positions as array indices.
    center : numpy.ndarray
        diffraction pattern center in array indices.
    calibration : float
        calibration in reciprocal Angstroms per pixels.

    Returns
    -------
    g : numpy.ndarray
        peak positions in calibrated units.

    """
    g = (z - center) * calibration
    return g


@deprecated(since="0.18.0", removal="1.0.0")
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
    numpy.arange : Produces suitable objects for std_dev_maxs

    """
    gauss_processed = np.empty(
        (len(std_dev_maxs), len(std_dev_mins), *sample_dp.axes_manager.signal_shape)
    )

    for i, std_dev_max in enumerate(tqdm(std_dev_maxs, leave=False)):
        for j, std_dev_min in enumerate(std_dev_mins):
            gauss_processed[i, j] = sample_dp.subtract_diffraction_background(
                "difference of gaussians",
                lazy_output=False,
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


def find_hot_pixels(z, threshold_multiplier=500, mask=None):
    """Find single pixels which have much larger values compared to neighbors.

    Finds pixels which have a gradient larger than the threshold multiplier.
    These are extremely sharp peaks with values on average much larger than
    the surrounding values.

    Parameters
    ----------
    z : numpy.ndarray
        Frame to operate on
    threshold_multiplier : scaler
        Used to threshold the dif.
    mask : numpy.ndarray, optional
        Array with bool values. The True values will be masked
        (i.e. ignored). Must have the same shape as the two
        last dimensions in dask_array.

    """
    # find the gradient of the image.
    footprint = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    median = ndi.median_filter(z, footprint=footprint)
    hot_pixels = (z - median) > threshold_multiplier
    if mask is not None:
        hot_pixels[mask] = False
    return hot_pixels


def remove_bad_pixels(z, bad_pixels):
    """Replace values in bad pixels with mean of neighbors.

    Parameters
    ----------
    z : numpy.ndarray
        A single frame
    bad_pixels : numpy.ndarray
        Must either have the same shape as dask_array,
        or the same shape as the two last dimensions of dask_array.

    Returns
    -------
    data_output : Dask array

    Examples
    --------
    >>> import pyxem.utils.dask_tools as dt
    >>> s = pxm.data.dummy_data.dummy_data.get_dead_pixel_signal(lazy=True)
    >>> dead_pixels = dt._find_dead_pixels(s.data)
    >>> data_output = dt._remove_bad_pixels(s.data, dead_pixels)

    """
    z[bad_pixels] = 0
    bad_pixels_ind = np.transpose(np.array(np.where(bad_pixels)))
    bad_slices = [tuple([slice(i - 1, i + 2) for i in ind]) for ind in bad_pixels_ind]
    values = [np.sum(z[s]) / 8 for s in bad_slices]
    z[bad_pixels] = values
    return z


def normalize_template_match(z, template, subtract_min=True, pad_input=True, **kwargs):
    """Matches a template with an image z. Preformed a normalized cross-correlation
    using the given templates. If subtract_min is True then the minimum value will
    be subtracted from the correlation.

    Parameters
    ----------
    z : numpy.ndarray
        Two-dimensional data array containing signal.
    template : numpy.ndarray
        Two-dimensional data array containing template.
    subtract_min : bool
        If True the minimum value will be subtracted from the correlation.
    pad_input : bool
        If True the input array will be padded. (This should be True otherwise
        the result will be shifted by half the template size)
    **kwargs :
        Keyword arguments to be passed to :func:`skimage.feature.match_template`
    """
    if kwargs.pop("circular_background", False):
        template_match = match_template_dilate(z, template, **kwargs)
    else:
        template_match = match_template(z, template, pad_input=pad_input, **kwargs)
    if subtract_min:
        template_match = template_match - np.min(template_match)
    return template_match


def _supported_float_type(input_dtype, allow_complex=False):
    """Return an appropriate floating-point dtype for a given dtype.

    float32, float64, complex64, complex128 are preserved.
    float16 is promoted to float32.
    complex256 is demoted to complex128.
    Other types are cast to float64.

    Parameters
    ----------
    input_dtype : np.dtype or tuple of np.dtype
        The input dtype. If a tuple of multiple dtypes is provided, each
        dtype is first converted to a supported floating point type and the
        final dtype is then determined by applying `np.result_type` on the
        sequence of supported floating point types.
    allow_complex : bool, optional
        If False, raise a ValueError on complex-valued inputs.

    Returns
    -------
    float_type : dtype
        Floating-point dtype for the image.
    """
    if isinstance(input_dtype, tuple):
        return np.result_type(*(_supported_float_type(d) for d in input_dtype))
    input_dtype = np.dtype(input_dtype)
    if not allow_complex and input_dtype.kind == "c":
        raise ValueError("complex valued input is not supported")
    return new_float_type.get(input_dtype.char, np.float64)


def match_template_dilate(
    image, template, template_dilation=2, mode="constant", constant_values=0
):
    """Matches a template with an image using a window normalized cross-correlation.

    This performs very well for image with different background intensities.  This is a slower version
    of the skimage :func:`skimage.feature.match_template` but performs better for images with circular
    variations in background intensity, specifically accounting for an amorphous halo around the
    diffraction pattern.

    Parameters
    ----------
    image : np.array
        Image to be matched
    template : np.array
        Template to preform the normalized cross-correlation with
    template_dilation : int
        The number of pixels to dilate the template by for the windowed cross-correlation
    mode : str
        Padding mode for the image. Options are 'constant', 'edge', 'wrap', 'reflect'
    constant_values : int
        Value to pad the image with if mode is 'constant'

    Returns
    -------
    response : np.array
        The windowed cross-correlation of the image and template
    """

    image = image.astype(float, copy=False)
    if image.ndim < template.ndim:  # pragma: no cover
        raise ValueError(
            "Dimensionality of template must be less than or "
            "equal to the dimensionality of image."
        )
    if np.any(np.less(image.shape, template.shape)):  # pragma: no cover
        raise ValueError("Image must be larger than template.")

    image_shape = image.shape
    float_dtype = _supported_float_type(image.dtype)

    template = np.pad(template, template_dilation)
    pad_width = tuple((width, width) for width in template.shape)
    if mode == "constant":
        image = np.pad(
            image, pad_width=pad_width, mode=mode, constant_values=constant_values
        )
    else:
        image = np.pad(image, pad_width=pad_width, mode=mode)

    dilated_template = morphology.dilation(
        template, footprint=morphology.disk(template_dilation)
    )
    # Use special case for 2-D images for much better performance in
    # computation of integral images
    image_window_sum = fftconvolve(image, dilated_template[::-1, ::-1], mode="valid")[
        1:-1, 1:-1
    ]
    image_window_sum2 = fftconvolve(
        image**2, dilated_template[::-1, ::-1], mode="valid"
    )[1:-1, 1:-1]

    template_mean = template[
        dilated_template.astype(bool)
    ].mean()  # only consider the pixels in the dilated template
    template_volume = np.sum(dilated_template)
    template_ssd = np.sum((template - template_mean) ** 2)

    xcorr = fftconvolve(image, template[::-1, ::-1], mode="valid")[1:-1, 1:-1]
    numerator = xcorr - image_window_sum * template_mean

    denominator = image_window_sum2
    np.multiply(image_window_sum, image_window_sum, out=image_window_sum)
    np.divide(image_window_sum, template_volume, out=image_window_sum)
    denominator -= image_window_sum
    denominator *= template_ssd
    np.maximum(denominator, 0, out=denominator)  # sqrt of negative number not allowed
    np.sqrt(denominator, out=denominator)

    response = np.zeros_like(xcorr, dtype=float_dtype)

    # avoid zero-division
    mask = denominator > np.finfo(float_dtype).eps

    response[mask] = numerator[mask] / denominator[mask]

    slices = []
    for i in range(template.ndim):
        d0 = (template.shape[i] - 1) // 2
        d1 = d0 + image_shape[i]

        slices.append(slice(d0, d1))

    return response[tuple(slices)]
