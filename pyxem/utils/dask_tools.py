# -*- coding: utf-8 -*-
# Copyright 2016-2020 The pyXem developers
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

import copy
import numpy as np
import dask.array as da
from skimage.feature import match_template, blob_dog, blob_log
import scipy.ndimage as ndi
from skimage import morphology


def _rechunk_signal2d_dim_one_chunk(dask_array):
    array_dims = len(dask_array.shape)
    if array_dims < 2:
        raise ValueError(
            "dask_array must be at least two dimensions, not {0}".format(array_dims)
        )
    detx, dety = dask_array.shape[-2:]
    chunks = [None] * array_dims
    chunks[-2] = detx
    chunks[-1] = dety
    chunks = tuple(chunks)
    dask_array_rechunked = dask_array.rechunk(chunks=chunks)
    return dask_array_rechunked


def _mask_array(dask_array, mask_array, fill_value=None):
    """Mask two last dimensions in a dask array.

    Parameters
    ----------
    dask_array : Dask array
    mask_array : NumPy array
        Array with bool values. The True values will be masked
        (i.e. ignored). Must have the same shape as the two
        last dimensions in dask_array.
    fill_value : scalar, optional

    Returns
    -------
    dask_array_masked : masked Dask array

    Examples
    --------
    >>> import dask.array as da
    >>> import pyxem.utils.dask_tools as dt
    >>> data = da.random.random(
    ...     size=(32, 32, 128, 128), chunks=(16, 16, 128, 128))
    >>> mask_array = np.ones(shape=(128, 128), dtype=bool)
    >>> mask_array[64-10:64+10, 64-10:64+10] = False
    >>> output_dask = dt._mask_array(data, mask_array=mask_array)
    >>> output = output_dask.compute()

    With fill value specified

    >>> output_dask = dt._mask_array(
    ...     data, mask_array=mask_array, fill_value=0.0)
    >>> output = output_dask.compute()

    """
    if not dask_array.shape[-2:] == mask_array.shape:
        raise ValueError(
            "mask_array ({0}) and last two dimensions in the "
            "dask_array ({1}) need to have the same shape.".format(
                mask_array.shape, dask_array.shape[-2:]
            )
        )
    mask_array_4d = da.ones_like(dask_array, dtype=np.bool)
    mask_array_4d = mask_array_4d[:, :] * mask_array
    dask_array_masked = da.ma.masked_array(
        dask_array, mask_array_4d, fill_value=fill_value
    )
    return dask_array_masked


def _threshold_array(dask_array, threshold_value=1, mask_array=None):
    """
    Parameters
    ----------
    dask_array : Dask array
        Must have either 2, 3 or 4 dimensions.
    threshold_value : scalar, optional
        Default value is 1.
    mask_array : NumPy array, optional
        Array with bool values. The True values will be masked
        (i.e. ignored). Must have the same shape as the two
        last dimensions in dask_array.

    Returns
    -------
    thresholded_array : Dask array

    Examples
    --------
    >>> import dask.array as da
    >>> import pyxem.utils.dask_tools as dt
    >>> data = da.random.random(
    ...     size=(32, 32, 128, 128), chunks=(16, 16, 128, 128))
    >>> output_dask = dt._threshold_array(data)
    >>> output = output_dask.compute()

    Non-default threshold value

    >>> output_dask = dt._threshold_array(data, threshold_value=1.5)
    >>> output = output_dask.compute()

    Masking everything except the center of the image

    >>> mask_array = np.ones(shape=(128, 128), dtype=bool)
    >>> mask_array[64-10:64+10, 64-10:64+10] = False
    >>> output_dask = dt._threshold_array(data, mask_array=mask_array)
    >>> output = output_dask.compute()

    """
    input_array = dask_array.copy()
    if mask_array is not None:
        input_array = _mask_array(input_array, mask_array)
        dask_array = dask_array * np.invert(mask_array)
    mean_array = da.mean(input_array, axis=(-2, -1))
    threshold_array = mean_array * threshold_value

    # Not very elegant solution, but works for the most common data dimensions
    if len(dask_array.shape) == 4:
        swaped_array = dask_array.swapaxes(0, 2).swapaxes(1, 3)
        thresholded_array = swaped_array > threshold_array
        thresholded_array = thresholded_array.swapaxes(1, 3).swapaxes(0, 2)
    elif len(dask_array.shape) == 3:
        swaped_array = dask_array.swapaxes(0, 1).swapaxes(1, 2)
        thresholded_array = swaped_array > threshold_array
        thresholded_array = thresholded_array.swapaxes(1, 2).swapaxes(0, 1)
    elif len(dask_array.shape) == 2:
        thresholded_array = dask_array > threshold_array
    else:
        raise ValueError(
            "dask_array need to have either 2, 3, or 4 dimensions. "
            "The input has {0} dimensions".format(len(dask_array.shape))
        )
    thresholded_array = da.ma.getdata(thresholded_array)
    return thresholded_array


def _template_match_binary_image_single_frame(frame, binary_image):
    """Template match a binary image (template) with a single image.

    Parameters
    ----------
    frame : NumPy 2D array
    binary_image : NumPy 2D array
        Must be smaller than frame

    Returns
    -------
    template_match : NumPy 2D array
        Same size as frame

    Examples
    --------
    >>> frame = np.random.randint(1000, size=(256, 256))
    >>> from skimage import morphology
    >>> binary_image = morphology.disk(4, np.uint16)
    >>> import pyxem.utils.dask_tools as dt
    >>> template_match = dt._template_match_binary_image_single_frame(
    ...     frame, binary_image)

    """
    template_match = match_template(frame, binary_image, pad_input=True)
    template_match = template_match - np.min(template_match)
    return template_match


def _template_match_binary_image_chunk(data, binary_image):
    """Template match a circular disk with a 4D dataset.

    Parameters
    ----------
    data : NumPy 4D array
    binary_image : NumPy 2D array
        Must be smaller than the two last dimensions in data

    Returns
    -------
    template_match : NumPy 4D array
        Same size as input data

    Examples
    --------
    >>> data = np.random.randint(1000, size=(10, 10, 256, 256))
    >>> from skimage import morphology
    >>> binary_image = morphology.disk(4, np.uint16)
    >>> import pyxem.utils.dask_tools as dt
    >>> template_match = dt._template_match_binary_image_chunk(
    ...     data, binary_image)

    """
    output_array = np.zeros_like(data, dtype=np.float32)
    frame = np.zeros(data.shape[-2:])
    for index in np.ndindex(data.shape[:-2]):
        islice = np.s_[index]
        frame[:] = data[islice]
        output_array[islice] = _template_match_binary_image_single_frame(
            frame, binary_image
        )
    return output_array


def _template_match_with_binary_image(dask_array, binary_image):
    """Template match a dask array with a binary image (template).

    Parameters
    ----------
    dask_array : Dask array
        The two last dimensions are the signal dimensions. Must have at least
        2 dimensions.
    binary_image : 2D NumPy array
        Must be smaller than the two last dimensions in dask_array

    Returns
    -------
    template_match : Dask array
        Same size as input data

    Examples
    --------
    >>> data = np.random.randint(1000, size=(20, 20, 256, 256))
    >>> import dask.array as da
    >>> dask_array = da.from_array(data, chunks=(5, 5, 128, 128))
    >>> from skimage import morphology
    >>> binary_image = morphology.disk(4, np.uint16)
    >>> import pyxem.utils.dask_tools as dt
    >>> template_match = dt._template_match_with_binary_image(
    ...     dask_array, binary_image)

    """
    if len(binary_image.shape) != 2:
        raise ValueError(
            "binary_image must have two dimensions, not {0}".format(
                len(binary_image.shape)
            )
        )
    dask_array_rechunked = _rechunk_signal2d_dim_one_chunk(dask_array)
    output_array = da.map_blocks(
        _template_match_binary_image_chunk,
        dask_array_rechunked,
        binary_image,
        dtype=np.float32,
    )
    return output_array


def _peak_find_dog_single_frame(
    image,
    min_sigma=0.98,
    max_sigma=55,
    sigma_ratio=1.76,
    threshold=0.36,
    overlap=0.81,
    normalize_value=None,
):
    """Find peaks in a single frame using skimage's blob_dog function.

    Parameters
    ----------
    image : NumPy 2D array
    min_sigma : float, optional
    max_sigma : float, optional
    sigma_ratio : float, optional
    threshold : float, optional
    overlap : float, optional
    normalize_value : float, optional
        All the values in image will be divided by this value.
        If no value is specified, the max value in the image will be used.

    Returns
    -------
    peaks : NumPy 2D array
        In the form [[x0, y0], [x1, y1], [x2, y2], ...]

    Example
    -------
    >>> s = pxm.dummy_data.dummy_data.get_cbed_signal()
    >>> import pyxem.utils.dask_tools as dt
    >>> peaks = _peak_find_dog_single_frame(s.data[0, 0])

    """

    if normalize_value is None:
        normalize_value = np.max(image)
    peaks = blob_dog(
        image / normalize_value,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        sigma_ratio=sigma_ratio,
        threshold=threshold,
        overlap=overlap,
        exclude_border=False,
    )
    peak = peaks[:, :2].astype(np.float64)

    return peak


def _peak_find_dog_chunk(data, **kwargs):
    """Find peaks in a chunk using skimage's blob_dog function.

    Parameters
    ----------
    data : NumPy array
    min_sigma : float, optional
    max_sigma : float, optional
    sigma_ratio : float, optional
    threshold : float, optional
    overlap : float, optional
    normalize_value : float, optional
        All the values in data will be divided by this value.
        If no value is specified, the max value in each individual image will
        be used.

    Returns
    -------
    peak_array : NumPy 2D object array
        Same size as the two last dimensions in data.
        The peak positions themselves are stored in 2D NumPy arrays
        inside each position in peak_array. This is done instead of
        making a 4D NumPy array, since the number of found peaks can
        vary in each position.

    Example
    -------
    >>> s = pxm.dummy_data.dummy_data.get_cbed_signal()
    >>> import pyxem.utils.dask_tools as dt
    >>> peak_array = dt._peak_find_dog_chunk(s.data)
    >>> peaks00 = peak_array[0, 0]
    >>> peaks23 = peak_array[2, 3]

    """
    output_array = np.empty(data.shape[:-2], dtype="object")
    for index in np.ndindex(data.shape[:-2]):
        islice = np.s_[index]
        output_array[islice] = _peak_find_dog_single_frame(image=data[islice], **kwargs)
    return output_array


def _peak_find_dog(dask_array, **kwargs):
    """Find peaks in a dask array using skimage's blob_dog function.

    Parameters
    ----------
    dask_array : Dask array
        Must be at least 2 dimensions.
    min_sigma : float, optional
    max_sigma : float, optional
    sigma_ratio : float, optional
    threshold : float, optional
    overlap : float, optional
    normalize_value : float, optional
        All the values in dask_array will be divided by this value.
        If no value is specified, the max value in each individual image will
        be used.

    Returns
    -------
    peak_array : dask object array
        Same size as the two last dimensions in data.
        The peak positions themselves are stored in 2D NumPy arrays
        inside each position in peak_array. This is done instead of
        making a 4D NumPy array, since the number of found peaks can
        vary in each position.

    Example
    -------
    >>> s = pxm.dummy_data.dummy_data.get_cbed_signal()
    >>> import dask.array as da
    >>> dask_array = da.from_array(s.data, chunks=(5, 5, 25, 25))
    >>> import pyxem.utils.dask_tools as dt
    >>> peak_array = _peak_find_dog(dask_array)
    >>> peak_array_computed = peak_array.compute()

    """
    dask_array_rechunked = _rechunk_signal2d_dim_one_chunk(dask_array)
    drop_axis = (dask_array_rechunked.ndim - 2, dask_array_rechunked.ndim - 1)
    output_array = da.map_blocks(
        _peak_find_dog_chunk,
        dask_array_rechunked,
        drop_axis=drop_axis,
        dtype=np.object,
        **kwargs
    )
    return output_array


def _peak_find_log_single_frame(
    image,
    min_sigma=0.98,
    max_sigma=55,
    num_sigma=10,
    threshold=0.36,
    overlap=0.81,
    normalize_value=None,
):
    """Find peaks in a single frame using skimage's blob_log function.

    Parameters
    ----------
    image : NumPy 2D array
    min_sigma : float, optional
    max_sigma : float, optional
    num_sigma : float, optional
    threshold : float, optional
    overlap : float, optional
    normalize_value : float, optional
        All the values in image will be divided by this value.
        If no value is specified, the max value in the image will be used.

    Returns
    -------
    peaks : NumPy 2D array
        In the form [[x0, y0], [x1, y1], [x2, y2], ...]

    Example
    -------
    >>> s = pxm.dummy_data.dummy_data.get_cbed_signal()
    >>> import pyxem.utils.dask_tools as dt
    >>> peaks = dt._peak_find_log_single_frame(s.data[0, 0])

    """

    if normalize_value is None:
        normalize_value = np.max(image)
    peaks = blob_log(
        image / normalize_value,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        overlap=overlap,
        exclude_border=False,
    )
    peak = peaks[:, :2].astype(np.float64)
    return peak


def _peak_find_log_chunk(data, **kwargs):
    """Find peaks in a chunk using skimage's blob_log function.

    Parameters
    ----------
    data : NumPy array
    min_sigma : float, optional
    max_sigma : float, optional
    num_sigma : float, optional
    threshold : float, optional
    overlap : float, optional
    normalize_value : float, optional
        All the values in data will be divided by this value.
        If no value is specified, the max value in each individual image will
        be used.

    Returns
    -------
    peak_array : NumPy 2D object array
        Same size as the two last dimensions in data.
        The peak positions themselves are stored in 2D NumPy arrays
        inside each position in peak_array. This is done instead of
        making a 4D NumPy array, since the number of found peaks can
        vary in each position.

    Example
    -------
    >>> s = pxm.dummy_data.dummy_data.get_cbed_signal()
    >>> import pyxem.utils.dask_tools as dt
    >>> peak_array = dt._peak_find_log_chunk(s.data)
    >>> peaks00 = peak_array[0, 0]
    >>> peaks23 = peak_array[2, 3]

    """
    output_array = np.empty(data.shape[:-2], dtype="object")
    for index in np.ndindex(data.shape[:-2]):
        islice = np.s_[index]
        output_array[islice] = _peak_find_log_single_frame(image=data[islice], **kwargs)
    return output_array


def _peak_find_log(dask_array, **kwargs):
    """Find peaks in a dask array using skimage's blob_log function.

    Parameters
    ----------
    dask_array : Dask array
        Must be at least 2 dimensions.
    min_sigma : float, optional
    max_sigma : float, optional
    num_sigma : float, optional
    threshold : float, optional
    overlap : float, optional
    normalize_value : float, optional
        All the values in dask_array will be divided by this value.
        If no value is specified, the max value in each individual image will
        be used.

    Returns
    -------
    peak_array : dask object array
        Same size as the two last dimensions in data.
        The peak positions themselves are stored in 2D NumPy arrays
        inside each position in peak_array. This is done instead of
        making a 4D NumPy array, since the number of found peaks can
        vary in each position.

    Example
    -------
    >>> s = pxm.dummy_data.dummy_data.get_cbed_signal()
    >>> import dask.array as da
    >>> dask_array = da.from_array(s.data, chunks=(5, 5, 25, 25))
    >>> import pyxem.utils.dask_tools as dt
    >>> peak_array = dt._peak_find_log(dask_array)
    >>> peak_array_computed = peak_array.compute()

    """
    dask_array_rechunked = _rechunk_signal2d_dim_one_chunk(dask_array)
    drop_axis = (dask_array_rechunked.ndim - 2, dask_array_rechunked.ndim - 1)
    output_array = da.map_blocks(
        _peak_find_log_chunk,
        dask_array_rechunked,
        drop_axis=drop_axis,
        dtype=np.object,
        **kwargs
    )
    return output_array


def _center_of_mass_array(dask_array, threshold_value=None, mask_array=None):
    """Find center of mass of last two dimensions for a dask array.

    The center of mass can be calculated using a mask and threshold.

    Parameters
    ----------
    dask_array : Dask array
        Must have either 2, 3 or 4 dimensions.
    threshold_value : scalar, optional
    mask_array : NumPy array, optional
        Array with bool values. The True values will be masked
        (i.e. ignored). Must have the same shape as the two
        last dimensions in dask_array.

    Returns
    -------
    center_of_mask_dask_array : Dask array

    Examples
    --------
    >>> import dask.array as da
    >>> import pyxem.utils.dask_tools as dt
    >>> data = da.random.random(
    ...     size=(64, 64, 128, 128), chunks=(16, 16, 128, 128))
    >>> output_dask = dt._center_of_mass_array(data)
    >>> output = output_dask.compute()

    Masking everything except the center of the image

    >>> mask_array = np.ones(shape=(128, 128), dtype=bool)
    >>> mask_array[64-10:64+10, 64-10:64+10] = False
    >>> output_dask = dt._center_of_mass_array(data, mask_array=mask_array)
    >>> output = output_dask.compute()

    Masking and thresholding

    >>> output_dask = dt._center_of_mass_array(
    ...     data, mask_array=mask_array, threshold_value=3)
    >>> output = output_dask.compute()

    """
    det_shape = dask_array.shape[-2:]
    y_grad, x_grad = np.mgrid[0 : det_shape[0], 0 : det_shape[1]]
    y_grad, x_grad = y_grad.astype(np.float64), x_grad.astype(np.float64)
    sum_array = np.ones_like(x_grad)

    if mask_array is not None:
        if not mask_array.shape == det_shape:
            raise ValueError(
                "mask_array ({0}) must have same shape as last two "
                "dimensions of the dask_array ({1})".format(mask_array.shape, det_shape)
            )
        x_grad = x_grad * np.invert(mask_array)
        y_grad = y_grad * np.invert(mask_array)
        sum_array = sum_array * np.invert(mask_array)
    if threshold_value is not None:
        dask_array = _threshold_array(
            dask_array, threshold_value=threshold_value, mask_array=mask_array
        )

    x_shift = da.multiply(dask_array, x_grad, dtype=np.float64)
    y_shift = da.multiply(dask_array, y_grad, dtype=np.float64)
    sum_array = da.multiply(dask_array, sum_array, dtype=np.float64)

    x_shift = np.sum(x_shift, axis=(-2, -1), dtype=np.float64)
    y_shift = np.sum(y_shift, axis=(-2, -1), dtype=np.float64)
    sum_array = np.sum(sum_array, axis=(-2, -1), dtype=np.float64)

    beam_shifts = da.stack((x_shift, y_shift))
    beam_shifts = da.divide(beam_shifts[:], sum_array, dtype=np.float64)
    return beam_shifts


def _get_border_slices(nav_dim_size):
    """Get a list of slices for doing pixel interpolation.

    Parameters
    ----------
    nav_dim_size : scalar

    Returns
    -------
    slice_mi, slice_xp, slice_xm, slice_yp, slice_ym : list of slices

    Examples
    --------
    >>> import pyxem.utils.dask_tools as dt
    >>> s_mi, s_xp, s_xm, s_yp, s_ym = dt._get_border_slices(2)

    """
    nav_slice = [slice(None, None, None)] * nav_dim_size

    slice_mi = copy.deepcopy(nav_slice)
    slice_mi.extend(np.s_[1:-1, 1:-1])
    slice_mi = tuple(slice_mi)

    slice_xp = copy.deepcopy(nav_slice)
    slice_xp.extend(np.s_[0:-2, 1:-1])
    slice_xp = tuple(slice_xp)

    slice_xm = copy.deepcopy(nav_slice)
    slice_xm.extend(np.s_[2:None, 1:-1])
    slice_xm = tuple(slice_xm)

    slice_yp = copy.deepcopy(nav_slice)
    slice_yp.extend(np.s_[1:-1, 0:-2])
    slice_yp = tuple(slice_yp)

    slice_ym = copy.deepcopy(nav_slice)
    slice_ym.extend(np.s_[1:-1, 2:None])
    slice_ym = tuple(slice_ym)

    return (slice_mi, slice_xp, slice_xm, slice_yp, slice_ym)


def _remove_bad_pixels(dask_array, bad_pixel_array):
    """Replace values in bad pixels with mean of neighbors.

    Parameters
    ----------
    dask_array : Dask array
        Must be at least two dimensions
    bad_pixel_array : array-like
        Must either have the same shape as dask_array,
        or the same shape as the two last dimensions of dask_array.

    Returns
    -------
    data_output : Dask array

    Examples
    --------
    >>> import pyxem.utils.dask_tools as dt
    >>> s = pxm.dummy_data.dummy_data.get_dead_pixel_signal(lazy=True)
    >>> dead_pixels = dt._find_dead_pixels(s.data)
    >>> data_output = dt._remove_bad_pixels(s.data, dead_pixels)

    """
    if len(dask_array.shape) < 2:
        raise ValueError(
            "dask_array {0} must be at least 2 dimensions".format(dask_array.shape)
        )
    if bad_pixel_array.shape == dask_array.shape:
        pass
    elif bad_pixel_array.shape == dask_array.shape[-2:]:
        temp_array = da.zeros_like(dask_array)
        bad_pixel_array = da.add(temp_array, bad_pixel_array)
    else:
        raise ValueError(
            "bad_pixel_array {0} must either 2-D and have the same shape "
            "as the two last dimensions in dask_array {1}. Or be "
            "the same shape as dask_array {2}".format(
                bad_pixel_array.shape, dask_array.shape[-2:], dask_array.shape
            )
        )
    dif0 = da.roll(dask_array, shift=1, axis=-2)
    dif1 = da.roll(dask_array, shift=-1, axis=-2)
    dif2 = da.roll(dask_array, shift=1, axis=-1)
    dif3 = da.roll(dask_array, shift=-1, axis=-1)

    dif = (dif0 + dif1 + dif2 + dif3) / 4
    dif = dif * bad_pixel_array

    data_output = da.multiply(dask_array, da.logical_not(bad_pixel_array))
    data_output = data_output + dif

    return data_output


def _find_dead_pixels(dask_array, dead_pixel_value=0, mask_array=None):
    """Find pixels which have the same value for all images.

    Useful for finding dead pixels.

    Parameters
    ----------
    dask_array : Dask array
        Must be at least 2 dimensions
    dead_pixel_value : scalar
        Default 0
    mask_array : NumPy array, optional
        Array with bool values. The True values will be masked
        (i.e. ignored). Must have the same shape as the two
        last dimensions in dask_array.

    Returns
    -------
    dead_pixels : Dask array

    Examples
    --------
    >>> import pyxem.utils.dask_tools as dt
    >>> s = pxm.dummy_data.dummy_data.get_dead_pixel_signal(lazy=True)
    >>> dead_pixels = dt._find_dead_pixels(s.data)

    With a mask

    >>> mask_array = np.zeros((128, 128), dtype=np.bool)
    >>> mask_array[:, :100] = True
    >>> dead_pixels = dt._find_dead_pixels(s.data, mask_array=mask_array)

    With a dead_pixel_value

    >>> dead_pixels = dt._find_dead_pixels(s.data, dead_pixel_value=2)

    """
    if len(dask_array.shape) < 2:
        raise ValueError("_find_dead_pixels must have at least 2 dimensions")
    nav_dim_size = len(dask_array.shape) - 2
    nav_axes = tuple(range(nav_dim_size))
    data_sum = dask_array.sum(axis=nav_axes, dtype=np.int64)
    dead_pixels = data_sum == dead_pixel_value
    if mask_array is not None:
        dead_pixels = dead_pixels * np.invert(mask_array)

    return dead_pixels


def _find_hot_pixels(dask_array, threshold_multiplier=500, mask_array=None):
    """Find single pixels which have much larger values compared to neighbors.

    Finds pixel which has very large value difference compared to its
    neighbors. The functions looks at both at the direct neighbors
    (x-1, y), and also the diagonal neighbors (x-1, y-1).

    Experimental function, so use with care.

    Parameters
    ----------
    dask_array : Dask array
        Must be have 4 dimensions.
    threshold_multiplier : scaler
        Used to threshold the dif.
    mask_array : NumPy array, optional
        Array with bool values. The True values will be masked
        (i.e. ignored). Must have the same shape as the two
        last dimensions in dask_array.

    """
    if len(dask_array.shape) < 2:
        raise ValueError("dask_array must have at least 2 dimensions")

    dask_array = dask_array.astype("float64")
    dif0 = da.roll(dask_array, shift=1, axis=-2)
    dif1 = da.roll(dask_array, shift=-1, axis=-2)
    dif2 = da.roll(dask_array, shift=1, axis=-1)
    dif3 = da.roll(dask_array, shift=-1, axis=-1)

    dif4 = da.roll(dask_array, shift=(1, 1), axis=(-2, -1))
    dif5 = da.roll(dask_array, shift=(-1, 1), axis=(-2, -1))
    dif6 = da.roll(dask_array, shift=(1, -1), axis=(-2, -1))
    dif7 = da.roll(dask_array, shift=(-1, -1), axis=(-2, -1))

    dif = dif0 + dif1 + dif2 + dif3 + dif4 + dif5 + dif6 + dif7
    dif = dif - (dask_array * 8)

    if mask_array is not None:
        data = _mask_array(dask_array, mask_array=mask_array)
    else:
        data = dask_array
    data_mean = data.mean() * threshold_multiplier
    data_threshold = dif < -data_mean
    if mask_array is not None:
        mask_array = np.invert(mask_array).astype(np.float64)
        data_threshold = data_threshold * mask_array
    return data_threshold


def _intensity_peaks_image_single_frame(frame, peaks, disk_r):
    """Intensity of the peaks is calculated by taking the mean value
    of the pixel values inside radius disk_r where the centers are the
    peak positions. If the peak position plus disk_r exceed the detector
    edges, then the intensity for that peak will be put to zero.

    Parameters
    ----------
    frame : NumPy 2D array
    peaks: Numpy 2D array with x and y coordinates of peaks
    disk : NumPy 2D array
        Must be smaller than frame
        peaks: NumPy Object
        can have multiple peaks per image

    Returns
    -------
    intensity_array : NumPy array with
        peak coordinates and intensity of peaks

    Examples
    --------
    >>> import pyxem.utils.dask_tools as dt
    >>> s = pxm.dummy_data.dummy_data.get_cbed_signal()
    >>> peaks = np.array(([50,50],[25,50]))
    >>> intensity = dt._intensity_peaks_image_single_frame(
    ...     s.data[0,0,:,:], peaks, 5)

    """
    array_shape = peaks.shape
    mask = morphology.disk(disk_r)
    size = np.shape(frame)
    intensity_array = np.zeros((array_shape[0], 3), dtype="float64")
    for i in range(array_shape[0]):
        cx = int(peaks[i, 0])
        cy = int(peaks[i, 1])
        intensity_array[i, 0] = peaks[i, 0]
        intensity_array[i, 1] = peaks[i, 1]
        if (
            (cx - disk_r < 0)
            | (cx + disk_r + 1 >= size[0])
            | (cy - disk_r < 0)
            | (cy + disk_r + 1 >= size[1])
        ):
            intensity_array[i, 2] = 0
        else:
            subframe = frame[
                cx - disk_r : cx + disk_r + 1, cy - disk_r : cy + disk_r + 1
            ]
            intensity_array[i, 2] = np.mean(mask * subframe)

    return intensity_array


def _intensity_peaks_image_chunk(data, peak_array, disk_r):
    """Intensity of the peaks is calculated by taking the mean value
    of the pixel values inside radius disk_r where the centers are the
    peak positions for the entire chunk. If the peak position plus disk_r
    exceed the detector edges, then the intensity for that peak will be
    put to zero.

    Parameters
    ----------
    data : NumPy 4D array
    peak_array: NumPy 2D array
        In the form [[x0, y0], [x1, y1], [x2, y2], ...]
    r__disk : Integer number which represents the radius of the discs

    Returns
    -------
    intensity array : NumPy object with x, y and intensity for every peak

    Examples
    --------
    >>> import pyxem.utils.dask_tools as dt
    >>> s = pxm.dummy_data.dummy_data.get_cbed_signal()
    >>> peak_array = dt._peak_find_dog_chunk(s.data)
    >>> intensity = dt._intensity_peaks_image_chunk(s.data, peak_array, 5)
    """

    output_array = np.empty(data.shape[:-2], dtype=np.object)
    if peak_array.ndim < data.ndim:
        dim_dif = data.ndim - peak_array.ndim
        for i in range(dim_dif):
            peak_array = np.expand_dims(peak_array, axis=peak_array.ndim)

    for index in np.ndindex(data.shape[:-2]):
        islice = np.s_[index]
        peaks = peak_array[islice][0, 0]
        output_array[islice] = _intensity_peaks_image_single_frame(
            data[islice], peaks, disk_r
        )
    return output_array


def _intensity_peaks_image(dask_array, peak_array, disk_r):
    """Intensity of the peaks is calculated by taking the mean value
    of the pixel values inside radius disk_r where the centers are the
    peak positions for the full dask array. If the peak position plus disk_r
    exceed the detector edges, then the intensity for that peak will be
    put to zero.

    Parameters
    ----------
    dask_array : Dask array
        The two last dimensions are the signal dimensions. Must have at least
        2 dimensions.
    peak_array: Dask array
        Must have the same shape as dask_array's navigation dimensions,
        ergo dask_array's dimensions, except the two last ones.
        So if dask_array has the shape [10, 15, 50, 50], the peak_array
        must have the shape [10, 15].
        In the form [[x0, y0], [x1, y1], [x2, y2], ...]
    r_disk : Integer number which represents the radius of the discs
        peak_array = Numpy object with x and y coordinates peaks

    Returns
    -------
    intensity_array : dask object array
        Same size as the two last dimensions in data.
        The x, y peak positions and intensities are stored in the
        three columns.

    Examples
    --------
    >>> import pyxem.utils.dask_tools as dt
    >>> import dask.array as da
    >>> s = pxm.dummy_data.dummy_data.get_cbed_signal()
    >>> dask_array = da.from_array(s.data, chunks=(5, 5, 25, 25))
    >>> peak_array = dt._peak_find_dog(dask_array)
    >>> intensity = dt._intensity_peaks_image(dask_array, peak_array, 5)

    """
    if dask_array.shape[:-2] != peak_array.shape:
        raise ValueError(
            "peak_array ({0}) must have the same shape as dask_array "
            "except the two last dimensions ({1})".format(
                peak_array.shape, dask_array.shape[:-2]
            )
        )
    if not hasattr(dask_array, "chunks"):
        raise ValueError("dask_array must be a Dask array")
    if not hasattr(peak_array, "chunks"):
        raise ValueError("peak_array must be a Dask array")

    dask_array_rechunked = _rechunk_signal2d_dim_one_chunk(dask_array)

    peak_array_rechunked = peak_array.rechunk(dask_array_rechunked.chunks[:-2])
    shape = list(peak_array_rechunked.shape)
    shape.extend([1, 1])
    peak_array_rechunked = peak_array_rechunked.reshape(shape)

    drop_axis = (dask_array_rechunked.ndim - 2, dask_array_rechunked.ndim - 1)
    kwargs_intensity_peaks = {"disk_r": disk_r}

    output_array = da.map_blocks(
        _intensity_peaks_image_chunk,
        dask_array_rechunked,
        peak_array_rechunked,
        drop_axis=drop_axis,
        dtype=np.object,
        **kwargs_intensity_peaks
    )
    return output_array


def _background_removal_single_frame_dog(frame, min_sigma=1, max_sigma=55):
    """Background removal using difference of Gaussians.

    Parameters
    ----------
    frame : NumPy 2D array
    min_sigma : float
    max_sigma : float

    Returns
    -------
    background_removed : Numpy 2D array

    Examples
    --------
    >>> import pyxem.utils.dask_tools as dt
    >>> s = pxm.dummy_data.dummy_data.get_cbed_signal()
    >>> s_rem = dt._background_removal_single_frame_dog(s.data[0, 0])

    """
    blur_max = ndi.gaussian_filter(frame, max_sigma)
    blur_min = ndi.gaussian_filter(frame, min_sigma)
    return np.maximum(np.where(blur_min > blur_max, frame, 0) - blur_max, 0)


def _background_removal_chunk_dog(data, **kwargs):
    """Background removal using difference of Gaussians.

    Parameters
    ----------
    data : NumPy array
    At least 2 dimensions
    min_sigma : float
    max_sigma : float

    Returns
    -------
    output_array : NumPy array
        Same dimensions as data

    Examples
    --------
    >>> import pyxem.utils.dask_tools as dt
    >>> s = pxm.dummy_data.dummy_data.get_cbed_signal()
    >>> s_rem = dt._background_removal_chunk_dog(s.data[0:10, 0:10,:,:])
    """
    output_array = np.zeros_like(data, dtype=np.float32)
    frame = np.zeros(data.shape[-2:])
    for index in np.ndindex(data.shape[:-2]):
        islice = np.s_[index]
        frame[:] = data[islice]
        output_array[islice] = _background_removal_single_frame_dog(frame, **kwargs)
    return output_array


def _background_removal_dog(dask_array, **kwargs):
    """Background removal using difference of Gaussians.

    Parameters
    ----------
    dask_array : Dask array
        At least 2 dimensions
    min_sigma : float
    max_sigma : float

    Returns
    -------
    output_array : Dask array
        Same dimensions as dask_array

    Examples
    --------
    >>> import pyxem.utils.dask_tools as dt
    >>> import dask.array as da
    >>> s = pxm.dummy_data.dummy_data.get_cbed_signal()
    >>> dask_array = da.from_array(s.data, chunks=(5,5,25, 25))
    >>> s_rem = pxm.Diffraction2D(
    ...     dt._background_removal_dog(dask_array))

    """
    dask_array_rechunked = _rechunk_signal2d_dim_one_chunk(dask_array)
    output_array = da.map_blocks(
        _background_removal_chunk_dog, dask_array_rechunked, dtype=np.float32, **kwargs
    )
    return output_array


def _background_removal_single_frame_median(frame, footprint=19):
    """Background removal using median filter.

    Parameters
    ----------
    frame : NumPy 2D array
    footprint : float

    Returns
    -------
    background_removed : Numpy 2D array

    Examples
    --------
    >>> import pyxem.utils.dask_tools as dt
    >>> s = pxm.dummy_data.dummy_data.get_cbed_signal()
    >>> s_rem = dt._background_removal_single_frame_median(s.data[0, 0])

    """
    bg_subtracted = frame - ndi.median_filter(frame, size=footprint)
    return bg_subtracted


def _background_removal_chunk_median(data, **kwargs):
    """Background removal using median filter.

    Parameters
    ----------
    data : NumPy array
        Must be at least 2 dimensions
    footprint : float

    Returns
    -------
    output_array : NumPy array
        Same dimensions as data

    Examples
    --------
    >>> import pyxem.utils.dask_tools as dt
    >>> s = pxm.dummy_data.dummy_data.get_cbed_signal()
    >>> s_rem = dt._background_removal_chunk_median(s.data[0:10, 0:10,:,:])

    """
    output_array = np.zeros_like(data, dtype=np.float32)
    frame = np.zeros(data.shape[-2:])
    for index in np.ndindex(data.shape[:-2]):
        islice = np.s_[index]
        frame[:] = data[islice]
        output_array[islice] = _background_removal_single_frame_median(frame, **kwargs)
    return output_array


def _background_removal_median(dask_array, **kwargs):
    """Background removal using median filter.

    Parameters
    ----------
    dask_array : Dask array
        Must be at least 2 dimensions
    footprint : float

    Returns
    -------
    output_array : Dask array
        Same dimensions as dask_array

    Examples
    --------
    >>> import pyxem.utils.dask_tools as dt
    >>> import dask.array as da
    >>> s = pxm.dummy_data.dummy_data.get_cbed_signal()
    >>> dask_array = da.from_array(s.data+1e3, chunks=(5,5,25, 25))
    >>> s_rem = pxm.Diffraction2D(
    ...     dt._background_removal_median(dask_array), footprint=20)

    """
    dask_array_rechunked = _rechunk_signal2d_dim_one_chunk(dask_array)
    output_array = da.map_blocks(
        _background_removal_chunk_median,
        dask_array_rechunked,
        dtype=np.float32,
        **kwargs
    )
    return output_array


def _background_removal_single_frame_radial_median(frame, centre_x=128, centre_y=128):
    """Background removal by subtracting median of pixel at the same
    radius from the center.

    Parameters
    ----------
    frame : NumPy 2D array
    centre_x : int
    centre_y : int

    Returns
    -------
    background_removed : Numpy 2D array

    Examples
    --------
    >>> import pyxem.utils.dask_tools as dt
    >>> s = pxm.dummy_data.dummy_data.get_cbed_signal()
    >>> s_rem = dt._background_removal_single_frame_radial_median(s.data[0, 0])
    """

    y, x = np.indices((frame.shape))
    r = np.hypot(x - centre_x, y - centre_y)
    r = r.astype(int)
    r_flat = r.ravel()
    diff_image_flat = frame.ravel()
    r_median = np.zeros(np.max(r) + 1, dtype=np.float64)

    for i in range(len(r_median)):
        if diff_image_flat[r_flat == i].size != 0:
            r_median[i] = np.median(diff_image_flat[r_flat == i])
    image = frame - r_median[r]

    return image


def _background_removal_chunk_radial_median(data, **kwargs):
    """Background removal by subtracting median of pixel at the same
    radius from the center.

    Parameters
    ----------
    data : NumPy array
        Must be at least 2 dimensions
    centre_x : int
    centre_y : int

    Returns
    -------
    output_array : NumPy array
        Same dimensions as data

    Examples
    --------
    >>> import pyxem.utils.dask_tools as dt
    >>> s = pxm.dummy_data.dummy_data.get_cbed_signal()
    >>> s_rem = _background_removal_chunk_radial_median(s.data[0:10, 0:10,:,:])
    """
    output_array = np.zeros_like(data, dtype=np.float32)
    frame = np.zeros(data.shape[-2:])
    for index in np.ndindex(data.shape[:-2]):
        islice = np.s_[index]
        frame[:] = data[islice]
        output_array[islice] = _background_removal_single_frame_radial_median(
            frame, **kwargs
        )
    return output_array


def _background_removal_radial_median(dask_array, **kwargs):
    """Background removal by subtracting median of pixel at the same
    radius from the center.

    Parameters
    ----------
    dask_array : Dask array
        Must be at least 2 dimensions
    centre_x : int
    centre_y : int

    Returns
    -------
    output_array : Dask array
        Same dimensions as dask_array

    Examples
    --------
    >>> import pyxem.utils.dask_tools as dt
    >>> s = pxm.dummy_data.dummy_data.get_cbed_signal()
    >>> dask_array = da.from_array(s.data+1e3, chunks=(5,5,25, 25))
    >>> s_r = pxm.Diffraction2D(
    ...     dt._background_removal_radial_median(
    ...     dask_array, centre_x=128, centre_y=128))

    """
    dask_array_rechunked = _rechunk_signal2d_dim_one_chunk(dask_array)
    output_array = da.map_blocks(
        _background_removal_chunk_radial_median,
        dask_array_rechunked,
        dtype=np.float32,
        **kwargs
    )
    return output_array


def _peak_refinement_centre_of_mass_frame(frame, peaks, square_size):
    """Refining the peak positions using the center of mass of the peaks.

    Parameters
    ----------
    frame : Numpy array
    peaks : Numpy array
    square_size : Even Int, this should be larger than the diffraction
        disc diameter. However not to large because then other discs will
        influence the center of mass.

    Returns
    -------
    peak_array : A Numpy array with the x and y positions of the refined peaks

    Examples
    --------
    >>> import pyxem.utils.dask_tools as dt
    >>> s = pxm.dummy_data.dummy_data.get_cbed_signal()
    >>> peak_array = np.array(([48.,48.],[25.,48.]))
    >>> r_p_array = dt._peak_refinement_centre_of_mass_frame(
    ...     s.data[0,0,:,:], peak_array, 20)

    """
    new_peak = np.zeros(peaks.shape, dtype="float64")

    if peaks.size == 0:
        return peaks
    else:
        for i in range(peaks.shape[0]):
            subframe = _center_of_mass_experimental_square(
                frame, np.asarray(peaks[i], dtype=np.uint16), square_size
            )
            if subframe is None:
                new_peak[i, 0] = peaks[i, 0].astype("float64")
                new_peak[i, 1] = peaks[i, 1].astype("float64")
            else:
                subframe = subframe.astype("float64")
                f_x = (subframe.shape[0]) / 2
                f_y = (subframe.shape[1]) / 2
                cx, cy = _center_of_mass_hs(subframe)
                new_peak[i, 0] = peaks[i, 0].astype("float64") + (cx - f_x)
                new_peak[i, 1] = peaks[i, 1].astype("float64") + (cy - f_y)

        return new_peak


def _peak_refinement_centre_of_mass_chunk(data, peak_array, square_size):
    """Refining the peak positions using the center of mass of the peaks.

    Parameters
    ----------
    data : Numpy array
        The two last dimensions are the signal dimensions. Must have at least
        2 dimensions.
    peak_array : Numpy array
    square_size : Even integer, this should be larger than the diffraction
        disc diameter. However not to large because then other discs will
        influence the center of mass.

    Returns
    -------
    peak_array : A NumPy array with the x and y positions of the refined peaks

    Examples
    --------
    >>> import pyxem.utils.dask_tools as dt
    >>> s = pxm.dummy_data.dummy_data.get_cbed_signal()
    >>> peak_array = dt._peak_find_dog_chunk(s.data)
    >>> r_p_array = dt._peak_refinement_centre_of_mass_chunk(
    ...     s.data, peak_array, 20)


    """
    output_array = np.empty(data.shape[:-2], dtype=np.object)
    if peak_array.ndim < data.ndim:
        dim_dif = data.ndim - peak_array.ndim
        for i in range(dim_dif):
            peak_array = np.expand_dims(peak_array, axis=peak_array.ndim)
    frame = np.zeros(data.shape[-2:])
    for index in np.ndindex(data.shape[:-2]):
        islice = np.s_[index]
        frame[:] = data[islice]
        peaks = peak_array[islice][0, 0]
        output_array[islice] = _peak_refinement_centre_of_mass_frame(
            frame, peaks, square_size
        )
    return output_array


def _peak_refinement_centre_of_mass(dask_array, peak_array, square_size):
    """Refining the peak positions using the center of mass of the peaks

    Parameters
    ----------
    dask_array : Dask array
        The two last dimensions are the signal dimensions. Must have at least
        2 dimensions.
    peak_array : Dask array
    square_size : Even integer, this should be larger than the diffraction
        disc diameter. However not to large because then other discs will
        influence the center of mass.

    Returns
    -------
    peak_array : A dask array with the x and y positions of the refined peaks

    Examples
    --------
    >>> import pyxem.utils.dask_tools as dt
    >>> import dask.array as da
    >>> s = ps.dummy_data.get_cbed_signal()
    >>> dask_array = da.from_array(s.data, chunks=(5, 5, 25, 25))
    >>> peak_array = dt._peak_find_dog(dask_array)
    >>> r_p_array = dt._peak_refinement_centre_of_mass(
    ...     dask_array, peak_array, 20)
    >>> ref_peak_array = r_p_array.compute()
    """
    if dask_array.shape[:-2] != peak_array.shape:
        raise ValueError(
            "peak_array ({0}) must have the same shape as dask_array "
            "except the two last dimensions ({1})".format(
                peak_array.shape, dask_array.shape[:-2]
            )
        )
    if not hasattr(dask_array, "chunks"):
        raise ValueError("dask_array must be a Dask array")
    if not hasattr(peak_array, "chunks"):
        raise ValueError("peak_array must be a Dask array")

    dask_array_rechunked = _rechunk_signal2d_dim_one_chunk(dask_array)

    peak_array_rechunked = peak_array.rechunk(dask_array_rechunked.chunks[:-2])
    shape = list(peak_array_rechunked.shape)
    shape.extend([1, 1])
    peak_array_rechunked = peak_array_rechunked.reshape(shape)

    drop_axis = (dask_array_rechunked.ndim - 2, dask_array_rechunked.ndim - 1)
    kwargs_refinement = {"square_size": square_size}

    output_array = da.map_blocks(
        _peak_refinement_centre_of_mass_chunk,
        dask_array_rechunked,
        peak_array_rechunked,
        drop_axis=drop_axis,
        dtype=np.object,
        **kwargs_refinement
    )
    return output_array


def _center_of_mass_hs(z):
    """Return the center of mass of an array with coordinates in the
    hyperspy convention.

    Parameters
    ----------
    z : 2D NumPy array

    Returns
    -------
    cx: The x location of the center of mass
    cy: The y location of the center of mass
    """
    s = np.sum(z)
    if s != 0:
        z *= 1 / s

    dx = np.sum(z, axis=0)
    dy = np.sum(z, axis=1)
    h, w = z.shape
    cx = np.sum(dx * np.arange(w))
    cy = np.sum(dy * np.arange(h))

    return cy, cx


def _center_of_mass_experimental_square(z, vector, square_size):
    """Wrapper for _get_experimental_square that makes the non-zero
    elements symmetrical around the 'unsubpixeled' peak by zeroing a
    'spare' row and column (top and left).

    Parameters
    ----------
    z : np.array
    vector : np.array([x,y])
    square_size : int (even)

    Returns
    -------
    z_adpt : np.array
        z, but with row and column zero set to 0
    """

    z_adpt = np.copy(
        _get_experimental_square(z, vector=vector, square_size=square_size)
    )

    if z_adpt.size == 0:
        return None
    else:
        z_adpt[:, 0] = 0
        z_adpt[0, :] = 0
        return z_adpt


def _get_experimental_square(z, vector, square_size):
    """Defines a square region around a given diffraction vector and returns an
    upsampled copy.

    Parameters
    ----------
    z : np.array()
        Single diffraction pattern
    vector : np.array()
        Single vector in pixels (int) [x,y] with top left as [0,0]
    square_size : int
        The length of one side of the bounding square (must be even)

    Returns
    -------
    _z : np.array()
        Of size (L,L) where L = square_size

    Examples
    --------
    >>> import pyxem.dummy_data.dummy_data as dd
    >>> d = dd.get_cbed_signal()
    >>> import pyxem.utils.dask_tools as dt
    >>> sub_d = dt._get_experimental_square(d.data[0,0,:,:], [50,50], 30)

    """
    if square_size % 2 != 0:
        raise ValueError("'square_size' must be an even number")

    cx, cy, half_ss = (vector[1]), (vector[0]), int(square_size / 2)
    _z = z[cy - half_ss : cy + half_ss, cx - half_ss : cx + half_ss]
    return _z
