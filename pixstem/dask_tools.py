import copy
import numpy as np
import dask.array as da


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
    >>> import pixstem.dask_tools as dt
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
                    mask_array.shape, dask_array.shape[-2:]))
    mask_array_4d = da.ones_like(dask_array, dtype=np.bool)
    mask_array_4d = mask_array_4d[:, :] * mask_array
    dask_array_masked = da.ma.masked_array(
            dask_array, mask_array_4d, fill_value=fill_value)
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
    >>> import pixstem.dask_tools as dt
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
                "The input has {0} dimensions".format(len(dask_array.shape)))
    thresholded_array = da.ma.getdata(thresholded_array)
    return thresholded_array


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
    >>> import pixstem.dask_tools as dt
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
    y_grad, x_grad = np.mgrid[0:det_shape[0], 0:det_shape[1]]
    y_grad, x_grad = y_grad.astype(np.float64), x_grad.astype(np.float64)
    sum_array = np.ones_like(x_grad)

    if mask_array is not None:
        if not mask_array.shape == det_shape:
            raise ValueError(
                    "mask_array ({0}) must have same shape as last two "
                    "dimensions of the dask_array ({1})".format(
                        mask_array.shape, det_shape))
        x_grad = x_grad * np.invert(mask_array)
        y_grad = y_grad * np.invert(mask_array)
        sum_array = sum_array * np.invert(mask_array)
    if threshold_value is not None:
        dask_array = _threshold_array(
                dask_array, threshold_value=threshold_value,
                mask_array=mask_array)

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
    >>> import pixstem.dask_tools as dt
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
    return(slice_mi, slice_xp, slice_xm, slice_yp, slice_ym)


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
    >>> import pixstem.api as ps
    >>> import pixstem.dask_tools as dt
    >>> s = ps.dummy_data.get_dead_pixel_signal(lazy=True)
    >>> dead_pixels = dt._find_dead_pixels(s.data)
    >>> data_output = dt._remove_bad_pixels(s.data, dead_pixels)

    """
    if len(dask_array.shape) < 2:
        raise ValueError("dask_array {0} must be at least 2 dimensions".format(
            dask_array.shape))
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
                    bad_pixel_array.shape,
                    dask_array.shape[-2:], dask_array.shape))
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
    >>> import pixstem.api as ps
    >>> import pixstem.dask_tools as dt
    >>> s = ps.dummy_data.get_dead_pixel_signal(lazy=True)
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

    dask_array = dask_array.astype('float64')
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
