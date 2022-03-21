import numpy as np


def _correlation(z, axis=0, mask=None, wrap=True, normalize=True):
    r"""A generic function for applying a correlation with a mask.

    Takes a nd image and then preforms a auto-correlation on some axis.
    Used in the electron correlation and angular correlation codes. Uses
    the fft to speed up the correlation.

    Parameters
    ----------
    z: np.array
        A nd numpy array
    axis: int
        The axis to apply the correlation to
    mask: np.array
        A boolean array of the same size as z
    wrap: bool
        Allow the function to wrap or add zeros to the beginning and the
        end of z on the specified axis
    normalize: bool
        Subtract <I(\theta)>^2 and divide by <I(\theta)>^2
    """
    if wrap is False:
        z_shape = np.shape(z)
        padder = [(0, 0)] * len(z_shape)
        pad = z_shape[axis]  # This will be faster if the length of the axis
        # is a power of 2.  Based on the numpy implementation.  Not terribly
        # faster I think..
        padder[axis] = (pad, pad)
        slicer = [
            slice(None),
        ] * len(z_shape)
        slicer[axis] = slice(0, -2 * pad)  # creating the proper slices
        if mask is None:
            mask = np.zeros(shape=np.shape(z))
        z = np.pad(z, padder, "constant")

    if mask is not None or wrap is False:  # Need to scale for wrapping properly
        m = np.array(mask, dtype=bool)  # make sure it is a boolean array
        # this is to determine how many of the variables were non zero... This is really dumb.  but...
        # it works and I should stop trying to fix it (wreak it)
        mask_boolean = ~m  # inverting the boolean mask
        if wrap is False:  # padding with zeros to the function along some axis
            m = np.pad(
                m, padder, "constant"
            )  # all the zeros are masked (should account for padding
            #  when normalized.
            mask_boolean = np.pad(mask_boolean, padder, "constant")
        mask_fft = np.fft.rfft(mask_boolean, axis=axis)
        number_unmasked = np.fft.irfft(
            mask_fft * np.conjugate(mask_fft), axis=axis
        ).real
        number_unmasked[
            number_unmasked < 1
        ] = 1  # get rid of divide by zero error for completely masked rows
        z[m] = 0

    # fast method uses a FFT and is a process which is O(n) = n log(n)
    I_fft = np.fft.rfft(z, axis=axis)
    a = np.fft.irfft(I_fft * np.conjugate(I_fft), axis=axis)

    if mask is not None:
        a = np.multiply(np.divide(a, number_unmasked), np.shape(z)[0])

    if normalize:  # simplified way to calculate the normalization
        row_mean = np.mean(a, axis=axis)
        row_mean[row_mean == 0] = 1
        row_mean = np.expand_dims(row_mean, axis=axis)
        a = np.divide(np.subtract(a, row_mean), row_mean)

    if wrap is False:
        print(slicer)
        a = a[slicer]
    return a


def _power(z, axis=0, mask=None, wrap=True, normalize=True):
    """The power spectrum of the correlation.

    This method is a little more complex if mask is not None due to
    the extra calculations necessary to ignore some of the pixels
    during the calculations.

    Parameters
    ----------
    z: np.array
        Some n-d array to get the power spectrum from.
    axis: int
        The axis to preform the operation on.
    mask: np.array
        A boolean mask to be applied.
    wrap: bool
        Choose if the function should wrap.  In most cases this will be True
        when calculating the power of some function
    normalize: bool
        Choose to normalize the function by the mean.

    Returns
    -------
    power: np.array
        The power spectrum along some axis
    """
    if mask is None:  # This might not normalize things as well
        I_fft = np.fft.rfft(z, axis=axis)
        return (I_fft * np.conjugate(I_fft)).real
    else:
        return np.power(
            np.fft.rfft(
                _correlation(z=z, axis=axis, mask=mask, wrap=wrap, normalize=normalize)
            ),
            2,
        ).real


def _pearson_correlation(z, mask=None):
    """
    Calculate Pearson cross-correlation of the image with itself
    after rotation as a function of rotation

     Parameters
    ----------
    z: np.array
        Input image in 2D array
    mask: np.array
        A boolean mask to be applied.

    Returns
    -------
    p_correlation: np.array
        Pearson correlation of the input image

    """
    if mask is not None:
        # this is to determine how many of the elements were unmasked for normalization
        m = np.array(mask, dtype=bool)
        mask_bool = ~m
        mask_fft = np.fft.fft(mask_bool, axis=1)
        n_unmasked = np.fft.ifft(mask_fft * mask_fft.conj()).real
        n_unmasked[n_unmasked < 1] = 1  # avoid dividing by zero for completely masked rows
        z[m] = 0  # set masked pixels to zero
        fft_intensity = np.divide(np.fft.fft(z, axis=1), n_unmasked)
        a = np.multiply(np.fft.ifft(fft_intensity * fft_intensity.conj()).real, n_unmasked)
    else:
        z_length = np.shape(z)[1]
        fft_intensity = np.fft.fft(z, axis=1) / z_length
        a = np.fft.ifft(fft_intensity * fft_intensity.conj(), axis=1).real * z_length

    p_correlation = (np.mean(a, axis=0) - np.mean(z) ** 2) / (np.mean(z ** 2) - np.mean(z) ** 2)
    return p_correlation


def _wrap_set_float(target, bottom, top, value):
    """This function sets values in a list assuming that
    the list is circular and allows for float bottom and float top
    which are equal to the residual times that value.

    Parameters
    ----------
    target: list
        The list or array to be adjusted
    bottom: float
        The bottom index. Can be a float in which case the value will be
        split.  i.e. 7.5 will set target[7]= value * .5
    top: float
        The top index. Can be a float in which case the value will be
        split.  i.e. 7.5 will set target[8]= value * .5
    value:
        The value to set the range as.
    """
    ceiling_bottom = int(np.ceil(bottom))
    residual_bottom = ceiling_bottom-bottom
    floor_top = int(np.floor(top))
    residual_top = top-floor_top
    if floor_top > len(target) - 1:
        target[ceiling_bottom:] = value
        new_floor_top = floor_top % len(target)
        target[new_floor_top] = value
        target[new_floor_top+1] = value*residual_top
    elif ceiling_bottom < 0:
        target[:floor_top] = value
        target[ceiling_bottom:] = value
        target[ceiling_bottom-1] = value*residual_bottom
    else:
        target[ceiling_bottom:floor_top+1] = value
        target[ceiling_bottom-1] = value*residual_bottom
        if floor_top + 1 > len(target) - 1:
            target[0] = value*residual_top
        else:
            target[floor_top+1] = value*residual_top
    return target


def _get_interpolation_matrix(angles, angular_range, num_points, method="average"):
    """Returns an interpolation matrix for slicing a dataset based on the given angles as
    well as an angular range.

    The method is separated into two based on if the angles are all treated equally or if
    only the first or max angle is considered for some list of angles.

    Parameters
    ----------
    angles: list
        A list of a list of angles where each row represents a grouping of angles to be considered.
        For example if you are trying to get the expression of 2-fold and 4-fold symmetry this
        would look like [[0,np.pi],[0,np.pi/2,np.pi,np.pi*3/2]]
    angular_range: float
        The angular range in rad to consider.  If zero only the nearest pixel will be considered
    num_points: int
        The number of points in the azimuthal range to consider
    method: str
        One of "average" "first" or "max".  Changes how the interpolation matrix is created
        for further processing

    """
    if method is "average":
        angular_ranges = [(angle - angular_range / (2*np.pi),
                           angle + angular_range / (2*np.pi)) for angle in angles]
        angular_ranges = np.multiply(angular_ranges, num_points)
        interpolation_matrix = np.zeros(num_points)
        for i, angle in enumerate(angular_ranges):
            _wrap_set_float(interpolation_matrix, top=angle[1], bottom=angle[0], value=1)
        return interpolation_matrix
    else:
        angular_ranges = [(angle - angular_range / (2*np.pi), angle + angular_range / (2*np.pi))
                          for angle in angles]
        angular_ranges = np.multiply(angular_ranges, num_points)
        interpolation_matrix = np.zeros((len(angles), num_points))
        for i, angle in enumerate(angular_ranges):
            _wrap_set_float(interpolation_matrix[i, :], top=angle[1], bottom=angle[0], value=1)
        return interpolation_matrix


def _symmetry_stem(signal, interpolation, method="average"):
    """Returns the "average" "max" or "first" value for some given signal and an interpolation matrix.

    The interpolation matrix is defined by the  `_get_interpolation_matrix` function which creates a
    matrix which when matrix multiplied by the signal returns the "average", "max" or "first" value for
    certain angles related to some symmetry operation.

    ie. For 4-Fold symmetry operations the angles are [0, pi/2, pi,3pi/2]

    Parameters
    ----------
    signal: np.array
        The signal from which to calculate the symmetry stem given an interpolation matrix
    interpolation: np.array
        The interpolation matrix for calculating the expression of certain symmetry operations.
    method:str
        One of "average", "max" or "first"
    """
    if method is "average":
        return np.matmul(signal, np.transpose(interpolation))
    elif method is "max":
        val = np.transpose([np.amax([np.matmul(signal, np.transpose(i))
                                     for i in interp], axis=0)
                            for interp in interpolation])
    elif method is "first":
        val = np.transpose([np.matmul(signal, np.transpose(interp[0]))
                            for interp in interpolation])
    else:
        raise ValueError("Method must be one of `average`, `max` or `first`")
    return val


def corr_to_power(z):
    return np.power(np.fft.rfft(z, axis=1), 2).real
