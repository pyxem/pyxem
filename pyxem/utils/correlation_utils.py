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


def corr_to_power(z):
    return np.power(np.fft.rfft(z, axis=1), 2).real
