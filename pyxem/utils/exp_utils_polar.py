import numpy as np


def _correlation(z, axis=0, mask=None, wrap=True, normalize=True):
    """A generic function for applying a correlation with a mask.

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
    m = mask
    if wrap is False:
        z_shape = np.shape(z)
        padder = [(0,0)]*len(z_shape)
        pad = z_shape[axis]//2  #  This will be faster if the length of the axis
        # is a power of 2.  Based on the numpy implementation.  Not terribly
        # faster I think..
        padder[axis]= z_shape[axis] = (pad, pad)
        slicer = (slice(None),)*len(z_shape)
        slicer[axis] = slice(pad,-pad)  # creating the proper slices

    if m is not None:
        # this is to determine how many of the variables were non zero... This is really dumb.  but...
        # it works and I should stop trying to fix it (wreak it)
        mask_boolean = ~ m  # inverting the boolean mask
        if wrap is False:  # padding with zeros to the function along some axis
            m = np.pad(m, padder,'constant')  # all the zeros are masked (should account for padding
            #  when normalized.
        mask_fft = np.fft.fft(mask_boolean, axis=axis)
        number_unmasked = np.fft.ifft(mask_fft*np.conjugate(mask_fft), axis=axis.real)
        number_unmasked[number_unmasked < 1] = 1  # get rid of divide by zero error for completely masked rows
        z[m] = 0

    # fast method uses a FFT and is a process which is O(n) = n log(n)
    z = np.pad(z, padder, 'constant')
    I_fft = np.fft.fft(z, axis=axis)
    a = np.fft.ifft(I_fft * np.conjugate(I_fft), axis=axis).real

    if m is not None:
        a = np.multiply(np.divide(a, np.transpose(number_unmasked)), np.shape(z)[axis])

    if normalize:
        row_mean = np.mean(a, axis=axis)
        row_mean[row_mean == 0] = 1
        np.expand_dims(row_mean, axis=axis)
        a = np.divide(np.subtract(a, row_mean), row_mean)
    if wrap is False:
        a = a[slicer]
    return a

def _power(z, axis=0, mask=None, wrap=True, normalize=True):
    """The power spectrum of the correlation.

    This method is a little more complex if mask is not None due to
    the extra calculations necessary to ignore some of the pixels
    during the calculations.

    Parameters
    ----------------
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
    -----------------
    power: np.array
        The power spectrum along some axis
    """
    if mask is None:  # This might not normalize things as well
        I_fft = np.fft.fft(z, axis=axis)
        return I_fft * np.conjugate(I_fft)
    return np.power(np.fft.fft(_correlation(z=z, axis=axis, mask=mask, wrap=wrap, normalize=normalize)),2)


def angular_correlation(z, mask=None, normalize=True):
    """ Performs some radial correlation on some image z. Assumes that
    the angular direction is axis=1 for z.

    Parameters
    -----------
    z: np.array
        A nd numpy array
    mask: np.array
        A boolean array of the same size as z
    normalize: bool
        Subtract <I(\theta)>^2 and divide by <I(\theta)>^2
    """
    return _correlation(z, axis=1, mask=mask, normalize=normalize, wrap=True)


def angular_power(z, mask=None, normalize=True):
    """ Returns the power of the angular correlation on some image z. Assumes that
    the angular direction is axis=1 for z.

    Parameters
    -----------
    z: np.array
        A nd numpy array
    mask: np.array
        A boolean array of the same size as z
    normalize: bool
        Subtract <I(\theta)>^2 and divide by <I(\theta)>^2
    """
    return _power(z,axis=1, mask=mask, normalize=normalize, wrap=True)


def variance(z, mask=None, axis=0):
    """Calculates the variance along some axis while applying some mask.

    Parameters
    ----------------
    z: np.array
        The array to be operated on
    mask: None or np.array
        A boolean mask masking some values
    axis: The axis to calculate the variance along.

    Returns
    --------------
    v: np.array
        The variance along some axis
    """
    if mask is not None:
        z[mask] = 0
        num_valid = np.shape(z)[axis] - np.sum(mask, axis=axis)
        num_valid[num_valid == 0] = 1
        bottom_mean = np.power(np.sum(z, axis=axis)/num_valid,2)
        top_mean = np.sum(np.power(z,2), axis=axis)/num_valid
    else:
        bottom_mean = np.power(np.mean(z, axis=axis),2)
        top_mean = np.mean(np.power(z,2))
    v = np.subtract(np.divide(top_mean,bottom_mean)-1)
    return v


def mean_mask(z, mask, axis):
    """Calculates the mean using a mask along some axis for the array z

    This method needs to be explicit to overcome the problems with defining a mask
    for a dask array.
    """
    z[mask] = 0
    num_valid = np.shape(z)[axis] - np.sum(mask, axis=axis)
    num_valid[num_valid == 0] = 1
    return np.sum(z, axis=axis) / num_valid

