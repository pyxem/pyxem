import numpy as np

def _correlation(z, axis=1, mask=None, wrap=True, normalize=True, ):
    """A generic function for applying a correlation with a mask.

     Takes a nd image and then preforms a auto-correlation on some axis.
     Used in the electron correlation and angular correlation codes. Uses
     the fft to speed up the correlation.
    Parameters
    ----------
    z: np.array
        A nd numpy array
    mask: np.array
        A boolean array of the same size as z
    wrap: bool
        Allow the function to wrap or add zeros to the beginning and the
        end of z on the specified axis
    normalize: bool
        Subtract <I(\theta)>^2 and divide by <I(\theta)>^2
    """
    m = mask
    if wrap:
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
    if wrap:
        a = a[slicer]
    return a


def angular_correlation(z, mask=None, normalize=True, radial_bin=1, angular_bin=1):
    """ Performs some radial correlation on some image z. Assumes that
    the angular direction is axis=1 for z.

    Parameters
    -----------
    z: np.array
        The image
    :param z:
    :param mask:
    :param normalize:
    :param radial_bin:
    :param angular_bin:
    :return:
    """
    pass
