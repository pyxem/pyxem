import numpy as np
from skimage._shared.fft import fftmodule, next_fast_len
from functools import partial


def _flip(arr, axes=None):
    """ Reverse array over many axes. Generalization of arr[::-1] for many
    dimensions. If `axes` is `None`, flip along all axes. """
    if axes is None:
        reverse = [slice(None, None, -1)] * arr.ndim
    else:
        reverse = [slice(None, None, None)] * arr.ndim
        for axis in axes:
            reverse[axis] = slice(None, None, -1)

    return arr[tuple(reverse)]


def _centered(arr, newshape, axes):
    """ Return the center `newshape` portion of `arr`, leaving axes not
    in `axes` untouched. """
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)

    slices = [slice(None, None)] * arr.ndim

    for ax in axes:
        startind = (currshape[ax] - newshape[ax]) // 2
        endind = startind + newshape[ax]
        slices[ax] = slice(startind, endind)

    return arr[tuple(slices)]

def autocorrelate(z,
                  mask=None,
                  mode="full",
                  axs=(-1),
                  pad_axis=None,
                  overlap_ratio=0.3):
    """
        Masked normalized cross-correlation between two arrays.
        Parameters
        ----------
        z1 : ndarray
            The stationary array for the correlation
        z2 : ndarray
            The moving array for the correaltion
        mode : {'full', 'same'}, optional
            'full':
                This returns the convolution at each point of overlap. At
                the end-points of the convolution, the signals do not overlap
                completely, and boundary effects may be seen.
            'same':
                The output is the same size as `arr1`, centered with respect
                to the `‘full’` output. Boundary effects are less prominent.
        axis : tuple of ints, optional
            Axes along which to compute the cross-correlation.
        pad_axis : tuple of ints, optional
            Axes along which pad the correlation (won't perform circular correlation)
        complex : bool
            If true then the complex value is returned otherwise only the real part is returned.
        Returns
        -------
        out : ndarray
            Masked normalized cross-correlation.
        Raises
        ------
        ValueError : if correlation `mode` is not valid, or array dimensions along
            non-transformation axes are not equal.
        References
        ----------
        .. [1] Dirk Padfield. Masked Object Registration in the Fourier Domain.
               IEEE Transactions on Image Processing, vol. 21(5),
               pp. 2706-2718 (2012). :DOI:`10.1109/TIP.2011.2181402`
        .. [2] D. Padfield. "Masked FFT registration". In Proc. Computer Vision and
               Pattern Recognition, pp. 2918-2925 (2010).
               :DOI:`10.1109/CVPR.2010.5540032`
        """
    if mask is None:
        mask = np.ones(np.shape(z))
    return _autocorrelation_masked(z=z,
                                   mask=mask,
                                   mode=mode,
                                   axis=axs,
                                   pad_axis=pad_axis,
                                   overlap_ratio=overlap_ratio)
def cross_correlate(z1,
                    z2,
                    mask1=None,
                    mask2=None,
                    mode="full",
                    axs=(-1),
                    pad_axis=(-1),
                    overlap_ratio=0.3):
    """
        Masked normalized cross-correlation between two arrays.
        Parameters
        ----------
        z1 : ndarray
            The stationary array for the correlation
        z2 : ndarray
            The moving array for the correaltion
        mode : {'full', 'same'}, optional
            'full':
                This returns the convolution at each point of overlap. At
                the end-points of the convolution, the signals do not overlap
                completely, and boundary effects may be seen.
            'same':
                The output is the same size as `arr1`, centered with respect
                to the `‘full’` output. Boundary effects are less prominent.
        axis : tuple of ints, optional
            Axes along which to compute the cross-correlation.
        pad_axis : tuple of ints, optional
            Axes along which pad the correlation (won't perform circular correlation)
        complex : bool
            If true then the complex value is returned otherwise only the real part is returned.
        Returns
        -------
        out : ndarray
            Masked normalized cross-correlation.
        Raises
        ------
        ValueError : if correlation `mode` is not valid, or array dimensions along
            non-transformation axes are not equal.
        References
        ----------
        .. [1] Dirk Padfield. Masked Object Registration in the Fourier Domain.
               IEEE Transactions on Image Processing, vol. 21(5),
               pp. 2706-2718 (2012). :DOI:`10.1109/TIP.2011.2181402`
        .. [2] D. Padfield. "Masked FFT registration". In Proc. Computer Vision and
               Pattern Recognition, pp. 2918-2925 (2010).
               :DOI:`10.1109/CVPR.2010.5540032`
        """
    axis=axs
    if mask1 is None:
        mask1 = np.ones(np.shape(z1))
    if mask2 is None:
        mask2 = np.ones(np.shape(z2))
    return _cross_correlate_masked(z1,
                                   z2,
                                   mask1,
                                   mask2,
                                   mode,
                                   axis,
                                   pad_axis,
                                   overlap_ratio)


def _cross_correlate_masked(z1,
                            z2,
                            mask1,
                            mask2,
                            mode="full",
                            axis=(0,1),
                            pad_axis=(-1),
                            overlap_ratio=0.3):
    """
        Masked normalized cross-correlation between two arrays.
        Parameters
        ----------
        z1 : ndarray
            The stationary array for the correlation
        z2 : ndarray
            The moving array for the correaltion
        mask1 : ndarray
            Mask of `z1`. The mask should evaluate to `True`
            (or 1) on valid pixels. `mask` should have the same shape as `z`.
        mask2 : ndarray
            Mask of `z1`. The mask should evaluate to `True`
            (or 1) on valid pixels. `mask` should have the same shape as `z`.
        mode : {'full', 'same'}, optional
            'full':
                This returns the convolution at each point of overlap. At
                the end-points of the convolution, the signals do not overlap
                completely, and boundary effects may be seen.
            'same':
                The output is the same size as `arr1`, centered with respect
                to the `‘full’` output. Boundary effects are less prominent.
        axis : tuple of ints, optional
            Axes along which to compute the cross-correlation.
        pad_axis : tuple of ints, optional
            Axes along which pad the correlation (won't perform circular correlation)
        overlap_ratio : float, optional
            Minimum allowed overlap ratio between images. The correlation for
            translations corresponding with an overlap ratio lower than this
            threshold will be ignored. A lower `overlap_ratio` leads to smaller
            maximum translation, while a higher `overlap_ratio` leads to greater
            robustness against spurious matches due to small overlap between
            masked images.
        complex : bool
            If true then the complex value is returned otherwise only the real part is returned.
        Returns
        -------
        out : ndarray
            Masked normalized cross-correlation.
        Raises
        ------
        ValueError : if correlation `mode` is not valid, or array dimensions along
            non-transformation axes are not equal.
        References
        ----------
        .. [1] Dirk Padfield. Masked Object Registration in the Fourier Domain.
               IEEE Transactions on Image Processing, vol. 21(5),
               pp. 2706-2718 (2012). :DOI:`10.1109/TIP.2011.2181402`
        .. [2] D. Padfield. "Masked FFT registration". In Proc. Computer Vision and
               Pattern Recognition, pp. 2918-2925 (2010).
               :DOI:`10.1109/CVPR.2010.5540032`
        """
    if isinstance(axis,int):
        axis = (axis,)
    if isinstance(pad_axis, int):
        pad_axis = (pad_axis,)
    if mode not in {'full', 'same'}:
        raise ValueError("Correlation mode '{}' is not valid.".format(mode))
    fixed_image = np.array(z1, dtype=float)
    fixed_mask = np.array(mask1, dtype=bool)
    moving_image = np.array(z2, dtype=float)
    moving_mask = np.array(mask2, dtype=bool)
    eps = np.finfo(float).eps

    # Array dimensions along non-transformation axes should be equal.
    # Determine final size along transformation axes
    # Note that it might be faster to compute Fourier transform in a slightly
    # larger shape (`fast_shape`). Then, after all fourier transforms are done,
    # we slice back to`final_shape` using `final_slice`.
    if pad_axis is not None:
        final_shape = list(z1.shape)
        for ax in axis:
            if ax in pad_axis:
                final_shape[ax] = (fixed_image.shape[ax] +
                                     moving_image.shape[ax] - 1)
        final_shape = tuple(final_shape)
        final_slice = tuple([slice(0, int(sz)) for sz in final_shape])
        # Extent transform axes to the next fast length (i.e. multiple of 3, 5, or 7)
        fast_shape = tuple([next_fast_len(final_shape[ax]) if ax in pad_axis else final_shape[ax] for ax in axis])
        # We use numpy.fft or the new scipy.fft because they allow leaving the
        # transform axes unchanged which was not possible with scipy.fftpack's
        # fftn/ifftn in older versions of SciPy.
        # E.g. arr shape (2, 3, 7), transform along axes (0, 1) with shape (4, 4)
        # results in arr_fft shape (4, 4, 7)
        print(fast_shape)
        fft = partial(fftmodule.fftn, s=fast_shape, axes=axis)
        ifft = partial(fftmodule.ifftn, s=fast_shape, axes=axis)
    else:
        fft = partial(fftmodule.fftn, axes=axis)
        ifft = partial(fftmodule.ifftn, axes=axis)

    fixed_image[np.logical_not(fixed_mask)] = 0.0
    moving_image[np.logical_not(moving_mask)] = 0.0

    # N-dimensional analog to rotation by 180deg is flip over all relevant axes.
    # See [1] for discussion.
    rotated_moving_image = _flip(moving_image, axes=axis)
    rotated_moving_mask = _flip(moving_mask, axes=axis)

    fixed_fft = fft(fixed_image)
    rotated_moving_fft = fft(rotated_moving_image)
    fixed_mask_fft = fft(fixed_mask)
    rotated_moving_mask_fft = fft(rotated_moving_mask)

    # Calculate overlap of masks at every point in the convolution.
    # Locations with high overlap should not be taken into account.
    number_overlap_masked_px = np.real(
        ifft(rotated_moving_mask_fft * fixed_mask_fft))
    number_overlap_masked_px[:] = np.round(number_overlap_masked_px)
    number_overlap_masked_px[:] = np.fmax(number_overlap_masked_px, eps)
    masked_correlated_fixed_fft = ifft(rotated_moving_mask_fft * fixed_fft)
    masked_correlated_rotated_moving_fft = ifft(
        fixed_mask_fft * rotated_moving_fft)

    numerator = ifft(rotated_moving_fft * fixed_fft)
    numerator -= masked_correlated_fixed_fft * \
                 masked_correlated_rotated_moving_fft / number_overlap_masked_px

    fixed_squared_fft = fft(np.square(fixed_image))
    fixed_denom = ifft(rotated_moving_mask_fft * fixed_squared_fft)
    fixed_denom -= np.square(masked_correlated_fixed_fft) / \
                   number_overlap_masked_px
    fixed_denom[:] = np.fmax(fixed_denom, 0.0)

    rotated_moving_squared_fft = fft(np.square(rotated_moving_image))
    moving_denom = ifft(fixed_mask_fft * rotated_moving_squared_fft)
    moving_denom -= np.square(masked_correlated_rotated_moving_fft) / \
                    number_overlap_masked_px
    moving_denom[:] = np.fmax(moving_denom, 0.0)

    denom = np.sqrt(fixed_denom * moving_denom)

    # Slice back to expected convolution shape.
    if pad_axis is not None:
        numerator = numerator[final_slice]
        denom = denom[final_slice]
        number_overlap_masked_px = number_overlap_masked_px[final_slice]

    if mode == 'same':
        _centering = partial(_centered,
                             newshape=fixed_image.shape,
                             axes=axis)
        denom = _centering(denom)
        numerator = _centering(numerator)
        number_overlap_masked_px = _centering(number_overlap_masked_px)

    # Pixels where `denom` is very small will introduce large
    # numbers after division. To get around this problem,
    # we zero-out problematic pixels.
    tol = 1e3 * eps * np.max(np.abs(denom), axis=axis, keepdims=True)
    nonzero_indices = denom > tol

    out = np.zeros_like(denom)
    out[nonzero_indices] = numerator[nonzero_indices] / denom[nonzero_indices]
    np.clip(out, a_min=-1, a_max=1, out=out)

    # Apply overlap ratio threshold
    number_px_threshold = overlap_ratio * np.max(number_overlap_masked_px,
                                                 axis=axis,
                                                 keepdims=True)
    out[number_overlap_masked_px < number_px_threshold] = 0.0

    return out.real


def _autocorrelation_masked(z,
                            mask,
                            axis=(-1,),
                            pad_axis=None,
                            mode="full",
                            overlap_ratio=0.3):
    """
    Masked normalized cross-correlation between arrays.
    Parameters
    ----------
    z : ndarray
        The array to Correlate
    mask : ndarray
        Mask of `z`. The mask should evaluate to `True`
        (or 1) on valid pixels. `mask` should have the same shape as `z`.
    axis : tuple of ints, optional
        Axes along which to compute the cross-correlation.
    overlap_ratio : float, optional
        Minimum allowed overlap ratio between images. The correlation for
        translations corresponding with an overlap ratio lower than this
        threshold will be ignored. A lower `overlap_ratio` leads to smaller
        maximum translation, while a higher `overlap_ratio` leads to greater
        robustness against spurious matches due to small overlap between
        masked images.
    Returns
    -------
    out : ndarray
        Masked normalized cross-correlation.
    Raises
    ------
    ValueError : if correlation `mode` is not valid, or array dimensions along
        non-transformation axes are not equal.
    References
    ----------
    .. [1] Dirk Padfield. Masked Object Registration in the Fourier Domain.
           IEEE Transactions on Image Processing, vol. 21(5),
           pp. 2706-2718 (2012). :DOI:`10.1109/TIP.2011.2181402`
    .. [2] D. Padfield. "Masked FFT registration". In Proc. Computer Vision and
           Pattern Recognition, pp. 2918-2925 (2010).
           :DOI:`10.1109/CVPR.2010.5540032`
    """
    return _cross_correlate_masked(z1=z,
                                   z2=z,
                                   mask1=mask,
                                   mask2=mask,
                                   mode=mode,
                                   axis=axis,
                                   pad_axis=pad_axis,
                                   overlap_ratio=overlap_ratio)

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


def corr_to_power(z):
    return np.power(np.fft.rfft(z, axis=1), 2).real
