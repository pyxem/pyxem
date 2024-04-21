"""Utils for Background Subtraction."""

import numpy as np

from scipy.ndimage import gaussian_filter, median_filter
from skimage.filters.rank import mean as rank_mean
from skimage.morphology import square
from pyxem.utils.diffraction import regional_filter


def _subtract_radial_median(frame, center_x=128, center_y=128):
    """Background removal by subtracting median of pixel at the same
    radius from the center.

    Parameters
    ----------
    frame : NumPy 2D array
    center_x : int
    center_y : int

    Returns
    -------
    background_removed : Numpy 2D array

    Examples
    --------
    >>> import pyxem.utils.dask_tools as dt
    >>> s = pxm.data.dummy_data.get_cbed_signal()
    >>> s_rem = dt._background_removal_single_frame_radial_median(s.data[0, 0])
    """

    y, x = np.indices((frame.shape))
    r = np.hypot(x - center_x, y - center_y)
    r = r.astype(int)
    r_flat = r.ravel()
    diff_image_flat = frame.ravel()
    r_median = np.zeros(np.max(r) + 1, dtype=np.float64)

    for i in range(len(r_median)):
        if diff_image_flat[r_flat == i].size != 0:
            r_median[i] = np.median(diff_image_flat[r_flat == i])
    image = frame - r_median[r]

    return image


def _subtract_dog(frame, min_sigma=1, max_sigma=55):
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
    >>> s = pxm.data.dummy_data.dummy_data.get_cbed_signal()
    >>> s_rem = dt._background_removal_single_frame_dog(s.data[0, 0])

    """
    blur_max = gaussian_filter(frame, max_sigma)
    blur_min = gaussian_filter(frame, min_sigma)
    return np.maximum(np.where(blur_min > blur_max, frame, 0) - blur_max, 0)


def _subtract_median(frame, footprint=19):
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
    >>> s = pxm.data.dummy_data.get_cbed_signal()
    >>> s_rem = dt._background_removal_single_frame_median(s.data[0, 0])

    """
    bg_subtracted = frame - median_filter(frame, size=footprint)
    return bg_subtracted


def _subtract_hdome(frame, **kwargs):
    """Background removal using h-dome filter."""
    max_value = np.max(frame)
    bg_subtracted = rank_mean(
        regional_filter(frame / max_value, **kwargs), footprint=square(3)
    )
    bg_subtracted = bg_subtracted / np.max(bg_subtracted)
    return bg_subtracted


def _polar_subtract_radial_median(frame):
    """Background removal using the radial median"""
    median = np.nanmedian(frame, axis=1)
    image = frame - median[:, np.newaxis]
    image[image < 0] = 0
    return image


def _polar_subtract_radial_percentile(frame, percentile: int):
    """Background removal using the specified radial percentile.

    Parameters
    ----------
    frame : NumPy 2D array
    percentile : percentile, between 0 and 100

    Note
    -------
    if `percentile` is 50, then this is equivalent to `_polar_subtract_radial_median`.
    """
    percentile = np.nanpercentile(frame, percentile, axis=1)
    image = frame - percentile[:, np.newaxis]
    image[image < 0] = 0
    return image
