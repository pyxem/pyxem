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

"""Tools for ellipse fitting using RANSAC."""

from tqdm import tqdm
import math
from functools import partial
import numpy as np
from skimage.measure import EllipseModel, ransac
import warnings
from hyperspy.signals import BaseSignal
from hyperspy.misc.utils import isiterable
import hyperspy.api as hs

__all__ = [
    "is_ellipse_good",
    "make_ellipse_data_points",
    "ellipse_to_markers",
    "get_ellipse_model_ransac",
    "get_ellipse_model_ransac_single_frame",
    "determine_ellipse",
]


def is_ellipse_good(
    ellipse_model,
    data,
    xf,
    yf,
    rf_lim,
    semi_len_min=None,
    semi_len_max=None,
    semi_len_ratio_lim=None,
):
    """Check if an ellipse model is within parameters.

    Parameters
    ----------
    ellipse_model : skimage EllipseModel
    data : Not used
    xf, yf : scalar
        Focus of the ellipse
    rf_lim : scalar
        If the distance from (xf, yf) and the centre of the
        ellipse is larger than rf_lim, False is returned.
    semi_len_min, semi_len_max : scalar
        Minimum and maximum semi length values for the ellipse, if any
        of the two semi lengths are outside this range, False is returned.
    semi_len_ratio_lim : scalar
        If the ratio between the largest and smallest semi length is larger
        than semi_len_ratio_lim, False is returned

    Returns
    -------
    is_good : bool

    Examples
    --------
    >>> import pyxem.utils.ransac_ellipse_tools as ret
    >>> model = ret.EllipseModel()
    >>> model.params = ret._make_ellipse_model_params_focus(30, 50, 30, 20, 0)
    >>> is_good = ret.is_ellipse_good(
    ...         ellipse_model=model, data=None, xf=30, yf=50, rf_lim=5)

    """
    xc, yc, a, b, r = ellipse_model.params
    x, y = _get_closest_focus(xf, yf, xc, yc, a, b, r)
    rf = math.hypot(x - xf, y - yf)
    if rf > rf_lim:
        return False
    if semi_len_min is not None:
        if a < semi_len_min:
            return False
        if b < semi_len_min:
            return False
    if semi_len_max is not None:
        if a > semi_len_max:
            return False
        if b > semi_len_max:
            return False
    if semi_len_ratio_lim is not None:
        semi_len_ratio = max(a, b) / min(a, b)
        if semi_len_ratio > semi_len_ratio_lim:
            return False
    return True


def _ellipse_centre_to_focus(x, y, a, b, r):
    """Get focus for an ellipse from EllipseModel.params

    Parameters
    ----------
    x, y : scalars
        Centre position of the ellipse.
    a, b : scalars
        Semi lengths
    r : scalar
        Rotation, in theta

    Returns
    -------
    foci : tuple of tuples
        (x focus 0, y focus 0), (x focus 1, y focus 1)

    Examples
    --------
    >>> import pyxem.utils.ransac_ellipse_tools as ret
    >>> f0, f1 = ret._ellipse_centre_to_focus(20, 32, 12, 9, 0.2)

    """
    if a < b:
        r += math.pi / 2
        a, b = b, a
    c = math.sqrt(math.pow(a, 2) - math.pow(b, 2))
    xf0, yf0 = x + c * math.cos(r), y + c * math.sin(r)
    xf1, yf1 = x - c * math.cos(r), y - c * math.sin(r)
    return ((xf0, yf0), (xf1, yf1))


def _get_closest_focus(x, y, xc, yc, a, b, r):
    """Get the focus closest to the centre from EllipseModel parameters

    Parameters
    ----------
    x, y : scalar
        Centre position of the diffraction pattern.
    xc, yc : scalars
        Centre position of the ellipse.
    a, b : scalars
        Semi lengths
    r : scalar
        Rotation, in theta

    Returns
    -------
    xf, yf : tuple
        Ellipse focus closest to the diffraction centre

    Examples
    --------
    >>> import pyxem.utils.ransac_ellipse_tools as ret
    >>> xf, yf = ret._get_closest_focus(25, 30, 20, 32, 12, 9, 0.2)

    """
    (xf0, yf0), (xf1, yf1) = _ellipse_centre_to_focus(xc, yc, a, b, r)
    rf0 = math.hypot(x - xf0, y - yf0)
    rf1 = math.hypot(x - xf1, y - yf1)
    if rf0 <= rf1:
        xf, yf = xf0, yf0
    else:
        xf, yf = xf1, yf1
    return (xf, yf)


def make_ellipse_data_points(x, y, a, b, r, nt=20, use_focus=True):
    """Get an ellipse position list.

    Parameters
    ----------
    x, y : scalars
        Centre position of the ellipse.
    a, b : scalars
        Semi lengths
    r : scalar
        Rotation, in theta
    nt : int, optional
        Number of data positions, default 20
    use_focus : bool
        If True, (x, y) will be the focus. If False,
        (x, y) will be centre of the ellipse.

    Returns
    -------
    data : numpy.ndarray
        [[x0, y0], [x1, y1], ...]

    Examples
    --------
    >>> import pyxem.utils.ransac_ellipse_tools as ret
    >>> data = ret.make_ellipse_data_points(5, 9, 8, 4, np.pi/3)

    Using all the arguments

    >>> data = ret.make_ellipse_data_points(5, 9, 8, 4, 0, nt=40,
    ...                                     use_focus=False)

    """
    if use_focus:
        params = _make_ellipse_model_params_focus(x, y, a, b, r)
    else:
        params = (x, y, a, b, r)
    theta_array = np.arange(0, 2 * np.pi, 2 * np.pi / nt)
    data = EllipseModel().predict_xy(theta_array, params=params)
    return data


def _ellipse_model_centre_to_focus(xc, yc, a, b, r, x, y):
    """Get focus params from centre params

    There are two different focuses, the one closest to
    (x, y) is returned.

    Parameters
    ----------
    xc, yc : scalar
        x and y position of the ellipse centre
    a, b : scalar
        Semi lengths
    r : scalar
        Rotation
    x, y : scalar

    Returns
    -------
    params : tuple
        (xf, yf, a, b, r)

    """
    xf, yf = _get_closest_focus(x, y, xc, yc, a, b, r)
    params = (xf, yf, a, b, r)
    return params


def _make_ellipse_model_params_focus(xf, yf, a, b, r):
    """
    Parameters
    ----------
    xf, yf : scalar
        x and y position of the focus
    a, b : scalar
        Semi lengths
    r : scalar
        Rotation

    Returns
    -------
    params : tuple
        (xc, yc, a, b, r)

    """
    if a < b:
        r += math.pi / 2
        a, b = b, a
    c = math.sqrt(math.pow(a, 2) - math.pow(b, 2))
    xc = xf - c * math.cos(r)
    yc = yf - c * math.sin(r)
    params = (xc, yc, a, b, r)
    return params


def get_ellipse_model_ransac_single_frame(
    data,
    xf=128,
    yf=128,
    rf_lim=30,
    semi_len_min=50,
    semi_len_max=90,
    semi_len_ratio_lim=1.2,
    min_samples=6,
    residual_threshold=10,
    max_trials=500,
):
    """Pick a random number of data points to fit an ellipse to.

    The ellipse's constraints can be specified.

    See skimage.measure.ransac for more information.

    Parameters
    ----------
    data : numpy.ndarray
        In the form [[x0, y0], [x1, y1], ...]
    xf, yf : scalar, optional
        Default 128
    rf_lim : scalar, optional
        How far the ellipse centre can be from (xf, yf)
    semi_len_min, semi_len_max : scalar, optional
        Limits of the semi lengths
    semi_len_ratio_lim : scalar, optional
        Limit of the ratio of the semi length, must be equal or larger
        than 1. This ratio is calculated by taking the largest semi length
        divided by the smallest semi length:
        max(semi0, semi1)/min(semi0, semi1). So for a perfect circle this
        ratio will be 1.
    min_samples : scalar, optional
        Minimum number of data points to fit the ellipse model to.
    residual_threshold : scalar, optional
        Maximum distance for a data point to be considered an inlier.
    max_trials : scalar, optional
        Maximum number of tries for the ransac algorithm.

    Returns
    -------
    model_ransac, inliers
        Model data is accessed in model_ransac.params:
        [x, y, semi_len0, semi_len1, rotation]

    Examples
    --------
    >>> import pyxem.utils.ransac_ellipse_tools as ret
    >>> data = ret.EllipseModel().predict_xy(
    ...        np.arange(0, 2*np.pi, 0.5), params=(128, 130, 50, 60, 0.2))
    >>> ellipse_model, inliers = ret.get_ellipse_model_ransac_single_frame(
    ...        data, xf=128, yf=128, rf_lim=5, semi_len_min=45,
    ...        semi_len_max=65, semi_len_ratio_lim=1.4, max_trials=1000)

    """
    is_model_valid = partial(
        is_ellipse_good,
        xf=xf,
        yf=yf,
        rf_lim=rf_lim,
        semi_len_min=semi_len_min,
        semi_len_max=semi_len_max,
        semi_len_ratio_lim=semi_len_ratio_lim,
    )
    if min_samples > len(data):
        min_samples = len(data) - 1
    # This for loop is here to avoid the returned model being outside the
    # specified limits especially semi_len_ratio_lim.
    # This can happen if only the ransac function is used with this check.
    # This is (probably) due to a valid model being found first, then
    # additional inliers are found afterwards.
    for i in range(3):
        model_ransac, inliers = ransac(
            data.astype(np.float32),
            EllipseModel,
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            max_trials=max_trials,
            is_model_valid=is_model_valid,
        )
        if model_ransac is not None:
            if is_model_valid(model_ransac, None):
                break
            else:
                model_ransac, inliers = None, None
        else:
            break
    return model_ransac, inliers


def get_ellipse_model_ransac(
    data,
    xf=128,
    yf=128,
    rf_lim=30,
    semi_len_min=70,
    semi_len_max=90,
    semi_len_ratio_lim=1.2,
    min_samples=6,
    residual_threshold=10,
    max_trials=500,
    show_progressbar=True,
):
    """Pick a random number of data points to fit an ellipse to.

    The ellipse's constraints can be specified.

    See skimage.measure.ransac for more information.

    Parameters
    ----------
    data : numpy.ndarray
        In the form [[[[x0, y0], [x1, y1], ...]]]
    xf, yf : scalar, optional
        Default 128 center of the diffraction pattern
    rf_lim : scalar, optional
        How far the ellipse centre can be from (xf, yf)
    semi_len_min, semi_len_max : scalar, optional
        Limits of the semi lengths
    semi_len_ratio_lim : scalar, optional
        Limit of the ratio of the semi length, must be equal or larger
        than 1. This ratio is calculated by taking the largest semi length
        divided by the smallest semi length:
        max(semi0, semi1)/min(semi0, semi1). So for a perfect circle this
        ratio will be 1.
    min_samples : scalar, optional
        Minimum number of data points to fit the ellipse model to.
    residual_threshold : scalar, optional
        Maximum distance for a data point to be considered an inlier.
    max_trials : scalar, optional
        Maximum number of tries for the ransac algorithm.
    show_progressbar : bool, optional
        Default True

    Returns
    -------
    ellipse_array, inlier_array : numpy.ndarray
        Model data is accessed in ellipse_array, where each probe position
        (for two axes) contain a list with the ellipse parameters:
        [y, x, semi_len0, semi_len1, rotation]. If no ellipse is found
        this is None.

    """
    if not isiterable(xf):
        xf = np.ones(data.shape[:2]) * xf
    if not isiterable(yf):
        yf = np.ones(data.shape[:2]) * yf

    ellipse_array = np.zeros(data.shape[:2], dtype=object)
    inlier_array = np.zeros(data.shape[:2], dtype=object)
    num_total = data.shape[0] * data.shape[1]
    t = tqdm(np.ndindex(data.shape[:2]), disable=not show_progressbar, total=num_total)

    new_peaks = np.empty(data.shape, dtype=object)
    for i in np.ndindex(data.shape):
        new_peaks[i] = data[i][:, ::-1]

    for iy, ix in t:
        temp_xf, temp_yf = xf[iy, ix], yf[iy, ix]
        ellipse_model, inliers = get_ellipse_model_ransac_single_frame(
            new_peaks[iy, ix],  # reverse x,y for pixel units
            xf=temp_xf,
            yf=temp_yf,
            rf_lim=rf_lim,
            semi_len_min=semi_len_min,
            semi_len_max=semi_len_max,
            semi_len_ratio_lim=semi_len_ratio_lim,
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            max_trials=max_trials,
        )
        if ellipse_model is not None:
            params = ellipse_model.params
        else:
            params = None
        ellipse_array[iy, ix] = params
        inlier_array[iy, ix] = inliers
    return ellipse_array, inlier_array


def _get_max_positions(signal, mask=None, num_points=5000):
    """Gets the top num_points pixels in the dataset.

    Parameters
    --------------
    signal : BaseSignal
        The signal which we want to find the max positions for.
    mask : np.array
        A mask to be applied to the data for values to ignore.
    num_points : int
        The number of points to be considered.

    """
    if isinstance(signal, BaseSignal):
        data = signal.data
    else:
        data = signal
    i_shape = np.shape(data)
    flattened_array = data.flatten()
    if mask is not None:
        flattened_mask = mask.flatten()
        flattened_array[flattened_mask] = 0
    # take top 5000 points make sure exclude zero beam
    indexes = np.argsort(flattened_array)
    cords = np.array(
        [
            np.floor_divide(indexes[-num_points:], i_shape[1]),
            np.remainder(indexes[-num_points:], i_shape[1]),
        ]
    )  # [x axis (row),y axis (col)]
    return cords.T


def _ellipse_to_affine(major, minor, rot):
    if minor > major:
        temp = minor
        minor = major
        major = temp
        rot = rot + np.pi / 2
    rot = np.pi / 2 - rot
    if rot < 0:
        rot = rot + np.pi

    Q = [[np.cos(rot), -np.sin(rot), 0], [np.sin(rot), np.cos(rot), 0], [0, 0, 1]]
    S = [[1, 0, 0], [0, minor / major, 0], [0, 0, 1]]
    C = np.matmul(np.matmul(Q, S), np.transpose(Q))
    return C


def mask_peak_array(array, mask, invert=False):
    """Return only the peaks in the array which are not masked. Works for
    both ragged and non-ragged arrays.

    Parameters
    ----------
    array: np.ndarray
        The array of peaks to be masked. Can be ragged or non-ragged.
    mask: np.ndarray
        The mask to be applied to the array. If the array is ragged, the mask
        must be ragged as well.
    invert: bool
        If True, the mask is inverted.
    """
    if array.dtype != object:
        if invert:
            mask = np.logical_not(mask)
        return array[mask]
    else:
        masked_array = np.empty(array.shape, dtype=object)
        for i in np.ndindex(array.shape):
            if invert:
                m = np.logical_not(mask[i])
            else:
                m = mask[i]
            masked_array[i] = array[i][m]
        return masked_array


def ellipse_to_markers(ellipse_array, points=None, inlier=None):
    """Convert an ellipse array to a :class:`hyperspy.api.plot.markers.Ellipses` object. If points and
    inlier are provided, then the points are also plotted. The inlier points are plotted in green
    and the outlier points are plotted in red.

    Parameters
    ----------
    ellipse_array: np.ndarray
        The array of ellipses parameters in the form [x_c, y_c, semi_len0, semi_len1, rotation]
    points: np.ndarray
        The array of points to be plotted. If None, then no points are plotted.
    inlier:np.ndarray
        The bool array of inlier points. If None, then no points are plotted.

    Returns
    -------

    """
    if not isinstance(ellipse_array, np.ndarray):
        ellipse_array = np.array(ellipse_array)
    ellipse_array = ellipse_array.T
    if ellipse_array.dtype == object:
        offsets = np.empty(ellipse_array.shape, dtype=object)
        heights = np.empty(ellipse_array.shape, dtype=object)
        widths = np.empty(ellipse_array.shape, dtype=object)
        angles = np.empty(ellipse_array.shape, dtype=object)
        for i in np.ndindex(ellipse_array.shape):
            offsets[i] = ellipse_array[i][:2][::-1]
            heights[i] = ellipse_array[i][2] * 2
            widths[i] = ellipse_array[i][3] * 2
            angles[i] = np.rad2deg(-ellipse_array[i][4])
    else:
        offsets = np.array(
            [
                ellipse_array[:2][::-1],
            ]
        )
        heights = ellipse_array[2] * 2
        widths = ellipse_array[3] * 2
        angles = np.rad2deg(-ellipse_array[4])

    el = hs.plot.markers.Ellipses(
        offsets=offsets,
        heights=heights,
        widths=widths,
        angles=angles,
        facecolor="none",
        edgecolor="white",
        lw=4,
    )

    if points is not None and inlier is not None:
        in_points = hs.plot.markers.Points(
            offsets=mask_peak_array(points[:, ::-1], inlier), color="green"
        )
        out_points = hs.plot.markers.Points(
            offsets=mask_peak_array(points[:, ::-1], inlier, invert=True),
            color="red",
            alpha=0.5,
        )
        return el, in_points, out_points
    elif points is not None:
        points = hs.plot.markers.Points(
            offsets=points[:, ::-1], color="green", alpha=0.5
        )
        return el, points
    else:
        return el


def determine_ellipse(
    signal=None,
    pos=None,
    mask=None,
    num_points=1000,
    use_ransac=False,
    guess_starting_params=True,
    return_params=False,
    **kwargs,
):
    """
    This method starts by taking some number of points which are the most intense
    in the signal or those points can be directly passed.  It then takes those points
    and guesses some starting parameters for the `get_ellipse_model_ransac_single_frame`
    function. From there it will try to determine the ellipse parameters.

    Parameters
    -----------
    signal : Signal2D
        The signal of interest.
    pos : np.ndarray
        The positions of the points to be used to determine the ellipse.
    mask : Array-like
        The mask to be applied to the data.  The True values are ignored.
    num_points : int
        The number of points to consider.
    use_ransac : bool
        If Ransac should be used to determine the ellipse. False is faster but less.
        robust with respect to noise.
    guess_starting_params : bool
        If True then the starting parameters will be guessed based on the points determined.
    return_params : bool
        If the ellipse parameters should be returned as well.
    **kwargs:
        Any other keywords for ``get_ellipse_model_ransac_single_frame``.

    Returns
    -------
    center : (x,y)
        The center of the diffraction pattern.
    affine :
        The affine transformation to make the diffraction pattern circular.

    Examples
    --------
    >>> import pyxem.utils.ransac_ellipse_tools as ret
    >>> import pyxem.data.dummy_data.make_diffraction_test_data as mdtd
    >>> test_data = mdtd.MakeTestData(200, 200, default=False)
    >>> test_data.add_disk(x0=100, y0=100, r=5, intensity=30)
    >>> test_data.add_ring_ellipse(x0=100, y0=100, semi_len0=63, semi_len1=70, rotation=45)
    >>> s = test_data.signal
    >>> s.set_signal_type("electron_diffraction")
    >>> import numpy as np
    >>> mask = np.zeros_like(s.data, dtype=bool)
    >>> mask[100 - 20:100 + 20, 100 - 20:100 + 20] = True # mask beamstop
    >>> center, affine = ret.determine_ellipse(s, mask=mask, use_ransac=False)
    >>> s_corr = s.apply_affine_transformation(affine, inplace=False)

    """
    if signal is not None:
        pos = _get_max_positions(signal, mask=mask, num_points=num_points)
    elif pos is None:
        raise ValueError("Either signal or pos must be specified")
    if use_ransac:
        if guess_starting_params:
            el, inlier = get_ellipse_model_ransac_single_frame(
                pos,
                xf=np.mean(pos[:, 0]),
                yf=np.mean(pos[:, 1]),
                rf_lim=np.shape(signal.data)[0] / 5,
                semi_len_min=np.std(pos[:, 1]),
                semi_len_max=np.std(pos[:, 1]) * 2,
                semi_len_ratio_lim=1.2,
                min_samples=6,
                residual_threshold=20,
                max_trials=1000,
            )
        else:
            el, inlier = get_ellipse_model_ransac_single_frame(pos, **kwargs)
    else:
        e = EllipseModel()
        converge = e.estimate(data=pos)
        el = e
    if el is not None:
        affine = _ellipse_to_affine(el.params[3], el.params[2], el.params[4])
        center = (el.params[0], el.params[1])
        if return_params and use_ransac:
            return center, affine, el.params, pos, inlier
        elif return_params:
            return center, affine, el.params, pos
        else:
            return center, affine
    else:  # pragma: no cover
        warnings.warn("Ransac Ellipse detection did not converge")
        return None
