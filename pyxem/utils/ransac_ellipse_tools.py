# -*- coding: utf-8 -*-
# Copyright 2016-2021 The pyXem developers
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

from tqdm import tqdm
import math
from functools import partial
import numpy as np
from skimage.measure import EllipseModel, ransac
from hyperspy.misc.utils import isiterable
import pyxem.utils.marker_tools as mt


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
    data : NumPy array
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
    max_trails=500,
):
    """Pick a random number of data points to fit an ellipse to.

    The ellipse's constraints can be specified.

    See skimage.measure.ransac for more information.

    Parameters
    ----------
    data : NumPy array
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
    max_trails : scalar, optional
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
    ...        semi_len_max=65, semi_len_ratio_lim=1.4, max_trails=1000)

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
            max_trials=max_trails,
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
    max_trails=500,
    show_progressbar=True,
):
    """Pick a random number of data points to fit an ellipse to.

    The ellipse's constraints can be specified.

    See skimage.measure.ransac for more information.

    Parameters
    ----------
    data : NumPy array
        In the form [[[[x0, y0], [x1, y1], ...]]]
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
    max_trails : scalar, optional
        Maximum number of tries for the ransac algorithm.
    show_progressbar : bool, optional
        Default True

    Returns
    -------
    ellipse_array, inlier_array : NumPy array
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
    for iy, ix in t:
        temp_xf, temp_yf = xf[iy, ix], yf[iy, ix]
        ellipse_model, inliers = get_ellipse_model_ransac_single_frame(
            data[iy, ix],
            xf=temp_yf,
            yf=temp_xf,
            rf_lim=rf_lim,
            semi_len_min=semi_len_min,
            semi_len_max=semi_len_max,
            semi_len_ratio_lim=semi_len_ratio_lim,
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            max_trails=max_trails,
        )
        if ellipse_model is not None:
            params = ellipse_model.params
        else:
            params = None
        ellipse_array[iy, ix] = params
        inlier_array[iy, ix] = inliers
    return ellipse_array, inlier_array


def _get_lines_list_from_ellipse_params(ellipse_params, nr=20):
    """Get a line vector list from ellipse params.

    Useful for making HyperSpy line segment markers.

    Parameters
    ----------
    ellipse_params : tuple
        (y, x, semi1, semi0, rotation)
    nr : scalar, optional
        Number of data points in the ellipse, default 20.

    Returns
    -------
    lines_list : list of list
        [[x0, y0, x1, y1], [x1, y1, x2, y2], ...]

    Examples
    --------
    >>> import pyxem.utils.ransac_ellipse_tools as ret
    >>> ellipse_params = (30, 70, 10, 20, 0.5)
    >>> lines_list = ret._get_lines_list_from_ellipse_params(ellipse_params)

    """
    ellipse_data_array = make_ellipse_data_points(
        *ellipse_params, nt=nr, use_focus=False
    )
    lines_list = []
    for i in range(len(ellipse_data_array) - 1):
        pos0 = ellipse_data_array[i]
        pos1 = ellipse_data_array[i + 1]
        lines_list.append([pos0[0], pos0[1], pos1[0], pos1[1]])
    pos0, pos1 = ellipse_data_array[-1], ellipse_data_array[0]
    lines_list.append([pos0[0], pos0[1], pos1[0], pos1[1]])
    return lines_list


def _get_lines_array_from_ellipse_array(ellipse_array, nr=20):
    """Get a line vector array from ellipse params.

    Useful for making HyperSpy line segment markers.

    Parameters
    ----------
    ellipse_array : tuple
        (y, x, semi1, semi0, rotation)
    nr : scalar, optional
        Number of data points in the ellipse, default 20.

    Returns
    -------
    lines_array : NumPy array
        [[[[x0, y0, x1, y1], [x1, y1, x2, y2], ...]]]

    Examples
    --------
    >>> import pyxem.utils.ransac_ellipse_tools as ret
    >>> ellipse_array = np.empty((2, 3), dtype=object)
    >>> ellipse_array[0, 0] = (30, 70, 10, 20, 0.5)
    >>> ellipse_array[1, 0] = (31, 69, 10, 21, 0.5)
    >>> ellipse_array[0, 1] = (29, 68, 10, 21, 0.1)
    >>> ellipse_array[0, 2] = (29, 68, 9, 21, 0.3)
    >>> ellipse_array[1, 1] = (28, 71, 9, 21, 0.5)
    >>> ellipse_array[1, 2] = (32, 68, 11, 22, 0.3)
    >>> larray = ret._get_lines_array_from_ellipse_array(ellipse_array, nr=20)

    """
    lines_array = np.empty(ellipse_array.shape[:2], dtype=object)
    for ix, iy in np.ndindex(ellipse_array.shape[:2]):
        ellipse_params = ellipse_array[ix, iy]
        if ellipse_params is not None:
            lines_list = _get_lines_list_from_ellipse_params(ellipse_params, nr=nr)
            lines_array[ix, iy] = lines_list
        else:
            lines_array[ix, iy] = None
    return lines_array


def _get_inlier_outlier_peak_arrays(peak_array, inlier_array):
    inlier_peak_array = np.empty(peak_array.shape[:2], dtype=object)
    outlier_peak_array = np.empty(peak_array.shape[:2], dtype=object)
    for ix, iy in np.ndindex(peak_array.shape[:2]):
        inliers = inlier_array[ix, iy]
        if inliers is not None:
            outliers = ~inlier_array[ix, iy]
            inlier_peaks = peak_array[ix, iy][inliers]
            outlier_peaks = peak_array[ix, iy][outliers]
        else:
            inlier_peaks = None
            outlier_peaks = peak_array[ix, iy]
        inlier_peak_array[ix, iy] = inlier_peaks
        outlier_peak_array[ix, iy] = outlier_peaks
    return inlier_peak_array, outlier_peak_array


def _get_ellipse_marker_list_from_ellipse_array(
    ellipse_array, nr=20, signal_axes=None, color="red", linewidth=1, linestyle="solid"
):
    lines_array = _get_lines_array_from_ellipse_array(ellipse_array, nr=nr)
    marker_lines_list = mt._get_4d_line_segment_list(
        lines_array,
        signal_axes=signal_axes,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    return marker_lines_list


def _get_ellipse_markers(
    ellipse_array,
    inlier_array=None,
    peak_array=None,
    nr=20,
    signal_axes=None,
    color_ellipse="blue",
    linewidth=1,
    linestyle="solid",
    color_inlier="blue",
    color_outlier="red",
    point_size=20,
):
    marker_list = _get_ellipse_marker_list_from_ellipse_array(
        ellipse_array,
        nr=nr,
        signal_axes=signal_axes,
        color=color_ellipse,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    if inlier_array is not None:
        inlier_parray, outlier_parray = _get_inlier_outlier_peak_arrays(
            peak_array, inlier_array
        )
        marker_in_list = mt._get_4d_points_marker_list(
            inlier_parray, signal_axes=signal_axes, color=color_inlier, size=point_size
        )
        marker_out_list = mt._get_4d_points_marker_list(
            outlier_parray,
            signal_axes=signal_axes,
            color=color_outlier,
            size=point_size,
        )
        marker_list.extend(marker_in_list)
        marker_list.extend(marker_out_list)
    return marker_list
