from tqdm import tqdm
import math
from functools import partial
import numpy as np
from skimage.measure import EllipseModel, ransac
import pixstem.marker_tools as mt


def is_ellipse_good(
        ellipse_model, data,
        xc, yc, r_elli_lim,
        semi_len_min, semi_len_max, semi_len_ratio_lim):
    """Check if an ellipse model is within parameters.

    Parameters
    ----------
    ellipse_model : skimage EllipseModel
    data : Not used
    xc, yc : scalar
    r_elli_lim : scalar
        If the distance from (xc, yc) and the centre of the
        ellipse is larger than r_elli_lim, False is returned.
    semi_len_min, semi_len_max : scalar
        Minimum and maximum semi length values for the ellipse, if any
        of the two semi lengths are outside this range, False is returned.
    semi_len_ratio_lim : scalar
        If the ratio between the largest and smallest semi length is larger
        than semi_len_ratio_lim, False is returned

    Returns
    -------
    is_good : bool

    """
    if semi_len_ratio_lim < 1:
        raise ValueError("semi_len_ratio_lim must be equal or larger than 1, "
                         "not {0}.".format(semi_len_ratio_lim))
    y, x, semi0, semi1, rot = ellipse_model.params
    rC = math.hypot(x - xc, y - yc)
    if rC > r_elli_lim:
        return False
    if not(semi_len_min < semi0 < semi_len_max):
        return False
    if not(semi_len_min < semi1 < semi_len_max):
        return False
    semi_len_ratio = max(semi0, semi1)/min(semi0, semi1)
    if semi_len_ratio > semi_len_ratio_lim:
        return False
    if semi_len_ratio > semi_len_ratio_lim:
        print(semi_len_ratio)
    return True


def is_data_good(data, xc, yc, r_peak_lim):
    """Returns False if any values in an array has points close to the centre.

    Parameters
    ----------
    data : NumPy array
        In the form [[y0, x0], [y1, x1], ...].
    xc, yc : scalar
    r_peak_lim : scalar
        If any of the data points are within r_peak_lim from (xc, yc),
        return False.

    Returns
    -------
    is_good : bool

    """
    dist = np.hypot(data[:, 1] - xc, data[:, 0] - yc)
    if not (dist > r_peak_lim).all():
        return False
    return True


def get_ellipse_model_ransac_single_frame(
        data, xc=128, yc=128, r_elli_lim=30, r_peak_lim=40,
        semi_len_min=50, semi_len_max=90, semi_len_ratio_lim=1.2,
        min_samples=6, residual_threshold=10, max_trails=500):
    """Pick a random number of data points to fit an ellipse to.

    The ellipse's constraints can be specified.

    See skimage.measure.ransac for more information.

    Parameters
    ----------
    data : NumPy array
        In the form [[x0, y0], [x1, y1], ...]
    xc, yc : scalar, optional
        Default 128
    r_elli_lim : scalar, optional
        How far the ellipse centre can be from (xc, yc)
    r_peak_lim : scalar, optional
        How close the individual data points can be from (xc, yc)
        So if any of the data points are too close to (xc, yc), that
        random selection will be rejected. Note, this will increase
        the max_trails counter, so in some situations it might be better
        to remove these data point before running this function.
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
        [y, x, semi_len0, semi_len1, rotation]

    Examples
    --------
    >>> import pixstem.ransac_ellipse_tools as ret
    >>> data = ret.EllipseModel().predict_xy(
    ...        np.arange(0, 2*np.pi, 0.5), params=(128, 130, 50, 60, 0.2))
    >>> ellipse_model, inliers = ret.get_ellipse_model_ransac_single_frame(
    ...        data, xc=128, yc=128, r_elli_lim=5, r_peak_lim=5,
    ...        semi_len_min=45, semi_len_max=65, semi_len_ratio_lim=1.4,
    ...        max_trails=1000)

    """
    is_model_valid = partial(
            is_ellipse_good,
            xc=xc, yc=yc, r_elli_lim=r_elli_lim,
            semi_len_min=semi_len_min, semi_len_max=semi_len_max,
            semi_len_ratio_lim=semi_len_ratio_lim)
    is_data_valid = partial(
            is_data_good,
            xc=xc, yc=yc, r_peak_lim=r_peak_lim)

    # This for loop is here to avoid the returned model being outside the
    # specified limits especially semi_len_ratio_lim.
    # This can happen if only the ransac function is used with this check.
    # This is (probably) due to a valid model being found first, then
    # additional inliers are found afterwards.
    for i in range(3):
        model_ransac, inliers = ransac(
                data.astype(np.float32), EllipseModel, min_samples=min_samples,
                residual_threshold=residual_threshold, max_trials=max_trails,
                is_model_valid=is_model_valid,
                is_data_valid=is_data_valid,
                )
        if model_ransac is not None:
            if is_model_valid(model_ransac, inliers):
                break
            else:
                model_ransac, inliers = None, None
        else:
            break
    return model_ransac, inliers


def get_ellipse_model_ransac(
        data, xc=128, yc=128, r_elli_lim=30, r_peak_lim=40,
        semi_len_min=70, semi_len_max=90, semi_len_ratio_lim=1.2,
        min_samples=6, residual_threshold=10, max_trails=500,
        show_progressbar=True):
    """Pick a random number of data points to fit an ellipse to.

    The ellipse's constraints can be specified.

    See skimage.measure.ransac for more information.

    Parameters
    ----------
    data : NumPy array
        In the form [[[[x0, y0], [x1, y1], ...]]]
    xc, yc : scalar, optional
        Default 128
    r_elli_lim : scalar, optional
        How far the ellipse centre can be from (xc, yc)
    r_peak_lim : scalar, optional
        How close the individual data points can be from (xc, yc)
        So if any of the data points are too close to (xc, yc), that
        random selection will be rejected. Note, this will increase
        the max_trails counter, so in some situations it might be better
        to remove these data point before running this function.
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
    ellipse_array = np.zeros(data.shape[:2], dtype=np.object)
    inlier_array = np.zeros(data.shape[:2], dtype=np.object)
    num_total = data.shape[0] * data.shape[1]
    t = tqdm(np.ndindex(data.shape[:2]), disable=not show_progressbar,
             total=num_total)
    for ix, iy in t:
        ellipse_model, inliers = get_ellipse_model_ransac_single_frame(
                data[ix, iy], xc=xc, yc=yc, r_elli_lim=r_elli_lim,
                r_peak_lim=r_peak_lim,
                semi_len_min=semi_len_min, semi_len_max=semi_len_max,
                semi_len_ratio_lim=semi_len_ratio_lim,
                min_samples=min_samples, residual_threshold=residual_threshold,
                max_trails=max_trails)
        if ellipse_model is not None:
            params = ellipse_model.params
        else:
            params = None
        ellipse_array[ix, iy] = params
        inlier_array[ix, iy] = inliers
    return ellipse_array, inlier_array


def _get_ellipse_model_data(ellipse_params, nr=20):
    """Get points along an ellipse from ellipse_params.

    Parameters
    ----------
    ellipse_params : tuple
        (y, x, semi1, semi0, rotation)
    nr : scalar, optional
        Number of data points in the ellipse, default 20.

    Returns
    -------
    ellipse_data : NumPy array
        In the form [[x0, y0], [x1, y1], ...]

    Examples
    --------
    >>> import pixstem.ransac_ellipse_tools as ret
    >>> ellipse_data = ret._get_ellipse_model_data((30, 70, 10, 20, 0.5))

    Different number of points

    >>> ellipse_data = ret._get_ellipse_model_data((7, 9, 10, 20, 0.5), nr=30)

    """
    phi_array = np.linspace(0, 2 * np.pi, nr+1)[:-1]
    ellipse_data = EllipseModel().predict_xy(
            phi_array, params=ellipse_params)
    return ellipse_data


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
    >>> import pixstem.ransac_ellipse_tools as ret
    >>> ellipse_params = (30, 70, 10, 20, 0.5)
    >>> lines_list = ret._get_lines_list_from_ellipse_params(ellipse_params)

    """
    ellipse_data_array = _get_ellipse_model_data(ellipse_params, nr=nr)
    lines_list = []
    for i in range(len(ellipse_data_array)-1):
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
    >>> import pixstem.ransac_ellipse_tools as ret
    >>> ellipse_array = np.empty((2, 3), dtype=np.object)
    >>> ellipse_array[0, 0] = (30, 70, 10, 20, 0.5)
    >>> ellipse_array[1, 0] = (31, 69, 10, 21, 0.5)
    >>> ellipse_array[0, 1] = (29, 68, 10, 21, 0.1)
    >>> ellipse_array[0, 2] = (29, 68, 9, 21, 0.3)
    >>> ellipse_array[1, 1] = (28, 71, 9, 21, 0.5)
    >>> ellipse_array[1, 2] = (32, 68, 11, 22, 0.3)
    >>> larray = ret._get_lines_array_from_ellipse_array(ellipse_array, nr=20)

    """
    lines_array = np.empty(ellipse_array.shape[:2], dtype=np.object)
    for ix, iy in np.ndindex(ellipse_array.shape[:2]):
        ellipse_params = ellipse_array[ix, iy]
        if ellipse_params is not None:
            lines_list = _get_lines_list_from_ellipse_params(ellipse_params,
                                                             nr=nr)
            lines_array[ix, iy] = lines_list
        else:
            lines_array[ix, iy] = None
    return lines_array


def _get_inlier_outlier_peak_arrays(peak_array, inlier_array):
    inlier_peak_array = np.empty(peak_array.shape[:2], dtype=np.object)
    outlier_peak_array = np.empty(peak_array.shape[:2], dtype=np.object)
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
        ellipse_array, nr=20, signal_axes=None, color='red',
        linewidth=1, linestyle='solid'):
    lines_array = _get_lines_array_from_ellipse_array(
            ellipse_array, nr=nr)
    marker_lines_list = mt._get_4d_line_segment_list(
            lines_array, signal_axes=signal_axes,
            color=color, linewidth=linewidth, linestyle=linestyle)
    return marker_lines_list


def _get_ellipse_markers(
        ellipse_array, inlier_array=None, peak_array=None, nr=20,
        signal_axes=None, color_ellipse='blue', linewidth=1, linestyle='solid',
        color_inlier='blue', color_outlier='red', point_size=20):
    marker_list = _get_ellipse_marker_list_from_ellipse_array(
            ellipse_array, nr=nr, signal_axes=signal_axes,
            color=color_ellipse, linewidth=linewidth, linestyle=linestyle)
    if inlier_array is not None:
        inlier_parray, outlier_parray = _get_inlier_outlier_peak_arrays(
                peak_array, inlier_array)
        marker_in_list = mt._get_4d_points_marker_list(
                inlier_parray, signal_axes=signal_axes, color=color_inlier,
                size=point_size)
        marker_out_list = mt._get_4d_points_marker_list(
                outlier_parray, signal_axes=signal_axes, color=color_outlier,
                size=point_size)
        marker_list.extend(marker_in_list)
        marker_list.extend(marker_out_list)
    return marker_list
