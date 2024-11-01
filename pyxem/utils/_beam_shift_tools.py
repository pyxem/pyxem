import copy
import math
import numpy as np
import scipy.optimize as opt
from matplotlib.colors import hsv_to_rgb
from hyperspy.signals import Signal2D


def _make_bivariate_histogram(
    x_position, y_position, histogram_range=None, masked=None, bins=200, spatial_std=3
):
    s0_flat = x_position.flatten()
    s1_flat = y_position.flatten()

    if masked is not None:
        temp_s0_flat = []
        temp_s1_flat = []
        for data0, data1, masked_value in zip(s0_flat, s1_flat, masked.flatten()):
            if not masked_value:
                temp_s0_flat.append(data0)
                temp_s1_flat.append(data1)
        s0_flat = np.array(temp_s0_flat)
        s1_flat = np.array(temp_s1_flat)

    if histogram_range is None:
        if s0_flat.std() > s1_flat.std():
            s0_range = (
                s0_flat.mean() - s0_flat.std() * spatial_std,
                s0_flat.mean() + s0_flat.std() * spatial_std,
            )
            s1_range = (
                s1_flat.mean() - s0_flat.std() * spatial_std,
                s1_flat.mean() + s0_flat.std() * spatial_std,
            )
        else:
            s0_range = (
                s0_flat.mean() - s1_flat.std() * spatial_std,
                s0_flat.mean() + s1_flat.std() * spatial_std,
            )
            s1_range = (
                s1_flat.mean() - s1_flat.std() * spatial_std,
                s1_flat.mean() + s1_flat.std() * spatial_std,
            )
    else:
        s0_range = histogram_range
        s1_range = histogram_range

    hist2d, xedges, yedges = np.histogram2d(
        s0_flat,
        s1_flat,
        bins=bins,
        range=[[s0_range[0], s0_range[1]], [s1_range[0], s1_range[1]]],
    )

    s_hist = Signal2D(hist2d).swap_axes(0, 1)
    s_hist.axes_manager[0].offset = xedges[0]
    s_hist.axes_manager[0].scale = xedges[1] - xedges[0]
    s_hist.axes_manager[1].offset = yedges[0]
    s_hist.axes_manager[1].scale = yedges[1] - yedges[0]
    return s_hist


def _get_corner_mask(s, corner_size=0.05):
    corner_slice_list = _get_corner_slices(s, corner_size=corner_size)
    mask = np.ones_like(s.data, dtype=bool)
    for corner_slice in corner_slice_list:
        mask[corner_slice] = False

    s_mask = s._deepcopy_with_new_data(mask).T
    return s_mask


def _get_corner_slices(s, corner_size=0.05):
    """Get signal slices for the corner positions for a Signal2D.

    Returns
    -------
    corner_array : slice tuple
        Tuple with the corner slices.

    """
    if s.axes_manager.signal_dimension != 0:
        raise ValueError("s need to have 0 signal dimensions")
    if s.axes_manager.navigation_dimension != 2:
        raise ValueError("s need to have 2 navigation dimensions")
    am = s.axes_manager.navigation_axes
    a0_range = (am[0].high_index - am[0].low_index) * corner_size
    a1_range = (am[1].high_index - am[1].low_index) * corner_size
    a0_range, a1_range = int(a0_range), int(a1_range)

    corner00_slice = np.s_[: a0_range + 1, : a1_range + 1]
    corner01_slice = np.s_[: a0_range + 1, am[1].high_index - a1_range :]
    corner10_slice = np.s_[am[0].high_index - a0_range :, : a1_range + 1]
    corner11_slice = np.s_[am[0].high_index - a0_range :, am[1].high_index - a1_range :]

    return (corner00_slice, corner01_slice, corner10_slice, corner11_slice)


def _f_min(X, p):
    plane_xyz = p[0:3]
    distance = (plane_xyz * X).sum(axis=1) + p[3]
    return distance / np.linalg.norm(plane_xyz)


def _residuals(params, X):
    return _f_min(X, params)


def _magnitude_deviations(params, X):
    plane_x_xy = params[0:2]
    plane_y_xy = params[3:5]
    distance_x = (plane_x_xy * X[:, :2]).sum(axis=1) + params[2]
    distance_y = (plane_y_xy * X[:, :2]).sum(axis=1) + params[5]
    distance = np.sqrt(((distance_x - X[:, 2]) ** 2 + (distance_y - X[:, 3]) ** 2))

    return distance - np.mean(distance)


def _plane_parameters_to_image(p, xaxis, yaxis):
    """Get a plane 2D array from plane parameters.

    Assumes a regularly spaced grid.

    Parameters
    ----------
    p : list
        4 values, like [0.1, 0.1, 0.1, 0.1]
    xaxis, yaxis : array-like
        List of x- and y-positions. For example via the axes_manager.
        s.axes_manager.signal_axes[0].axis

    Returns
    -------
    plane_image : 2D NumPy array

    Example
    -------
    >>> plane = _plane_parameters_to_image([1.2, 0.2, 3.2, 0.5], range(50), range(60))

    """
    x, y = np.meshgrid(xaxis, yaxis)
    z = (-p[0] * x - p[1] * y - p[3]) / p[2]
    return z


def _get_linear_plane_from_signal2d(signal, mask=None, initial_values=None):
    if len(signal.axes_manager.navigation_axes) != 0:
        raise ValueError("signal need to have 0 navigation dimensions")
    if len(signal.axes_manager.signal_axes) != 2:
        raise ValueError("signal need to have 2 signal dimensions")
    if initial_values is None:
        initial_values = [0.1, 0.1, 0.1, 0.1]

    sam = signal.axes_manager.signal_axes
    xaxis, yaxis = sam[0].axis, sam[1].axis
    x, y = np.meshgrid(xaxis, yaxis)
    xx, yy = x.flatten(), y.flatten()
    values = signal.data.flatten()
    points = np.stack((xx, yy, values)).T
    if mask is not None:
        if mask.__array__().shape != signal.__array__().shape:
            raise ValueError("signal and mask need to have the same shape")
        points = points[np.invert(mask).flatten()]

    p = opt.leastsq(_residuals, initial_values, args=points)[0]

    plane = _plane_parameters_to_image(p, xaxis, yaxis)
    return plane


def _get_linear_plane_by_minimizing_magnitude_variance(
    signal, mask=None, initial_values=None
):
    if len(signal.axes_manager.navigation_axes) != 2:
        raise ValueError("signal needs to have 2 navigation dimensions")
    if len(signal.axes_manager.signal_axes) != 1:
        raise ValueError("signal needs to have 1 signal dimension")
    if initial_values is None:
        initial_values = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    signal = signal.T
    sam = signal.axes_manager.signal_axes
    xaxis, yaxis = sam[0].axis, sam[1].axis
    x, y = np.meshgrid(xaxis, yaxis)
    xx, yy = x.flatten(), y.flatten()
    values_x = signal.data[0].flatten()
    values_y = signal.data[1].flatten()
    points = np.stack((xx, yy, values_x, values_y)).T
    if mask is not None:
        if mask.__array__().shape != signal.T.__array__().shape[:2]:
            raise ValueError("signal and mask need to have the same navigation shape")
        points = points[np.invert(mask).flatten()]

    p = opt.leastsq(_magnitude_deviations, initial_values, args=points)[0]

    x, y = np.meshgrid(xaxis, yaxis)
    z_x = p[0] * x + p[1] * y + p[2]
    z_y = p[3] * x + p[4] * y + p[5]

    return np.stack((z_x, z_y), axis=-1)


def _get_limits_from_array(data, sigma=4, ignore_zeros=False, ignore_edges=False):
    if ignore_edges:
        x_lim = int(data.shape[0] * 0.05)
        y_lim = int(data.shape[1] * 0.05)
        data_array = copy.deepcopy(data[x_lim:-x_lim, y_lim:-y_lim])
    else:
        data_array = copy.deepcopy(data)
    if ignore_zeros:
        data_array = np.ma.masked_values(data_array, 0.0)
    mean = data_array.mean()
    data_variance = data_array.std() * sigma
    clim = (mean - data_variance, mean + data_variance)
    if data_array.min() > clim[0]:
        clim = list(clim)
        clim[0] = data_array.min()
        clim = tuple(clim)
    if data_array.max() < clim[1]:
        clim = list(clim)
        clim[1] = data_array.max()
        clim = tuple(clim)
    return clim


def _get_rgb_phase_array(phase, rotation=None, max_phase=2 * np.pi, phase_lim=None):
    phase = _find_phase(phase, rotation=rotation, max_phase=max_phase)
    phase = phase / (2 * np.pi)
    S = np.ones_like(phase)
    HSV = np.dstack((phase, S, S))
    RGB = hsv_to_rgb(HSV)
    return RGB


def _find_phase(phase, rotation=None, max_phase=2 * np.pi):
    if rotation is not None:
        phase = phase + math.radians(rotation)
    phase = phase % max_phase
    return phase


def _get_rgb_phase_magnitude_array(
    phase, magnitude, rotation=None, magnitude_limits=None, max_phase=2 * np.pi
):
    phase = _find_phase(phase, rotation=rotation, max_phase=max_phase)
    phase = phase / (2 * np.pi)

    if magnitude_limits is not None:
        np.clip(magnitude, magnitude_limits[0], magnitude_limits[1], out=magnitude)
    magnitude_max = magnitude.max()
    if magnitude_max == 0:
        magnitude_max = 1
    magnitude = magnitude / magnitude_max
    S = np.ones_like(phase)
    HSV = np.dstack((phase, S, magnitude))
    RGB = hsv_to_rgb(HSV)
    return RGB
