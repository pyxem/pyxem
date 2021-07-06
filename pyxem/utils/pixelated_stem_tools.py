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

import copy
import math
import numpy as np
from scipy.ndimage import measurements, shift
from scipy.optimize import leastsq
from hyperspy.misc.utils import isiterable
from matplotlib.colors import hsv_to_rgb
import pyxem.utils.lazy_tools as lt


def _threshold_and_mask_single_frame(im, threshold=None, mask=None):
    image = copy.deepcopy(im)
    if mask is not None:
        image *= mask
    if threshold is not None:
        mean_value = measurements.mean(image, mask) * threshold
        image[image <= mean_value] = 0
        image[image > mean_value] = 1
    return image


def _radial_average_dask_array(
    dask_array,
    return_sig_size,
    centre_x,
    centre_y,
    normalize,
    mask_array=None,
    show_progressbar=True,
):
    func_args = {
        "mask": mask_array,
        "radial_array_size": return_sig_size,
        "normalize": normalize,
    }
    func_iterating_args = {"centre_x": centre_x, "centre_y": centre_y}
    data = lt._calculate_function_on_dask_array(
        dask_array,
        _get_radial_profile_of_diff_image,
        func_args=func_args,
        func_iterating_args=func_iterating_args,
        return_sig_size=return_sig_size,
        show_progressbar=show_progressbar,
    )
    return data


def _shift_single_frame(im, shift_x, shift_y, interpolation_order=1):
    im_shifted = shift(im, (-shift_y, -shift_x), order=interpolation_order)
    return im_shifted


def _make_circular_mask(centerX, centerY, imageSizeX, imageSizeY, radius):
    """Make a circular mask in a bool array for masking a region in an image.

    Parameters
    ----------
    centreX, centreY : float
        Centre point of the mask.
    imageSizeX, imageSizeY : int
        Size of the image to be masked.
    radius : float
        Radius of the mask.

    Returns
    -------
    Boolean Numpy 2D Array
        Array with the shape (imageSizeX, imageSizeY) with the mask.

    Examples
    --------
    >>> import numpy as np
    >>> import pyxem.utils.pixelated_stem_tools as pst
    >>> image = np.ones((9, 9))
    >>> mask = pst._make_circular_mask(4, 4, 9, 9, 2)
    >>> image_masked = image*mask
    >>> import matplotlib.pyplot as plt
    >>> cax = plt.imshow(image_masked)
    """
    x, y = np.ogrid[-centerY : imageSizeY - centerY, -centerX : imageSizeX - centerX]
    mask = x * x + y * y <= radius * radius
    return mask


def _get_signal_mean_position_and_value(signal):
    """Get the scaled position and mean data value from a signal.

    Note: due to how HyperSpy numbers the axis values, the results
    can sometimes be not as expected. For example, the signal
    initialized Signal2D(np.zeros((10, 10))) will the output of this
    function will be (4.5, 4.5, 0), due to the axis values being
    from 0 to 9 (see example).

    Parameters
    ----------
    signal : HyperSpy signal
        Must have 2 signal dimensions and 0 navigation dimensions.

    Returns
    -------
    output_values : tuple (x_mean, y_mean, value_mean)

    Examples
    --------
    >>> import numpy as np
    >>> import hyperspy.api as hs
    >>> import pyxem.utils.pixelated_stem_tools as pst
    >>> s = hs.signals.Signal2D(np.zeros((10, 10)))
    >>> pst._get_signal_mean_position_and_value(s)
    (4.5, 4.5, 0.0)

    """
    if len(signal.axes_manager.navigation_axes) != 0:
        raise ValueError("signal need to have 0 navigation dimensions")
    if len(signal.axes_manager.signal_axes) != 2:
        raise ValueError("signal need to have 2 signal dimensions")
    sam = signal.axes_manager.signal_axes
    x_mean = sam[0].axis.mean()
    y_mean = sam[1].axis.mean()
    value_mean = signal.data.mean()
    return (x_mean, y_mean, value_mean)


def _get_corner_values(s, corner_size=0.05):
    """Get the corner positions and mean values from a 2D signal.

    Returns
    -------
    corner_array : NumPy array
        corner_array[:, 0] top left corner,
        corner_array[:, 1] bottom left corner,
        corner_array[:, 2] top right corner,
        corner_array[:, 3] bottom right corner.

    """
    if len(s.axes_manager.navigation_axes) != 0:
        raise ValueError("s need to have 0 navigation dimensions")
    if len(s.axes_manager.signal_axes) != 2:
        raise ValueError("s need to have 2 signal dimensions")
    am = s.axes_manager.signal_axes
    a0_range = (am[0].high_index - am[0].low_index) * corner_size
    a1_range = (am[1].high_index - am[1].low_index) * corner_size
    a0_range, a1_range = int(a0_range), int(a1_range)

    s_corner00 = s.isig[: a0_range + 1, : a1_range + 1]
    s_corner01 = s.isig[: a0_range + 1, am[1].high_index - a1_range :]
    s_corner10 = s.isig[am[0].high_index - a0_range :, : a1_range + 1]
    s_corner11 = s.isig[am[0].high_index - a0_range :, am[1].high_index - a1_range :]

    corner00 = _get_signal_mean_position_and_value(s_corner00)
    corner01 = _get_signal_mean_position_and_value(s_corner01)
    corner10 = _get_signal_mean_position_and_value(s_corner10)
    corner11 = _get_signal_mean_position_and_value(s_corner11)

    return np.array((corner00, corner01, corner10, corner11)).T


def _f_min(X, p):
    plane_xyz = p[0:3]
    distance = (plane_xyz * X.T).sum(axis=1) + p[3]
    return distance / np.linalg.norm(plane_xyz)


def _residuals(params, signal, X):
    return _f_min(X, params)


def _fit_ramp_to_image(signal, corner_size=0.05):
    if len(signal.axes_manager.navigation_axes) != 0:
        raise ValueError("s need to have 0 navigation dimensions")
    if len(signal.axes_manager.signal_axes) != 2:
        raise ValueError("s need to have 2 signal dimensions")
    corner_values = _get_corner_values(signal, corner_size=corner_size)
    p0 = [0.1, 0.1, 0.1, 0.1]

    p = leastsq(_residuals, p0, args=(None, corner_values))[0]

    sam = signal.axes_manager.signal_axes
    xx, yy = np.meshgrid(sam[0].axis, sam[1].axis)
    zz = (-p[0] * xx - p[1] * yy - p[3]) / p[2]
    return zz


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


def _make_color_wheel(ax, rotation=None):
    x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
    r = (x ** 2 + y ** 2) ** 0.5
    t = np.arctan2(x, y)
    del x, y
    if rotation is not None:
        t += math.radians(rotation)
        t = (t + np.pi) % (2 * np.pi) - np.pi

    r_masked = np.ma.masked_where((2.0 < r) | (r < 1.0), r)
    r_masked -= 1.0

    mask = r_masked.mask
    r_masked.data[r_masked.mask] = r_masked.mean()
    rgb_array = _get_rgb_phase_magnitude_array(t, r_masked.data)
    rgb_array = np.dstack((rgb_array, np.invert(mask)))

    ax.imshow(rgb_array, interpolation="quadric", origin="lower")
    ax.set_axis_off()


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


def _find_longest_distance(
    imX,
    imY,
    centreX_min,
    centreY_min,
    centreX_max,
    centreY_max,
):
    max_value = max(
        int(((imX - centreX_min) ** 2 + (imY - centreY_min) ** 2) ** 0.5),
        int(((centreX_max) ** 2 + (imY - centreY_min) ** 2) ** 0.5),
        int(((imX - centreX_min) ** 2 + (centreY_max) ** 2) ** 0.5),
        int((centreX_max ** 2 + centreY_max ** 2) ** 0.5),
    )
    return max_value


def _make_centre_array_from_signal(signal, x=None, y=None):
    a_m = signal.axes_manager
    shape = a_m.navigation_shape[::-1]
    if x is None:
        centre_x_array = np.ones(shape) * a_m.signal_axes[0].value2index(0)
    else:
        centre_x_array = np.ones(shape) * x
    if y is None:
        centre_y_array = np.ones(shape) * a_m.signal_axes[1].value2index(0)
    else:
        centre_y_array = np.ones(shape) * y
    if not isiterable(centre_x_array):
        centre_x_array = np.array([centre_x_array])
    if not isiterable(centre_y_array):
        centre_y_array = np.array([centre_y_array])
    return (centre_x_array, centre_y_array)


def _get_radial_profile_of_diff_image(
    diff_image, centre_x, centre_y, normalize, radial_array_size, mask=None
):
    """Radially average a single diffraction image around a centre position.

    Radially profiles the data, averaging the intensity in rings
    out from the centre. Unreliable as we approach the edges of the
    image as it just profiles the corners. Less pixels there so become
    effectively zero after a certain point.

    Parameters
    ----------
    diff_image : 2-D numpy array
        Array consisting of a single diffraction image.
    centre_x : number
        Centre x position of the diffraction image.
    centre_y : number
        Centre y position of the diffraction image.
    radial_array_size : number
    mask : numpy bool array, optional
        Mask parts of the diffraction image, regions where
        the mask is True will be included in the radial profile.

    Returns
    -------
    1-D numpy array of the radial profile.

    """
    radial_array = np.zeros(shape=radial_array_size, dtype=np.float64)
    y, x = np.indices((diff_image.shape))
    r = np.sqrt((x - centre_x) ** 2 + (y - centre_y) ** 2)
    r = r.astype(int)
    if mask is None:
        r_flat = r.ravel()
        diff_image_flat = diff_image.ravel()
    else:
        r_flat = r[mask].ravel()
        diff_image_flat = diff_image[mask].ravel()
    tbin = np.bincount(r_flat, diff_image_flat)
    nr = np.bincount(r_flat)
    nr.clip(1, out=nr)  # To avoid NaN in data due to dividing by 0
    if normalize:
        radial_profile = tbin / nr
    else:
        radial_profile = tbin
    radial_array[0 : len(radial_profile)] = radial_profile
    return radial_array


def _get_angle_sector_mask(
    signal, angle0, angle1, centre_x_array=None, centre_y_array=None
):
    """Get a bool array with True values between angle0 and angle1.
    Will use the (0, 0) point as given by the signal as the centre,
    giving an "angular" slice. Useful for analysing anisotropy in
    diffraction patterns.

    Parameters
    ----------
    signal : HyperSpy 2-D signal
        Can have several navigation dimensions.
    angle0, angle1 : numbers

    Returns
    -------
    Mask : Numpy array
        The True values will be the region between angle0 and angle1.
        The array will have the same dimensions as the input signal.

    Examples
    --------
    >>> import numpy as np
    >>> import pyxem.utils.pixelated_stem_tools as pst
    >>> s = pxm.signals.Diffraction2D(np.arange(100).reshape(10, 10))
    >>> s.axes_manager.signal_axes[0].offset = -5
    >>> s.axes_manager.signal_axes[1].offset = -5
    >>> mask = pst._get_angle_sector_mask(s, 0.5*np.pi, np.pi)

    """
    if angle0 > angle1:
        raise ValueError(
            "angle1 ({0}) needs to be larger than angle0 ({1})".format(angle1, angle0)
        )

    bool_array = np.zeros_like(signal.data, dtype=bool)
    for s in signal:
        indices = signal.axes_manager.indices[::-1]
        signal_axes = s.axes_manager.signal_axes
        if centre_x_array is not None:
            if indices == ():
                signal_axes[0].offset = -centre_x_array[0]
            else:
                signal_axes[0].offset = -centre_x_array[indices]
        if centre_y_array is not None:
            if indices == ():
                signal_axes[1].offset = -centre_y_array[0]
            else:
                signal_axes[1].offset = -centre_y_array[indices]
        x_size = signal_axes[1].size * 1j
        y_size = signal_axes[0].size * 1j
        x, y = np.mgrid[
            signal_axes[1].low_value : signal_axes[1].high_value : x_size,
            signal_axes[0].low_value : signal_axes[0].high_value : y_size,
        ]
        t = np.arctan2(x, y) + np.pi
        if (angle1 - angle0) >= (2 * np.pi):
            bool_array[indices] = True
        else:
            angle0 = angle0 % (2 * np.pi)
            angle1 = angle1 % (2 * np.pi)
            if angle0 < angle1:
                bool_array[indices] = (t > angle0) * (t < angle1)
            elif angle1 < angle0:
                bool_array[indices] = (t > angle0) * (t <= (2 * np.pi))
                bool_array[indices] += (t >= 0) * (t < angle1)
    return bool_array


def _copy_signal2d_axes_manager_metadata(signal_original, signal_new):
    ax_o = signal_original.axes_manager.signal_axes
    ax_n = signal_new.axes_manager.signal_axes
    _copy_axes_object_metadata(ax_o[0], ax_n[0])
    _copy_axes_object_metadata(ax_o[1], ax_n[1])


def _copy_axes_object_metadata(axes_original, axes_new):
    axes_new.scale = axes_original.scale
    axes_new.offset = axes_original.offset
    axes_new.name = axes_original.name
    axes_new.units = axes_original.units


def _copy_signal_all_axes_metadata(signal_original, signal_new):
    if signal_original.axes_manager.shape != signal_new.axes_manager.shape:
        raise ValueError(
            "signal_original and signal_new must have the same shape, not "
            "{0} and {1}".format(
                signal_original.axes_manager.shape, signal_new.axes_manager.shape
            )
        )
    for iax in range(len(signal_original.axes_manager.shape)):
        ax_o = signal_original.axes_manager[iax]
        ax_n = signal_new.axes_manager[iax]
        _copy_axes_object_metadata(ax_o, ax_n)
