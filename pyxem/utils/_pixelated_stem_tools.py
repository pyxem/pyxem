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

"""Utils for operating on pixelated signals."""

import copy
import numpy as np
import scipy.ndimage as ndi
from hyperspy.misc.utils import isiterable


def _threshold_and_mask_single_frame(im, threshold=None, mask=None):
    image = copy.deepcopy(im)
    if mask is not None:
        image *= mask
    if threshold is not None:
        mean_value = ndi.mean(image, mask) * threshold
        image[image <= mean_value] = 0
        image[image > mean_value] = 1
    return image


def _shift_single_frame(im, shift_x, shift_y, interpolation_order=1):
    im_shifted = ndi.shift(im, (-shift_y, -shift_x), order=interpolation_order)
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
        int((centreX_max**2 + centreY_max**2) ** 0.5),
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
