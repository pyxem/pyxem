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

import numpy as np
import hyperspy.utils.markers as hm


def _get_4d_points_marker_list(
    peaks_list,
    signal_axes=None,
    color="red",
    size=20,
    bool_array=None,
    bool_invert=False,
):
    """Get a list of 4 dimensional point markers.

    The markers will be displayed on the signal dimensions.

    Parameters
    ----------
    peaks_list : 4D NumPy array
    signal_axes : HyperSpy axes_manager object
    color : string, optional
        Color of point marker. Default 'red'.
    size : scalar, optional
        Size of the point marker. Default 20.
    bool_array : NumPy array, optional
        Same shape as peaks_list.
    bool_invert : bool, optional
        Default False.

    Returns
    -------
    marker_list : list of HyperSpy marker objects

    Example
    -------
    >>> s = pxm.dummy_data.get_cbed_signal()
    >>> peak_array = s.find_peaks(lazy_result=False, show_progressbar=False)
    >>> import pyxem.utils.marker_tools as mt
    >>> marker_list = mt._get_4d_points_marker_list(
    ...     peak_array, s.axes_manager.signal_axes)

    """
    if bool_array is not None:
        peaks_list = _filter_peak_array_with_bool_array(
            peaks_list, bool_array, bool_invert=bool_invert
        )
    max_peaks = 0
    if peaks_list.dtype == object:
        peaks_list_shape = peaks_list.shape
    else:
        peaks_list_shape = peaks_list.shape[:-2]
    for index in np.ndindex(peaks_list_shape):
        islice = np.s_[index]
        peak_list = peaks_list[islice]
        if peak_list is not None:
            n_peaks = len(peak_list)
            if n_peaks > max_peaks:
                max_peaks = n_peaks

    marker_array_shape = list(peaks_list_shape)
    marker_array_shape.append(max_peaks)
    marker_x_array = np.ones(marker_array_shape) * -1000
    marker_y_array = np.ones(marker_array_shape) * -1000
    for index in np.ndindex(peaks_list_shape):
        islice = np.s_[index]
        peak_list = peaks_list[islice]
        if peak_list is not None:
            for i_p, peak in enumerate(peak_list):
                i2slice = list(islice)
                i2slice.append(i_p)
                i2slice = tuple(i2slice)
                if signal_axes is None:
                    marker_x_array[i2slice] = peak[1]
                    marker_y_array[i2slice] = peak[0]
                else:
                    i0min = signal_axes[0].low_index
                    i0max = signal_axes[0].high_index
                    i1min = signal_axes[1].low_index
                    i1max = signal_axes[1].high_index
                    bool0 = i0min <= peak[1] <= i0max
                    bool1 = i1min <= peak[0] <= i1max
                    if bool0 and bool1:
                        vx = _pixel_to_scaled_value(signal_axes[0], peak[1])
                        vy = _pixel_to_scaled_value(signal_axes[1], peak[0])
                        marker_x_array[i2slice] = vx
                        marker_y_array[i2slice] = vy
    marker_list = []
    for i_p in range(max_peaks):
        marker = hm.point(
            marker_x_array[..., i_p], marker_y_array[..., i_p], color=color, size=size
        )
        marker_list.append(marker)
    return marker_list


def _pixel_to_scaled_value(axis, pixel_value):
    offset = axis.offset
    scale = axis.scale
    scaled_value = (pixel_value * scale) + offset
    return scaled_value


def _filter_peak_array_with_bool_array(peak_array, bool_array, bool_invert=False):
    if bool_array.shape != peak_array.shape:
        raise ValueError(
            "bool_array {0} and peak_array {1} must have the"
            " same shape".format(bool_array.shape, peak_array.shape)
        )
    peak_array_filter = np.empty(shape=(peak_array.shape[:2]), dtype=object)
    for ix, iy in np.ndindex(peak_array.shape[:2]):
        peak_list = np.array(peak_array[ix, iy])
        bool_list = np.array(bool_array[ix, iy], dtype=bool)
        if bool_invert:
            bool_list = ~bool_list
        peak_list = peak_list[bool_list]
        peak_array_filter[ix, iy] = peak_list
    return peak_array_filter


def _get_4d_line_segment_list(
    lines_array, signal_axes=None, color="red", linewidth=1, linestyle="solid"
):
    """Get a list of 4 dimensional line segments markers.

    The markers will be displayed on the signal dimensions.

    Parameters
    ----------
    lines_array : 4D NumPy array
    signal_axes : HyperSpy axes_manager object
    color : string, optional
        Color of point marker. Default 'red'.
    linewidth : scalar, optional
        Default 2
    linestyle : string, optional
        Default 'solid'

    Returns
    -------
    marker_list : list of HyperSpy marker objects

    """
    max_lines = 0
    for ix, iy in np.ndindex(lines_array.shape[:2]):
        lines_list = lines_array[ix, iy]
        if lines_list is not None:
            n_lines = len(lines_list)
            if n_lines > max_lines:
                max_lines = n_lines

    marker_array_shape = (lines_array.shape[0], lines_array.shape[1], max_lines)
    marker_x1_array = np.ones(marker_array_shape) * -1000
    marker_y1_array = np.ones(marker_array_shape) * -1000
    marker_x2_array = np.ones(marker_array_shape) * -1000
    marker_y2_array = np.ones(marker_array_shape) * -1000
    for ix, iy in np.ndindex(marker_x1_array.shape[:2]):
        lines_list = lines_array[ix, iy]
        if lines_list is not None:
            for i_p, line in enumerate(lines_list):
                if signal_axes is None:
                    marker_x1_array[ix, iy, i_p] = line[1]
                    marker_y1_array[ix, iy, i_p] = line[0]
                    marker_x2_array[ix, iy, i_p] = line[3]
                    marker_y2_array[ix, iy, i_p] = line[2]
                else:
                    if _check_line_segment_inside(signal_axes, line):
                        sa0iv = signal_axes[0].index2value
                        sa1iv = signal_axes[1].index2value
                        marker_x1_array[ix, iy, i_p] = sa0iv(int(line[1]))
                        marker_y1_array[ix, iy, i_p] = sa1iv(int(line[0]))
                        marker_x2_array[ix, iy, i_p] = sa0iv(int(line[3]))
                        marker_y2_array[ix, iy, i_p] = sa1iv(int(line[2]))

    marker_list = []
    for i_p in range(max_lines):
        marker = hm.line_segment(
            marker_x1_array[..., i_p],
            marker_y1_array[..., i_p],
            marker_x2_array[..., i_p],
            marker_y2_array[..., i_p],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
        )
        marker_list.append(marker)
    return marker_list


def _check_line_segment_inside(signal_axes, line):
    sa0_li = signal_axes[0].low_index
    sa0_hi = signal_axes[0].high_index
    sa1_li = signal_axes[1].low_index
    sa1_hi = signal_axes[1].high_index
    if line[1] > sa0_hi:
        return False
    if line[1] < sa0_li:
        return False
    if line[3] > sa0_hi:
        return False
    if line[3] < sa0_li:
        return False
    if line[0] > sa1_hi:
        return False
    if line[0] < sa1_li:
        return False
    if line[2] > sa1_hi:
        return False
    if line[2] < sa1_li:
        return False
    return True


def _add_permanent_markers_to_signal(signal, marker_list):
    """Add a list of markers to a signal.

    Parameters
    ----------
    signal : PixelatedSTEM or Signal2D
    marker_list : list of markers

    Example
    -------
    >>> s = pxm.dummy_data.get_cbed_signal()
    >>> peak_array = s.find_peaks(lazy_result=False, show_progressbar=False)
    >>> import pyxem.utils.marker_tools as mt
    >>> marker_list = mt._get_4d_points_marker_list(
    ...     peak_array, s.axes_manager.signal_axes)
    >>> mt._add_permanent_markers_to_signal(s, marker_list)
    >>> s.plot()

    """
    if not hasattr(signal.metadata, "Markers"):
        signal.metadata.add_node("Markers")
    marker_extra = len(signal.metadata.Markers)
    for imarker, marker in enumerate(marker_list):
        marker_name = "marker{0}".format(imarker + marker_extra)
        signal.metadata.Markers[marker_name] = marker


def add_peak_array_to_signal_as_markers(
    signal, peak_array, color="red", size=20, bool_array=None, bool_invert=False
):
    """Add an array of points to a signal as HyperSpy markers.

    Parameters
    ----------
    signal : PixelatedSTEM or Signal2D
    peak_array : 4D NumPy array
    color : string, optional
        Default 'red'
    size : scalar, optional
        Default 20
    bool_array : NumPy array, optional
        Same shape as peaks_list.
    bool_invert : bool, optional
        Default False.

    Example
    -------
    >>> s = pxm.dummy_data.get_cbed_signal()
    >>> peak_array = s.find_peaks(lazy_result=False, show_progressbar=False)
    >>> import pyxem.utils.marker_tools as mt
    >>> mt.add_peak_array_to_signal_as_markers(s, peak_array)
    >>> s.plot()

    """
    if hasattr(peak_array, "chunks"):
        raise ValueError(
            "peak_array must be a NumPy array, not dask array. "
            "Run peak_array_computed = peak_array.compute()"
        )
    marker_list = _get_4d_points_marker_list(
        peak_array,
        signal.axes_manager.signal_axes,
        color=color,
        size=size,
        bool_array=bool_array,
        bool_invert=bool_invert,
    )
    _add_permanent_markers_to_signal(signal, marker_list)
