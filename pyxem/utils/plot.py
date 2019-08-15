# -*- coding: utf-8 -*-
# Copyright 2017-2019 The pyXem developers
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
import itertools

from mpl_toolkits.axisartist.floating_axes import GridHelperCurveLinear, \
    FloatingSubplot


def _find_max_length_peaks(peaks):
    """Worker function for generate_marker_inputs_from_peaks.

    Parameters
    ----------
    peaks : :class:`pyxem.diffraction_vectors.DiffractionVectors`
        Identified peaks in a diffraction signal.

    Returns
    -------
    longest_length : int
        The length of the longest peak list.

    """
    # FIX ME
    x_size, y_size = peaks.axes_manager.navigation_shape[0], peaks.axes_manager.navigation_shape[1]
    length_of_longest_peaks_list = 0
    for x in np.arange(0, x_size):
        for y in np.arange(0, y_size):
            if peaks.data[y, x].shape[0] > length_of_longest_peaks_list:
                length_of_longest_peaks_list = peaks.data[y, x].shape[0]
    return length_of_longest_peaks_list


def generate_marker_inputs_from_peaks(peaks):
    """Takes a peaks (defined in 2D) object from a STEM (more than 1 image) scan
    and returns markers.

    Parameters
    ----------
    peaks : :class:`pyxem.diffraction_vectors.DiffractionVectors`
        Identifies peaks in a diffraction signal.

    Example
    -------
    How to get these onto images::

        mmx,mmy = generate_marker_inputs_from_peaks(found_peaks)
        dp.plot(cmap='viridis')
        for mx,my in zip(mmx,mmy):
            m = hs.markers.point(x=mx,y=my,color='red',marker='x')
            dp.add_marker(m,plot_marker=True,permanent=False)

    """
    # XXX: non-square signals
    max_peak_len = _find_max_length_peaks(peaks)
    pad = np.array(list(itertools.zip_longest(*np.concatenate(peaks.data), fillvalue=[np.nan, np.nan])))
    pad = pad.reshape((max_peak_len), peaks.data.shape[0], peaks.data.shape[1], 2)
    xy_cords = np.transpose(pad, [3, 0, 1, 2])  # move the x,y pairs to the front
    x = xy_cords[0]
    y = xy_cords[1]

    return x, y
