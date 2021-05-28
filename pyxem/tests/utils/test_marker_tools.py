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

import pytest
import numpy as np
from numpy.testing import assert_equal
import dask.array as da

from pyxem.signals import Diffraction2D
import pyxem.utils.marker_tools as mt


class TestGet4DMarkerList:
    def test_simple(self):
        peak_array = np.empty((2, 3), dtype=object)
        peak_array[0, 0] = [[2, 4]]
        peak_array[0, 1] = [[8, 2]]
        peak_array[0, 2] = [[1, 8]]
        peak_array[1, 0] = [[3, 1]]
        peak_array[1, 1] = [[9, 1]]
        peak_array[1, 2] = [[6, 3]]
        s = Diffraction2D(np.zeros(shape=(2, 3, 10, 10)))
        marker_list = mt._get_4d_points_marker_list(
            peak_array, s.axes_manager.signal_axes, color="red"
        )
        mt._add_permanent_markers_to_signal(s, marker_list)
        assert len(marker_list) == 1
        marker = marker_list[0]
        assert marker.marker_properties["color"] == "red"
        s.plot()
        for iy, ix in np.ndindex(peak_array.shape[:2]):
            peak = peak_array[iy, ix]
            s.axes_manager.indices = (ix, iy)
            assert marker.get_data_position("x1") == peak[0][1]
            assert marker.get_data_position("y1") == peak[0][0]

    def test_color(self):
        color = "blue"
        peak_array = np.zeros(shape=(3, 2, 1, 2))
        s = Diffraction2D(np.zeros(shape=(3, 2, 10, 10)))
        marker_list = mt._get_4d_points_marker_list(
            peak_array, s.axes_manager.signal_axes, color=color
        )
        assert marker_list[0].marker_properties["color"] == "blue"

    def test_size(self):
        size = 12
        peak_array = np.zeros(shape=(3, 2, 1, 2))
        s = Diffraction2D(np.zeros(shape=(3, 2, 10, 10)))
        marker_list = mt._get_4d_points_marker_list(
            peak_array, s.axes_manager.signal_axes, size=size
        )
        assert marker_list[0].get_data_position("size") == size

    def test_several_markers(self):
        peak_array = np.zeros(shape=(3, 2, 3, 2))
        s = Diffraction2D(np.zeros(shape=(3, 2, 10, 10)))
        marker_list = mt._get_4d_points_marker_list(
            peak_array, s.axes_manager.signal_axes
        )
        assert len(marker_list) == 3

    def test_bool_array(self):
        peak_array = np.empty((2, 3), dtype=object)
        bool_array = np.empty((2, 3), dtype=object)
        for ix, iy in np.ndindex(peak_array.shape):
            peak_array[ix, iy] = np.random.randint(9, size=(1, 2))
            bool_array[ix, iy] = np.random.randint(0, 2, size=1, dtype=bool)

        s = Diffraction2D(np.zeros(shape=(2, 3, 10, 10)))
        marker_list = mt._get_4d_points_marker_list(
            peak_array, s.axes_manager.signal_axes, color="red", bool_array=bool_array
        )
        mt._add_permanent_markers_to_signal(s, marker_list)
        marker = marker_list[0]
        s.plot()

        for iy, ix in np.ndindex(peak_array.shape[:2]):
            peak = peak_array[iy, ix][0]
            boolean = bool_array[iy, ix][0]
            s.axes_manager.indices = (ix, iy)
            if boolean:
                assert marker.get_data_position("x1") == peak[1]
                assert marker.get_data_position("y1") == peak[0]
            else:
                assert marker.get_data_position("x1") == -1000.0
                assert marker.get_data_position("y1") == -1000.0

    def test_several_markers_different_peak_array_size(self):
        peak_array = np.empty((2, 3), dtype=object)
        peak_array[0, 0] = [[2, 4], [1, 9]]
        peak_array[0, 1] = [[8, 2]]
        s = Diffraction2D(np.zeros(shape=(2, 3, 10, 10)))
        marker_list = mt._get_4d_points_marker_list(
            peak_array, s.axes_manager.signal_axes, color="red"
        )
        assert len(marker_list) == 2


class TestFilterPeakArrayListBoolArray:
    def test_wrong_size_input(self):
        peak_array, bool_array = np.empty((2, 4)), np.empty((2, 3))
        with pytest.raises(ValueError):
            mt._filter_peak_array_with_bool_array(peak_array, bool_array)

    def test_filter(self):
        peak_array = np.empty((2, 3), dtype=object)
        bool_array = np.empty((2, 3), dtype=object)
        peak_array[0, 0] = [[2, 4], [1, 9], [4, 5]]
        peak_array[0, 1] = [[8, 2]]
        bool_array[0, 0] = [True, False, True]
        bool_array[0, 1] = [True]
        peak_array_filter = mt._filter_peak_array_with_bool_array(
            peak_array, bool_array
        )
        assert len(peak_array_filter[0, 0]) == 2
        assert_equal(peak_array_filter[0, 0], [[2, 4], [4, 5]])
        assert_equal(peak_array_filter[0, 1], [[8, 2]])

    def test_bool_invert(self):
        peak_array = np.empty((2, 3), dtype=object)
        bool_array = np.empty((2, 3), dtype=object)
        peak_array[0, 0] = [[2, 4], [1, 9], [4, 5]]
        peak_array[0, 1] = [[8, 2]]
        bool_array[0, 0] = [True, False, True]
        bool_array[0, 1] = [True]
        peak_array_filter = mt._filter_peak_array_with_bool_array(
            peak_array, bool_array, bool_invert=True
        )
        assert len(peak_array_filter[0, 0]) == 1
        assert_equal(peak_array_filter[0, 0], [[1, 9]])
        assert len(peak_array_filter[0, 1]) == 0


class TestAddPeakArrayToSignalAsMarkers:
    def test_simple(self):
        peak_array = np.zeros(shape=(3, 2, 3, 2))
        s = Diffraction2D(np.zeros(shape=(3, 2, 10, 10)))
        mt.add_peak_array_to_signal_as_markers(s, peak_array)
        assert len(s.metadata.Markers) == 3

    def test_color(self):
        color = "blue"
        peak_array = np.zeros(shape=(3, 2, 3, 2))
        s = Diffraction2D(np.zeros(shape=(3, 2, 10, 10)))
        mt.add_peak_array_to_signal_as_markers(s, peak_array, color=color)
        marker = list(s.metadata.Markers)[0][1]
        assert marker.marker_properties["color"] == color

    def test_size(self):
        size = 17
        peak_array = np.zeros(shape=(3, 2, 3, 2))
        s = Diffraction2D(np.zeros(shape=(3, 2, 10, 10)))
        mt.add_peak_array_to_signal_as_markers(s, peak_array, size=size)
        marker = list(s.metadata.Markers)[0][1]
        assert marker.get_data_position("size") == size

    def test_dask_input(self):
        s = Diffraction2D(np.zeros((2, 3, 20, 20)))
        peak_array = da.zeros((2, 3, 10, 2), chunks=(1, 1, 10, 2))
        with pytest.raises(ValueError):
            s.add_peak_array_as_markers(peak_array)
        peak_array_computed = peak_array.compute()
        s.add_peak_array_as_markers(peak_array_computed)


class TestPixelToScaledValue:
    def test_simple(self):
        s = Diffraction2D(np.zeros((50, 60)))
        axis = s.axes_manager[-1]
        value = mt._pixel_to_scaled_value(axis, 4.5)
        assert value == 4.5

    def test_scale(self):
        s = Diffraction2D(np.zeros((50, 60)))
        axis = s.axes_manager[-1]
        axis.scale = 0.5
        value = mt._pixel_to_scaled_value(axis, 4.5)
        assert value == 2.25

    def test_offset(self):
        s = Diffraction2D(np.zeros((50, 60)))
        axis = s.axes_manager[-1]
        axis.offset = 6
        value = mt._pixel_to_scaled_value(axis, 4.5)
        assert value == 10.5

    def test_scale_offset(self):
        s = Diffraction2D(np.zeros((50, 60)))
        axis = s.axes_manager[-1]
        axis.scale = 0.5
        axis.offset = 6
        value = mt._pixel_to_scaled_value(axis, 4.5)
        assert value == 8.25


class TestCheckLineSegmentInside:
    @pytest.mark.parametrize(
        "line",
        [
            [-10, 20, 30, 40],
            [60, 20, 30, 40],
            [10, -20, 30, 40],
            [10, 60, 30, 40],
            [10, 20, -30, 40],
            [10, 20, 60, 40],
            [10, 20, 30, -40],
            [10, 20, 30, 60],
        ],
    )
    def test_outside(self, line):
        s = Diffraction2D(np.zeros((2, 3, 50, 50), dtype=np.uint16))
        signal_axes = s.axes_manager.signal_axes
        assert not mt._check_line_segment_inside(signal_axes, line)

    @pytest.mark.parametrize("line", [[10, 20, 30, 40], [20, 20, 40, 40]])
    def test_inside(self, line):
        s = Diffraction2D(np.zeros((2, 3, 50, 50), dtype=np.uint16))
        signal_axes = s.axes_manager.signal_axes
        assert mt._check_line_segment_inside(signal_axes, line)
