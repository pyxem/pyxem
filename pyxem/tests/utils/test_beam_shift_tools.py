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


import pytest
from pytest import approx
import numpy as np
from numpy.testing import assert_allclose

import hyperspy.api as hs
import pyxem.utils._beam_shift_tools as bst


class TestGetRgbPhaseMagnitudeArray:
    def test_simple(self):
        phase = np.zeros((50, 50))
        magnitude = np.zeros((50, 50))
        rgb_array = bst._get_rgb_phase_magnitude_array(phase, magnitude)
        assert (rgb_array == 0.0).all()

    def test_magnitude_zero(self):
        phase = np.random.random((50, 50))
        magnitude = np.zeros((50, 50))
        rgb_array = bst._get_rgb_phase_magnitude_array(phase, magnitude)
        assert (rgb_array == 0.0).all()

    def test_all_same(self):
        phase = np.ones((50, 50))
        magnitude = np.ones((50, 50))
        rgb_array = bst._get_rgb_phase_magnitude_array(phase, magnitude)
        assert (rgb_array == rgb_array[0][0]).all()


class TestGetRgbPhaseArray:
    def test_all_same0(self):
        phase = np.zeros((50, 50))
        rgb_array = bst._get_rgb_phase_array(phase)
        assert (rgb_array == rgb_array[0][0]).all()

    def test_all_same1(self):
        phase = np.ones((50, 50))
        rgb_array = bst._get_rgb_phase_array(phase)
        assert (rgb_array == rgb_array[0][0]).all()


class TestFindPhase:
    def test_simple(self):
        phase = np.zeros((50, 50))
        phase0 = bst._find_phase(phase)
        assert (phase0 == 0.0).all()

    def test_rotation(self):
        phase = np.zeros((50, 50))
        phase0 = bst._find_phase(phase, rotation=90)
        assert (phase0 == np.pi / 2).all()
        phase1 = bst._find_phase(phase, rotation=45)
        assert (phase1 == np.pi / 4).all()
        phase2 = bst._find_phase(phase, rotation=180)
        assert (phase2 == np.pi).all()
        phase3 = bst._find_phase(phase, rotation=360)
        assert (phase3 == 0).all()
        phase4 = bst._find_phase(phase, rotation=-90)
        assert (phase4 == 3 * np.pi / 2).all()

    def test_max_phase(self):
        phase = (np.ones((50, 50)) * np.pi * 0.5) + np.pi
        phase0 = bst._find_phase(phase, max_phase=np.pi)
        assert (phase0 == np.pi / 2).all()


class TestGetCornerSlices:
    @pytest.mark.parametrize("corner_size", [0.02, 0.05, 0.20, 0.23])
    def test_corner_size(self, corner_size):
        size = 100
        s = hs.signals.Signal2D(np.zeros((size, size))).T
        corner_slice_list = bst._get_corner_slices(s, corner_size=corner_size)
        corner_shape = (round(size * corner_size), round(size * corner_size))
        for corner_slice in corner_slice_list:
            s_corner = s.inav[corner_slice]
            assert s_corner.axes_manager.shape == corner_shape

    def test_signal_slice_values(self):
        data = np.zeros((100, 200), dtype=np.uint16)
        data[:20, :20] = 2
        data[:20, -20:] = 5
        data[-20:, :20] = 10
        data[-20:, -20:] = 8
        s = hs.signals.Signal2D(data).T
        corner_slice_list = bst._get_corner_slices(s, corner_size=0.05)
        assert (s.inav[corner_slice_list[0]].data == 2).all()
        assert (s.inav[corner_slice_list[1]].data == 10).all()
        assert (s.inav[corner_slice_list[2]].data == 5).all()
        assert (s.inav[corner_slice_list[3]].data == 8).all()

    def test_non_square_signal(self):
        corner_size = 0.05
        size_x, size_y = 100, 200
        s = hs.signals.Signal2D(np.zeros((size_y, size_x))).T
        corner_slice_list = bst._get_corner_slices(s, corner_size=0.05)
        corner_shape = (round(size_x * corner_size), round(size_y * corner_size))
        for corner_slice in corner_slice_list:
            s_corner = s.inav[corner_slice]
            assert s_corner.axes_manager.shape == corner_shape

    def test_wrong_input_dimensions(self):
        s = hs.signals.Signal2D(np.ones((2, 10, 10)))
        with pytest.raises(ValueError):
            bst._get_corner_slices(s)
        s = hs.signals.Signal2D(np.ones((2, 2, 10, 10)))
        with pytest.raises(ValueError):
            bst._get_corner_slices(s)
        s = hs.signals.Signal1D(np.ones(10))
        with pytest.raises(ValueError):
            bst._get_corner_slices(s)


class TestPlaneParametersToImage:
    def test_simple(self):
        p = [0, 0, 1, 0]
        xaxis, yaxis = range(100), range(110)
        image = bst._plane_parameters_to_image(p, xaxis, yaxis)
        assert_allclose(image, 0)

    def test_offset(self):
        p = [0, 0, 1, 3]
        xaxis, yaxis = range(100), range(110)
        image = bst._plane_parameters_to_image(p, xaxis, yaxis)
        assert_allclose(image, -3)

    def test_x_plane(self):
        p = [1, 0, 1, 0]
        xaxis, yaxis = range(100), range(110)
        image = bst._plane_parameters_to_image(p, xaxis, yaxis)
        assert image[0, 0] > image[0, -1]

    def test_y_plane(self):
        p = [0, 1, 1, 0]
        xaxis, yaxis = range(100), range(110)
        image = bst._plane_parameters_to_image(p, xaxis, yaxis)
        assert image[0, 0] > image[-1, 0]

    def test_last_parameter(self):
        p = [0, 0, 2, 4]
        xaxis, yaxis = range(100), range(110)
        image = bst._plane_parameters_to_image(p, xaxis, yaxis)
        assert_allclose(image, -2)


class TestGetLinearPlaneFromSignal2d:
    def test_linear_ramp(self):
        s0, s1 = hs.signals.Signal2D(np.meshgrid(range(100), range(110)))
        s0.change_dtype("float64")
        s1.change_dtype("float64")
        s0_plane = bst._get_linear_plane_from_signal2d(s0)
        s1_plane = bst._get_linear_plane_from_signal2d(s1)
        np.testing.assert_almost_equal(s0_plane.data, s0.data)
        np.testing.assert_almost_equal(s1_plane.data, s1.data)

    def test_zeros_values(self):
        s = hs.signals.Signal2D(np.zeros((100, 200), dtype=np.float32))
        s_plane = bst._get_linear_plane_from_signal2d(s)
        np.testing.assert_almost_equal(s_plane.data, s.data)

    def test_ones_values(self):
        s = hs.signals.Signal2D(np.ones((100, 200), dtype=np.float32))
        s_plane = bst._get_linear_plane_from_signal2d(s)
        np.testing.assert_almost_equal(s_plane.data, s.data)

    def test_negative_values(self):
        data = np.ones((110, 100)) * -10
        s = hs.signals.Signal2D(data)
        s_plane = bst._get_linear_plane_from_signal2d(s)
        np.testing.assert_almost_equal(s_plane.data, s.data)

    def test_mask(self):
        data = np.ones((110, 200), dtype=np.float32)
        mask = np.zeros_like(data, dtype=bool)
        data[50, 51] = 10000
        mask[50, 51] = True
        s = hs.signals.Signal2D(data)
        plane_no_mask = bst._get_linear_plane_from_signal2d(s)
        plane_mask = bst._get_linear_plane_from_signal2d(s, mask=mask)
        assert plane_no_mask != approx(1.0)
        assert plane_mask == approx(1.0)

    def test_offest_and_scale(self):
        s, _ = hs.signals.Signal2D(np.meshgrid(range(100), range(110)))
        s.axes_manager[0].offset = -40
        s.axes_manager[1].offset = 50
        s.axes_manager[0].scale = -0.22
        s.axes_manager[1].scale = 5.2
        s_orig = s.deepcopy()
        mask = np.zeros_like(s.data, dtype=bool)
        s.data[50, 51] = 10000
        mask[50, 51] = True
        plane_mask = bst._get_linear_plane_from_signal2d(s, mask=mask)
        np.testing.assert_allclose(plane_mask, s_orig.data, atol=1e-6)

    def test_crop_signal(self):
        s, _ = hs.signals.Signal2D(np.meshgrid(range(100), range(110)))
        s.change_dtype("float32")
        s.axes_manager[0].offset = -40
        s.axes_manager[1].offset = 50
        s.axes_manager[0].scale = -0.22
        s.axes_manager[1].scale = 5.2
        s_crop = s.isig[10:-20, 21:-11]
        s_crop_orig = s_crop.deepcopy()
        s_crop.data[50, 51] = 10000
        mask = np.zeros_like(s_crop.data, dtype=bool)
        mask[50, 51] = True
        plane = bst._get_linear_plane_from_signal2d(s_crop, mask=mask)
        np.testing.assert_almost_equal(plane, s_crop_orig.data, decimal=6)

    def test_wrong_input_dimensions(self):
        s = hs.signals.Signal2D(np.ones((2, 10, 10)))
        with pytest.raises(ValueError):
            bst._get_linear_plane_from_signal2d(s)
        s = hs.signals.Signal2D(np.ones((2, 2, 10, 10)))
        with pytest.raises(ValueError):
            bst._get_linear_plane_from_signal2d(s)
        s = hs.signals.Signal1D(np.ones(10))
        with pytest.raises(ValueError):
            bst._get_linear_plane_from_signal2d(s)

    def test_wrong_mask_dimensions(self):
        s = hs.signals.Signal2D(np.ones((10, 10)))
        mask = np.zeros((11, 9), dtype=bool)
        with pytest.raises(ValueError):
            bst._get_linear_plane_from_signal2d(s, mask=mask)


class TestGetLimitsFromArray:
    def test_simple(self):
        data_array0 = np.array((5, 10))
        clim0 = bst._get_limits_from_array(data_array0, sigma=4)
        assert (data_array0.min(), data_array0.max()) == clim0
        data_array1 = np.array((5, -10))
        clim1 = bst._get_limits_from_array(data_array1, sigma=4)
        assert (data_array1.min(), data_array1.max()) == clim1
        data_array2 = np.array((-5, 10))
        clim2 = bst._get_limits_from_array(data_array2, sigma=4)
        assert (data_array2.min(), data_array2.max()) == clim2
        data_array3 = np.array((-5, -10))
        clim3 = bst._get_limits_from_array(data_array3, sigma=4)
        assert (data_array3.min(), data_array3.max()) == clim3

    def test_simple_sigma(self):
        data_array0 = np.array((5, 10))
        clim0 = bst._get_limits_from_array(data_array0, sigma=0)
        assert (data_array0.mean(), data_array0.mean()) == clim0
        data_array1 = np.array((5, -10))
        clim1 = bst._get_limits_from_array(data_array1, sigma=0)
        assert (data_array1.mean(), data_array1.mean()) == clim1
        data_array2 = np.array((-5, 10))
        clim2 = bst._get_limits_from_array(data_array2, sigma=0)
        assert (data_array2.mean(), data_array2.mean()) == clim2
        data_array3 = np.array((-5, -10))
        clim3 = bst._get_limits_from_array(data_array3, sigma=0)
        assert (data_array3.mean(), data_array3.mean()) == clim3

    def test_ignore_zeros(self):
        data_array0 = np.zeros(shape=(100, 100))
        value = 50
        data_array0[:, 70:80] = value
        clim0_0 = bst._get_limits_from_array(data_array0, ignore_zeros=True)
        assert (value, value) == clim0_0
        clim0_1 = bst._get_limits_from_array(data_array0, sigma=0)
        assert (data_array0.mean(), data_array0.mean()) == clim0_1
        clim0_2 = bst._get_limits_from_array(data_array0, sigma=1)
        assert (0.0, 20.0) == clim0_2

    def test_ignore_edges(self):
        data_array = np.ones(shape=(100, 100)) * 5000
        value = 50
        data_array[1:-1, 1:-1] = value
        clim0 = bst._get_limits_from_array(data_array, ignore_edges=True)
        assert (value, value) == clim0
        clim1 = bst._get_limits_from_array(data_array, ignore_edges=False)
        assert not ((value, value) == clim1)


class TestMakeBivariateHistogram:
    def test_make_bivariate_histogram(self):
        x, y = np.ones((100, 100)), np.ones((100, 100))
        bst._make_bivariate_histogram(
            x_position=x,
            y_position=y,
            histogram_range=None,
            masked=None,
            bins=200,
            spatial_std=3,
        )

    def test_single_x(self):
        size = 100
        x, y = np.ones(size), np.zeros(size)
        s = bst._make_bivariate_histogram(x, y)
        hist_iX = s.axes_manager[0].value2index(1.0)
        hist_iY = s.axes_manager[1].value2index(0.0)
        assert s.data[hist_iY, hist_iX] == size
        s.data[hist_iY, hist_iX] = 0
        assert not s.data.any()

    def test_single_negative_x(self):
        size = 100
        x, y = -np.ones(size), np.zeros(size)
        s = bst._make_bivariate_histogram(x, y)
        hist_iX = s.axes_manager[0].value2index(-1)
        hist_iY = s.axes_manager[1].value2index(0)
        assert s.data[hist_iY, hist_iX] == size
        s.data[hist_iY, hist_iX] = 0
        assert not s.data.any()

    def test_single_negative_x_y(self):
        size = 100
        x, y = -np.ones(size), np.ones(size)
        s = bst._make_bivariate_histogram(x, y)
        hist_iX = s.axes_manager[0].value2index(-1)
        hist_iY = s.axes_manager[1].value2index(1)
        assert s.data[hist_iY, hist_iX] == size
        s.data[hist_iY, hist_iX] = 0
        assert not s.data.any()
