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
import dask.array as da
from hyperspy.signals import Signal2D
import pyxem.data.dummy_data.dummy_data as dd
from pyxem.signals import BeamShift, LazyBeamShift, Diffraction2D


class TestMakeLinearPlane:
    def test_simple(self):
        data_x, data_y = np.meshgrid(
            np.arange(-50, 50, dtype=np.float32), np.arange(-256, 0, dtype=np.float32)
        )
        data = np.stack((data_y, data_x), -1)
        s = BeamShift(data)
        s.change_dtype("float32")
        s_orig = s.deepcopy()
        s.make_linear_plane()
        assert s.data == approx(s_orig.data, abs=1e-6)


class TestGetLinearPlane:
    def test_simple(self):
        data_x, data_y = np.meshgrid(
            np.arange(-50, 50, dtype=np.float32), np.arange(-256, 0, dtype=np.float32)
        )
        data = np.stack((data_y, data_x), -1)
        s = BeamShift(data)
        s.change_dtype("float32")
        s_lp = s.get_linear_plane()
        assert s_lp.data == approx(s.data, abs=1e-6)

    def test_mask(self):
        data_x, data_y = np.meshgrid(
            np.arange(-50, 50, dtype=np.float32), np.arange(-256, 0, dtype=np.float32)
        )
        data = np.stack((data_y, data_x), -1)
        mask = np.zeros_like(data[:, :, 0], dtype=bool)
        mask[45:50, 36:41] = True
        s_mask = Signal2D(mask)
        s = BeamShift(data)
        s.change_dtype("float32")
        s_orig = s.deepcopy()
        s.data[45:50, 36:41, 0] = 100000
        s.data[45:50, 36:41, 1] = -100000
        s_lp = s.get_linear_plane(mask=s_mask)
        assert s_lp.data == approx(s_orig.data, abs=1e-6)

    def test_constrain_magnitude_variance(self):
        p = [0.5] * 6  # Plane parameters
        x, y = np.meshgrid(np.arange(256), np.arange(256))
        base_plane_x = p[0] * x + p[1] * y + p[2]
        base_plane_y = p[3] * x + p[4] * y + p[5]

        base_plane = np.stack((base_plane_x, base_plane_y)).T
        data = base_plane.copy()

        shifts = np.zeros_like(data)
        shifts[:128, 128:] = (10, 10)
        shifts[:128, :128] = (-10, -10)
        shifts[128:, 128:] = (-10, 10)
        shifts[128:, :128] = (10, 10)
        data += shifts

        s = BeamShift(data)

        s_lp = s.get_linear_plane()
        assert not np.allclose(s_lp.data, base_plane.data, rtol=1e-7)

        s_lp = s.get_linear_plane(constrain_magnitude_variance=True)
        assert np.allclose(s_lp.data, base_plane.data, rtol=1e-7)

    def test_constrain_magnitude_variance_mask(self):
        p = [0.5] * 6  # Plane parameters
        x, y = np.meshgrid(np.arange(256), np.arange(256))
        base_plane_x = p[0] * x + p[1] * y + p[2]
        base_plane_y = p[3] * x + p[4] * y + p[5]

        base_plane = np.stack((base_plane_x, base_plane_y)).T
        data = base_plane.copy()

        shifts = np.zeros_like(data)
        shifts[:128, 128:] = (10, 10)
        shifts[:128, :128] = (-10, -10)
        shifts[128:, 128:] = (-10, 10)
        shifts[128:, :128] = (10, 10)
        shifts[:, -10:] = (9999, 1321)
        shifts[:, :10] = (2213, -9879)
        data += shifts

        mask = np.zeros((256, 256), dtype=bool)
        mask[:, -10:] = True
        mask[:, :10] = True

        s = BeamShift(data)

        s_lp = s.get_linear_plane(constrain_magnitude_variance=True)
        assert not np.allclose(s_lp.data, base_plane.data, rtol=1e-7)

        s_lp = s.get_linear_plane(constrain_magnitude_variance=True, mask=mask)
        assert np.allclose(s_lp.data, base_plane.data, rtol=1e-7)

        # Check wrong mask dimensions
        mask = mask[:255]
        with pytest.raises(ValueError):
            s_lp = s.get_linear_plane(constrain_magnitude_variance=True, mask=mask)

    def test_constrain_magnitude_variance_initial_values(self):
        p = [0.5] * 6  # Plane parameters
        x, y = np.meshgrid(np.arange(256), np.arange(256))
        base_plane_x = p[0] * x + p[1] * y + p[2]
        base_plane_y = p[3] * x + p[4] * y + p[5]

        base_plane = np.stack((base_plane_x, base_plane_y)).T
        data = base_plane.copy()

        shifts = np.zeros_like(data)
        shifts[:128, 128:] = (10, 10)
        shifts[:128, :128] = (-10, -10)
        shifts[128:, :] = (-10, 10)
        data += shifts

        s = BeamShift(data)

        # Plane fitting does poorly here, likely due to not enough different domains
        s_lp = s.get_linear_plane(constrain_magnitude_variance=True)
        assert not np.allclose(s_lp.data, base_plane.data, rtol=1e-7)

        # Varying the initial values around can help find different planes
        initial_values = [1.0] * 6
        s_lp = s.get_linear_plane(
            constrain_magnitude_variance=True, initial_values=initial_values
        )
        assert np.allclose(s_lp.data, base_plane.data, rtol=1e-7)

    def test_lazy_input_error(self):
        s = LazyBeamShift(da.zeros((50, 40, 2)))
        with pytest.raises(ValueError):
            s.get_linear_plane()
        s.compute()
        s.get_linear_plane()

    @pytest.mark.parametrize("constrain_magnitude_variance", [False, True])
    def test_wrong_navigation_input_1d_error(self, constrain_magnitude_variance):
        s = BeamShift(np.zeros((50, 2)))
        with pytest.raises(NotImplementedError):
            s.get_linear_plane(
                constrain_magnitude_variance=constrain_magnitude_variance
            )

    @pytest.mark.parametrize("constrain_magnitude_variance", [False, True])
    def test_wrong_navigation_input_3d_error(self, constrain_magnitude_variance):
        s = BeamShift(np.zeros((50, 40, 30, 2)))
        with pytest.raises(NotImplementedError):
            s.get_linear_plane(
                constrain_magnitude_variance=constrain_magnitude_variance
            )
        s1 = s.inav[:, :, 10]
        s1.get_linear_plane(constrain_magnitude_variance=constrain_magnitude_variance)

    def test_wrong_input_fit_corners_and_mask(self):
        s = BeamShift(np.zeros((5, 5, 2)))
        with pytest.raises(ValueError):
            s.get_linear_plane(fit_corners=0.05, mask=np.ones((5, 5)))

    def test_non_bool_mask(self):
        s = BeamShift(np.zeros((50, 40, 2)))
        mask = np.zeros((50, 40), dtype=np.int32)
        s_mask = Signal2D(mask)
        with pytest.raises(ValueError):
            s.get_linear_plane(mask=s_mask)
        s_mask.change_dtype(bool)
        s.get_linear_plane(mask=s_mask)


class TestBeamShiftFitCorners:
    def test_fit_corners_flat(self):
        data0 = np.ones(shape=(64, 64, 2))
        s0 = BeamShift(data0)
        s0_linear_plane = s0.get_linear_plane(fit_corners=0.05)
        s0_corr = s0 - s0_linear_plane
        assert (s0.data == data0).all()
        assert s0_corr.data == approx(0.0, abs=1e-7)

    def test_fit_corners_x_y(self):
        array_x, array_y = np.meshgrid(range(64), range(64))
        data_x = np.dstack((array_x, array_x)).astype("float64")
        data_y = np.dstack((array_y, array_y)).astype("float64")
        s_x = BeamShift(data_x)
        s_y = BeamShift(data_y)

        s_x_corr = s_x - s_x.get_linear_plane(fit_corners=0.05)
        s_y_corr = s_y - s_y.get_linear_plane(fit_corners=0.05)
        assert s_x_corr.data == approx(0.0, abs=1e-6)
        assert s_y_corr.data == approx(0.0, abs=1e-6)

        data_xy = np.dstack((array_x, array_y)).astype("float64")
        data_yx = np.dstack((array_y, array_x)).astype("float64")
        s_xy = BeamShift(data_xy)
        s_yx = BeamShift(data_yx)
        s_xy_corr = s_xy - s_xy.get_linear_plane(fit_corners=0.05)
        s_yx_corr = s_yx - s_yx.get_linear_plane(fit_corners=0.05)
        assert s_xy_corr.data == approx(0.0, abs=1e-6)
        assert s_yx_corr.data == approx(0.0, abs=1e-6)

        data_tilt = np.dstack((array_x + array_y, np.fliplr(array_x) + array_y))
        s_tilt = BeamShift(data_tilt.astype("float64"))
        s_tilt_corr = s_tilt - s_tilt.get_linear_plane(fit_corners=0.05)
        assert s_tilt_corr.data == approx(0.0, abs=1e-6)

    def test_fit_corners_one_large_value(self):
        array_x, array_y = np.meshgrid(range(64), range(64))
        array_x, array_y = array_x.astype("float64"), array_y.astype("float64")
        data = np.dstack((array_x + array_y, np.fliplr(array_x) + array_y))
        add_number_slice = np.s_[20:30, 30:40, :]
        data[add_number_slice] += 1000
        s = BeamShift(data)
        s_corr = s - s.get_linear_plane(fit_corners=0.05)
        s_corr.data[add_number_slice] -= 1000
        assert s_corr.data == approx(0.0, abs=1e-8)

    def test_large_values_not_in_corners(self):
        s = BeamShift(np.zeros((100, 100, 2)))
        s.isig[0].data[:5, :5], s.isig[0].data[:5, -5:] = 10, 10
        s.isig[0].data[-5:, :5], s.isig[0].data[-5:, -5:] = 10, 10
        s.isig[1].data[:5, :5], s.isig[1].data[:5, -5:] = 30, 30
        s.isig[1].data[-5:, :5], s.isig[1].data[-5:, -5:] = 30, 30

        cross_array = np.zeros((100, 100, 2))
        cross_array[30:70, :, :], cross_array[:, 30:70, :] = 1000, 1000
        s.data = s.data + cross_array

        s1 = s - s.get_linear_plane(fit_corners=0.05)
        s1.data = s1.data - cross_array

        assert s1.isig[0].data[5:95, :] == approx(np.ones((90, 100)) * -10)
        assert s1.isig[0].data[:, 5:95] == approx(np.ones((100, 90)) * -10)
        assert s1.isig[0].data[5:95, :] == approx(np.ones((90, 100)) * -10)
        assert s1.isig[1].data[5:95, :] == approx(np.ones((90, 100)) * -30)
        assert s1.isig[1].data[:, 5:95] == approx(np.ones((100, 90)) * -30)
        assert s1.isig[1].data[5:95, :] == approx(np.ones((90, 100)) * -30)
        assert s1.inav[:5, :5].data == approx(np.zeros((5, 5, 2)), abs=1e-7)
        assert s1.inav[-5:, :5].data == approx(np.zeros((5, 5, 2)), abs=1e-7)
        assert s1.inav[:5, -5:].data == approx(np.zeros((5, 5, 2)), abs=1e-7)
        assert s1.inav[-5:, -5:].data == approx(np.zeros((5, 5, 2)), abs=1e-7)

    def test_cropped_beam_shift_signal(self):
        s = BeamShift(np.random.random((200, 200, 2)))
        s_crop = s.inav[50:150, 50:150]
        s_crop_corr = s_crop - s_crop.get_linear_plane(fit_corners=0.05)
        assert s_crop_corr.axes_manager.shape == s_crop.axes_manager.shape

    def test_different_fit_corners_value(self):
        data = np.zeros((100, 100, 2))
        data[10:-10, 10:-10] = 100000
        s = BeamShift(data)
        s_lp_05 = s.get_linear_plane(fit_corners=0.05)
        s_lp_15 = s.get_linear_plane(fit_corners=0.15)
        assert s_lp_05.data == approx(0.0, abs=1e-8)
        assert s_lp_15.data != approx(0.0, abs=1e-8)


class TestFullDirectBeamCentering:
    def setup_method(self):
        data = np.zeros((3, 3, 16, 16), dtype=np.float32)
        data[0, 0, 6, 7] = 10
        data[0, 1, 6, 8] = 10
        data[0, 2, 6, 9] = 10
        data[1, 0, 7, 7] = 10
        data[1, 1, 7, 8] = 10
        data[1, 2, 7, 9] = 10
        data[2, 0, 8, 7] = 10
        data[2, 1, 8, 8] = 10
        data[2, 2, 8, 9] = 10
        s = Diffraction2D(data)
        s.axes_manager[0].scale = 0.5
        s.axes_manager[1].scale = 1.5
        s.axes_manager[2].scale = 5
        s.axes_manager[3].scale = 5
        s.axes_manager[0].offset = -10
        s.axes_manager[1].offset = 20
        s.axes_manager[2].offset = 30
        s.axes_manager[3].offset = 30
        self.s = s

    def test_simple(self):
        s = self.s
        s_beam_shift = s.get_direct_beam_position(method="blur", sigma=1)
        s.center_direct_beam(shifts=s_beam_shift)
        assert (s.data[:, :, 8, 8] == 10).all()
        s.data[:, :, 8, 8] = 0
        assert not s.data.any()

    def test_simple_1d(self):
        s = self.s.inav[0]
        s_beam_shift = s.get_direct_beam_position(method="blur", sigma=1)
        s.center_direct_beam(shifts=s_beam_shift)
        assert (s.data[:, 8, 8] == 10).all()
        s.data[:, 8, 8] = 0
        assert not s.data.any()

    def test_simple_lazy(self):
        s = self.s
        s = s.as_lazy()
        s_beam_shift = s.get_direct_beam_position(method="blur", sigma=1)
        s.center_direct_beam(shifts=s_beam_shift)
        s.compute()
        assert (s.data[:, :, 8, 8] == 10).all()
        s.data[:, :, 8, 8] = 0
        assert not s.data.any()

    def test_mask(self):
        s = self.s
        s.data[1, 2, 2, 3] = 1000
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 2] = True
        s_mask = Signal2D(mask)
        s_beam_shift = s.get_direct_beam_position(method="blur", sigma=1)
        s_linear_plane = s_beam_shift.get_linear_plane(mask=s_mask)
        s.center_direct_beam(shifts=s_linear_plane)
        assert s.data[:, :, 8, 8] == approx(10, abs=1e-5)
        s.data[:, :, 8, 8] = 0
        s.data[1, 2, 3, 2] = 0
        assert s.data == approx(0.0, abs=1e-4)

    def test_mask_lazy(self):
        s = self.s
        s.data[1, 2, 2, 3] = 1000
        s = self.s.as_lazy()
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 2] = True
        s_mask = Signal2D(mask)
        s_beam_shift = s.get_direct_beam_position(method="blur", sigma=1)
        s_beam_shift.compute()
        s_linear_plane = s_beam_shift.get_linear_plane(mask=s_mask)
        s.center_direct_beam(shifts=s_linear_plane)
        s.compute()
        assert s.data[:, :, 8, 8] == approx(10, abs=1e-5)
        s.data[:, :, 8, 8] = 0
        s.data[1, 2, 3, 2] = 0
        assert s.data == approx(0.0, abs=1e-4)


class TestGetMagnitudeSignal:
    def test_get_magnitude_signal_zeros(self):
        s = BeamShift(np.zeros((100, 100, 2)))
        s_magnitude = s.get_magnitude_signal()
        assert (s_magnitude.data == 0).all()

    def test_get_magnitude_signal_non_zero(self):
        data0 = np.zeros((100, 100, 2))
        data0[:, :, 0] = 1.5
        s0 = BeamShift(data0)
        s0_magnitude = s0.get_magnitude_signal()
        assert (s0_magnitude.data == 1.5).all()

        data1 = np.zeros((100, 100, 2))
        data1[:, :, 1] = 1.5
        s1 = BeamShift(data1)
        s1_magnitude = s1.get_magnitude_signal()
        assert (s1_magnitude.data == 1.5).all()

        data2 = np.zeros((100, 100, 2))
        data2[:, :, :] = 1.5
        s2 = BeamShift(data2)
        s2_magnitude = s2.get_magnitude_signal()
        assert s2_magnitude.data == approx(2.12132, abs=1e-5)

    def test_get_magnitude_single_value_non_zero(self):
        data = np.zeros((100, 50, 2))
        data[10, 15, 0] = 50
        s = BeamShift(data)
        s_magnitude = s.get_magnitude_signal(autolim=False)
        assert s_magnitude.data[10, 15] == 50
        s_magnitude.data[10, 15] = 0
        assert (s_magnitude.data == 0).all()

    def test_wrong_input_autolim_mangnitude_limits(self):
        s = BeamShift(np.zeros((10, 10, 2)))
        with pytest.raises(ValueError):
            s.get_magnitude_signal(autolim=True, magnitude_limits=(0, 30))


class TestGetPhaseSignal:
    def test_get_phase_signal(self):
        s = BeamShift(np.zeros((100, 100, 2)))
        s_p = s.get_phase_signal()
        assert s_p.axes_manager.shape == (100, 100)
        assert s_p.axes_manager.navigation_dimension == 0
        assert s_p.axes_manager.signal_dimension == 2

    def test_get_phase_signal_with_rotation(self):
        s = BeamShift(np.random.random(size=(100, 100, 2)))
        s.get_phase_signal(rotation=45)

    @pytest.mark.xfail(reason="Does not work with 1D navigation signals")
    def test_1d_get_phase_signal(self):
        s = BeamShift(np.zeros((100, 2)))
        s.get_phase_signal()


class TestGetColorSignal:
    def test_get_color_signal(self):
        s = BeamShift(np.random.random(size=(10, 10, 2)))
        s_color = s.get_color_signal()
        s_color.axes_manager.shape == (10, 10)


class TestGetMagnitudePhaseSignal:
    def test_get_magnitude_phase_signal(self):
        data_random = np.random.random(size=(64, 64, 2))
        s_random = BeamShift(data_random)
        s_random.get_magnitude_phase_signal()
        s_random.get_magnitude_phase_signal(rotation=45)
        s_random.get_magnitude_phase_signal(autolim=False, magnitude_limits=(0, 30))

    def test_get_magnitude_phase_signal_errors(self):
        data_random = np.random.random(size=(64, 64, 2))
        s_random = BeamShift(data_random)
        with pytest.raises(ValueError):
            s_random.get_magnitude_phase_signal(autolim=True, magnitude_limits=(0, 30))

    def test_get_magnitude_phase_signal_zeros(self):
        s = BeamShift(np.zeros((100, 100, 2)))
        s_color = s.get_magnitude_phase_signal()
        assert (s_color.data["R"] == 0).all()
        assert (s_color.data["G"] == 0).all()
        assert (s_color.data["B"] == 0).all()


class TestPlot:
    def test_simple_plot(self):
        s = BeamShift(np.zeros((10, 10, 2)))
        s.plot()

    def test_lazy_error(self):
        s = BeamShift(np.zeros((10, 10, 2))).as_lazy()
        with pytest.raises(ValueError):
            s.plot()


class TestGetBivariateHistogram:
    def test_get_bivariate_histogram_1d(self):
        s = BeamShift(np.random.random((10, 2)))
        s.get_bivariate_histogram()

    def test_get_bivariate_histogram_2d(self):
        s_random = BeamShift(np.random.random((64, 64, 2)))
        s_random.get_bivariate_histogram()

    def test_get_bivariate_histogram_3d(self):
        s_random = BeamShift(np.random.random((20, 64, 64, 2)))
        s_random.get_bivariate_histogram()

    def test_masked_get_bivariate_histogram(self):
        s = BeamShift(np.zeros((5, 5, 2)))
        value = 3
        s.data[0, 0, 0] = value
        s_hist = s.get_bivariate_histogram(bins=10, histogram_range=(-5, 5))
        assert s_hist.isig[3.0, 0.0].data[0] == 1.0
        assert s_hist.isig[0.0, 0.0].data[0] == (5 * 5) - 1

        masked = np.zeros((11, 11), dtype=bool)
        masked[0, 0] = True
        s_hist = s.get_bivariate_histogram(
            bins=10, histogram_range=(-5, 5), masked=masked
        )
        assert s_hist.isig[3.0, 0.0].data[0] == 0.0
        assert s_hist.isig[0.0, 0.0].data[0] == (5 * 5) - 1


class TestRotateBeamShifts:
    def setup_method(self):
        data = np.zeros((90, 70, 2))
        data[:, :, 0] = 1
        self.s = BeamShift(data)

    def test_clockwise_90_degrees(self):
        s = self.s
        s_rot = s.rotate_beam_shifts(90)
        data_x, data_y = s_rot.isig[0].data, s_rot.isig[1].data
        np.testing.assert_almost_equal(data_x, np.zeros_like(data_x))
        np.testing.assert_almost_equal(data_y, np.ones_like(data_y))

    def test_counterclockwise_90_degrees(self):
        s = self.s
        s_rot = s.rotate_beam_shifts(-90)
        data_x, data_y = s_rot.isig[0].data, s_rot.isig[1].data
        np.testing.assert_almost_equal(data_x, np.zeros_like(data_x))
        np.testing.assert_almost_equal(data_y, -np.ones_like(data_y))

    def test_180_degrees(self):
        s = self.s
        s_rot = s.rotate_beam_shifts(180)
        data_x, data_y = s_rot.isig[0].data, s_rot.isig[1].data
        np.testing.assert_almost_equal(data_x, -np.ones_like(data_x))
        np.testing.assert_almost_equal(data_y, np.zeros_like(data_y))

    def test_clockwise_45_degrees(self):
        s = self.s
        sin_rad = np.sin(np.deg2rad(45))
        s_rot = s.rotate_beam_shifts(45)
        data_x, data_y = s_rot.isig[0].data, s_rot.isig[1].data
        np.testing.assert_almost_equal(data_x, np.ones_like(data_x) * sin_rad)
        np.testing.assert_almost_equal(data_y, np.ones_like(data_y) * sin_rad)

    def test_counterclockwise_45_degrees(self):
        s = self.s
        sin_rad = np.sin(np.deg2rad(45))
        s_rot = s.rotate_beam_shifts(-45)
        data_x, data_y = s_rot.isig[0].data, s_rot.isig[1].data
        np.testing.assert_almost_equal(data_x, np.ones_like(data_x) * sin_rad)
        np.testing.assert_almost_equal(data_y, -np.ones_like(data_y) * sin_rad)


class TestRotateScanDimensions:
    def test_clockwise(self):
        s = dd.get_simple_beam_shift_signal()
        s_rot = s.rotate_scan_dimensions(1)
        assert not (s_rot.data[0, 0:10, 0] == 0).all()
        assert (s_rot.data[0, -10:, 0] == 0).all()
        assert not (s_rot.data[-10:, 0, 0] == 0).all()
        assert (s_rot.data[0:10, 0, 0] == 0).all()

    def test_counterclockwise(self):
        s = dd.get_simple_beam_shift_signal()
        s_rot = s.rotate_scan_dimensions(-1)
        assert (s_rot.data[0, 0:10, 0] == 0).all()
        assert not (s_rot.data[0, -10:, 0] == 0).all()
        assert (s_rot.data[-10:, 0, 0] == 0).all()
        assert not (s_rot.data[0:10, 0, 0] == 0).all()


class TestPhaseRetrieval:
    def setup_method(self):
        # construct the surface, two point with Gaussian distribution
        coords = np.linspace(-20, 10, num=512)
        x, y = np.meshgrid(coords, coords)
        surface = np.exp(-(x**2 + y**2) / 2) + np.exp(
            -((x - 2) ** 2 + (y + 4) ** 2) / 2
        )

        # x and y phase gradient of the Gaussians, analytical form
        dx = x * (-np.exp(-(x**2) / 2 - y**2 / 2)) + (x - 2) * -np.exp(
            (-0.5 * (x - 2) ** 2 - 0.5 * (y + 4) ** 2)
        )
        dy = y * (-np.exp(-(x**2) / 2 - y**2 / 2)) + (y + 4) * -np.exp(
            (-0.5 * (x - 2) ** 2 - 0.5 * (y + 4) ** 2)
        )

        data = np.empty((512, 512, 2))
        data[:, :, 0] = dx
        data[:, :, 1] = dy
        s = BeamShift(data)
        s.axes_manager.navigation_axes[0].axis = coords
        s.axes_manager.navigation_axes[1].axis = coords
        self.s = s

        # normalise for comparison later
        surface -= surface.mean()
        surface /= surface.std()
        self.surface = surface

    @pytest.mark.parametrize("method", ["kottler", "arnison", "frankot"])
    def test_kottler(self, method):
        s_recon = self.s.phase_retrieval(method=method)
        recon = s_recon.data
        recon -= recon.mean()
        recon /= recon.std()

        assert np.isclose(self.surface, recon, atol=1e-3).all()

    @pytest.mark.parametrize("method", ["kottler", "arnison", "frankot"])
    def test_mirroring(self, method):
        s_recon = self.s.phase_retrieval(method, mirroring=True)
        recon = s_recon.data
        recon -= recon.mean()
        recon /= recon.std()

        assert np.isclose(self.surface, recon, atol=1e-3).all()

    @pytest.mark.parametrize("method", ["kottler", "arnison", "frankot"])
    def test_mirror_flip(self, method):
        s_noflip = self.s.phase_retrieval(method, mirroring=True, mirror_flip=False)
        s_flip = self.s.phase_retrieval(method, mirroring=True, mirror_flip=True)
        noflip = s_noflip.data
        noflip -= noflip.mean()
        noflip /= noflip.std()
        flip = s_flip.data
        flip -= flip.mean()
        flip /= flip.std()

        noflip_sum_diff = np.abs(self.surface - noflip).sum()
        flip_sum_diff = np.abs(self.surface - flip).sum()

        assert noflip_sum_diff != flip_sum_diff

    @pytest.mark.xfail(reason="invalid_method")
    def test_unavailable_method(self):
        self.s.phase_retrieval("magic!")
