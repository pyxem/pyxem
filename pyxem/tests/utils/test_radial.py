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
from pytest import approx
import numpy as np

from pyxem.signals import Diffraction2D
import pyxem.utils.radial_utils as ra
from pyxem.dummy_data import make_diffraction_test_data as mdtd


@pytest.mark.slow
class TestRadialModule:
    def test_centre_comparison(self):
        s = Diffraction2D(np.ones((20, 20)))
        s_list = ra._centre_comparison(s, 1, 1)
        assert len(s_list) == 9

        s1 = Diffraction2D(np.ones((5, 20, 20)))
        with pytest.raises(ValueError):
            ra._centre_comparison(s1, 1, 1)

        s2 = Diffraction2D(np.ones((30, 30)))
        s2_list = ra._centre_comparison(s2, 1, 1, angleN=20)
        for temp_s in s2_list:
            assert temp_s.axes_manager.navigation_shape == (20,)

        s3 = Diffraction2D(np.ones((40, 40)))
        s3_list = ra._centre_comparison(s3, 1, 1, angleN=10, crop_radial_signal=(3, 8))
        for temp_s in s3_list:
            assert temp_s.axes_manager.signal_shape == (5,)
            assert temp_s.axes_manager.navigation_shape == (10,)

    def test_get_coordinate_of_min(self):
        array = np.ones((10, 10)) * 10
        x, y = 7, 5
        array[y, x] = 1  # In NumPy the order is [y, x]
        s = Diffraction2D(array)
        s.axes_manager[0].offset = 55
        s.axes_manager[1].offset = 50
        s.axes_manager[0].scale = 0.5
        s.axes_manager[1].scale = 0.4
        min_pos = ra.get_coordinate_of_min(s)
        # min_pos[0] (x) should be at (7*0.5) + 55 = 58.5
        # min_pos[1] (y) should be at (5*0.4) + 50 = 52.
        assert min_pos == (58.5, 52.0)

    def test_get_optimal_centre_position(self):
        x0, y0 = 300.0, 300.0
        test_data = mdtd.MakeTestData(size_x=600, size_y=600, default=False)
        test_data.add_ring(x0=x0, y0=y0, r=200, intensity=10, lw_pix=2)
        s = test_data.signal
        s.axes_manager[0].offset = -301.0
        s.axes_manager[1].offset = -301.0
        s_centre_position = ra.get_optimal_centre_position(
            s, radial_signal_span=(180, 210), steps=2, step_size=1
        )
        x, y = ra.get_coordinate_of_min(s_centre_position)
        assert (x0 - 0.5) <= x and x <= (x0 + 0.5)
        assert (x0 - 0.5) <= x and x <= (x0 + 0.5)

    def test_get_optimal_centre_position_off_centre_non_square(self):
        test_data = mdtd.MakeTestData(
            size_x=300, size_y=400, default=False, blur=True, downscale=False
        )
        x0, y0 = 150, 170
        test_data.add_ring(x0=x0, y0=y0, r=100, intensity=10, lw_pix=1)
        s = test_data.signal
        s.axes_manager[0].offset = -x0 - 2
        s.axes_manager[1].offset = -y0 - 3
        s_c = ra.get_optimal_centre_position(
            test_data.signal,
            radial_signal_span=(90, 110),
            steps=3,
            step_size=1.0,
            angleN=8,
        )
        min_pos0 = ra.get_coordinate_of_min(s_c)
        s.axes_manager[0].offset = -x0 + 2
        s.axes_manager[1].offset = -y0 - 3
        s_c = ra.get_optimal_centre_position(
            test_data.signal,
            radial_signal_span=(90, 110),
            steps=3,
            step_size=1.0,
            angleN=8,
        )
        min_pos1 = ra.get_coordinate_of_min(s_c)
        s.axes_manager[0].offset = -x0 - 2
        s.axes_manager[1].offset = -y0 + 3
        s_c = ra.get_optimal_centre_position(
            test_data.signal,
            radial_signal_span=(90, 110),
            steps=3,
            step_size=1.0,
            angleN=8,
        )
        min_pos2 = ra.get_coordinate_of_min(s_c)
        s.axes_manager[0].offset = -x0 + 2
        s.axes_manager[1].offset = -y0 + 3
        s_c = ra.get_optimal_centre_position(
            test_data.signal,
            radial_signal_span=(90, 110),
            steps=3,
            step_size=1.0,
            angleN=8,
        )
        min_pos3 = ra.get_coordinate_of_min(s_c)

        for min_pos in [min_pos0, min_pos1, min_pos2, min_pos3]:
            assert approx(min_pos[0]) == x0
            assert approx(min_pos[1]) == y0

    def test_get_radius_vs_angle(self):
        test_data = mdtd.MakeTestData(
            101, 121, blur=False, downscale=False, default=False
        )
        x, y, r0, r1 = 51, 55, 21, 35
        test_data.add_ring(x, y, r0)
        test_data.add_ring(x, y, r1)
        s = test_data.signal
        s.axes_manager.signal_axes[0].offset = -x
        s.axes_manager.signal_axes[1].offset = -y
        s_ar0 = ra.get_radius_vs_angle(
            s, radial_signal_span=(15, 25), show_progressbar=False
        )
        s_ar1 = ra.get_radius_vs_angle(
            s, radial_signal_span=(28, 40), show_progressbar=False
        )
        np.testing.assert_allclose(s_ar0.data, r0, 3)
        np.testing.assert_allclose(s_ar1.data, r1, 3)

    def test_refine_signal_centre_position(self):
        test_data = mdtd.MakeTestData(
            70, 70, blur=False, downscale=False, default=False
        )
        x, y, r = 35, 35, 15
        test_data.add_ring(x, y, r)
        s = test_data.signal
        s.axes_manager.signal_axes[0].offset = -x + 1
        s.axes_manager.signal_axes[1].offset = -y - 1
        ra.refine_signal_centre_position(s, (10.0, 20.0), angleN=6)
        assert s.axes_manager.signal_axes[0].offset == -x
        assert s.axes_manager.signal_axes[1].offset == -y


@pytest.mark.slow
class TestGetAngleImageComparison:
    def setup_method(self):
        self.r0, self.r1 = 10, 20
        test_data0 = mdtd.MakeTestData(100, 100)
        test_data0.add_ring(50, 50, self.r0)
        test_data1 = mdtd.MakeTestData(100, 100)
        test_data1.add_ring(50, 50, self.r1)
        s0, s1 = test_data0.signal, test_data1.signal
        s0.axes_manager[0].offset, s0.axes_manager[1].offset = -50, -50
        s1.axes_manager[0].offset, s1.axes_manager[1].offset = -50, -50
        self.s0, self.s1 = s0, s1

    def test_different_angleN(self):
        s0, s1 = self.s0, self.s1
        for i in range(1, 10):
            ra.get_angle_image_comparison(s0, s1, angleN=i)

    def test_correct_radius(self):
        s0, s1, r0, r1 = self.s0, self.s1, self.r0, self.r1
        s = ra.get_angle_image_comparison(s0, s1, angleN=2)
        assert s.axes_manager.signal_shape == (100, 100)
        # Check that radius is correct by getting a line profile
        s_top = s.isig[0.0, :0.0]
        s_bot = s.isig[0.0, 0.0:]
        argmax0 = s_top.data.argmax()
        argmax1 = s_bot.data.argmax()
        assert abs(s_top.axes_manager[0].index2value(argmax0)) == r0
        assert abs(s_bot.axes_manager[0].index2value(argmax1)) == r1

    def test_different_signal_size(self):
        s0 = mdtd.MakeTestData(100, 100).signal
        s1 = mdtd.MakeTestData(100, 150).signal
        with pytest.raises(ValueError):
            ra.get_angle_image_comparison(s0, s1)

    def test_mask(self):
        s0, s1 = self.s0, self.s1
        s_no_mask = ra.get_angle_image_comparison(s0, s1)
        s_mask = ra.get_angle_image_comparison(s0, s1, mask_radius=40)
        assert s_no_mask.data.sum() != 0.0
        assert s_mask.data.sum() == 0.0


@pytest.mark.slow
class TestFitEllipse:
    def setup_method(self):
        axis1, axis2 = 40, 70
        s = Diffraction2D(np.zeros((200, 220)))
        s.axes_manager[0].offset, s.axes_manager[1].offset = -100, -110
        xx, yy = np.meshgrid(s.axes_manager[0].axis, s.axes_manager[1].axis)
        ellipse_ring = mdtd._get_elliptical_ring(
            xx, yy, 0, 0, axis1, axis2, 0.8, lw_r=1
        )
        s.data += ellipse_ring
        self.s = s
        self.axis1, self.axis2 = axis1, axis2

    def test_find_parameters(self):
        axis1, axis2 = self.axis1, self.axis2
        s = self.s
        s_ra = ra.get_radius_vs_angle(s, (30.0, 80.0), angleN=20)
        x, y = ra._get_xy_points_from_radius_angle_plot(s_ra)
        ellipse_parameters = ra._fit_ellipse_to_xy_points(x, y)
        xC, yC, semi_len0, semi_len1, rot, eccen = ra._get_ellipse_parameters(
            ellipse_parameters
        )
        assert approx(xC, abs=0.1) == 0.0
        assert approx(yC, abs=0.1) == 0.0
        assert approx(semi_len0, abs=1.0) == axis2
        assert approx(semi_len1, abs=1.0) == axis1

    def test_get_signal_with_markers(self):
        s = self.s
        s_ra = ra.get_radius_vs_angle(s, (30.0, 80.0), angleN=20)
        x, y = ra._get_xy_points_from_radius_angle_plot(s_ra)
        ellipse_parameters = ra._fit_ellipse_to_xy_points(x, y)
        ra._get_marker_list(ellipse_parameters, x_list=x, y_list=y)

    def test_fit_single_ellipse_to_signal(self):
        s = Diffraction2D(np.zeros((200, 220)))
        s.axes_manager[0].offset, s.axes_manager[1].offset = -100, -110
        xx, yy = np.meshgrid(s.axes_manager[0].axis, s.axes_manager[1].axis)
        ellipse_ring = mdtd._get_elliptical_ring(xx, yy, 0, 0, 60, 60, 0.8, lw_r=1)
        s.data += ellipse_ring
        output = ra.fit_single_ellipse_to_signal(
            s, (50, 70), angleN=10, show_progressbar=False
        )
        output[0].plot()
        assert approx(output[1], abs=0.01) == 0.0
        assert approx(output[2], abs=0.01) == 0.0
        assert approx(output[3], abs=0.6) == 60
        assert approx(output[4], abs=0.6) == 60
        assert approx(output[6], abs=0.000001) == 1.0

    def test_fit_single_ellipse_to_signal_rotation(self):
        rot_list = [
            -np.pi / 16,
            -np.pi / 8,
            -np.pi / 4,
            -np.pi / 2,
            -0.1,
            0.1,
            np.pi / 16,
            np.pi / 8,
            np.pi / 4,
            np.pi / 2,
            np.pi + 0.1,
            np.pi * 2 + 0.1,
            np.pi * 2.5,
            np.pi * 3 + 0.1,
            np.pi * 3.2,
        ]
        for rot in rot_list:
            s = Diffraction2D(np.zeros((200, 200)))
            s.axes_manager[0].offset, s.axes_manager[1].offset = -100, -100
            xx, yy = np.meshgrid(s.axes_manager[0].axis, s.axes_manager[1].axis)
            s.data += mdtd._get_elliptical_ring(xx, yy, 0, 0, 70, 60, rot, lw_r=1)
            output = ra.fit_single_ellipse_to_signal(
                s, (50, 80), angleN=10, show_progressbar=False
            )
            output_rot = output[5] % np.pi
            assert approx(output_rot, abs=0.1) == (rot % np.pi)
        for rot in rot_list:
            s = Diffraction2D(np.zeros((200, 200)))
            s.axes_manager[0].offset, s.axes_manager[1].offset = -100, -100
            xx, yy = np.meshgrid(s.axes_manager[0].axis, s.axes_manager[1].axis)
            s.data += mdtd._get_elliptical_ring(xx, yy, 0, 0, 60, 70, rot, lw_r=1)
            output = ra.fit_single_ellipse_to_signal(
                s, (50, 80), angleN=10, show_progressbar=False
            )
            output_rot = (output[5] + np.pi / 2) % np.pi
            assert approx(output_rot, abs=0.1) == (rot % np.pi)

    def test_fit_ellipses_to_signal(self):
        s = Diffraction2D(np.zeros((200, 220)))
        s.axes_manager[0].offset, s.axes_manager[1].offset = -100, -110
        xx, yy = np.meshgrid(s.axes_manager[0].axis, s.axes_manager[1].axis)
        ellipse_ring0 = mdtd._get_elliptical_ring(xx, yy, 2, -1, 60, 60, 0.8)
        ellipse_ring1 = mdtd._get_elliptical_ring(xx, yy, 1, -2, 80, 80, 0.8)
        s.data += ellipse_ring0
        s.data += ellipse_ring1
        output0 = ra.fit_ellipses_to_signal(
            s, [(50, 70), (70, 95)], angleN=20, show_progressbar=False
        )
        output0[0].plot()
        output1 = ra.fit_ellipses_to_signal(
            s, [(50, 70), (70, 95)], angleN=[20, 30], show_progressbar=False
        )
        output1[0].plot()
        with pytest.raises(ValueError):
            ra.fit_ellipses_to_signal(
                s,
                [(50, 70), (70, 95), (80, 105)],
                angleN=[20, 30],
                show_progressbar=False,
            )

    def test_private_functionality_uncovered_above(self):
        from pyxem.utils.radial_utils import _get_ellipse_parameters

        g_a_greater_than_c_b_zero = np.asarray([2, 0, 1, 0, 0, 0])
        g_a_less_than_c_b_zero = np.asarray([1, 0, 2, 0, 0, 0])
        for g in [g_a_greater_than_c_b_zero, g_a_less_than_c_b_zero]:
            _ = _get_ellipse_parameters(g)


@pytest.mark.slow
class TestHolzCalibration:
    def test_get_holz_angle(self):
        wavelength = 2.51 / 1000
        lattice_parameter = 0.3905 * 2 ** 0.5
        angle = ra._get_holz_angle(wavelength, lattice_parameter)
        assert approx(95.37805 / 1000) == angle

    def test_scattering_angle_to_lattice_parameter(self):
        wavelength = 2.51 / 1000
        angle = 95.37805 / 1000
        lattice_size = ra._scattering_angle_to_lattice_parameter(wavelength, angle)
        assert approx(0.55225047) == lattice_size
