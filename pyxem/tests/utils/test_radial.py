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

from pyxem.signals import Diffraction2D
import pyxem.utils.radial_utils as ra
from pyxem.data.dummy_data import make_diffraction_test_data as mdtd


class TestRadialModule:
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

    def test_private_functionality_uncovered_above(self):
        from pyxem.utils.radial_utils import _get_ellipse_parameters

        g_a_greater_than_c_b_zero = np.asarray([2, 0, 1, 0, 0, 0])
        g_a_less_than_c_b_zero = np.asarray([1, 0, 2, 0, 0, 0])
        for g in [g_a_greater_than_c_b_zero, g_a_less_than_c_b_zero]:
            _ = _get_ellipse_parameters(g)


class TestHolzCalibration:
    def test_get_holz_angle(self):
        wavelength = 2.51 / 1000
        lattice_parameter = 0.3905 * 2**0.5
        angle = ra._get_holz_angle(wavelength, lattice_parameter)
        assert approx(95.37805 / 1000) == angle

    def test_scattering_angle_to_lattice_parameter(self):
        wavelength = 2.51 / 1000
        angle = 95.37805 / 1000
        lattice_size = ra._scattering_angle_to_lattice_parameter(wavelength, angle)
        assert approx(0.55225047) == lattice_size
