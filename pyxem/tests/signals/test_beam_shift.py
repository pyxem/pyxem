# -*- coding: utf-8 -*-
# Copyright 2016-2022 The pyXem developers
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
import dask.array as da
from hyperspy.signals import Signal2D
from pyxem.signals import BeamShift, LazyBeamShift, Diffraction2D


class TestToDPCSignal:
    def test_to_dpcsignal2d(self):
        probe_x, probe_y = 100, 200
        s = BeamShift(np.zeros((probe_y, probe_x, 2)))
        s_dpc = s.to_dpcsignal()
        assert s_dpc.axes_manager.shape == (2, probe_x, probe_y)

    def test_to_dpcsignal1d(self):
        probe_x = 200
        s = BeamShift(np.zeros((probe_x, 2)))
        s_dpc = s.to_dpcsignal()
        assert s_dpc.axes_manager.shape == (2, probe_x)

    def test_to_dpcsignal2d_lazy(self):
        probe_x, probe_y = 100, 200
        s = LazyBeamShift(da.zeros((probe_y, probe_x, 2)))
        s_dpc = s.to_dpcsignal()
        assert s_dpc._lazy
        assert s_dpc.axes_manager.shape == (2, probe_x, probe_y)

    def test_to_dpcsignal1d_lazy(self):
        probe_x = 200
        s = LazyBeamShift(da.zeros((probe_x, 2)))
        s_dpc = s.to_dpcsignal()
        assert s_dpc._lazy
        assert s_dpc.axes_manager.shape == (2, probe_x)


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
        np.testing.assert_almost_equal(s.data, s_orig, decimal=7)

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
        s.make_linear_plane(mask=s_mask)
        np.testing.assert_almost_equal(s.data, s_orig, decimal=7)

    def test_lazy_input_error(self):
        s = LazyBeamShift(da.zeros((50, 40, 2)))
        with pytest.raises(ValueError):
            s.make_linear_plane()
        s.compute()
        s.make_linear_plane()


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
        s_beam_shift.make_linear_plane(mask=s_mask)
        s.center_direct_beam(shifts=s_beam_shift)
        np.testing.assert_almost_equal(s.data[:, :, 8, 8], 10, decimal=5)
        s.data[:, :, 8, 8] = 0
        s.data[1, 2, 3, 2] = 0
        np.testing.assert_almost_equal(s.data, 0.0, decimal=4)

    def test_mask_lazy(self):
        s = self.s
        s.data[1, 2, 2, 3] = 1000
        s = self.s.as_lazy()
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 2] = True
        s_mask = Signal2D(mask)
        s_beam_shift = s.get_direct_beam_position(method="blur", sigma=1)
        s_beam_shift.compute()
        s_beam_shift.make_linear_plane(mask=s_mask)
        s.center_direct_beam(shifts=s_beam_shift)
        s.compute()
        np.testing.assert_almost_equal(s.data[:, :, 8, 8], 10,decimal=5)
        s.data[:, :, 8, 8] = 0
        s.data[1, 2, 3, 2] = 0
        np.testing.assert_almost_equal(s.data, 0.0, decimal=4)
