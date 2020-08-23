# -*- coding: utf-8 -*-
# Copyright 2016-2020 The pyXem developers
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
import pyxem.dummy_data.dummy_data as dd


@pytest.mark.skip(reason="These tests are slow")
class TestDummyDataModule:
    def test_simple_disk_shift(self):
        s = dd.get_disk_shift_simple_test_signal()
        assert hasattr(s, "plot")
        assert not s._lazy

        s = dd.get_disk_shift_simple_test_signal(lazy=True)
        assert s._lazy

    def test_simple_holz_signal(self):
        s = dd.get_holz_simple_test_signal()
        assert hasattr(s, "plot")
        assert not s._lazy

        s = dd.get_holz_simple_test_signal(lazy=True)
        assert s._lazy

    def test_single_ring_diffraction_signal(self):
        s = dd.get_single_ring_diffraction_signal()
        assert hasattr(s, "plot")

    def test_get_simple_dpc_signal(self):
        s = dd.get_simple_dpc_signal()
        assert hasattr(s, "plot")

    def test_get_holz_heterostructure_test_signal(self):
        s = dd.get_holz_heterostructure_test_signal()
        assert hasattr(s, "plot")
        assert not s._lazy

        s = dd.get_holz_heterostructure_test_signal(lazy=True)
        assert s._lazy

    def test_get_stripe_pattern_dpc_signal(self):
        s = dd.get_stripe_pattern_dpc_signal()
        assert hasattr(s, "plot")

    def test_get_square_dpc_signal(self):
        s = dd.get_square_dpc_signal()
        assert hasattr(s, "plot")
        s_ramp = dd.get_square_dpc_signal(add_ramp=True)
        s_ramp.plot()

    def test_get_dead_pixel_signal(self):
        s = dd.get_dead_pixel_signal()
        assert hasattr(s, "plot")
        assert (s.data == 0).any()

    def test_get_cbed_signal(self):
        s = dd.get_cbed_signal()
        s.plot()

    def test_get_fem_signal(self):
        s = dd.get_fem_signal()
        s.plot()

    def test_get_simple_fem_signal(self):
        s = dd.get_simple_fem_signal()
        s.plot()

    def test_get_nanobeam_electron_diffraction_signal(self):
        s = dd.get_nanobeam_electron_diffraction_signal()
        s.plot()

    def test_get_generic_fem_signal(self):
        s = dd.get_generic_fem_signal(probe_x=2, probe_y=3, image_x=20, image_y=25)
        s.plot()
        assert s.axes_manager.shape == (2, 3, 20, 25)
        assert hasattr(s, "plot")

    def test_get_simple_ellipse_signal_peak_array(self):
        s, peak_array = dd.get_simple_ellipse_signal_peak_array()
        s.add_peak_array_as_markers(peak_array=peak_array)
        assert hasattr(s, "plot")
        assert hasattr(s, "axes_manager")
        assert s.data.shape[:2] == peak_array.shape[:2]
