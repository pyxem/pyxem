# -*- coding: utf-8 -*-
# Copyright 2016-2025 The pyXem developers
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

from pyxem.data.dummy_data import CrystalSTEMSimulation

from pyxem.data import (
    simulated_overlap,
    simulated_pn_junction,
    simulated_constant_shift_magnitude,
)
from pyxem.signals import ElectronDiffraction2D, Diffraction2D


class TestMakeOverlapData:
    def test_init(self):
        s = simulated_overlap()


class TestDPC:
    def test_pn_junction(self):
        s = simulated_pn_junction()
        assert isinstance(s, ElectronDiffraction2D)
        assert s.metadata["title"] == "Simulated pn-junction"
        assert s.axes_manager.signal_axes[0].units == "nm^-1"
        assert s.axes_manager.signal_axes[1].units == "nm^-1"

    def test_simulated_constant_shift_magnitude(self):
        s = simulated_constant_shift_magnitude()
        assert isinstance(s, Diffraction2D)
        assert s.metadata["title"] == "Simulated Constant Shift Magnitude"
        assert s.axes_manager.signal_axes[0].units == "px"
        assert s.axes_manager.signal_axes[1].units == "px"
