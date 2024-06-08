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
import dask.array as da
import numpy as np

from pyxem.signals import (
    DPCSignal1D,
    DPCSignal2D,
    LazyDPCSignal1D,
    LazyDPCSignal2D,
)


class TestDpcSignal1dCreate:
    def test_create(self):
        data = np.ones(shape=(2, 10))
        DPCSignal1D(data)

    def test_create_lazy(self):
        data = da.ones(shape=(2, 10))
        LazyDPCSignal1D(data)


class TestDpcSignal2dCreate:
    def test_create(self):
        data = np.ones(shape=(2, 10, 10))
        DPCSignal2D(data)
        with pytest.raises(ValueError):
            DPCSignal2D(np.zeros(10))

    def test_create_lazy(self):
        data = da.ones(shape=(2, 10, 10))
        LazyDPCSignal2D(data)


class TestToBeamShift:
    def test_dpc_signal2d(self):
        probe_x, probe_y = 100, 200
        s = DPCSignal2D(np.zeros((2, probe_y, probe_x)))
        s_beam_shift = s.to_beamshift()
        assert s_beam_shift.axes_manager.shape == (probe_x, probe_y, 2)

    def test_dpc_signal1d(self):
        probe_x = 200
        s = DPCSignal1D(np.zeros((2, probe_x)))
        s_beam_shift = s.to_beamshift()
        assert s_beam_shift.axes_manager.shape == (probe_x, 2)

    def test_lazy_dpc_signal2d(self):
        probe_x, probe_y = 100, 200
        s = LazyDPCSignal2D(da.zeros((2, probe_y, probe_x)))
        s_beam_shift = s.to_beamshift()
        assert s_beam_shift._lazy
        assert s_beam_shift.axes_manager.shape == (probe_x, probe_y, 2)

    def test_lazy_dpc_signal1d(self):
        probe_x = 200
        s = LazyDPCSignal1D(da.zeros((2, probe_x)))
        s_beam_shift = s.to_beamshift()
        assert s_beam_shift._lazy
        assert s_beam_shift.axes_manager.shape == (probe_x, 2)
