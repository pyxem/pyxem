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

import numpy as np
import pyxem as pxm


class TestApi:
    def test_diffraction2d(self):
        s = pxm.Diffraction2D(np.ones((10, 5, 3, 6)))
        assert s.axes_manager.signal_shape == (6, 3)
        assert s.axes_manager.navigation_shape == (5, 10)

    def test_dpcbasesignal(self):
        s = pxm.DPCBaseSignal(np.ones((2))).T
        assert s.axes_manager.signal_shape == ()
        assert s.axes_manager.navigation_shape == (2,)

    def test_dpcsignal1d(self):
        s = pxm.DPCSignal1D(np.ones((2, 10)))
        assert s.axes_manager.signal_shape == (10,)
        assert s.axes_manager.navigation_shape == (2,)

    def test_dpcsignal2d(self):
        s = pxm.DPCSignal2D(np.ones((2, 10, 15)))
        assert s.axes_manager.signal_shape == (15, 10)
        assert s.axes_manager.navigation_shape == (2,)
