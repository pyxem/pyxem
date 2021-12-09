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

import numpy as np

from pyxem.signals import Diffraction2D


class TestAdjustMetadata:
    def test_add_navigation_signal(self):
        d = Diffraction2D(np.ones((10, 10, 10, 10)))
        data = np.ones((10, 10))
        d.add_navigation_signal(data=data, name="Test", unit="nm", nav_plot=True)
        d.add_navigation_signal(data=data, name="Test2", unit="nm", nav_plot=True)
        np.testing.assert_equal(d.metadata.Navigation_signals.Test["data"], data)
        np.testing.assert_equal(d.metadata.Navigation_signals.Test2["data"], data)
