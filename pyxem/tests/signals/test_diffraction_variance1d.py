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

from hyperspy.signals import Signal1D
from pyxem.signals import DiffractionVariance1D


class TestDiffractionVariance1D:
    def test_get_electron_diffraction1D(self):
        rad_signal = Signal1D(np.array([0, 4, 3, 5, 1, 4, 6, 2]))
        difprof = DiffractionVariance1D(rad_signal)
        assert isinstance(difprof, DiffractionVariance1D)
