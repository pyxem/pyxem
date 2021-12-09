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
import numpy as np

from hyperspy.signals import Signal1D

from pyxem.signals.symmetry1d import Symmetry1D

class TestSymmetry:
    @pytest.fixture
    def flat_pattern(self):
        pd = Symmetry1D(data=np.ones(shape=(2, 2, 360)))
        pd.axes_manager[2].scale = np.pi/180
        return pd

    def test_get_symmetry_coefficient(self, flat_pattern):
        sn = flat_pattern.get_symmetry_coefficient()
        assert isinstance(sn, Symmetry1D)
        assert (sn.data.shape[-1] == 11)
