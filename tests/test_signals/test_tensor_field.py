# -*- coding: utf-8 -*-
# Copyright 2017-2018 The pyXem developers
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

from pyxem.signals.tensor_field import _polar_decomposition, _get_rotation_angle
from pyxem.signals.tensor_field import DisplacementGradientMap

@pytest.mark.parametrize('D, R, U',[
    (np.array([[100,10,0],
               [200,20,0],
               [150,-15,0]]),
     np.array([[100,10,0],
                [200,20,0],
                [150,-15,0]]),
     np.array([[100,10,0],
                [200,20,0],
                [150,-15,0]])),
])
def test_polar_decomposition(D, R, U):
    Rc, Uc = _polar_decomposition(D)
    np.testing.assert_almost_equal(Rc, R)
    np.testing.assert_almost_equal(Uc, U)

@pytest.mark.parametrize('R, theta',[
    (np.array([[100,10,0],
               [200,20,0],
               [150,-15,0]]),
     15),
])
def test_get_rotation_angle(R, theta):
    tc = _get_rotation_angle(R)
    np.testing.assert_almost_equal(tc, theta)
