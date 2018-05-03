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
import numpy
from pyxem.utils.sim_utils import *

@pytest.mark.parametrize('accelerating_voltage, wavelength',[
    (100, 0.0370143659),
    (200, 0.0250793403),
    (300, 0.0196874888),
])
def test_get_electron_wavelength(accelerating_voltage, wavelength):
    val = get_electron_wavelength(accelerating_voltage=accelerating_voltage)
    np.testing.assert_almost_equal(val, wavelength)
