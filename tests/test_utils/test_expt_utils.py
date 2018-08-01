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
from pyxem.utils.expt_utils import *

@pytest.mark.parametrize('x, y, r, theta',[
    (100, 0.0370143659),
    (200, 0.0250793403),
    (300, 0.0196874888),
])
def test_cart2polar(x, y, r, theta):
    rc, thetac = _cart2polar(x=x, y=y)
    np.testing.assert_almost_equal(rc, r)
    np.testing.assert_almost_equal(thetac, theta)

@pytest.mark.parametrize('r, theta, x, y',[
    (100, 0.0370143659),
    (200, 0.0250793403),
    (300, 0.0196874888),
])
def test_polar2cart(r, theta, x, y):
    xc, yc = _polar2cart(r=r, theta=theta)
    np.testing.assert_almost_equal(xc, x)
    np.testing.assert_almost_equal(yc, y)

@pytest.mark.parametrize('z, center, calibration, g',[
    (100, 0.0370143659),
    (200, 0.0250793403),
    (300, 0.0196874888),
])
def test_polar2cart(z, center, calibration, g):
    gc = peaks_as_gvectors(z=z, center=center, calibration=calibration)
    np.testing.assert_almost_equal(gc, g)
