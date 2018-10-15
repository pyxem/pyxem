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

from pyxem.utils.subpixel_refinements_utils import *
from pyxem.utils.subpixel_refinements_utils import _sobel_filtered_xc,_conventional_xc

from skimage.transform import rescale

@pytest.fixture()
def square_as_spot_proxy():
    z = np.zeros((200,200))
    z[120:160,40:80] = 1
    return z

@pytest.fixture()
def sim_disc():
    return get_simulated_disc(60,20,upsample_factor=10)

@pytest.fixture()
def fake_square():
    z = np.zeros((60,60))
    z[9:40,9:40] = 1
    z = rescale(z,10)
    return z

def test___sobel_filtered_xc(square_as_spot_proxy,fake_square):
    exp_disc = get_experimental_square(square_as_spot_proxy,[140,60],square_size=60,upsample_factor=10)
    s = _sobel_filtered_xc(exp_disc,fake_square)

def test___conventional_xc(square_as_spot_proxy,fake_square):
    exp_disc = get_experimental_square(square_as_spot_proxy,[140,60],square_size=60,upsample_factor=10)
    s = _conventional_xc(exp_disc,fake_square)
    
