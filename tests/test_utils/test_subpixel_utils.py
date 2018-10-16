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
def exp_disc():
    square_size = 60
    disc_radius = 5 #note if these are different you get incorrect answers
    upsample_factor = 50

    upsss = int(square_size*upsample_factor) #upsample square size
    arr = np.zeros((upsss,upsss))
    rr, cc = draw.circle(int(upsss/2)+20, int(upsss/2)-50, radius=disc_radius*upsample_factor, shape=arr.shape)
    arr[rr, cc] = 0.9
    return arr

@pytest.fixture()
def sim_disc():
    return get_simulated_disc(60,5,upsample_factor=50)

def test___sobel_filtered_xc(exp_disc,sim_disc):
    s = _sobel_filtered_xc(exp_disc,sim_disc)
    assert np.all(s==[20,-50])

def test___conventional_xc(exp_disc,sim_disc):
    s = _conventional_xc(exp_disc,sim_disc)
    assert np.all(s==[20,-50])
