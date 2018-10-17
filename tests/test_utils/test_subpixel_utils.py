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
from pyxem.utils.subpixel_refinements_utils import _conventional_xc

from skimage.transform import rescale

@pytest.fixture()
def exp_disc():
    ss,disc_radius,upsample_factor = int(60),6,10

    arr = np.zeros((ss,ss))
    rr, cc = draw.circle(int(ss/2)+20, int(ss/2)-10, radius=disc_radius, shape=arr.shape)
    arr[rr, cc] = 1
    arr = rescale(arr,upsample_factor)
    return arr

@pytest.fixture()
def sim_disc():
    return get_simulated_disc(60,5,upsample_factor=10)

def test___conventional_xc(exp_disc,sim_disc):
    s = _conventional_xc(exp_disc,sim_disc)
    error = np.subtract(s,np.asarray([200,-100]))
    rms = np.sqrt(error[0]**2+error[1]**2)
    assert rms < 1 #which corresponds to a 10th of a pixel
