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

import pyxem as pxm
from pyxem.generators.subpixelrefinement_generator import SubpixelrefinementGenerator
from skimage import draw


@pytest.fixture
def SPR_generator(diffraction_pattern,vect = np.asarray([[4,4]])):
    return SubpixelrefinementGenerator(diffraction_pattern,vect)

def test_conventional_xc(SPR_generator):
    diff_vect = SPR_generator.conventional_xc(4,2,100)
    assert True

def test_sobel_filtered_xc(SPR_generator):
    diff_vect = SPR_generator.sobel_filtered_xc(4,2,100)
    assert True

@pytest.fixture()
def create_spot():
    z = np.zeros((128,128))
    for r in [4,3,2]:
        rr, cc = draw.circle(90,30,radius=r,shape=z.shape)
        z[rr, cc] = 1 / r
    dp = pxm.ElectronDiffraction(np.asarray([[z,z],[z,z]])) #this needs to be in 2x2
    return dp


def test_assertioned_xc(create_spot):
    spr = SubpixelrefinementGenerator(create_spot,np.asarray([[90,30]]))
    s = spr.conventional_xc(12,4,8)
    error = np.subtract(spr.vectors_out[0,0],np.asarray([[90,30]]))
    rms_error = np.sqrt(error[0,0]**2+error[0,1]**2)
    assert rms_error < 0.2 #1/5th a pixel
    assert rms_error == 1
