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

import numpy as np
import pytest
import hyperspy.api as hs

from pyxem.generators.displacement_gradient_tensor_generator import *

def rotation(z):
     theta = np.deg2rad(3)
     c,s = np.cos(theta),np.sin(theta)
     R = np.asarray(([c,-s],[s,c]))
     return np.matmul(R,z)

def uniform_expansion(z):
    return (1.1 * z)

def stretch_in_x(z):
    M = np.asarray([[1.1,0],[0,1]])
    return np.matmul(M,z)

def generate_test_vectors(v):
    return np.asarray([[v,rotation(v)],
                       [uniform_expansion(v),stretch_in_x(v)]])

def generate_strain_map(vectors):
    dp = hs.signals.Signal2D(generate_test_vectors(vectors))
    st = get_DisplacementGradientMap(dp,vectors).get_strain_maps()
    return st

def test_strain_mapping():
    xy = np.asarray([[1,0],[0,2]])
    oo = np.asarray(([1,2],[3,-4]))
    s_xy = generate_strain_map(xy)
    s_oo = generate_strain_map(oo)
    np.testing.assert_almost_equal(s_xy.data,s_oo.data)
    for s in [s_xy,s_oo]:
        # ALERT to the minus sign we have had to drop in
        #only one rotations occurs so you can use sum
        np.testing.assert_almost_equal(np.sum(s.inav[3].data),-1*np.deg2rad(3))
    return None
