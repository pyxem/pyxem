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


def test_strain_mapping():
    # this could be rephrased in terms of fixtures
    xy = np.asarray([[1,0],[0,2]])
    oo = np.asarray(([1,2],[3,-4]))
    dp_xy = hs.signals.Signal2D(generate_test_vectors(xy))
    dp_oo = hs.signals.Signal2D(generate_test_vectors(oo))
    s_xy = get_DisplacementGradientMap(dp_xy,[[1,0],[0,2]]).get_strain_maps()
    s_oo = get_DisplacementGradientMap(dp_oo,[[1,2],[3,-4]]).get_strain_maps()
    np.testing.assert_almost_equal(s_xy.data,s_oo.data)
    ### ALERT to the minus sign we have had to drop in
    np.testing.assert_almost_equal(np.sum(s_xy.inav[3].data),-1*np.deg2rad(3)) #only one rotations occurs so you can use sum
    np.testing.assert_almost_equal(np.sum(s_oo.inav[3].data),-1*np.deg2rad(3))
