# -*- coding: utf-8 -*-
# Copyright 2017-2019 The pyXem developers
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


@pytest.mark.parametrize('D, side, R, U', [
    (np.array([[0.98860899, -0.2661997, 0.],
               [0.2514384, 0.94324267, 0.],
               [0., 0., 1.]]),
     'right',
     np.array([[0.96592583, -0.25881905, 0.],
               [0.25881905, 0.96592583, 0.],
               [0., 0., 1.]]),
     np.array([[1.02, -0.013, 0.],
               [-0.013, 0.98, 0.],
               [0., 0., 1.]])),
])
def test_polar_decomposition(D, side, R, U):
    Rc, Uc = _polar_decomposition(D, side=side)
    np.testing.assert_almost_equal(Rc, R)
    np.testing.assert_almost_equal(Uc, U)


@pytest.mark.parametrize('R, theta', [
    (np.array([[0.76604444, 0.64278761, 0.],
               [-0.64278761, 0.76604444, 0.],
               [0., 0., 1.]]),
     0.6981317007977318),
    (np.array([[0.96592583, -0.25881905, 0.],
               [0.25881905, 0.96592583, 0.],
               [0., 0., 1.]]),
     -0.2617993877991494),
])
def test_get_rotation_angle(R, theta):
    tc = _get_rotation_angle(R)
    np.testing.assert_almost_equal(tc, theta)


@pytest.mark.parametrize('dgm, rotation_map, distortion_map', [
    (DisplacementGradientMap(np.array([[[[0.98860899, -0.2661997, 0.],
                                         [0.2514384, 0.94324267, 0.],
                                         [0., 0., 1.]],
                                        [[0.98860899, -0.2661997, 0.],
                                         [0.2514384, 0.94324267, 0.],
                                         [0., 0., 1.]]],
                                       [[[0.98860899, -0.2661997, 0.],
                                         [0.2514384, 0.94324267, 0.],
                                         [0., 0., 1.]],
                                        [[0.98860899, -0.2661997, 0.],
                                         [0.2514384, 0.94324267, 0.],
                                         [0., 0., 1.]]]])),
     np.array([[[[0.96592583, -0.25881905, 0.],
                 [0.25881905, 0.96592583, 0.],
                 [0., 0., 1.]],
                [[0.96592583, -0.25881905, 0.],
                 [0.25881905, 0.96592583, 0.],
                 [0., 0., 1.]]],
               [[[0.96592583, -0.25881905, 0.],
                 [0.25881905, 0.96592583, 0.],
                 [0., 0., 1.]],
                [[0.96592583, -0.25881905, 0.],
                 [0.25881905, 0.96592583, 0.],
                 [0., 0., 1.]]]]),
     np.array([[[[1.02, -0.013, 0.],
                 [-0.013, 0.98, 0.],
                 [0., 0., 1.]],
                [[1.02, -0.013, 0.],
                 [-0.013, 0.98, 0.],
                 [0., 0., 1.]]],
               [[[1.02, -0.013, 0.],
                 [-0.013, 0.98, 0.],
                 [0., 0., 1.]],
                [[1.02, -0.013, 0.],
                 [-0.013, 0.98, 0.],
                 [0., 0., 1.]]]])),
])
def test_map_polar_decomposition(dgm,
                                 rotation_map,
                                 distortion_map):
    Rc, Uc = dgm.polar_decomposition()
    np.testing.assert_almost_equal(Rc.data, rotation_map)
    np.testing.assert_almost_equal(Uc.data, distortion_map)


@pytest.mark.parametrize('dgm, strain_answers', [
    (DisplacementGradientMap(np.array([[[[0.98860899, -0.2661997, 0.],
                                         [0.2514384, 0.94324267, 0.],
                                         [0., 0., 1.]],
                                        [[0.98860899, -0.2661997, 0.],
                                         [0.2514384, 0.94324267, 0.],
                                         [0., 0., 1.]]],
                                       [[[0.98860899, -0.2661997, 0.],
                                         [0.2514384, 0.94324267, 0.],
                                         [0., 0., 1.]],
                                        [[0.98860899, -0.2661997, 0.],
                                         [0.2514384, 0.94324267, 0.],
                                         [0., 0., 1.]]]])),
     np.array([[[-0.02, -0.02],
                [-0.02, -0.02]],
               [[0.02, 0.02],
                [0.02, 0.02]],
               [[-0.013, -0.013],
                [-0.013, -0.013]],
               [[-0.26179939, -0.26179939],
                [-0.26179939, -0.26179939]]])),
])
def test_get_strain_maps(dgm,
                         strain_answers):
    strain_results = dgm.get_strain_maps()
    np.testing.assert_almost_equal(strain_results.data, strain_answers)
