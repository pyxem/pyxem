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

import hyperspy.api as hs
import pytest
import numpy as np
from pyxem.tests.test_generators.test_displacement_gradient_tensor_generator import generate_test_vectors
import hyperspy.api as hs
from pyxem.generators.displacement_gradient_tensor_generator import get_DisplacementGradientMap

@pytest.fixture()
def Displacement_Grad_Map():
    xy = np.asarray([[1, 0], [0, 1]])
    deformed = hs.signals.Signal2D(generate_test_vectors(xy))
    D = get_DisplacementGradientMap(deformed,xy)
    return D

def test__init__(Displacement_Grad_Map):
    strain_map = Displacement_Grad_Map.get_strain_maps()
    assert strain_map.axes_manager.navigation_size == 4


""" These are change of basis tests """

#@pytest.mark.skip(reason="Failing test above")
def test_something_changes(Displacement_Grad_Map):
    oneone_strain_original = Displacement_Grad_Map.get_strain_maps()
    local_D  = Displacement_Grad_Map
    strain_alpha = local_D.get_strain_maps()
    oneone_strain_alpha = strain_alpha.rotate_strain_basis([1.3,+1.9])
    assert not np.allclose(oneone_strain_original.data,oneone_strain_alpha.data, atol=0.01)


@pytest.mark.skip(reason="Failing test above")
def test_rotation(Displacement_Grad_Map):  # pragma: no cover
    """
    We should always measure the same rotations, regardless of basis
    """
    local_D  = Displacement_Grad_Map
    original = local_D.get_strain_maps().inav[3].data
    local_D.rotate_strain_basis([1.3,+1.9]) #works in place
    rotation_alpha =  local_D.get_strain_maps().inav[3].data
    local_D = Displacement_Grad_Map
    local_D.rotate_strain_basis([1.7,-0.3])
    rotation_beta = local_D.get_strain_maps().inav[3].data

    # check the functionality has left invarient quantities invarient
    np.testing.assert_almost_equal(original, rotation_alpha, decimal=2)  # rotations
    np.testing.assert_almost_equal(original, rotation_beta, decimal=2)  # rotations


@pytest.mark.skip(reason="basis change functionality not yet implemented")
def test_trace(xy_vectors, right_handed, multi_vector):  # pragma: no cover
    """
    Basis does effect strain measurement, but we can simply calculate suitable invarients.
    See https://en.wikipedia.org/wiki/Infinitesimal_strain_theory for details.
    """
    np.testing.assert_almost_equal(
        np.add(
            xy_vectors.inav[0].data, xy_vectors.inav[1].data), np.add(
            right_handed.inav[0].data, right_handed.inav[1].data), decimal=2)
    np.testing.assert_almost_equal(
        np.add(
            xy_vectors.inav[0].data, xy_vectors.inav[1].data), np.add(
            multi_vector.inav[0].data, multi_vector.inav[1].data), decimal=2)
