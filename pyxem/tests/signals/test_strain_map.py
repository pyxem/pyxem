# -*- coding: utf-8 -*-
# Copyright 2016-2021 The pyXem developers
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

from pyxem.tests.generators.test_displacement_gradient_tensor_generator import (
    generate_test_vectors,
)
from pyxem.generators import get_DisplacementGradientMap
from pyxem.signals.strain_map import _get_rotation_matrix


@pytest.fixture()
def Displacement_Grad_Map():
    xy = np.asarray([[1, 0], [0, 1]])
    deformed = hs.signals.Signal2D(generate_test_vectors(xy))
    D = get_DisplacementGradientMap(deformed, xy)
    return D


def test_rotation_matrix_formation():
    x_new = [np.random.rand(), np.random.rand()]
    R = _get_rotation_matrix(x_new)
    ratio_array = np.divide(x_new, np.matmul(R, [1, 0]))
    assert np.allclose(ratio_array[0], ratio_array[1])


def test__init__(Displacement_Grad_Map):
    strain_map = Displacement_Grad_Map.get_strain_maps()
    assert strain_map.axes_manager.navigation_size == 4


def test_signal_axes_carry_through(Displacement_Grad_Map):
    """ A strain map that is calibrated, should stay calibrated when we change basis """
    strain_map = Displacement_Grad_Map.get_strain_maps()
    strain_map.axes_manager.signal_axes[1].units = "nm"
    strain_map.axes_manager.signal_axes[0].scale = 19
    strain_alpha = strain_map.rotate_strain_basis([np.random.rand(), np.random.rand()])
    assert strain_alpha.axes_manager.signal_axes[1].units == "nm"
    assert strain_alpha.axes_manager.signal_axes[0].scale == 19


""" These are change of basis tests """


def test_something_changes(Displacement_Grad_Map):
    oneone_strain_original = Displacement_Grad_Map.get_strain_maps()
    local_D = Displacement_Grad_Map
    strain_alpha = local_D.get_strain_maps()
    oneone_strain_alpha = strain_alpha.rotate_strain_basis([1, 1])
    assert not np.allclose(
        oneone_strain_original.data, oneone_strain_alpha.data, atol=0.01
    )


def test_90_degree_rotation(Displacement_Grad_Map):
    oneone_strain_original = Displacement_Grad_Map.get_strain_maps()
    local_D = Displacement_Grad_Map
    strain_alpha = local_D.get_strain_maps()
    oneone_strain_alpha = strain_alpha.rotate_strain_basis([0, 1])
    assert np.allclose(
        oneone_strain_original.inav[2:].data,
        oneone_strain_alpha.inav[2:].data,
        atol=0.01,
    )
    assert np.allclose(
        oneone_strain_original.inav[0].data, oneone_strain_alpha.inav[1].data, atol=0.01
    )
    assert np.allclose(
        oneone_strain_original.inav[1].data, oneone_strain_alpha.inav[0].data, atol=0.01
    )


def test_going_back_and_forward_between_bases(Displacement_Grad_Map):
    """ Checks that going via an intermediate strain map doesn't give incorrect answers"""
    strain_original = Displacement_Grad_Map.get_strain_maps()
    local_D = Displacement_Grad_Map
    temp_strain = local_D.get_strain_maps()
    temp_strain = temp_strain.rotate_strain_basis([np.random.rand(), np.random.rand()])
    fixed_xnew = [3.1, 4.1]
    alpha = strain_original.rotate_strain_basis(fixed_xnew)
    beta = temp_strain.rotate_strain_basis(fixed_xnew)
    assert np.allclose(alpha, beta, atol=0.01)


def test_rotation(Displacement_Grad_Map):
    """
    We should always measure the same rotations, regardless of basis
    """
    local_D = Displacement_Grad_Map
    original = local_D.get_strain_maps()
    rotation_alpha = original.rotate_strain_basis([np.random.rand(), np.random.rand()])
    rotation_beta = original.rotate_strain_basis([np.random.rand(), -np.random.rand()])

    # check the functionality has left invarient quantities invarient
    np.testing.assert_almost_equal(
        original.inav[3].data, rotation_alpha.inav[3].data, decimal=2
    )  # rotations
    np.testing.assert_almost_equal(
        original.inav[3].data, rotation_beta.inav[3].data, decimal=2
    )  # rotations


def test_trace(Displacement_Grad_Map):
    """
    Basis does effect strain measurement, but we can simply calculate suitable invariants.
    See https://en.wikipedia.org/wiki/Infinitesimal_strain_theory for details.
    """

    local_D = Displacement_Grad_Map
    original = local_D.get_strain_maps()
    rotation_alpha = original.rotate_strain_basis([1.3, +1.9])
    rotation_beta = original.rotate_strain_basis([1.7, -0.3])

    np.testing.assert_almost_equal(
        np.add(original.inav[0].data, original.inav[1].data),
        np.add(rotation_alpha.inav[0].data, rotation_alpha.inav[1].data),
        decimal=2,
    )
    np.testing.assert_almost_equal(
        np.add(original.inav[0].data, original.inav[1].data),
        np.add(rotation_beta.inav[0].data, rotation_beta.inav[1].data),
        decimal=2,
    )
