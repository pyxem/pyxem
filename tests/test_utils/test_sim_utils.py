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
import diffpy

from transforms3d.euler import euler2mat

from pyxem.signals.electron_diffraction import ElectronDiffraction
from pyxem.utils.sim_utils import *


@pytest.mark.parametrize('accelerating_voltage, wavelength', [
    (100, 0.0370143659),
    (200, 0.0250793403),
    (300, 0.0196874888),
])
def test_get_electron_wavelength(accelerating_voltage, wavelength):
    val = get_electron_wavelength(accelerating_voltage=accelerating_voltage)
    np.testing.assert_almost_equal(val, wavelength)


@pytest.mark.parametrize('accelerating_voltage, interaction_constant', [
    (100, 1.0066772603317773e-16),
    (200, 2.0133545206634971e-16),
    (300, 3.0200317809952176e-16),
])
def test_get_interaction_constant(accelerating_voltage, interaction_constant):
    val = get_interaction_constant(accelerating_voltage=accelerating_voltage)
    np.testing.assert_almost_equal(val, interaction_constant)


def test_get_unique_families():
    hkls = ((0, 1, 1), (1, 1, 0))
    unique_families = get_unique_families(hkls)
    assert unique_families == {(1, 1, 0): 2}


def test_get_points_in_sphere():
    latt = diffpy.structure.lattice.Lattice(0.5, 0.5, 0.5, 90, 90, 90)
    ind, cord, dist = get_points_in_sphere(latt, 0.6)
    assert len(ind) == len(cord)
    assert len(ind) == len(dist)
    assert len(dist) == 1 + 6


def make_structure_hexagonal(rotation=None):
    """Construct a hexagonal P63mc GaAs Wurtzite structure. """
    if rotation is None:
        rotation = np.eye(3)
    a = 4.053
    c = 6.680
    lattice = diffpy.structure.lattice.Lattice(a, a, c, 90, 90, 120)
    atom_list = []
    for x, y, z in [(1/3, 2/3, 0), (2/3, 1/3, 1/2)]:
        atom_list.append(
                diffpy.structure.atom.Atom(
                    atype='Ga',
                    xyz=[x, y, z],
                    lattice=lattice))
        atom_list.append(
                diffpy.structure.atom.Atom(
                    atype='As',
                    xyz=[x + 3/8, y + 3/8, z + 3/8],
                    lattice=lattice))
    return diffpy.structure.Structure(atoms=atom_list, lattice=lattice)


def test_kinematic_intensities_rotation():
    """Test that kinematically forbidden diffraction spots gets zero intensity also after rotation."""
    rotation = euler2mat(0, np.pi/2, 0, 'rzxz')
    structure = make_structure_hexagonal(rotation)
    reciprocal_lattice = structure.lattice.reciprocal()
    g_indices = [(0, 0, 1)]
    g_hkls = np.array([reciprocal_lattice.dist(g_indices, [0, 0, 0])])

    intensities = get_kinematical_intensities(
            structure,
            g_indices,
            g_hkls,
            excitation_error=0,
            maximum_excitation_error=1,
            debye_waller_factors={})

    np.testing.assert_almost_equal(intensities, [0])


def test_kinematic_simulator_plane_wave():
    atomic_coordinates = np.asarray([[0, 0, 0]])  # structure.cart_coords
    sim = simulate_kinematic_scattering(atomic_coordinates, "Si", 300.,
                                        simulation_size=32)
    assert isinstance(sim, ElectronDiffraction)


def test_kinematic_simulator_gaussian_probe():
    atomic_coordinates = np.asarray([[0, 0, 0]])  # structure.cart_coords
    sim = simulate_kinematic_scattering(atomic_coordinates, "Si", 300.,
                                        simulation_size=32,
                                        illumination='gaussian_probe')
    assert isinstance(sim, ElectronDiffraction)


@pytest.mark.xfail(raises=ValueError)
def test_kinematic_simulator_invalid_illumination():
    atomic_coordinates = np.asarray([[0, 0, 0]])  # structure.cart_coords
    sim = simulate_kinematic_scattering(atomic_coordinates, "Si", 300.,
                                        simulation_size=32,
                                        illumination='gaussian')
    assert isinstance(sim, ElectronDiffraction)
