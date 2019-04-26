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
import diffpy

from pyxem.signals.electron_diffraction import ElectronDiffraction
from pyxem.utils.sim_utils import get_electron_wavelength, \
    get_interaction_constant, get_unique_families, get_kinematical_intensities,\
    get_vectorized_list_for_atomic_scattering_factors, get_points_in_sphere, \
    simulate_kinematic_scattering, peaks_from_best_template, \
    is_lattice_hexagonal, transfer_navigation_axes, uvtw_to_uvw, \
    rotation_list_stereographic


def create_lattice_structure(a, b, c, alpha, beta, gamma):
    lattice = diffpy.structure.lattice.Lattice(a, b, c, alpha, beta, gamma)
    atom = diffpy.structure.atom.Atom(atype='Si', xyz=[0, 0, 0], lattice=lattice)
    return diffpy.structure.Structure(atoms=[atom], lattice=lattice)


def create_structure_cubic():
    return create_lattice_structure(1, 1, 1, 90, 90, 90)


def create_structure_hexagonal():
    return create_lattice_structure(1, 1, 1, 90, 90, 120)


def create_structure_orthorombic():
    return create_lattice_structure(1, 2, 3, 90, 90, 90)


def create_structure_tetragonal():
    return create_lattice_structure(1, 1, 2, 90, 90, 90)


def create_structure_trigonal():
    return create_lattice_structure(1, 1, 1, 100, 100, 100)


def create_structure_monoclinic():
    return create_lattice_structure(1, 2, 3, 90, 100, 90)


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


def test_kinematic_simulator_xtables_scattering_params():
    atomic_coordinates = np.asarray([[0, 0, 0]])  # structure.cart_coords
    sim = simulate_kinematic_scattering(atomic_coordinates, "Si", 300.,
                                        simulation_size=32,
                                        illumination='gaussian_probe',
                                        scattering_params='xtables')
    assert isinstance(sim, ElectronDiffraction)


@pytest.mark.xfail(raises=NotImplementedError)
def test_kinematic_simulator_invalid_scattering_params():
    atomic_coordinates = np.asarray([[0, 0, 0]])  # structure.cart_coords
    sim = simulate_kinematic_scattering(atomic_coordinates, "Si", 300.,
                                        simulation_size=32,
                                        illumination='gaussian_probe',
                                        scattering_params='_empty')
    #assert isinstance(sim, ElectronDiffraction)


@pytest.mark.xfail(raises=ValueError)
def test_kinematic_simulator_invalid_illumination():
    atomic_coordinates = np.asarray([[0, 0, 0]])  # structure.cart_coords
    sim = simulate_kinematic_scattering(atomic_coordinates, "Si", 300.,
                                        simulation_size=32,
                                        illumination='gaussian')
    #assert isinstance(sim, ElectronDiffraction)


@pytest.mark.parametrize('uvtw, uvw', [
    ((0, 0, 0, 1), (0, 0, 1)),
    ((1, 0, 0, 1), (2, 1, 1)),
    ((2, 2, 0, 0), (1, 1, 0)),
])
def test_uvtw_to_uvw(uvtw, uvw):
    val = uvtw_to_uvw(uvtw)
    np.testing.assert_almost_equal(val, uvw)


# Three corners of the rotation lists, for comparison
structure_cubic_rotations = [
    [0, 0, 0],
    [90, 45, 0],
    [135, 54.73561032, 0]
]

structure_hexagonal_rotations = [
    [0, 0, 0],
    [90, 90, 0],
    [120, 90, 0]
]

structure_orthogonal_rotations = [
    [0, 0, 0],
    [90, 90, 0],
    [180, 90, 0]
]

structure_tetragonal_rotations = [
    [0, 0, 0],
    [90, 90, 0],
    [135, 90, 0]
]

structure_trigonal_rotations = [
    [0, 0, 0],
    [-28.64458044, 75.45951959, 0],
    [38.93477108, 90, 0]
]

structure_monoclinic_rotations = [
    [0, 0, 0],
    [0, 90, 0],
    [180, 90, 0]
]


@pytest.mark.parametrize('structure, corner_a, corner_b, corner_c, rotation_list', [
    (create_structure_cubic(), (0, 0, 1), (1, 0, 1), (1, 1, 1), structure_cubic_rotations),
    (create_structure_hexagonal(), (0, 0, 0, 1), (1, 0, -1, 0), (1, 1, -2, 0), structure_hexagonal_rotations),
    (create_structure_orthorombic(), (0, 0, 1), (1, 0, 0), (0, 1, 0), structure_orthogonal_rotations),
    (create_structure_tetragonal(), (0, 0, 1), (1, 0, 0), (1, 1, 0), structure_tetragonal_rotations),
    (create_structure_trigonal(), (0, 0, 0, 1), (0, -1, 1, 0), (1, -1, 0, 0), structure_trigonal_rotations),
    (create_structure_monoclinic(), (0, 0, 1), (0, 1, 0), (0, -1, 0), structure_monoclinic_rotations),
])
def test_rotation_list_stereographic(structure, corner_a, corner_b, corner_c, rotation_list):
    val = rotation_list_stereographic(structure, corner_a, corner_b, corner_c, [0], np.deg2rad(10))
    for expected in rotation_list:
        assert any((np.allclose(expected, actual) for actual in val))


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize('structure, corner_a, corner_b, corner_c, inplane_rotations, resolution, rotation_list', [
    (create_structure_cubic(), (0, 0, 1), (0, 0, 1), (1, 1, 1), [0], np.deg2rad(10), structure_cubic_rotations),
    (create_structure_cubic(), (0, 0, 1), (1, 0, 1), (0, 0, 1), [0], np.deg2rad(10), structure_cubic_rotations)
])
def test_rotation_list_stereographic_raises_invalid_corners(
        structure, corner_a, corner_b, corner_c, inplane_rotations, resolution, rotation_list):
    rotation_list_stereographic(structure, corner_a, corner_b, corner_c, inplane_rotations, resolution)
