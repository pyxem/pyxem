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
import pymatgen as pmg

from pyxem.signals.electron_diffraction import ElectronDiffraction
from pyxem.utils.sim_utils import *

@pytest.mark.parametrize('accelerating_voltage, wavelength',[
    (100, 0.0370143659),
    (200, 0.0250793403),
    (300, 0.0196874888),
])
def test_get_electron_wavelength(accelerating_voltage, wavelength):
    val = get_electron_wavelength(accelerating_voltage=accelerating_voltage)
    np.testing.assert_almost_equal(val, wavelength)

@pytest.mark.parametrize('accelerating_voltage, interaction_constant',[
    (100, 1.0066772603317773e-16),
    (200, 2.0133545206634971e-16),
    (300, 3.0200317809952176e-16),
])
def test_get_interaction_constant(accelerating_voltage, interaction_constant):
    val = get_interaction_constant(accelerating_voltage=accelerating_voltage)
    np.testing.assert_almost_equal(val, interaction_constant)

def test_kinematic_simulator_plane_wave():
    si = pmg.Element("Si")
    lattice = pmg.Lattice.cubic(5.431)
    structure = pmg.Structure.from_spacegroup("Fd-3m",lattice, [si], [[0, 0, 0]])
    atomic_coordinates = structure.cart_coords
    sim = simulate_kinematic_scattering(atomic_coordinates, "Si", 300.,
                                        simulation_size=32)
    assert isinstance(sim, ElectronDiffraction)

def test_kinematic_simulator_gaussian_probe():
    si = pmg.Element("Si")
    lattice = pmg.Lattice.cubic(5.431)
    structure = pmg.Structure.from_spacegroup("Fd-3m",lattice, [si], [[0, 0, 0]])
    atomic_coordinates = structure.cart_coords
    sim = simulate_kinematic_scattering(atomic_coordinates, "Si", 300.,
                                        simulation_size=32,
                                        illumination='gaussian_probe')
    assert isinstance(sim, ElectronDiffraction)

@pytest.mark.xfail(raises=ValueError)
def test_kinematic_simulator_invalid_illumination():
    si = pmg.Element("Si")
    lattice = pmg.Lattice.cubic(5.431)
    structure = pmg.Structure.from_spacegroup("Fd-3m",lattice, [si], [[0, 0, 0]])
    atomic_coordinates = structure.cart_coords
    sim = simulate_kinematic_scattering(atomic_coordinates, "Si", 300.,
                                        simulation_size=32,
                                        illumination='gaussian')
    assert isinstance(sim, ElectronDiffraction)
