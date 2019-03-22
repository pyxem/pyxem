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

import numpy as np
import pytest
from pyxem.signals.diffraction_simulation import DiffractionSimulation
from pyxem.signals.diffraction_simulation import ProfileSimulation
from pyxem.generators.diffraction_generator import DiffractionGenerator
import diffpy.structure


@pytest.fixture(params=[(300, 0.02, None), ])
def diffraction_calculator(request):
    return DiffractionGenerator(*request.param)


def make_structure(lattice_parameter=None):
    """
    We construct an Fd-3m silicon (with lattice parameter 5.431 as a default)
    """
    if lattice_parameter is not None:
        a = lattice_parameter
    else:
        a = 5.431
    latt = diffpy.structure.lattice.Lattice(a, a, a, 90, 90, 90)
    # TODO - Make this construction with internal diffpy syntax
    atom_list = []
    for coords in [[0, 0, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0]]:
        x, y, z = coords[0], coords[1], coords[2]
        atom_list.append(
            diffpy.structure.atom.Atom(
                atype='Si', xyz=[
                    x, y, z], lattice=latt))  # Motif part A
        atom_list.append(
            diffpy.structure.atom.Atom(
                atype='Si',
                xyz=[
                    x + 0.25,
                    y + 0.25,
                    z + 0.25],
                lattice=latt))  # Motif part B
    return diffpy.structure.Structure(atoms=atom_list, lattice=latt)


@pytest.fixture()
def local_structure():
    return make_structure()


class TestDiffractionCalculator:

    def test_init(self, diffraction_calculator: DiffractionGenerator):
        assert diffraction_calculator.debye_waller_factors == {}

    def test_matching_results(self, diffraction_calculator, local_structure):
        diffraction = diffraction_calculator.calculate_ed_data(
            local_structure, reciprocal_radius=5.)
        assert len(diffraction.indices) == len(diffraction.coordinates)
        assert len(diffraction.coordinates) == len(diffraction.intensities)

    def test_appropriate_scaling(self, diffraction_calculator: DiffractionGenerator):
        """Tests that doubling the unit cell halves the pattern spacing."""
        silicon = make_structure(5)
        big_silicon = make_structure(10)
        diffraction = diffraction_calculator.calculate_ed_data(
            structure=silicon, reciprocal_radius=5.)
        big_diffraction = diffraction_calculator.calculate_ed_data(
            structure=big_silicon, reciprocal_radius=5.)
        indices = [tuple(i) for i in diffraction.indices]
        big_indices = [tuple(i) for i in big_diffraction.indices]
        assert (2, 2, 0) in indices
        assert (2, 2, 0) in big_indices
        coordinates = diffraction.coordinates[indices.index((2, 2, 0))]
        big_coordinates = big_diffraction.coordinates[big_indices.index((2, 2, 0))]
        assert np.allclose(coordinates, big_coordinates * 2)

    def test_appropriate_intensities(self, diffraction_calculator, local_structure):
        """Tests the central beam is strongest."""
        diffraction = diffraction_calculator.calculate_ed_data(
            local_structure, reciprocal_radius=5.)
        indices = [tuple(i) for i in diffraction.indices]
        central_beam = indices.index((0, 0, 0))
        smaller = np.greater_equal(diffraction.intensities[central_beam], diffraction.intensities)
        assert np.all(smaller)

    def test_calculate_profile_class(self, local_structure, diffraction_calculator):
        # tests the non-hexagonal (cubic) case
        profile = diffraction_calculator.calculate_profile_data(local_structure,
                                                                reciprocal_radius=1.)
        assert isinstance(profile, ProfileSimulation)

        latt = diffpy.structure.lattice.Lattice(3, 3, 5, 90, 90, 120)
        atom = diffpy.structure.atom.Atom(atype='Ni', xyz=[0, 0, 0], lattice=latt)
        hexagonal_structure = diffpy.structure.Structure(atoms=[atom], lattice=latt)
        hexagonal_profile = diffraction_calculator.calculate_profile_data(structure=hexagonal_structure,
                                                                          reciprocal_radius=1.)
        assert isinstance(hexagonal_profile, ProfileSimulation)


scattering_params = ['lobato', 'xtables']


@pytest.mark.parametrize('scattering_param', scattering_params)
def test_param_check(scattering_param):
    generator = DiffractionGenerator(300, 0.2, None,
                                     scattering_params=scattering_param)


@pytest.mark.xfail(raises=NotImplementedError)
def test_invalid_scattering_params():
    scattering_param = '_empty'
    generator = DiffractionGenerator(300, 0.2, None,
                                     scattering_params=scattering_param)
