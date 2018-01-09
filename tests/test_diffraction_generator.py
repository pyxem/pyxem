# -*- coding: utf-8 -*-
# Copyright 2017 The PyCrystEM developers
#
# This file is part of PyCrystEM.
#
# PyCrystEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyCrystEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyCrystEM.  If not, see <http://www.gnu.org/licenses/>.

import pytest
import numpy as np
import pymatgen as pmg

from pyxem.diffraction_generator import ElectronDiffractionCalculator


@pytest.fixture(params=[
    (300, 0.02, None),
])
def diffraction_calculator(request):
    return ElectronDiffractionCalculator(*request.param)

@pytest.fixture(params=[
    "Si",
])
def element(request):
    return pmg.Element(request.param)


@pytest.fixture(params=[
    5.431
])
def lattice(request):
    return pmg.Lattice.cubic(request.param)


@pytest.fixture(params=[
    "Fd-3m"
])
def structure(request, lattice, element):
    return pmg.Structure.from_spacegroup(request.param, lattice, [element], [[0, 0, 0]])


class TestDiffractionCalculator:

    def test_init(self, diffraction_calculator: ElectronDiffractionCalculator):
        assert diffraction_calculator.debye_waller_factors == {}

    def test_appropriate_scaling(self, diffraction_calculator: ElectronDiffractionCalculator):
        """Tests that doubling the unit cell halves the pattern spacing."""
        si = pmg.Element("Si")
        lattice = pmg.Lattice.cubic(5.431)
        big_lattice = pmg.Lattice.cubic(10.862)
        silicon = pmg.Structure.from_spacegroup("Fd-3m", lattice, [si], [[0, 0, 0]])
        big_silicon = pmg.Structure.from_spacegroup("Fd-3m", big_lattice, [si], [[0, 0, 0]])
        diffraction = diffraction_calculator.calculate_ed_data(structure=silicon, reciprocal_radius=5.)
        big_diffraction = diffraction_calculator.calculate_ed_data(structure=big_silicon, reciprocal_radius=5.)
        indices = [tuple(i) for i in diffraction.indices]
        big_indices = [tuple(i) for i in big_diffraction.indices]
        assert (2, 2, 0) in indices
        assert (2, 2, 0) in big_indices
        coordinates = diffraction.coordinates[indices.index((2, 2, 0))]
        big_coordinates = big_diffraction.coordinates[big_indices.index((2, 2, 0))]
        assert np.allclose(coordinates, big_coordinates * 2)

    @pytest.mark.parametrize('structure, expected, expected_extinction', [
        ('Fd-3m', (2, 2, 0), (2, 1, 0)),
        ('Im-3m', (1, 1, 0), (2, 1, 0))
    ], indirect=['structure'])
    def test_correct_extinction(self, diffraction_calculator, structure, expected, expected_extinction):
        diffraction = diffraction_calculator.calculate_ed_data(structure=structure, reciprocal_radius=5.)
        indices = [tuple(i) for i in diffraction.indices]
        assert expected in indices
        assert expected_extinction not in indices




class TestDiffractionSimulation:

    def test_calibration(self):
        pass

    def test_coordinates(self):
        pass

    def test_intensities(self):
        pass
