# -*- coding: utf-8 -*-
# Copyright 2018 The pyXem developers
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
import pymatgen as pmg

from pyxem.generators.diffraction_generator import DiffractionGenerator
from pyxem.generators.library_generator import DiffractionLibraryGenerator
from pyxem.signals.diffraction_library import DiffractionLibrary


@pytest.fixture
def diffraction_calculator():
    return DiffractionGenerator(300., 0.02)


@pytest.fixture
def library_generator(diffraction_calculator):
    return DiffractionLibraryGenerator(diffraction_calculator)

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

@pytest.fixture
def structure_library(structure):
    return {'Si': (structure, [(0, 0, 0)])}


class TestDiffractionLibraryGenerator:

    @pytest.mark.parametrize('calibration, reciprocal_radius, half_shape, representation', [
        (0.017, 2.4, (72,72) ,'euler'),
    ])
    def test_get_diffraction_library(
            self,
            library_generator: DiffractionLibraryGenerator,
            structure_library, calibration, reciprocal_radius, half_shape, representation
    ):
        library = library_generator.get_diffraction_library(
            structure_library, calibration, reciprocal_radius,half_shape, representation)
        assert isinstance(library, DiffractionLibrary)