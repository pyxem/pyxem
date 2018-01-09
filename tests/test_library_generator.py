# -*- coding: utf-8 -*-
# Copyright 2017 The pyXem developers
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

from pyxem.diffraction_generator import ElectronDiffractionCalculator
from pyxem.library_generator import DiffractionLibraryGenerator, DiffractionLibrary
from hyperspy.signals import Signal1D, Signal2D

@pytest.fixture
def diffraction_calculator():
    return ElectronDiffractionCalculator(300., 0.02)

@pytest.fixture
def library_generator(diffraction_calculator):
    return DiffractionLibraryGenerator(diffraction_calculator)


class TestDiffractionLibraryGenerator:

    def test_get_diffraction_library(
            self,
            library_generator: DiffractionLibraryGenerator,
            structure_library, calibration, reciprocal_radius, representation
    ):
        library = library_generator.get_diffraction_library(
            structure_library, calibration, reciprocal_radius, representation)
        assert isinstance(library, DiffractionLibrary)