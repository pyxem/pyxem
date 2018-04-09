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

from pyxem.generators.library_generator import library_generator
import pymatgen


@pytest.fixture
def get_library():
        diffraction_calculator = DiffractionGenerator(300., 0.02)
        DiffractionLibraryGenerator(diffraction_calculator)

        element = pmg.Element('Si')
        lattice = pmg.Lattice.cubic(5)

        structure = pmg.Structure.from_spacegroup("F-43m", lattice, [element], [[0, 0, 0]])
        structure_library = {'Si': (structure, [(0, 0, 0)])}

        return library_generator.get_diffraction_library(
            structure_library, 0.017, 2.4, (72,72) ,'euler')


def test_get_pattern(get_library):
        assert isinstance(get_library.get_pattern(),DiffractionSimulation)
        assert isinstance(get_library.get_pattern(phase='Si'),DiffractionSimulation)
        assert isinstance(get_library.get_pattern(phase='Si',angle=(0,0,0)),DiffractionSimulation)

def test_library_io()
    pass
