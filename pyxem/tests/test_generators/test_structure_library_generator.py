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

from pyxem.generators.structure_library_generator import StructureLibraryGenerator
from pyxem.tests.test_utils.test_sim_utils import create_structure_cubic


def test_orientations_from_list():
    expected_orientations = [(0, 0, 0), (0, 90, 0)]
    structure_library_generator = StructureLibraryGenerator([
        ('a', None, 'cubic'),
        ('b', None, 'hexagonal')
    ])
    structure_library = structure_library_generator.get_orientations_from_list(expected_orientations)
    assert structure_library.identifiers == ['a', 'b']
    assert structure_library.structures == [None, None]
    np.testing.assert_almost_equal(structure_library.orientations, expected_orientations)


def test_orientations_from_stereographic_triangle():
    structure_cubic = create_structure_cubic()
    structure_library_generator = StructureLibraryGenerator([('a', structure_cubic, 'cubic')])
    structure_library = structure_library_generator.get_orientations_from_stereographic_triangle([[0]], np.pi / 8)
    assert structure_library.identifiers == ['a']
    assert structure_library.structures == [structure_cubic]
    # Tests for rotation_list_stereographic checks correctness of list content
    assert len(structure_library.orientations) == 1
