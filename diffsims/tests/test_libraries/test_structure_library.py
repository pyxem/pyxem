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

from pyxem.libraries.structure_library import StructureLibrary


def test_constructor():
    identifiers = ['a', 'b']
    # Arbitrary values for tracking
    structures = [1, 2]
    orientations = [3, 4]
    library = StructureLibrary(identifiers, structures, orientations)

    np.testing.assert_equal(library.identifiers, identifiers)
    np.testing.assert_equal(library.structures, structures)
    np.testing.assert_equal(library.orientations, orientations)
    np.testing.assert_equal(library.struct_lib['a'], (1, 3))
    np.testing.assert_equal(library.struct_lib['b'], (2, 4))


@pytest.mark.parametrize('identifiers, structures, orientations', [
    (['a'], [1, 2], [3, 4]),
    (['a'], [1], [3, 4]),
])
@pytest.mark.xfail(raises=ValueError)
def test_constructor_parameter_validation_errors(identifiers, structures, orientations):
    StructureLibrary(identifiers, structures, orientations)
