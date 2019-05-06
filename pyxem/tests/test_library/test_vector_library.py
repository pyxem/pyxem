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
import pyxem as pxm
import os
import numpy as np
import diffpy.structure

from pyxem.libraries.vector_library import load_VectorLibrary
from pyxem.libraries.structure_library import StructureLibrary


@pytest.fixture
def get_library(default_structure):
    structure_library = StructureLibrary(['Phase'], [default_structure], [[(0, 0, 0), (0, 0.2, 0)]])
    vlg = pxm.VectorLibraryGenerator(structure_library)
    return vlg.get_vector_library(0.5)


def test_library_io(get_library):
    get_library.pickle_library('file_01.pickle')
    loaded_library = load_VectorLibrary('file_01.pickle', safety=True)
    os.remove('file_01.pickle')
    # we can't check that the entire libraries are the same as the memory
    # location of the 'Sim' changes
    np.testing.assert_allclose(get_library['Phase']['measurements'], loaded_library['Phase']['measurements'])
    np.testing.assert_allclose(get_library['Phase']['indices'], loaded_library['Phase']['indices'])


@pytest.mark.xfail(raises=RuntimeError)
def test_unsafe_loading(get_library):
    get_library.pickle_library('file_01.pickle')
    loaded_library = load_VectorLibrary('file_01.pickle')
