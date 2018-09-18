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
import pyxem as pxm
import os
import numpy as np
import diffpy.structure

from pyxem.signals.diffraction_simulation import DiffractionSimulation
from pyxem.libraries.diffraction_library import load_DiffractionLibrary
from pyxem.libraries.structure_library import StructureLibrary

@pytest.fixture
def get_library(default_structure):
        diffraction_calculator = pxm.DiffractionGenerator(300., 0.02)
        dfl = pxm.DiffractionLibraryGenerator(diffraction_calculator)
        structure_library = StructureLibrary(['Phase'],[default_structure],[[(0, 0, 0),(0,0.2,0)]])

        return dfl.get_diffraction_library(structure_library, 0.017, 2.4, (72,72))

def test_get_library_entry_assertionless(get_library):
        assert isinstance(get_library.get_library_entry()['Sim'],DiffractionSimulation)
        assert isinstance(get_library.get_library_entry(phase='Phase')['Sim'],DiffractionSimulation)

def test_get_library_small_offeset(get_library):
    alpha = get_library.get_library_entry(phase='Phase',angle=(0,0,0))['intensities']
    beta  = get_library.get_library_entry(phase='Phase',angle=(1e-8,0,0))['intensities']
    assert np.allclose(alpha,beta)


def test_library_io(get_library):
    get_library.pickle_library('file_01.pickle')
    loaded_library = load_DiffractionLibrary('file_01.pickle',safety=True)
    os.remove('file_01.pickle')
    # we can't check that the entire libraries are the same as the memory location of the 'Sim' changes
    assert np.allclose(get_library['Phase'][(0,0,0)]['intensities'],loaded_library['Phase'][(0,0,0)]['intensities'])

@pytest.mark.xfail(raises=ValueError)
def test_unknown_library_entry(get_library):
    # The angle we have asked for is not in the library
    assert isinstance(get_library.get_library_entry(phase='Phase',angle=(1e-3,0,0))['Sim'],DiffractionSimulation)

@pytest.mark.xfail(raises=RuntimeError)
def test_unsafe_loading(get_library):
    get_library.pickle_library('file_01.pickle')
    loaded_library = load_DiffractionLibrary('file_01.pickle')
