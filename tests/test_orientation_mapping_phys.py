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

import numpy as np
import pytest
import pyxem as pxm
import hyperspy.api as hs

from pyxem.generators.indexation_generator import IndexationGenerator
from pyxem.libraries.structure_library import StructureLibrary
from pyxem.utils.sim_utils import peaks_from_best_template

"""
The test are designed to make sure orientation mapping works when actual
rotation are considered.

Specifically we test (for both an orthorhombic and hexagonal samples) that:

- The algorithm can tell the difference between down a, down b and down c axes
- The algorithm can tell that +0.2 is better than -0.4 etc for zone axis rotation
"""


half_side_length = 72

@pytest.fixture
def create_Ortho():
    Zn = pmg.Element("Zn")
    lattice = pmg.Lattice.orthorhombic(4.5,4.3,5.7)
    return pmg.Structure.from_spacegroup("P2",lattice, [Zn], [[0, 0, 0]])

@pytest.fixture
def create_Hex():
    Ni = pmg.Element("Ni")
    lattice = pmg.Lattice.hexagonal(3.5,5)
    return pmg.Structure.from_spacegroup(162,lattice, [Ni], [[0, 0, 0]])

@pytest.fixture
def edc():
    return pxm.DiffractionGenerator(300, 5e-2)

@pytest.fixture()
def rot_list():
    from itertools import product
    # about the zone axis
    a,b,c = np.arange(0,0.1,step=0.01),[0],[0]
    rot_list_temp = list(product(a,b,c))
    # to y direction
    a,b,c = [0],[np.deg2rad(90)],np.arange(0,0.1,step=0.01)
    rot_list_temp += list(product(a,b,c))
    # to z direction
    a,b,c = [np.deg2rad(90)],[np.deg2rad(90)],np.arange(0,0.1,step=0.01)
    rot_list_temp += list(product(a,b,c))
    return rot_list_temp

def pattern_rot_list():
    return [(0,0,0.012)] #alpha and gamma are equiv if beta == 0

def get_template_library(structure,rot_list,edc):
    diff_gen = pxm.DiffractionLibraryGenerator(edc)
    struc_lib = StructureLibrary(['A'],[structure],[rot_list])
    library = diff_gen.get_diffraction_library(struc_lib,
                                           calibration=1/half_side_length,
                                           reciprocal_radius=0.8,
                                           half_shape=(half_side_length,
                                                       half_side_length),
                                           with_direct_beam=False)
    return library

@pytest.mark.parametrize("structure",[create_Ortho(),create_Hex()])
@pytest.mark.parametrize("rot_list,edc,pattern_list",[[rot_list(),edc(),
                                                       pattern_rot_list()]])
def test_orientation_mapping_physical(structure,rot_list,pattern_list,edc):
    dp_library = get_template_library(structure,pattern_list,edc)
    for key in dp_library['A']:
        pattern = (dp_library['A'][key]['Sim'].as_signal(2*half_side_length,0.025,1).data)
    dp = pxm.ElectronDiffraction([[pattern,pattern],[pattern,pattern]])
    library = get_template_library(structure,rot_list,edc)
    indexer = IndexationGenerator(dp,library)
    M = indexer.correlate()
    return pattern
    assert np.all(M.inav[0,0] == M.inav[1,0])
    assert np.allclose(M.inav[0,0].isig[:,0].data,[0,0.01,0,0,2],atol=1e-3)

@pytest.mark.parametrize("structure,rot_list,edc,pattern_list",
                         [[create_Ortho(),rot_list(),edc(),pattern_rot_list()]])

def test_masked_OM(structure,rot_list,pattern_list,edc):
    dp_library = get_template_library(structure,pattern_list,edc)
    for key in dp_library['A']:
        pattern = (dp_library['A'][key]['Sim'].as_signal(2*half_side_length,0.025,1).data)
    dp = pxm.ElectronDiffraction([[pattern,pattern],[pattern,pattern]])
    library = get_template_library(structure,rot_list,edc)
    indexer = IndexationGenerator(dp,library)
    mask = hs.signals.Signal1D(([[[1],[1]],[[0],[1]]]))
    M = indexer.correlate(mask=mask)
    assert np.all(np.isnan(M.inav[0,1].data))

@pytest.mark.skip()
def test_generate_peaks_from_best_template():
    # also test peaks from best template
    peaks = M.map(peaks_from_best_template,
                  phase=["A"],library=library,inplace=False)
    assert peaks.inav[0,0] == library["A"][(0,0,0)]['Sim'].coordinates[:,:2]
