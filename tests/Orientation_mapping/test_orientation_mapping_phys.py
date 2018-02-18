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
from pyxem.signals.diffraction_simulation import DiffractionSimulation
from pyxem.generators.indexation_generator import IndexationGenerator
from OM_fixtures import create_GaAs, build_linear_grid_in_euler,half_side_length,create_sample
from pyxem.utils.sim_utils import peaks_from_best_template
from pyxem.utils.plot import generate_marker_inputs_from_peaks

def build_structure_lib(structure,rot_list):
    struc_lib = dict()
    struc_lib["A"] = (structure,rot_list)
    return struc_lib    

rot_list = build_linear_grid_in_euler(12,10,5,1)
structure = create_GaAs()
edc = pxm.DiffractionGenerator(300, 5e-2)

@pytest.fixture
def get_template_library(structure,rot_list,edc):    
    diff_gen = pxm.DiffractionLibraryGenerator(edc)
    struc_lib = build_structure_lib(structure,rot_list)
    library = diff_gen.get_diffraction_library(struc_lib,
                                           calibration=1/half_side_length,
                                           reciprocal_radius=0.6,
                                           half_shape=(half_side_length,half_side_length),
                                           representation='euler',
                                           with_direct_beam=False)
    return library

### Test two rotation direction on axis and an arbitary rotation direction

"""
Case A -
We rotate about 4 on the first, z axis, as we don't rotate around x at all we can
then rotate again around the second z axis in a similar way
CLASS THIS
"""

def TestClassA(self,structure,rot_list,edc):
    dp = create_sample(edc,structure,[0,0,0],[4,0,0])
    indexer = IndexationGenerator(dp,get_template_library())
    match_results_A = indexer.correlate()

    def test_match_results_essential():
        assert np.all(match_results.inav[0,0] == match_results.inav[1,0])
        assert np.all(match_results.inav[0,1] == match_results.inav[1,1])

    def test_peak_from_best_template():
        # Will fail if top line of test_match_results failed
        peaks = match_results.map(peaks_from_best_template,phase=["A"],library=library,inplace=False)
        assert peaks.inav[0,0] == library["A"][(0,0,0)]['Sim'].coordinates[:,:2] 

    def test_match_results_caseA():
        assert np.all(match_results_A.inav[0,0].data[0,1:4] == np.array([0,0,0]))
        assert match_results_A.inav[1,1].data[0,2]   == 0 #no rotation in z for the twinning
        #rotation totals must equal 4, and each must give the same coefficient
        assert np.all(np.sum(match_results_A.inav[1,1].data[:,1:4],axis=1) == 4)
        assert np.all(match_results_A.inav[1,1].data[:,4] == match_results_A.inav[1,1].data[0,4])

    #test_match_results_essential()
    #test_peak_from_best_template()
    test_match_results_caseA()

"""

#Case B -
#We rotate all 3 and test that we get good answers
#CLASS THIS


dp = create_sample(edc,structure,[0,0,0],[3,7,1])
indexer = IndexationGenerator(dp,library)
match_results = indexer.correlate()

test_match_results_essential()

def test_match_results_caseB():
    assert np.all(match_results.inav[1,1].data[0,1:4] == np.array([3,7,1]))

test_match_results_caseB()

#Case C - Use non-integers
#CLASS THIS

dp = create_sample(edc,structure,[0,0,0],[3,7.01,0.99])
indexer = IndexationGenerator(dp,library)
match_results = indexer.correlate()

test_match_results_essential()

def test_match_results_caseC():
    assert np.all(match_results.inav[1,1].data[0,1:4] == np.array([3,7,1]))
    
test_match_results_caseC()
"""
"""
# Visualization Code 
peaks = match_results.map(peaks_from_best_template,phase=["A"],library=library,inplace=False)
mmx,mmy = generate_marker_inputs_from_peaks(peaks)
dp.plot(cmap='viridis')
for mx,my in zip(mmx,mmy):
    ## THERE IS A GOTCHA HERE DUE TO WEAK REFLECTION
    m = hs.markers.point(x=mx,y=my,color='red',marker='x') #see visual test
    dp.add_marker(m,plot_marker=True,permanent=True)
"""
