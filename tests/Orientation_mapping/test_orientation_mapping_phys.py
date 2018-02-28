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
import pymatgen as pmg

from pyxem.signals.diffraction_simulation import DiffractionSimulation
from pyxem.generators.indexation_generator import IndexationGenerator
from pyxem.utils.sim_utils import peaks_from_best_template
from pyxem.utils.plot import generate_marker_inputs_from_peaks
from transforms3d.euler import euler2axangle
from pymatgen.transformations.standard_transformations import RotationTransformation

half_side_length = 72

def create_GaAs():
    Ga = pmg.Element("Ga")
    As = pmg.Element("As")
    lattice = pmg.Lattice.cubic(5.6535)
    return pmg.Structure.from_spacegroup("F23",lattice, [Ga,As], [[0, 0, 0],[0.25,0.25,0.25]])

def create_pair(angle_start,angle_change):
    """ Lists for angles """
    angle_2 = np.add(angle_start,angle_change)
    return [angle_start,angle_start,angle_2,angle_2]

def build_linear_grid_in_euler(alpha_max,beta_max,gamma_max,resolution):
    a = np.arange(0,alpha_max,step=resolution)
    b = np.arange(0,beta_max,step=resolution)
    c = np.arange(0,gamma_max,step=resolution)
    from itertools import product
    return list(product(a,b,c))

def create_sample(edc,structure,angle_start,angle_change):
    dps = []
    for orientation in create_pair(angle_start,angle_change):
        axis, angle = euler2axangle(orientation[0], orientation[1],orientation[2], 'rzxz')
        rotation = RotationTransformation(axis, angle,angle_in_radians=True)
        rotated_structure = rotation.apply_transformation(structure)
        data = edc.calculate_ed_data(rotated_structure,
                                 reciprocal_radius=0.9, #avoiding a reflection issue
                                 with_direct_beam=False)
        dps.append(data.as_signal(2*half_side_length,0.025,1).data)
    dp = pxm.ElectronDiffraction([dps[0:2],dps[2:]])
    dp.set_calibration(1/half_side_length)
    return dp

def build_structure_lib(structure,rot_list):
    struc_lib = dict()
    struc_lib["A"] = (structure,rot_list)
    return struc_lib    

rot_list = build_linear_grid_in_euler(12,10,5,1)
structure = create_GaAs()
edc = pxm.DiffractionGenerator(300, 5e-2)

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

def get_match_results_case_a(structure,rot_list,edc): #get params
    """
    Case A -
    We rotate about 4 on the first, z axis, as we don't rotate around x at all we can
    then rotate again around the second z axis in a similar way
    """
    dp = create_sample(edc,structure,[0,0,0],[4,0,0])
    library = get_template_library(structure,rot_list,edc)
    indexer = IndexationGenerator(dp,library)
    match_results = indexer.correlate()
    return match_results

def get_match_results_case_b(structure,rot_list,edc):
    """
    Case B -
    Now an arbitary rotation
    """
    dp = create_sample(edc,structure,[0,0,0],[3,7,1])
    library = get_template_library(structure,rot_list,edc)
    indexer = IndexationGenerator(dp,library)
    match_results = indexer.correlate()
    return match_results


def get_match_results_case_c(structure,rot_list,edc):
    """
    Case C -
    Use non-ints for the rotation in B
    """
    dp = create_sample(edc,structure,[0,0,0],[3,7.01,0.99])
    library = get_template_library(structure,rot_list,edc)
    indexer = IndexationGenerator(dp,library)
    match_results = indexer.correlate()
    return match_results

"""
#This runs the test twice, but only the test directly below -

rot_list_1 = build_linear_grid_in_euler(12,10,5,1)
rot_list_2 = build_linear_grid_in_euler(12,10,7,5)
@pytest.mark.parametrize("structure",[create_GaAs()])
@pytest.mark.parametrize("rot_list",[rot_list_1,rot_list_2])
@pytest.mark.parametrize("edc",[edc])
"""

@pytest.mark.parametrize("structure",[create_GaAs()])
@pytest.mark.parametrize("rot_list",[rot_list])
@pytest.mark.parametrize("edc",[edc])
    
def test_match_results_essential(structure,rot_list,edc):
    M = get_match_results_case_a(structure,rot_list,edc) #for concision
    assert np.all(M.inav[0,0] == M.inav[1,0])
    assert np.all(M.inav[0,1] == M.inav[1,1])
        
    # also test peaks from best template
    library = get_template_library(structure,rot_list,edc)
    peaks = M.map(peaks_from_best_template,phase=["A"],library=library,inplace=False)
    assert peaks.inav[0,0] == library["A"][(0,0,0)]['Sim'].coordinates[:,:2] 

@pytest.mark.parametrize("structure",[create_GaAs()])
@pytest.mark.parametrize("rot_list",[rot_list])
@pytest.mark.parametrize("edc",[edc])

def test_match_results_caseA(structure,rot_list,edc):
    M = get_match_results_case_a(structure,rot_list,edc)
    assert np.all(M.inav[0,0].data[0,1:4] == np.array([0,0,0]))
    assert M.inav[1,1].data[0,2]   == 0 #no rotation in z 
    
    #for the twinning the rotation totals must equal 4 
    assert np.all(np.sum(M.inav[1,1].data[:,1:4],axis=1) == 4)
    #and each must give the same coefficient
    assert np.all(M.inav[1,1].data[:,4] == M.inav[1,1].data[0,4])

@pytest.mark.parametrize("structure",[create_GaAs()])
@pytest.mark.parametrize("rot_list",[rot_list])
@pytest.mark.parametrize("edc",[edc])

def test_match_results_caseB(structure,rot_list,edc):
    M = get_match_results_case_b(structure,rot_list,edc)
    assert np.all(M.inav[1,1].data[0,1:4] == np.array([3,7,1]))

@pytest.mark.parametrize("structure",[create_GaAs()])
@pytest.mark.parametrize("rot_list",[rot_list])
@pytest.mark.parametrize("edc",[edc])
     
def test_match_results_caseC(structure,rot_list,edc):
    M = get_match_results_case_c(structure,rot_list,edc)
    assert np.all(M.inav[1,1].data[0,1:4] == np.array([3,7,1]))
    
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
