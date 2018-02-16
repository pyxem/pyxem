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

from pyxem.signals.diffraction_simulation import DiffractionSimulation
from pyxem.generators.indexation_generator import IndexationGenerator
from transforms3d.euler import euler2axangle
from pymatgen.transformations.standard_transformations import RotationTransformation
from OM_fixtures import create_GaAs,create_twin_angles,half_side_length, build_linear_grid_in_euler
from pyxem.utils.sim_utils import peaks_from_best_template
from pyxem.utils.plot import generate_marker_inputs_from_peaks
import hyperspy.api as hs

structure = create_GaAs()
edc = pxm.DiffractionGenerator(300, 5e-2)

dps = []
for orientation in create_twin_angles()*2:
    axis, angle = euler2axangle(orientation[0], orientation[1],orientation[2], 'rzxz')
    rotation = RotationTransformation(axis, angle,angle_in_radians=True)
    rotated_structure = rotation.apply_transformation(structure)
    data = edc.calculate_ed_data(rotated_structure,
                                 reciprocal_radius=0.9, #avoiding a reflection issue
                                 with_direct_beam=False)
    dps.append(data.as_signal(2*half_side_length,0.025,1).data)

dp = pxm.ElectronDiffraction([dps[0:2],dps[2:]])
dp.set_calibration(1/half_side_length)

diff_gen = pxm.DiffractionLibraryGenerator(edc)
rot_list = build_linear_grid_in_euler(20,5,5,1) 
struc_lib = dict()
struc_lib["A"] = (structure,rot_list)
library = diff_gen.get_diffraction_library(struc_lib,
                                            calibration=1/half_side_length,
                                            reciprocal_radius=0.6,
                                            half_shape=(half_side_length,half_side_length),
                                            representation='euler',
                                            with_direct_beam=False)
        
indexer = IndexationGenerator(dp,library)
match_results = indexer.correlate()

def test_match_results():
    assert np.all(match_results.inav[0,0].data[0,1:4] == np.array([0,0,0]))
    assert match_results.inav[1,1].data[0,2]   == 0 #no rotation in z for the twinning
    
# Visualization Code 

peaks = match_results.map(peaks_from_best_template,
                          phase=["A"],library=library,inplace=False)

def test_peak_from_best_template():
    # Will fail if top line of test_match_results failed
    assert peaks.inav[0,0] == library["A"][(0,0,0)]['Sim'].coordinates[:,:2] 

mmx,mmy = generate_marker_inputs_from_peaks(peaks)
dp.plot(cmap='viridis')
for mx,my in zip(mmx,mmy):
    ## THERE IS A GOTCHA HERE DUE TO WEAK REFLECTION
    m = hs.markers.point(x=mx,y=my,color='red',marker='x') #see visual test
    dp.add_marker(m,plot_marker=True,permanent=True)

