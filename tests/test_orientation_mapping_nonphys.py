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
from pyxem.libraries.diffraction_library import DiffractionLibrary

# This test suite is aimed at checking the basic functionality of the Omapping process, obviously to have a succesful OM process
# many other components will also need to be correct

def create_library_entry(library,rotation,pattern):
    library["Phase"][rotation] = {}
    library["Phase"][rotation]['Sim'] = pattern
    library["Phase"][rotation]['intensities']  = pattern.intensities
    library["Phase"][rotation]['pixel_coords'] = (pattern.calibrated_coordinates[:,:2]+half_shape).astype(int)
    library["Phase"][rotation]['pattern_norm'] = np.sqrt(np.dot(pattern.intensities,pattern.intensities))
    return DiffractionLibrary(library)

dps, dp_sim_list = [],[]
half_side_length = 72
library = dict()
half_shape = (half_side_length,half_side_length)
library["Phase"] = {}

# Creating the matchresults.

for alpha in [0,1,2,3,4,5,6,7,8,9,10]:
    coords = (np.random.rand(5,2)-0.5)*2 #zero mean, range from -1 to +1
    dp_sim = DiffractionSimulation(coordinates=coords,
                                   intensities=np.ones_like(coords[:,0]),
                                   calibration=1/half_side_length)
    if alpha < 4:
        dps.append(dp_sim.as_signal(2*half_side_length,0.075,1).data) #stores a numpy array of pattern

    library = create_library_entry(library,(alpha,alpha,alpha),dp_sim) #this add a new entry to the library

dp = pxm.ElectronDiffraction([dps[0:2],dps[2:]]) #now from a 2x2 array of patterns

indexer = IndexationGenerator(dp,library)
match_results = indexer.correlate()

def test_match_results():
    # Note the random number generator may give a different assertion failure
    # This should always work regardless of the RNG.
    assert match_results.inav[0,0].data[0][1] == 0
    assert match_results.inav[1,0].data[0][1] == 1
    assert match_results.inav[0,1].data[0][1] == 2
    assert match_results.inav[1,1].data[0][1] == 3

def test_visuals():
    ## This functions will need to abuse globals.
    ## & Can be removed if we trust the other tests
    from pyxem.utils.sim_utils import peaks_from_best_template
    from pyxem.utils.plot import generate_marker_inputs_from_peaks
    import hyperspy.api as hs

    peaks = match_results.map(peaks_from_best_template,
                          phase=["Phase"],library=library,inplace=False)
    mmx,mmy = generate_marker_inputs_from_peaks(peaks)
    dp.set_diffraction_calibration(2/144)
    dp.plot(cmap='viridis')
    for mx,my in zip(mmx,mmy):
        m = hs.markers.point(x=mx,y=my,color='red',marker='x')
        dp.add_marker(m,plot_marker=True,permanent=True)

    # Hand checking again
    assert True
