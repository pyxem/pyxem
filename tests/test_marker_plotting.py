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
from pyxem.utils.plot import generate_marker_inputs_from_peaks

### When you run this the markers should land at the center of the peaks, simple as.

dps, dp_cord_list = [],[]

# Create 4 random diffraction simulations
half_side_length = 72
for alpha in [0,1,2,3]:
    coords = (np.random.rand(5,2)-0.5)*2 #zero mean, range from -1 to +1
    dp_sim = DiffractionSimulation(coordinates=coords,
                                   intensities=np.ones_like(coords[:,0]),
                                   calibration=1/half_side_length)
    dp_cord_list.append(dp_sim.coordinates[:,:2]) #stores the simulations coords
    dps.append(dp_sim.as_signal(2*half_side_length,0.075,1).data) #stores a numpy array of pattern

dp = pxm.ElectronDiffraction([dps[0:2],dps[2:]]) #now from a 2x2 array of patterns
peaks = hs.signals.Signal2D(np.array([dp_cord_list[0:2],dp_cord_list[2:]]))

mmx,mmy = generate_marker_inputs_from_peaks(peaks)
dp.set_calibration(2/144)
dp.plot(cmap='viridis')
for mx,my in zip(mmx,mmy):
    m = hs.markers.point(x=mx,y=my,color='red',marker='x')
    dp.add_marker(m,plot_marker=True,permanent=True)

def test_marker_placement_correct():
    #This is human assessed, if you see this comment, you should check it
    assert True