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
import pytest
import pyxem as pxm
import hyperspy.api as hs
from pyxem.utils.plot import generate_marker_inputs_from_peaks
from pyxem.signals.diffraction_simulation import DiffractionSimulation

"""
When you run this the markers should land at the center of the peaks
near the dots.
"""


def generate_dp_cord_list():
    dp_cord_list = []
    for alpha in [0, 1, 2, 3]:
        coords = np.array([5 + 3 * alpha, 20 * alpha, 0]).reshape(1, 3)
        dp_cord_list.append(coords[:, :2])
    return dp_cord_list


def local_plotter(dp, dp_cord_list):
    peaks = hs.signals.Signal2D(np.array([dp_cord_list[0:2], dp_cord_list[2:]]))
    mmx, mmy = generate_marker_inputs_from_peaks(peaks)
    dp.plot(cmap='viridis')
    for mx, my in zip(mmx, mmy):
        m = hs.markers.point(x=mx, y=my, color='red', marker='x')
        dp.add_marker(m, plot_marker=True, permanent=True)

    return None


def test_marker_placement_correct_alpha():
    dps = []
    dp_cord_list = generate_dp_cord_list()
    for coords in dp_cord_list:
        back = np.zeros((144, 144))
        x = coords.astype(int)[0, 0]
        y = coords.astype(int)[0, 1]
        back[x, y] = 1
        dps.append(back.T)  # stores a numpy array of pattern, This is dangerous (T everywhere)

    dp = pxm.ElectronDiffraction(np.array([dps[0:2], dps[2:]]))
    local_plotter(dp, dp_cord_list)

    # This is human assessed, if you see this comment, you should check it
    assert True

# Now get .as_signal()


def test_marker_placement_correct_beta():
    dps = []
    dp_cord_list = np.divide(generate_dp_cord_list(), 80)
    max_r = np.max(dp_cord_list) + 0.1
    for coords in dp_cord_list:
        dp_sim = DiffractionSimulation(coordinates=coords,
                                       intensities=np.ones_like(coords[:, 0]))
        dps.append(dp_sim.as_signal(144, 0.025, max_r).data)  # stores a numpy array of pattern
    dp = pxm.ElectronDiffraction(np.array([dps[0:2], dps[2:]]))  # now from a 2x2 array of patterns
    dp.set_diffraction_calibration(2 * max_r / (144))
    local_plotter(dp, dp_cord_list)

    # This is human assessed, if you see this comment, you should check it
    assert True
