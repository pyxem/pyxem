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
import pyxem as pxm
from diffsims.sims.diffraction_simulation import DiffractionSimulation
from pyxem.generators.indexation_generator import IndexationGenerator
from diffsims.libraries.diffraction_library import DiffractionLibrary

from pyxem.utils.sim_utils import sim_as_signal

# This test suite is aimed at checking the basic functionality of the
# orientation mapping process, obviously to have a succesful OM process
# many other components will also need to be correct


def create_library():
    dps = []
    half_side_length = 72
    half_shape = (half_side_length, half_side_length)
    num_orientations = 11
    simulations = np.empty(num_orientations, dtype='object')
    orientations = np.empty(num_orientations, dtype='object')
    pixel_coords = np.empty(num_orientations, dtype='object')
    intensities = np.empty(num_orientations, dtype='object')

    # Creating the matchresults.
    for alpha in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        coords = (np.random.rand(5, 2) - 0.5) * 2  # zero mean, range from -1 to +1
        dp_sim = DiffractionSimulation(coordinates=coords,
                                       intensities=np.ones_like(coords[:, 0]),
                                       calibration=1 / half_side_length)
        simulations[alpha] = dp_sim
        orientations[alpha] = (alpha, alpha, alpha)
        pixel_coords[alpha] = (
            dp_sim.calibrated_coordinates[:, :2] + half_shape).astype(int)
        intensities[alpha] = dp_sim.intensities
        if alpha < 4:
            dps.append(
                sim_as_signal(dp_sim,
                    2 * half_side_length,
                    0.075,
                    1).data)  # stores a numpy array of pattern

    library = DiffractionLibrary()
    library["Phase"] = {
        'simulations': simulations,
        'orientations': orientations,
        'pixel_coords': pixel_coords,
        'intensities': intensities,
    }
    dp = pxm.ElectronDiffraction2D([dps[0:2], dps[2:]])  # now from a 2x2 array of patterns
    return dp, library


dp, library = create_library()

indexer = IndexationGenerator(dp, library)
match_results = indexer.correlate(inplane_rotations=[0])


def test_match_results():
    # Note the random number generator may give a different assertion failure
    # This should always work regardless of the RNG.
    assert match_results.inav[0, 0].data[0][1][0] == 0
    assert match_results.inav[1, 0].data[0][1][0] == 1
    assert match_results.inav[0, 1].data[0][1][0] == 2
    assert match_results.inav[1, 1].data[0][1][0] == 3


def test_plot_best_matching_results_on_signal():
    match_results.plot_best_matching_results_on_signal(dp, library=library)
