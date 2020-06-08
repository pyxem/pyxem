# -*- coding: utf-8 -*-
# Copyright 2016-2020 The pyXem developers
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


def create_library_and_diffraction_pattern():
    """ This creates a library, the first 4 entries of which are used to
    create the relevant diffraction patterns, we then test the we get suitable
    results for matching """

    dps = []
    half_shape = (72, 72)
    num_orientations = 11
    simulations = np.empty(num_orientations, dtype="object")
    orientations = np.empty(num_orientations, dtype="object")
    pixel_coords = np.empty(num_orientations, dtype="object")
    intensities = np.empty(num_orientations, dtype="object")

    # Creating the matchresults.
    for alpha in np.arange(11):
        coords = (np.random.rand(5, 2) - 0.5) * 2  # zero mean, range from -1 to +1
        simulated_pattern = DiffractionSimulation(
            coordinates=coords,
            intensities=np.ones_like(coords[:, 0]),
            calibration=1 / 72,
        )

        simulations[alpha] = simulated_pattern
        orientations[alpha] = (alpha, alpha, alpha)
        intensities[alpha] = simulated_pattern.intensities
        pixel_coords[alpha] = (
            simulated_pattern.calibrated_coordinates[:, :2] + half_shape
        ).astype(int)

        if alpha < 4:
            z = sim_as_signal(simulated_pattern, 2 * 72, 0.075, 1)
            dps.append(z.data)  # stores a numpy array of pattern

    dp = pxm.ElectronDiffraction2D(
        [dps[0:2], dps[2:]]
    )  # now from a 2x2 array of patterns
    library = DiffractionLibrary()
    library["Phase"] = {
        "simulations": simulations,
        "orientations": orientations,
        "pixel_coords": pixel_coords,
        "intensities": intensities,
    }

    return dp, library


dp, library = create_library_and_diffraction_pattern()
indexer = IndexationGenerator(dp, library)
match_results = indexer.correlate()


def test_match_results():
    # This should always work regardless of the RNG.
    for zxz_angle in [0, 1, 2]:
        assert match_results.inav[0, 0].data[0][1][zxz_angle] == 0
        assert match_results.inav[1, 0].data[0][1][zxz_angle] == 1
        assert match_results.inav[0, 1].data[0][1][zxz_angle] == 2
        assert match_results.inav[1, 1].data[0][1][zxz_angle] == 3


def test_plot_best_template_matching_results_on_signal():
    # for coverage
    match_results.plot_best_matching_results_on_signal(dp, library=library)
