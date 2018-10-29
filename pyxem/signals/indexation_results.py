# -*- coding: utf-8 -*-
# Copyright 2017-2018 The pyXem developers
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
from hyperspy.signal import BaseSignal

from pyxem import CrystallographicMap


def crystal_from_matching_results(z_matches):
    """Takes matching results for a single navigation position
    and returns the best matching phase and orientation with correlation
    and reliability/ies to define a crystallographic map.

    inputs: z_matches a numpy.array (m,5)

    outputs: np.array of shape (6) or (7)
    phase, angle,angle,angle, correlation, R_orientation,(R_phase)
    """

    # count the phases
    if np.unique(z_matches[:, 0]).shape[0] == 1:
        # these case is easier as output is correctly ordered
        results_array = np.zeros(6)
        results_array[:5] = z_matches[0, :5]
        results_array[5] = 100 * (1 -
                                  z_matches[1, 4] / results_array[4])
    else:
        results_array = np.zeros(7)
        index_best_match = np.argmax(z_matches[:, 4])
        # store phase,angle,angle,angle,correlation
        results_array[:5] = z_matches[index_best_match, :5]
        # do reliability_orientation
        z = z_matches[z_matches[:, 0] == results_array[0]]
        second_score = np.partition(z[:, 4], -2)[-2]
        results_array[5] = 100 * (1 -
                                  second_score / results_array[4])
        # and reliability phase
        z = z_matches[z_matches[:, 0] != results_array[0]]
        second_score = np.max(z[:, 4])
        results_array[6] = 100 * (1 -
                                  second_score / results_array[4])

    return results_array


class IndexationResults(BaseSignal):
    _signal_type = "matching_results"
    _signal_dimension = 2

    def __init__(self, *args, **kwargs):
        BaseSignal.__init__(self, *args, **kwargs)
        self.axes_manager.set_signal_dimension(2)

    def get_crystallographic_map(self,
                                 *args, **kwargs):
        """Obtain a crystallographic map specifying the best matching
        phase and orientation at each probe position with corresponding
        correlation and reliabilty scores.

        """
        # TODO: Add alternative methods beyond highest correlation score at each
        # navigation position.
        # TODO Only keep a subset of the data for the map
        cryst_map = self.map(crystal_from_matching_results,
                             inplace=False,
                             *args, **kwargs)
        return CrystallographicMap(cryst_map)
