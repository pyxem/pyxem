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


def crystal_from_matching_results(matching_results):
    """Takes matching results for a single navigation position and returns the
    best matching phase and orientation with correlation and reliability to
    define a crystallographic map.
    """
    res_arr = np.zeros(6)
    top_index = np.where(matching_results.T[-1]==matching_results.T[-1].max())
    res_arr[:5] = matching_results[top_index][0]
    res_arr[5] = res_arr[4] - np.partition(matching_results.T[-1], -2)[-2]
    return res_arr


def phase_specific_results(matching_results, phaseid):
    """Takes matching results for a single navigation position and returns the
    matching results for a phase specified by a phase id.
    """
    return matching_results.T[:,:len(np.where(matching_results.T[0]==phaseid)[0])].T


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
        #TODO: Add alternative methods beyond highest correlation score at each
        #navigation position.
        cryst_map = self.map(crystal_from_matching_results,
                             inplace=False,
                             *args, **kwargs)
        return CrystallographicMap(cryst_map)

    def get_phase_results(self,
                          phaseid,
                          *args, **kwargs):
        """Obtain matching results for specified phase.

        Parameters
        ----------
        phaseid : int
            Identifying integer of phase to obtain results for.

        Returns
        -------
        phase_matching_results: IndexationResults
            Matching results for the specified phase.

        """
        return self.map(phase_specific_results,
                        phaseid=phaseid,
                        inplace=False,
                        *args, **kwargs)
