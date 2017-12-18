# -*- coding: utf-8 -*-
# Copyright 2017 The PyCrystEM developers
#
# This file is part of PyCrystEM.
#
# PyCrystEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyCrystEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyCrystEM.  If not, see <http://www.gnu.org/licenses/>.

"""Indexation generator and associated tools.

"""

import numpy as np
from hyperspy.signals import BaseSignal
from heapq import nlargest
from operator import itemgetter
from scipy.constants import pi

from .utils import correlate
from .crystallographic_map import CrystallographicMap

def correlate_library(image, library, n_largest):
    """Correlates all simulated diffraction templates in a DiffractionLibrary
    with a particular experimental diffraction pattern (image) stored as a
    numpy array.
    """
    i=0
    out_arr = np.zeros((n_largest * len(library),5))
    for key in library.keys():
        if n_largest:
            pass
        else:
            n_largest=len(library[key])
        correlations = dict()
        for orientation, diffraction_pattern in library[key].items():
            correlation = correlate(image, diffraction_pattern)
            correlations[orientation] = correlation
        res = nlargest(n_largest, correlations.items(), key=itemgetter(1))
        for j in np.arange(n_largest):
            out_arr[j + i*n_largest][0] = i
            out_arr[j + i*n_largest][1] = res[j][0][0]
            out_arr[j + i*n_largest][2] = res[j][0][1]
            out_arr[j + i*n_largest][3] = res[j][0][2]
            out_arr[j + i*n_largest][4] = res[j][1]
        i = i + 1
    return out_arr

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


class IndexationGenerator():
    """Generates an indexer for data using a number of methods.

    Parameters
    ----------
    signal : ElectronDiffraction
        The signal of electron diffraction patterns to be indexed.
    library : DiffractionLibrary
        The library of simulated diffraction patterns for indexation

    """
    def __init__(self, signal, library):
        self.signal = signal
        self.library = library

    def correlate(self,
                  n_largest=5,
                  *args, **kwargs):
        """Correlates the library of simulated diffraction patterns with the
        electron diffraction signal.

        Parameters
        ----------
        n_largest : integer
            The n orientations with the highest correlation values are returned.

        *args/**kwargs : keyword arguments
            Keyword arguments passed to the HyperSpy map() function. Important
            options include...

        Returns
        -------
        matching_results : MatchingResults
            Navigation axes of the electron diffraction signal containing correlation 
            results for each diffraction pattern. As an example, the signal in
            Euler reads ( Library Key , X , Z , X , Correlation Score )

        """
        signal = self.signal
        library = self.library
        matching_results = signal.map(correlate_library,
                                      library=library,
                                      n_largest=n_largest,
                                      inplace=False,
                                      *args, **kwargs)
        return MatchingResults(matching_results)


class MatchingResults(BaseSignal):
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
        """Obtain matching results for speicified phase.

        Paramters
        ---------
        phaseid = int
            Identifying integer of phase to obtain results for.

        Returns
        -------
        phase_matching_results: MatchingResults
            Matching results for the specified phase.

        """
        return self.map(phase_specific_results,
                        phaseid=phaseid,
                        inplace=False,
                        *args, **kwargs)
