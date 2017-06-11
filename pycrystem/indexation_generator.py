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

from __future__ import division

import numpy as np
from hyperspy.signals import BaseSignal
from tqdm import tqdm
from heapq import nlargest
from operator import itemgetter
from transforms3d.euler import euler2axangle
from scipy.constants import pi

from .utils import correlate
from pycrystem.orientation_map import OrientationMap

def correlate_library(image, library, n_largest=None):
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
    res_arr = np.zeros(6)
    top_index = np.where(matching_results.T[-1]==matching_results.T[-1].max())
    res_arr[:5] = matching_results[top_index][0]
    res_arr[5] = res_arr[4] - np.partition(matching_results.T[-1], -2)[-2]
    return res_arr

def euler2axangle_signal(euler):
    return np.array(euler2axangle(euler[0], euler[1], euler[2])[1])


class IndexationGenerator():
    """Generates an indexer for data using a number of methods.
    """
    def __init__(self, signal, library):
        """Initialises the indexer with a diffraction signal and library to be
        correlated in template matching.

        Parameters
        ----------
        signal : :class:`ElectronDiffraction`
            The signal of electron diffraction patterns to be indexed.

        library : :class: `DiffractionLibrary`
            The library of simulated diffraction patterns for indexation

        """
        self.signal = signal
        self.library = library

    def correlate(self,
                  n_largest=5,
                  show_progressbar=True,
                  *args, **kwargs):
        """Correlates the library of simulated diffraction patterns with the
        electron diffraction signal.

        Parameters
        ----------
        n_largest : integer
            The n orientations with the highest correlation values are returned.

        show_progressbar : boolean
            If True a progress bar is shown.

        Returns
        -------
        matching_results : array
            Numpy array with the same shape as the the navigation axes of the
            electron diffraction signal containing correlation results for each
            diffraction pattern.

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
                          phaseid):
        """Obtain matching results for speicified phase.

        Paramters
        ---------
        phaseid = int
            Identifying integer of phase to obtain results for.

        Returns
        -------
        phase_matching_results: Matcsults
            Matching results for the specified phase

        """
        pass


class CrystallographicMap(BaseSignal):

    def __init__(self, *args, **kwargs):
        BaseSignal.__init__(self, *args, **kwargs)
        self.axes_manager.set_signal_dimension(1)

    def get_phase_map(self):
        """Obtain a map of the best matching phase at each navigation position.

        """
        return self.isig[0].as_signal2D((0,1))

    def get_orientation_image(self):
        """Obtain an orientation image of the rotational angle associated with
        the crystal orientation at each navigation position.

        """
        eulers = ori_map.isig[1:4]
        return eulers.map(euler2axangle_signal, inplace=False)

    def get_correlation_map(self):
        """Obtain a correlation map showing the highest correlation score at
        each navigation position.

        """
        return self.isig[4].as_signal2D((0,1))

    def get_reliability_map(self):
        """Obtain a reliability map showing the difference between the highest
        correlation scor and the next best score at each navigation position.
        """
        return self.isig[5].as_signal2D((0,1))

    def savetxt(self):
        pass
