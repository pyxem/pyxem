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
from .utils.plot import plot_correlation_map
from pycrystem.orientation_map import OrientationMap

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
                  show_progressbar=True):
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
        #Specify structured array to contain the matching results.
        output_array = np.zeros(signal.axes_manager.navigation_shape,
                                dtype=object).T
        #Iterate through the electron diffraction signal.
        for image, index in tqdm(zip(signal._iterate_signal(),
                signal.axes_manager._array_indices_generator()),
                disable=not show_progressbar,
                total=signal.axes_manager.navigation_size):
            #Specify empty correlation class object to contain correlations.
            phase_correlations = dict()
            #Iterate through the phases in the library.
            for key in library.keys():
                diff_lib = library[key]
                #Specify empty dictionary for orientation correlations of phase
                correlations = dict()
                #Iterate through orientations of the phase.
                for orientation, diffraction_pattern in diff_lib.items():
                    correlation = correlate(image, diffraction_pattern)
                    correlations[orientation] = correlation
                #return n best correlated orientations for phase
                if n_largest:
                    phase_correlations[key] = Correlation(nlargest(n_largest,
                                                          correlations.items(),
                                                          key=itemgetter(1)))
                else:
                    phase_correlations[key] = Correlation(correlations)
            #Put correlation results for navigation position in output array.
            output_array[index] = phase_correlations
        return output_array.T


class MatchingResults(np.ndarray):
    """Container for pattern matching results as a structured array of
    correlations at each navigation index of the indexed diffraction signal.

    """

    def __new__(cls, input_array, info=None):
        obj = np.asarray(input_array).view(cls)
        obj.info = info
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'info', None)

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def get_crystallographic_map(self):
        """Obtain an crystallographic map of phase and orientation specifed by
        an Euler angle triple at each navigation position.

        Returns
        -------
        crystallographic_map : BaseSignal
            Orientation map specifying an Euler angle triple in the rxzx
            convention at each navigation position.

        """
        #TODO:Add smoothing optimisation method with flag.
        best_angle = dict()
        for i in np.arange(self.shape[0]):
            for j in np.arange(self.shape[1]):
                for key in self[i,j].keys():
                    best_angle[key] = (max(self[i,j][key],
                                       key=self[i,j][key].get))
        angles = np.asarray(angle)
        euler_map = BaseSignal(angles.reshape(self.shape[0],
                                              self.shape[1],
                                              3))
        euler_map.axes_manager.set_signal_dimension(1)
        euler_map = euler_map.swap_axes(0,1)
        return OrientationMap(euler_map)


class Correlation(dict):
    """Maps angles to correlation indices.

    Some useful properties and methods are defined.

    """

    @property
    def angles(self):
        """Returns the angles (keys) as a list."""
        return list(self.keys())

    @property
    def correlations(self):
        """Returns the correlations (values) as a list."""
        return list(self.values())

    @property
    def best_angle(self):
        """Returns the angle with the highest correlation index."""
        return max(self, key=self.get)

    @property
    def best_correlation(self):
        """Returns the highest correlation index."""
        return self[self.best_angle]

    @property
    def best(self):
        """Returns angle and value of the highest correlation index."""
        return self.best_angle, self.best_correlation
