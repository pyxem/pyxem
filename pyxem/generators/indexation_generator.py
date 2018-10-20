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

"""Indexation generator and associated tools.

"""

from heapq import nlargest
from operator import itemgetter

import numpy as np
from pyxem.signals.indexation_results import IndexationResults

from pyxem.utils import correlate

import hyperspy.api as hs

def correlate_library(image, library,n_largest,mask,keys=[]):
    """Correlates all simulated diffraction templates in a DiffractionLibrary
    with a particular experimental diffraction pattern (image) stored as a
    numpy array. See the correlate method of IndexationGenerator for details.
    """
    i=0
    out_arr = np.zeros((n_largest * len(library),5))
    if mask == 1:
        for key in library.keys():
            correlations = dict()
            for orientation, diffraction_pattern in library[key].items():
                #diffraction_pattern here is in fact a library of
                #diffraction_pattern_properties
                correlation = correlate(image, diffraction_pattern)
                correlations[orientation] = correlation
                res = nlargest(n_largest, correlations.items(),
                               key=itemgetter(1))
            for j in np.arange(n_largest):
                out_arr[j + i*n_largest][0] = i
                out_arr[j + i*n_largest][1] = res[j][0][0]
                out_arr[j + i*n_largest][2] = res[j][0][1]
                out_arr[j + i*n_largest][3] = res[j][0][2]
                out_arr[j + i*n_largest][4] = res[j][1]
            i = i + 1

    else:
        for j in np.arange(n_largest):
            for k in [0,1,2,3,4]:
                out_arr[j + i*n_largest][k] = np.nan
        i = i + 1
    return out_arr


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
                  keys=[],
                  mask=None,
                  *args,
                  **kwargs):
        """Correlates the library of simulated diffraction patterns with the
        electron diffraction signal.

        Parameters
        ----------
        n_largest : int
            The n orientations with the highest correlation values are returned.
        keys : list
            If more than one phase present in library it is recommended that
            these are submitted. This allows a mapping from the number to the
            phase.  For example, keys = ['si','ga'] will have an output with 0
            for 'si' and 1 for 'ga'.
        mask : Array
            Array with the same size as signal (in navigation) True False
        *args
            Arguments passed to map().
        **kwargs
            Keyword arguments passed map().

        Returns
        -------
        matching_results : pyxem.signals.indexation_results.IndexationResults
            Navigation axes of the electron diffraction signal containing
            correlation results for each diffraction pattern. As an example, the
            signal in Euler reads:
                    ( Library Number , Z , X , Z , Correlation Score)

        """
        signal = self.signal
        library = self.library
        if mask is None:
           #index at all real space pixels
           sig_shape = signal.axes_manager.navigation_shape
           mask = hs.signals.Signal1D(np.ones((sig_shape[0],sig_shape[1],1)))

        matches = signal.map(correlate_library,
                             library=library,
                             n_largest=n_largest,
                             keys=keys,
                             mask=mask,
                             inplace=False,
                             **kwargs)
        matching_results = IndexationResults(matching_results)

        #Set calibration to same as signal
        x = matching_results.axes_manager.navigation_axes[0]
        y = matching_results.axes_manager.navigation_axes[1]

        x.name = 'x'
        x.scale = signal.axes_manager.navigation_axes[0].scale
        x.units = 'nm'

        y.name = 'y'
        y.scale = signal.axes_manager.navigation_axes[0].scale
        y.units = 'nm'

        return matching_results


class ProfileIndexationGenerator():
    """Generates an indexer for data using a number of methods.

    Parameters
    ----------
    profile : ElectronDiffractionProfile
        The signal of diffraction profiles to be indexed.
    library : ProfileSimulation
        The simulated profile data.

    """
    def __init__(self, magnitudes, simulation):
        self.magnitudes = magnitudes
        self.simulation = simulation

    def index_peaks(self,
                    tolerance=0.1,
                    **kwargs):
        """Assigns hkl indices to peaks in the diffraction profile.

        Parameters
        ----------
        tolerance : float
            The n orientations with the highest correlation values are returned.
        keys : list
            If more than one phase present in library it is recommended that
            these are submitted. This allows a mapping from the number to the
            phase.  For example, keys = ['si','ga'] will have an output with 0
            for 'si' and 1 for 'ga'.
        **kwargs
            Keyword arguments passed to the HyperSpy map() function.

        Returns
        -------
        matching_results : pyxem.signals.indexation_results.IndexationResults
            Navigation axes of the electron diffraction signal containing
            correlation results for each diffraction pattern. As an example, the
            signal in Euler reads:
                    ( Library Number , Z , X , Z , Correlation Score)

        """
        mags = self.magnitudes
        simulation = self.simulation

        mags = np.array(mags)
        sim_mags = np.array(simulation.magnitudes)
        sim_hkls = np.array(simulation.hkls)
        indexation = np.zeros(len(mags), dtype=object)

        for i in np.arange(len(mags)):
            diff = np.absolute((sim_mags - mags.data[i]) / mags.data[i] * 100)

            hkls = sim_hkls[np.where(diff < tolerance)]
            diffs = diff[np.where(diff < tolerance)]

            indices = np.array((hkls, diffs))
            indexation[i] = np.array((mags.data[i], indices))

        return indexation
