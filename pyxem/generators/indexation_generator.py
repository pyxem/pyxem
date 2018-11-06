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

import numpy as np
import hyperspy.api as hs
from math import acos, cos, sin, pi, radians, degrees
import itertools

from pyxem.signals.indexation_results import IndexationResults

from pyxem.utils.sim_utils import carry_through_navigation_calibration

from pyxem.utils.indexation_utils import index_magnitudes
from pyxem.utils.indexation_utils import correlate_library

import hyperspy.api as hs


class IndexationGenerator():
    """Generates an indexer for data using a number of methods.

    Parameters
    ----------
    signal : ElectronDiffraction
        The signal of electron diffraction patterns to be indexed.
    diffraction_library : DiffractionLibrary
        The library of simulated diffraction patterns for indexation.
    """

    def __init__(self,
                 signal,
                 diffraction_library):
        self.signal = signal
        self.library = diffraction_library

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
        *args : arguments
            Arguments passed to map().
        **kwargs : arguments
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
            # index at all real space pixels
            sig_shape = signal.axes_manager.navigation_shape
            mask = hs.signals.Signal1D(np.ones((sig_shape[0], sig_shape[1], 1)))

        matches = signal.map(correlate_library,
                             library=library,
                             n_largest=n_largest,
                             keys=keys,
                             mask=mask,
                             inplace=False,
                             **kwargs)
        matching_results = IndexationResults(matches)

        matching_results = carry_through_navigation_calibration(matching_results, signal)

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

    def __init__(self, magnitudes, simulation, mapping=True):
        self.map = mapping
        self.magnitudes = magnitudes
        self.simulation = simulation

    def index_peaks(self,
                    tolerance=0.1,
                    *args,
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
        **kwargs : arguments
            Keyword arguments passed to the HyperSpy map() function.

        Returns
        -------
        matching_results : pyxem.signals.indexation_results.IndexationResults
            Navigation axes of the electron diffraction signal containing
            correlation results for each diffraction pattern. As an example, the
            signal in Euler reads:
                    ( Library Number , Z , X , Z , Correlation Score)

        """
        mapping = self.map
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


class VectorIndexationGenerator():
    """Generates an indexer for DiffractionVectors using a number of methods.
    Parameters
    ----------
    vectors : DiffractionVectors
        DiffractionVectors to be indexed.
    vector_library : DiffractionVectorLibrary
        The library of simulated diffraction patterns for indexation.
    """

    def __init__(self,
                 vectors,
                 vector_library):
        self.vectors = vectors
        self.library = vector_library

    def index_vectors(self,
                      max_length,
                      mag_threshold,
                      angle_threshold,
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
            Navigation axes of the electron diffraction signal containing vector
            indexation results for each diffraction pattern.
        """
        mapping = self.map
        vectors = self.vectors
        structure = self.strucutre
        edc = self.edc

        # set up simulator
        sim_prof = edc.calculate_profile_data(structure=structure,
                                              reciprocal_radius=max_length)
        # get theoretical g-vector magnitudes from family indexation
        magnitudes = np.array(sim_prof.magnitudes)
        # assign possible indices based on magnitude alone
        mags = vectors.get_magnitudes()
        mag_index = ProfileIndexationGenerator(mags, sim_prof, mapping=False)
        indexation = mag_index.index_peaks(tolerance=mag_threshold)

        if mapping == True:
            indexation = vectors.map(get_vector_pair_indexation,
                                     structure=structure,
                                     edc=edc,
                                     magnitudes=magnitudes,
                                     indexation=indexation,
                                     max_length=max_length,
                                     mag_threshold=mag_threshold,
                                     angle_threshold=angle_threshold,
                                     **kwargs)

        else:
            # compare theory with experiment with threshold on mag of difference
            phi_diffs = phis - np.absolute(phi_expt)
            valid_pairs = np.array(np.where(np.abs(phi_diffs) < angle_threshold))
            # obtain Miller indices corresponding to planes satisfying mag + angle.
            indexed_pairs.append([vectors.data[i], hkls1[valid_pairs[0]], vectors.data[j], hkls2[valid_pairs[1]]])
            # results give two arrays containing Miller indices for each reflection in pair that are self consistent.
            indexation = np.array(indexed_pairs)

        return indexation
