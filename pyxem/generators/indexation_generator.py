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
from math import acos, cos, sin, pi, radians, degrees
import itertools

from pyxem.signals.indexation_results import IndexationResults
from pyxem.utils import correlate
from pyxem.utils.expt_utils import _cart2polar
from pyxem.utils.indexation_utils import index_magnitudes
from pyxem.utils.sim_utils import get_interplanar_angle


def correlate_library(image, library,n_largest,keys=[]):
    """Correlates all simulated diffraction templates in a DiffractionLibrary
    with a particular experimental diffraction pattern (image) stored as a
    numpy array. See the correlate method of IndexationGenerator for details.
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
            #diffraction_pattern here is in fact a library of diffraction_pattern_properties
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

def get_vector_pair_indexation(structure, edc, vectors, maximum_length,
                               mag_threshold, angle_threshold):
    """Determine valid indexations for pairs of vectors based on
    their magnitude and the angle between them.

    Parameters
    ----------
    structure : Structure
        The structure for which to calculate the interplanar angle.
    edc : DiffractionGenerator
        Miller indices of first plane.
    guni : DiffractionVectors
        Miller indices of second plane.
    maximum_length : float
        maximum g-vector magnitude for simulation.
    mag_threshold : float
        maximum percentage error in g-vector magnitude for indexation.
    angle_threshold : float
        maximum angular error in radians for g-vector pair interplanar angle.

    Returns
    -------
    indexed_pairs : array
        Array containing possible consistent indexations for each pair
        of g-vectors.

    """

    sim_prof = edc.calculate_profile_data(structure=structure,
                                          reciprocal_radius=maximum_length)
    #get theoretical g-vector magnitudes from family indexation
    magnitudes = np.array(sim_prof.magnitudes)

    rl = structure.lattice.reciprocal_lattice_crystallographic
    recip_pts = rl.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], maximum_length)
    calc_peaks = np.asarray(sorted(recip_pts, key=lambda i: (i[1], -i[0][0], -i[0][1], -i[0][2])))

    mags = vectors.get_magnitudes()
    mag_index = ProfileIndexationGenerator(mags, sim_prof, mapping=False)
    indexation = mag_index.index_peaks(tolerance=mag_threshold)

    #convert array of unique vectors to polar coordinates
    gpolar = np.array(_cart2polar(vectors.data.T[0], vectors.data.T[1]))

    #get array of indices comparing all values in list with all other values in same list.
    reduced_pairs = []
    for pair in np.array(list(itertools.product(np.arange(len(vectors)), np.arange(len(vectors))))):
        #this loop removes self comparisons
        if pair[0]==pair[1]:
            pass
        else:
            reduced_pairs.append(pair)
    reduced_pairs = np.array(reduced_pairs)

    #iterate through all pairs checking theoretical interplanar angle
    indexed_pairs = []

    for pair in reduced_pairs:

        i, j = pair[0], pair[1]

        #get hkl values for all planes in indexed family
        hkls1 = calc_peaks.T[0][np.where(np.isin(calc_peaks.T[1], indexation[i][1][1]))]
        hkls2 = calc_peaks.T[0][np.where(np.isin(calc_peaks.T[1], indexation[j][1][1]))]

        phis = np.zeros((len(hkls1), len(hkls2)))

        for m in np.arange(len(hkls1)):
            for n in np.arange(len(hkls2)):
                hkl1 = hkls1[m]
                hkl2 = hkls2[n]
                #These two special cases give math domain errors so treat separately.
                if np.array_equal(hkl1, hkl2):
                    phis[m,n] = 0
                elif np.array_equal(-hkl1, hkl2):
                    phis[m,n] = pi
                else:
                    phis[m,n] = get_interplanar_angle(structure, hkl1, hkl2)

        phi_expt = gpolar[1][j] - gpolar[1][i]
        phi_diffs = phis - phi_expt

        valid_pairs = np.array(np.where(phi_diffs<angle_threshold))

        indexed_pairs.append([hkls1[valid_pairs[0]], hkls2[valid_pairs[1]]])
    #results give two arrays containing Miller indices for each reflection in pair that are self consistent.
    return np.array(indexed_pairs)


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
        **kwargs
            Keyword arguments passed to the HyperSpy map() function.

        Returns
        -------
        matching_results : pyxem.signals.indexation_results.IndexationResults
            Navigation axes of the electron diffraction signal containing correlation
            results for each diffraction pattern. As an example, the signal in
            Euler reads ( Library Number , Z , X , Z , Correlation Score )


        """
        signal = self.signal
        library = self.library
        matching_results = signal.map(correlate_library,
                                      library=library,
                                      n_largest=n_largest,
                                      keys=keys,
                                      inplace=False,
                                      **kwargs)
        return IndexationResults(matching_results)


class ProfileIndexationGenerator():
    """Generates an indexer for data using a number of methods.

    Parameters
    ----------
    profile : DiffractionProfile
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
            Navigation axes of the electron diffraction signal containing correlation
            results for each diffraction pattern. As an example, the signal in
            Euler reads ( Library Number , Z , X , Z , Correlation Score )


        """
        mapping = self.map
        mags = self.magnitudes
        simulation = self.simulation

        if mapping==True:
            indexation = mags.map(index_magnitudes,
                                  simulation=simulation,
                                  tolerance=tolerance,
                                  **kwargs)

        else:
            mags = np.array(mags)
            sim_mags = np.array(simulation.magnitudes)
            sim_hkls = np.array(simulation.hkls)
            indexation = np.zeros(len(mags), dtype=object)

            for i in np.arange(len(mags)):
                diff = np.absolute((sim_mags - mags.data[i]) / mags.data[i] * 100)

                hkls = sim_hkls[np.where(diff < tolerance)]
                mags_out = sim_mags[np.where(diff < tolerance)]
                diffs = diff[np.where(diff < tolerance)]

                indices = np.array((hkls, mags_out, diffs))
                indexation[i] = np.array((mags.data[i], indices))

        return indexation
