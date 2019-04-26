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

"""Indexation generator and associated tools.

"""

from heapq import nlargest
from operator import itemgetter

import numpy as np
import hyperspy.api as hs
from math import acos, cos, sin, pi, radians, degrees
import itertools

from pyxem.signals.indexation_results import TemplateMatchingResults
from pyxem.signals.indexation_results import VectorMatchingResults

from pyxem.utils.sim_utils import transfer_navigation_axes

from pyxem.utils.indexation_utils import correlate_library
from pyxem.utils.indexation_utils import index_magnitudes
from pyxem.utils.indexation_utils import match_vectors

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
                  mask=None,
                  inplane_rotations=np.arange(0, 360, 1),
                  max_peaks=100,
                  *args,
                  **kwargs):
        """Correlates the library of simulated diffraction patterns with the
        electron diffraction signal.

        Parameters
        ----------
        n_largest : int
            The n orientations with the highest correlation values are returned.
        mask : Array
            Array with the same size as signal (in navigation) True False
        inplane_rotations : ndarray
            Array of inplane rotation angles in degrees. Defaults to 0-360 degrees
            at 1 degree resolution.
        max_peaks : int
            Maximum number of peaks to consider when comparing a template to
            the diffraction pattern. The strongest peaks are kept.
        *args : arguments
            Arguments passed to map().
        **kwargs : arguments
            Keyword arguments passed map().

        Returns
        -------
        matching_results : TemplateMatchingResults
            Navigation axes of the electron diffraction signal containing
            correlation results for each diffraction pattern, in the form
            [Library Number , [z, x, z], Correlation Score]

        """
        signal = self.signal
        library = self.library
        inplane_rotations = np.deg2rad(inplane_rotations)
        num_inplane_rotations = inplane_rotations.shape[0]
        sig_shape = signal.axes_manager.signal_shape
        signal_half_width = sig_shape[0] / 2

        if mask is None:
            # Index at all real space pixels
            mask = 1

        # Create a copy of the library, cropping and padding the peaks to match
        # max_peaks. Also create rotated pixel coordinates according to
        # inplane_rotations
        rotation_matrices_2d = np.array([[[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]] for t in inplane_rotations])
        cropped_library = {}

        for phase_name, phase_entry in library.items():
            num_orientations = len(phase_entry['orientations'])
            intensities_jagged = phase_entry['intensities']
            intensities = np.zeros((num_orientations, max_peaks))
            pixel_coords_jagged = phase_entry['pixel_coords']
            pixel_coords = np.zeros((num_inplane_rotations, num_orientations, max_peaks, 2))
            for i in range(num_orientations):
                num_peaks = min(pixel_coords_jagged[i].shape[0], max_peaks)
                intensities[i, :num_peaks] = intensities_jagged[i][:num_peaks]
                # Get and compute pixel coordinates for all rotations about the
                # center, clipped to the detector size and rounded to integer positions.
                pixel_coords[:, i, :num_peaks] = np.clip(
                    (signal_half_width + rotation_matrices_2d @ (
                        pixel_coords_jagged[i][:num_peaks].T - signal_half_width)).transpose(0, 2, 1),
                    a_min=0,
                    a_max=np.array(sig_shape) - 1)

            np.rint(pixel_coords, out=pixel_coords)
            cropped_library[phase_name] = {
                'orientations': phase_entry['orientations'],
                'pixel_coords': pixel_coords.astype('int'),
                'intensities': intensities,
                'pattern_norms': np.linalg.norm(intensities, axis=1),
            }

        matches = signal.map(correlate_library,
                             library=cropped_library,
                             n_largest=n_largest,
                             mask=mask,
                             inplace=False,
                             **kwargs)

        matching_results = TemplateMatchingResults(matches)
        matching_results = transfer_navigation_axes(matching_results, signal)

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
        *args : arguments
            Arguments passed to the map() function.
        **kwargs : arguments
            Keyword arguments passed to the map() function.

        Returns
        -------
        matching_results : ProfileIndexation

        """
        return index_magnitudes(np.array(self.magnitudes), self.simulation, tolerance)


class VectorIndexationGenerator():
    """Generates an indexer for DiffractionVectors using a number of methods.

    Attributes
    ----------
    vectors : DiffractionVectors
        DiffractionVectors to be indexed.
    vector_library : DiffractionVectorLibrary
        Library of theoretical diffraction vector magnitudes and inter-vector
        angles for indexation.

    Parameters
    ----------
    vectors : DiffractionVectors
        DiffractionVectors to be indexed.
    vector_library : DiffractionVectorLibrary
        Library of theoretical diffraction vector magnitudes and inter-vector
        angles for indexation.
    """

    def __init__(self,
                 vectors,
                 vector_library):
        if vectors.cartesian is None:
            raise ValueError("Cartesian coordinates are required in order to index "
                             "diffraction vectors. Use the calculate_cartesian_coordinates "
                             "method of DiffractionVectors to obtain these.")
        else:
            self.vectors = vectors
            self.library = vector_library

    def index_vectors(self,
                      mag_tol,
                      angle_tol,
                      index_error_tol,
                      n_peaks_to_index,
                      n_best,
                      keys=[],
                      *args,
                      **kwargs):
        """Assigns hkl indices to diffraction vectors.

        Parameters
        ----------
        mag_tol : float
            The maximum absolute error in diffraction vector magnitude, in units
            of reciprocal Angstroms, allowed for indexation.
        angle_tol : float
            The maximum absolute error in inter-vector angle, in units of
            degrees, allowed for indexation.
        index_error_tol : float
            Max allowed error in peak indexation for classifying it as indexed,
            calculated as :math:`|hkl_calculated - round(hkl_calculated)|`.
        n_peaks_to_index : int
            The maximum number of peak to index.
        n_best : int
            The maximum number of good solutions to be retained.
        keys : list
            If more than one phase present in library it is recommended that
            these are submitted. This allows a mapping from the number to the
            phase.  For example, keys = ['si','ga'] will have an output with 0
            for 'si' and 1 for 'ga'.
        *args : arguments
            Arguments passed to the map() function.
        **kwargs : arguments
            Keyword arguments passed to the map() function.

        Returns
        -------
        indexation_results : VectorMatchingResults
            Navigation axes of the diffraction vectors signal containing vector
            indexation results for each probe position.
        """
        vectors = self.vectors
        library = self.library

        matched = vectors.cartesian.map(match_vectors,
                                        library=library,
                                        mag_tol=mag_tol,
                                        angle_tol=np.deg2rad(angle_tol),
                                        index_error_tol=index_error_tol,
                                        n_peaks_to_index=n_peaks_to_index,
                                        n_best=n_best,
                                        keys=keys,
                                        inplace=False,
                                        *args,
                                        **kwargs)
        indexation = np.array(matched.isig[0].data.tolist(), dtype='object')
        rhkls = matched.isig[1].data

        indexation_results = VectorMatchingResults(indexation)
        indexation_results.vectors = vectors
        indexation_results.hkls = rhkls
        indexation_results = transfer_navigation_axes(indexation_results,
                                                      vectors.cartesian)

        vectors.hkls = rhkls

        return indexation_results
