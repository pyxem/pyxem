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

from heapq import nlargest
from itertools import combinations
from operator import itemgetter

from pyxem.utils import correlate
from pyxem.utils.vector_utils import get_rotation_matrix_between_vectors
from pyxem.utils.vector_utils import get_angle_cartesian

from transforms3d.euler import mat2euler


def correlate_library(image, library, n_largest, mask, keys=[]):
    """Correlates all simulated diffraction templates in a DiffractionLibrary
    with a particular experimental diffraction pattern (image).

    Parameters
    ----------
    image : np.array()
        The experimental diffraction pattern of interest.
    library : DiffractionLibrary
        The library of diffraction simulations to be correlated with the
        experimental data.
    n_largest : int
        The number of well correlated simulations to be retained.
    mask : bool array
        A mask for navigation axes 1 indicates positions to be indexed.

    Returns
    -------
    top_matches : (<num phases>*n_largest, 5), np.array()
        A numpy array containing the top n correlated simulations for the
        experimental pattern of interest.

    See also
    --------
    pyxem.utils.correlate and the correlate method of IndexationGenerator.
    """
    top_matches = np.zeros((len(library), n_largest, 5))
    if mask == 1:
        for phase_index, key in enumerate(library.keys()):
            correlations = np.empty((len(library[key]), 4))
            # Use enumerate to index, i, each (orientation, diffraction_pattern) in list
            for i, (orientation, diffraction_pattern) in enumerate(library[key].items()):
                correlation = correlate(image, diffraction_pattern)
                correlations[i, :] = *orientation, correlation

            # Partition to get the n_largest best matches
            top_n = correlations[correlations[:, 3].argpartition(-n_largest)[-n_largest:]]
            # Sort the matches by correlation score, descending
            top_n = top_n[top_n[:, 3].argsort()][::-1]

            top_matches[phase_index, :, 0] = phase_index
            top_matches[phase_index, :, 1:] = top_n
    else:
        top_matches.fill(np.nan)
    return top_matches.reshape((len(library) * n_largest, 5))


def index_magnitudes(z, simulation, tolerance):
    """Assigns hkl indices to peaks in the diffraction profile.

    Parameters
    ----------
    simulation : DiffractionProfileSimulation
        Simulation of the diffraction profile.
    tolerance : float
        The n orientations with the highest correlation values are returned.

    Returns
    -------
    indexation : np.array()
        indexation results.

    """
    mags = z
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


def match_vectors(ks,
                  library,
                  mag_tol,
                  angle_tol,
                  seed_pool_size,
                  n_best,
                  keys=[],
                  *args,
                  **kwargs):
    """Assigns hkl indices to pairs of diffraction vectors.

    Parameters
    ----------
    ks : np.array()
        The experimentally measured diffraction vectors, associated with a
        particular probe position, to be indexed. In Cartesian coordinates.
    library : VectorLibrary
        Library of reciprocal space vectors to be matched to the vectors.
    mag_tol : float
        The number of well correlated simulations to be retained.
    angle_tol : float
        A mask for navigation axes 1 indicates positions to be indexed.
    seed_pool_size : int
        The maximum number of peak pairs to check.
    n_best : int
        The maximum number of good solutions to be retained.

    Returns
    -------
    indexation : np.array()
        A numpy array containing the indexation results, each result consisting of 5 entries:
            [phase index, rotation matrix, match rate, error hkls, total error]

    """
    # Initialise for loop with first entry & assign empty array to hold
    # indexation results.
    top_matches = np.empty((len(library), n_best, 5), dtype='object')
    res_rhkls = []  # TODO: Correct format?
    peaks = ks[0]
    # Iterate over phases in DiffractionVectorLibrary and perform indexation
    # with respect to each phase.
    for phase_index, (key, structure) in enumerate(zip(library.keys(), library.structures)):
        solutions = []
        # TODO: Testing, think this is what SPIND calculates. What is the physical meaning of the matrix transformations?
        recip_lattice = structure.lattice.recbase
        recip_lattice_inv = np.linalg.inv(recip_lattice)
        # Pair unindexed peaks into combinations inluding up to seed_pool_size
        # many peak pairs to define the seed_pool
        unindexed_peak_ids = list(range(min(peaks.shape[0], seed_pool_size)))
        seed_pool = list(combinations(unindexed_peak_ids, 2))
        # Determine overall indexations associated with each seed in the
        # seed_pool to generate a solution pool.
        # TODO: Do we include the [0, 0, 0] vector?
        for seed in seed_pool:
            # Consider a seed pair of vectors.
            q1, q2 = peaks[seed, :]
            q1_len, q2_len = np.linalg.norm(q1), np.linalg.norm(q2)
            # Ensure q1 is longer than q2 so cominations in correct order.
            if q1_len < q2_len:
                q1, q2 = q2, q1
                q1_len, q2_len = q2_len, q1_len
            # Calculate the angle between experimental scattering vectors.
            angle = get_angle_cartesian(q1, q2)

            # Get library indices for hkls matching peaks within tolerances.
            match_ids = np.where((np.abs(q1_len - library[key][:, 2]) < mag_tol) &
                                 (np.abs(q2_len - library[key][:, 3]) < mag_tol) &
                                 (np.abs(angle - library[key][:, 4]) < angle_tol))[0]
            # Iterate over matched seed vectors determining the error in the
            # associated indexation and finding the minimum error cases.
            for match_id in match_ids:
                hkl1 = library[key][:, 0][match_id]
                hkl2 = library[key][:, 1][match_id]
                # Reference vectors are cartesian coordinates of hkls
                ref_q1, ref_q2 = recip_lattice.dot(hkl1), recip_lattice.dot(hkl2)
                R = get_rotation_matrix_between_vectors(q1, q2,
                                                        ref_q1, ref_q2)
                # Evaluate error on seed point, total error & match rate
                # hkls are the peak positions converted to Miller indices
                R_inv = R.T  # Inverse rotation from tranposed matrix
                hkls = recip_lattice_inv.dot(R_inv.dot(peaks.T)).T
                rhkls = np.rint(hkls)
                ehkls = np.abs(hkls - rhkls)
                res_rhkls.append(rhkls)

                # indices of matched peaks
                eval_tol = 0.25  # TODO: Parameter, better name
                pair_ids = np.where(np.max(ehkls, axis=1) < eval_tol)[0]
                # TODO: SPIND allows trying to match multiple crystals
                # (until) match_rate == 0 by filtering already in indexed peaks
                # pair_ids = list(set(pair_ids) - set(indexed_peak_ids))

                # calculate match_rate as fraction of peaks indexed
                num_pairs = len(pair_ids)
                num_peaks = len(peaks)
                match_rate = num_pairs / num_peaks

                if len(pair_ids) == 0:
                    # no matching peaks, set error to 1
                    total_error = 1.0
                else:
                    # naive error of matching peaks
                    total_error = ehkls[pair_ids].mean()

                solutions.append([
                    R,
                    match_rate,
                    ehkls,
                    total_error
                ])

        solutions = np.array(solutions)
        # TODO: SPIND sorts by highest match rate then lowest total error and
        # returns the single best solution. Here, we instead return the n best
        # solutions. Correct approach for pyXem?
        # best_match_rate_solutions = solutions[solutions[6].argmax()]
        if solutions.shape[0] > n_best:
            match_rate_index = 1
            top_n = solutions[solutions[:, match_rate_index].argpartition(-n_best)[-n_best:]]

            # Put the top n ranked solutions in the output array
            top_matches[phase_index, :, 0] = phase_index
            top_matches[phase_index, :, 1:] = top_n
        # TODO: Refine?

    return top_matches.reshape((len(library) * n_best, 5)), res_rhkls


def crystal_from_template_matching(z_matches):
    """Takes template matching results for a single navigation position and
    returns the best matching phase and orientation with correlation and
    reliability/ies to define a crystallographic map.

    Parameters
    ----------
    z_matches : np.array()
        Template matching results in an array of shape (m,5) with entries
        [phase, z, x, z, correlation]

    Returns
    -------
    results_array : np.array()
        Crystallographic mapping results in an array (3) with entries
        [phase, np.array((z,x,z)), dict(metrics)]
    """
    # Create empty array for results.
    results_array = np.empty(3, dtype='object')
    # Consider single phase and multi-phase matching cases separately
    if np.unique(z_matches[:, 0]).shape[0] == 1:
        # get best matching phase (there is only one here)
        results_array[0] = z_matches[0, 0]
        # get best matching orientation Euler angles
        results_array[1] = np.array(z_matches[0, 1:4])
        # get template matching metrics
        metrics = dict()
        metrics['correlation'] = z_matches[0, 4]
        metrics['orientation_reliability'] = 100 * (1 - z_matches[1, 4] / z_matches[0, 4])
        results_array[2] = metrics
    else:
        # get best matching result
        index_best_match = np.argmax(z_matches[:, 4])
        # get best matching phase
        results_array[0] = z_matches[index_best_match, 0]
        # get best matching orientation Euler angles.
        results_array[1] = np.array(z_matches[index_best_match, 1:4])
        # get second highest correlation orientation for orientation_reliability
        z = z_matches[z_matches[:, 0] == results_array[0]]
        second_orientation = np.partition(z[:, 4], -2)[-2]
        # get second highest correlation phase for phase_reliability
        z = z_matches[z_matches[:, 0] != results_array[0]]
        second_phase = np.max(z[:, 4])
        # get template matching metrics
        metrics = dict()
        metrics['correlation'] = z_matches[index_best_match, 4]
        metrics['orientation_reliability'] = 100 * (1 - second_orientation / z_matches[index_best_match, 4])
        metrics['phase_reliability'] = 100 * (1 - second_phase / z_matches[index_best_match, 4])
        results_array[2] = metrics

    return results_array


def crystal_from_vector_matching(z_matches):
    """Takes vector matching results for a single navigation position and
    returns the best matching phase and orientation with correlation and
    reliability/ies to define a crystallographic map.

    Parameters
    ----------
    z_matches : np.array()
        Template matching results in an array of shape (m,5) with entries
        [phase, R, match_rate, ehkls, total_error]

    Returns
    -------
    results_array : np.array()
        Crystallographic mapping results in an array (3) with entries
        [phase, np.array((z,x,z)), dict(metrics)]
    """
    # Create empty array for results.
    results_array = np.empty(3, dtype='object')
    # Consider single phase and multi-phase matching cases separately
    if np.unique(z_matches[:, 0]).shape[0] == 1:
        # get best matching phase (there is only one here)
        results_array[0] = z_matches[0, 0]
        # get best matching orientation Euler angles
        results_array[1] = mat2euler(z_matches[0, 1], 'rzxz')
        # get template matching metrics
        metrics = dict()
        metrics['match_rate'] = z_matches[0, 2]
        metrics['ehkls'] = z_matches[0, 3]
        metrics['total_error'] = z_matches[0, 4]
        metrics['orientation_reliability'] = 100 * (1 - z_matches[0, 4] / z_matches[1, 4])
        results_array[2] = metrics

    else:
        # get best matching result, with minimal total_error
        index_best_match = np.argmin(z_matches[:, 4])
        # get best matching phase
        results_array[0] = z_matches[index_best_match, 0]
        # get best matching orientation Euler angles.
        results_array[1] = mat2euler(z_matches[index_best_match, 1], 'rzxz')

        # get second smallest total error for orientation_reliability
        z = z_matches[z_matches[:, 0] == results_array[0]]
        second_orientation = np.partition(z[:, 4], 1)[1]
        # get second highest correlation phase for phase_reliability
        z = z_matches[z_matches[:, 0] != results_array[0]]
        second_phase = np.min(z[:, 4])
        # get template matching metrics
        metrics = dict()
        metrics['match_rate'] = z_matches[index_best_match, 2]
        metrics['ehkls'] = z_matches[index_best_match, 3]
        metrics['total_error'] = z_matches[index_best_match, 4]
        metrics['orientation_reliability'] = 100 * (1 - z_matches[index_best_match, 4] / second_orientation)
        metrics['phase_reliability'] = 100 * (1 - z_matches[index_best_match, 4] / second_phase)
        results_array[2] = metrics

    return results_array
