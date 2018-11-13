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
    out_arr : np.array()
        A numpy array containing the top n correlated simulations for the
        experimental pattern of interest.

    See also
    --------
    pyxem.utils.correlate and the correlate method of IndexationGenerator.

    """
    i = 0
    out_arr = np.zeros((n_largest * len(library), 5))
    if mask == 1:
        for key in library.keys():
            correlations = dict()
            for orientation, diffraction_pattern in library[key].items():
                # diffraction_pattern here is in fact a library of
                # diffraction_pattern_properties
                correlation = correlate(image, diffraction_pattern)
                correlations[orientation] = correlation
                res = nlargest(n_largest, correlations.items(),
                               key=itemgetter(1))
            # put top n results in output array
            for j in np.arange(n_largest):
                # get phase identifying integer
                out_arr[j + i * n_largest][0] = i
                # get Euler angles z, x, z
                out_arr[j + i * n_largest][1] = res[j][0][0]
                out_arr[j + i * n_largest][2] = res[j][0][1]
                out_arr[j + i * n_largest][3] = res[j][0][2]
                # get correlation score
                out_arr[j + i * n_largest][4] = res[j][1]
            i = i + 1

    else:
        for j in np.arange(n_largest):
            for k in [0, 1, 2, 3, 4]:
                out_arr[j + i * n_largest][k] = np.nan
        i = i + 1
    return out_arr


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

    Returns
    -------
    indexation : np.array()
        A numpy array containing the indexation results.

    """
    # Initialise for loop with first entry & assign empty array to hold
    # indexation results.
    i = 0
    out_arr = np.zeros((n_largest * len(library), 5))
    # Iterate over phases in DiffractionVectorLibrary and perform indexation
    # with respect to each phase.
    for key in library.keys():
        strucure = library[key][0]
        # pair unindexed peaks into combinations inluding up to seed_pool_size
        # many pairs to define the seed_pool
        unindexed_peak_ids = list(set(range(min(ks.shape[0], seed_pool_size))))
        seed_pool = list(combinations(unindexed_peak_ids, 2))
        # Determine overall indexations associated with each seed in the
        # seed_pool to generate a solution pool.
        solution_pool = []
        for i in tqdm(range(len(seed_pool))):
            seed = seed_pool[i]
            # Consider a seed pair of vectors.
            q1, q2 = ks[seed, :]
            q1_len, q2_len = norm(q1), norm(q2)
            # Ensure q1 is longer than q2 so cominations in correct order.
            if q1_len < q2_len:
                q1, q2 = q2, q1
                q1_len, q2_len = q2_len, q1_len
            # Calculate the angle between experimental scattering vectors.
            angle = get_angle_cartesian(q1, q2)
            # Get library indices for hkls matching peaks within tolerances.
            match_ids = np.where((np.abs(q1_len - library[key][1][:, 2]) < mag_tol) *
                                 (np.abs(q2_len - library[key][1][:, 3]) < mag_tol) *
                                 (np.abs(angle - library[key][1][:, 4]) < angle_tol))[0]
            # Iterate over matched seed vectors determining the error in the
            # associated indexation and finding the minimum error cases.
            for match_id in match_ids:
                #
                hkl1 = library[key][:, 0][match_id]
                hkl2 = library[key][:, 0][match_id]
                # reference vectors are cartesian coordinates of hkls
                # TODO: could put this in the library?
                ref_q1, ref_q2 = A0.dot(hkl1), A0.dot(hkl2)
                R = get_rotation_matrix_between_vectors(q1, q2,
                                                        ref_q1, ref_q2)
                # Evaluate error on seed point, total error & match rate
                R_inv = np.linalg.inv(R)
                hkls = A0_inv.dot(R_inv.dot(ks.T)).T
                rhkls = np.rint(hkls)
                ehkls = np.abs(hkls - rhkls)

                # indices of matched peaks
                pair_ids = np.where(np.max(ehkls, axis=1) < eval_tol)[0]
                pair_ids = list(set(pair_ids) - set(indexed_peak_ids))

                # calculate match_rate as fraction of peaks indexed
                nb_pairs = len(pair_ids)
                nb_peaks = len(ks)
                match_rate = float(nb_pairs) / float(nb_peaks)

                # set solution attributes
                solution.hkls = hkls
                solution.rhkls = rhkls
                solution.ehkls = ehkls
                solution.pair_ids = pair_ids
                solution.nb_pairs = nb_pairs

                # evaluate / store indexation metrics
                solution.seed_error = ehkls[seed, :].max()
                solution.match_rate = match_rate
                if len(pair_ids) == 0:
                    # no matching peaks, set error to 1
                    solution.total_error = 1.
                else:
                    # naive error of matching peaks
                    solution.total_error = ehkls[pair_ids].mean()
                # Put solutions in the solution_pool
                solution_pool.append(solution)

            # Sort solutions in the solution_pool
            if len(solution_pool) > 0:
                # best solution has highest total score and lowest total error
                good_solutions.sort(key=lambda x: x.match_rate, reverse=True)
                best_score = good_solutions[0].match_rate
                best_solutions = [solution for solution in solution_pool if solution.match_rate == best_score]
                best_solutions.sort(key=lambda x: x.total_error, reverse=False)
                best_solution = best_solutions[0]
            else:
                best_solution = None

            # Put the top n ranked solutions in the output array
            for j in np.arange(n_largest):
                # store phase identifying integer
                out_arr[j + i * n_largest][0] = i
                # store rotation matrix
                out_arr[j + i * n_largest][1] = ranked_solutions[j][0][0]
                # store match_rate
                out_arr[j + i * n_largest][2] = ranked_solutions[j][0][1]
                # store ehkls
                out_arr[j + i * n_largest][3] = res[j][0][2]
                # store total_error
                out_arr[j + i * n_largest][4] = res[j][1]
            i = i + 1

    return out_arr, rhkls


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
    results_array = np.zeros(3)
    # Consider single phase and multi-phase matching cases separately
    if np.unique(z_matches[:, 0]).shape[0] == 1:
        # get best matching phase (there is only one here)
        results_array[0] = z_matches[0, 0]
        # get best matching orientation Euler angles
        results_array[1] = np.array(z_matches[0, 1:4])
        # get template matching metrics
        metrics = dict()
        mectrics['correlation'] = z_matches[0, 4]
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
        mectrics['correlation'] = z_matches[0, 4]
        metrics['orientation_reliability'] = 100 * (1 - z_matches[1, 4] / z_matches[0, 4])
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
    results_array = np.zeros(3)
    # Consider single phase and multi-phase matching cases separately
    if np.unique(z_matches[:, 0]).shape[0] == 1:
        # get best matching phase (there is only one here)
        results_array[0] = z_matches[0, 0]
        # get best matching orientation Euler angles
        results_array[1] = mat2euler(z_matches[0, 1])
        # get template matching metrics
        metrics = dict()
        mectrics['match_rate'] = z_matches[0, 2]
        metrics['ehkls'] = z_matches[0, 3]
        metrics['total_error'] = z_matches[0, 4]
        metrics['orientation_reliability'] = 100 * (1 - z_matches[1, 4] / z_matches[0, 4])
        results_array[2] = metrics

    else:
        # get best matching result
        index_best_match = np.argmax(z_matches[:, 4])
        # get best matching phase
        results_array[0] = z_matches[index_best_match, 0]
        #get best matching orientation Euler angles.
        results_array[1] = mat2euler(z_matches[index_best_match, 1])
        # get second highest correlation orientation for orientation_reliability
        z = z_matches[z_matches[:, 0] == results_array[0]]
        second_orientation = np.partition(z[:, 4], -2)[-2]
        # get second highest correlation phase for phase_reliability
        z = z_matches[z_matches[:, 0] != results_array[0]]
        second_phase = np.max(z[:, 4])
        # get template matching metrics
        metrics = dict()
        mectrics['match_rate'] = z_matches[index_best_match, 2]
        metrics['ehkls'] = z_matches[index_best_match, 3]
        metrics['total_error'] = z_matches[index_best_match, 4]
        metrics['orientation_reliability'] = 100 * (1 - second_orienation / z_matches[index_best_match, 4])
        metrics['phase_reliability'] = 100 * (1 - second_phase / z_matches[index_best_match, 4])
        results_array[2] = metrics

    return best_solution
