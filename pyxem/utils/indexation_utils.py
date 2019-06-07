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


from heapq import nlargest
from itertools import combinations
import math
from operator import itemgetter

import numpy as np

from pyxem.utils.expt_utils import _cart2polar
from pyxem.utils.vector_utils import get_rotation_matrix_between_vectors
from pyxem.utils.vector_utils import get_angle_cartesian

from transforms3d.euler import mat2euler
from transforms3d.quaternions import mat2quat


def correlate_library(image, library, n_largest, mask):
    """Correlates all simulated diffraction templates in a DiffractionLibrary
    with a particular experimental diffraction pattern (image).

    Calculated using the normalised (see return type documentation) dot
    product, or cosine distance,

    .. math::
        \\frac{\\sum_{j=1}^m P(x_j, y_j) T(x_j, y_j)}{\\sqrt{\\sum_{j=1}^m T^2(x_j, y_j)}}

    for a template T and an experimental pattern P.

    Parameters
    ----------
    image : numpy.array
        The experimental diffraction pattern of interest.
    library : DiffractionLibrary
        The library of diffraction simulations to be correlated with the
        experimental data.
    n_largest : int
        The number of well correlated simulations to be retained.
    mask : bool
        A mask for navigation axes. 1 indicates positions to be indexed.

    Returns
    -------
    top_matches : numpy.array
        Array of shape (<num phases>*n_largest, 3) containing the top n
        correlated simulations for the experimental pattern of interest, where
        each entry is on the form [phase index, [z, x, z], correlation].

    See also
    --------
    IndexationGenerator.correlate

    Notes
    -----
    Correlation results are defined as,
        phase_index : int
            Index of the phase, following the ordering of the library keys
        [z, x, z] : ndarray
            numpy array of three floats, specifying the orientation in the
            Bunge convention, in degrees.
        correlation : float
            A coefficient of correlation, only normalised to the template
            intensity. This is in contrast to the reference work.

    References
    ----------
    E. F. Rauch and L. Dupuy, “Rapid Diffraction Patterns identification through
       template matching,” vol. 50, no. 1, pp. 87–99, 2005.
    """
    top_matches = np.empty((len(library), n_largest, 3), dtype='object')

    if mask == 1:
        for phase_index, library_entry in enumerate(library.values()):
            orientations = library_entry['orientations']
            pixel_coords = library_entry['pixel_coords']
            intensities = library_entry['intensities']
            pattern_norms = library_entry['pattern_norms']

            # Extract experimental intensities from the diffraction image
            image_intensities = image[pixel_coords[:, :, :, 1], pixel_coords[:, :, :, 0]]
            # Correlation is the normalized dot product
            correlations = np.sum(image_intensities * intensities, axis=2) / pattern_norms

            # Find the top n correlations in sorted order
            top_n_indices = correlations.argpartition(-n_largest, axis=None)[-n_largest:]
            top_n_correlations = correlations.ravel()[top_n_indices]
            top_n_indices = top_n_indices[top_n_correlations.argsort()[::-1]]

            # Store the results in top_matches
            top_matches[phase_index, :, 0] = phase_index
            inplane_rotation_angle = 360 / pixel_coords.shape[0]
            for i in range(n_largest):
                inplane_index, orientation_index = np.unravel_index(top_n_indices[i], correlations.shape)
                top_matches[phase_index, i, 1] = orientations[orientation_index] + np.array(
                    [0, 0, inplane_index * inplane_rotation_angle])
            top_matches[phase_index, :, 2] = correlations.ravel()[top_n_indices]

    return top_matches.reshape(-1, 3)


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


def _choose_peak_ids(peaks, n_peaks_to_index):
    """Choose `n_peaks_to_index` indices from `peaks`.

    This implementation sorts by angle and then picks every
    len(peaks)/n_peaks_to_index element to get an even distribution of angles.

    Parameters
    ----------
    peaks : array_like
        Array of peak positions.
    n_peaks_to_index : int
        Number of indices to return.

    Returns
    -------
    peak_ids : numpy.array
        Array of indices of the chosen peaks.
    """
    r, angles = _cart2polar(peaks[:, 0], peaks[:, 1])
    return angles.argsort()[np.linspace(0, angles.shape[0] - 1, n_peaks_to_index, dtype=np.int)]


def _sort_solutions(solutions, n_solutions):
    """Sort the solutions by quality.

    Parameters
    ----------
    solutions : numpy.array
        Potential solutions on the form used by `match_vectors`.
    n_solutions : int
        Number of solutions to return.

    Results
    -------
    solutions : numpy.array
        `n_solutions` best solutions, sorted.
    """
    # Sort by match rate descending
    n_solutions = min(n_solutions, solutions.shape[0])
    return solutions[solutions[:, 1].argsort()[::-1][:n_solutions]]


def match_vectors(peaks,
                  library,
                  mag_tol,
                  angle_tol,
                  index_error_tol,
                  n_peaks_to_index,
                  n_best):
    # TODO: Sort peaks by intensity or SNR
    """Assigns hkl indices to pairs of diffraction vectors.

    Parameters
    ----------
    peaks : np.array()
        The experimentally measured diffraction vectors, associated with a
        particular probe position, to be indexed. In Cartesian coordinates.
    library : VectorLibrary
        Library of reciprocal space vectors to be matched to the vectors.
    mag_tol : float
        Max allowed magnitude difference when comparing vectors.
    angle_tol : float
        Max allowed angle difference in radians when comparing vector pairs.
    index_error_tol : float
        Max allowed error in peak indexation for classifying it as indexed,
        calculated as :math:`|hkl_calculated - round(hkl_calculated)|`.
    n_peaks_to_index : int
        The maximum number of peak to index.
    n_best : int
        The maximum number of good solutions to be retained.

    Returns
    -------
    indexation : np.array()
        A numpy array containing the indexation results, each result consisting of 5 entries:
            [phase index, rotation matrix, match rate, error hkls, total error]

    """
    if peaks.shape == (1,) and peaks.dtype == np.object:
        peaks = peaks[0]

    # Assign empty array to hold indexation results. The n_best best results
    # from each phase is returned.
    top_matches = np.empty((len(library), n_best, 5), dtype=np.object)
    res_rhkls = []

    # Iterate over phases in DiffractionVectorLibrary and perform indexation
    # on each phase, storing the best results in top_matches.
    for phase_index, (phase, structure) in enumerate(zip(library.values(), library.structures)):
        solutions = []
        lattice_recip = structure.lattice.reciprocal()
        phase_indices = phase['indices']
        phase_measurements = phase['measurements']

        if peaks.shape[0] >= 2:
            # Choose up to n_peaks_to_index unindexed peaks to be paired in all
            # combinations.
            # TODO: Matching can be done iteratively where successfully indexed
            #       peaks are removed after each iteration. This can possibly
            #       handle overlapping patterns.
            # unindexed_peak_ids = range(min(peaks.shape[0], n_peaks_to_index))
            # TODO: Better choice of peaks (longest, highest SNR?)
            # TODO: Inline after choosing the best, and possibly require external sorting (if using sorted)?
            unindexed_peak_ids = _choose_peak_ids(peaks, n_peaks_to_index)

            # Find possible solutions for each pair of peaks.
            for vector_pair_index, peak_pair_indices in enumerate(list(combinations(unindexed_peak_ids, 2))):
                # Consider a pair of experimental scattering vectors.
                q1, q2 = peaks[peak_pair_indices, :]
                q1_len, q2_len = np.linalg.norm(q1), np.linalg.norm(q2)

                # Ensure q1 is longer than q2 for consistent order.
                if q1_len < q2_len:
                    q1, q2 = q2, q1
                    q1_len, q2_len = q2_len, q1_len

                # Calculate the angle between experimental scattering vectors.
                angle = get_angle_cartesian(q1, q2)

                # Get library indices for hkls matching peaks within tolerances.
                # TODO: phase are object arrays. Test performance of direct float arrays
                tolerance_mask = np.abs(phase_measurements[:, 0] - q1_len) < mag_tol
                tolerance_mask[tolerance_mask] &= np.abs(phase_measurements[tolerance_mask, 1] - q2_len) < mag_tol
                tolerance_mask[tolerance_mask] &= np.abs(phase_measurements[tolerance_mask, 2] - angle) < angle_tol

                # Iterate over matched library vectors determining the error in the
                # associated indexation.
                if np.count_nonzero(tolerance_mask) == 0:
                    continue

                # Reference vectors are cartesian coordinates of hkls
                reference_vectors = lattice_recip.cartesian(phase_indices[tolerance_mask])

                # Rotation from experimental to reference frame
                rotations = get_rotation_matrix_between_vectors(q1, q2, reference_vectors[:, 0], reference_vectors[:, 1])

                # Index the peaks by rotating them to the reference coordinate
                # system. Use rotation directly since it is multiplied from the
                # right. Einsum gives list of peaks.dot(rotation).
                hklss = lattice_recip.fractional(np.einsum('ijk,lk->ilj', rotations, peaks))

                # Evaluate error of peak hkl indexation
                rhklss = np.rint(hklss)
                ehklss = np.abs(hklss - rhklss)
                valid_peak_mask = np.max(ehklss, axis=-1) < index_error_tol
                valid_peak_counts = np.count_nonzero(valid_peak_mask, axis=-1)
                error_means = ehklss.mean(axis=(1, 2))

                num_peaks = len(peaks)
                match_rates = (valid_peak_counts * (1 / num_peaks)) if num_peaks else 0

                possible_solution_mask = match_rates > 0
                solutions += [[
                    R,
                    match_rate,
                    ehkls,  # TODO: Needed?
                    error_mean,
                    vector_pair_index
                ] for R, match_rate, ehkls, error_mean in zip(
                    rotations[possible_solution_mask],
                    match_rates[possible_solution_mask],
                    ehklss[possible_solution_mask],
                    error_means[possible_solution_mask])]
                res_rhkls += rhklss[possible_solution_mask].tolist()

        n_solutions = min(n_best, len(solutions))
        if n_solutions > 0:
            top_n = _sort_solutions(np.array(solutions), n_solutions)

            # Put the top n ranked solutions in the output array
            top_matches[phase_index, :, 0] = phase_index
            top_matches[phase_index, :n_solutions, 1:] = top_n[:, :4]

        if n_solutions < n_best:
            # Fill with dummy values
            top_matches[phase_index, n_solutions:] = [
                0,
                np.identity(3),
                0,
                np.array([]),
                1.0
            ]

        # TODO: Refine?

    # Because of a bug in numpy (https://github.com/numpy/numpy/issues/7453),
    # triggered by the way HyperSpy reads results (np.asarray(res), which fails
    # when the two tuple values have the same first dimension), we cannot
    # return a tuple directly, but instead have to format the result as an
    # array ourselves.
    res = np.empty(2, dtype=np.object)
    res[0] = top_matches.reshape((len(library) * n_best, 5))
    res[1] = np.asarray(res_rhkls)
    return res


def crystal_from_template_matching(z_matches):
    """Takes template matching results for a single navigation position and
    returns the best matching phase and orientation with correlation and
    reliability to define a crystallographic map.

    Parameters
    ----------
    z_matches : numpy.array
        Template matching results in an array of shape (m,3) sorted by
        correlation (descending) within each phase, with entries
        [phase, [z, x, z], correlation]

    Returns
    -------
    results_array : numpy.array
        Crystallographic mapping results in an array of shape (3) with entries
        [phase, np.array((z, x, z)), dict(metrics)]

    """
    # Create empty array for results.
    results_array = np.empty(3, dtype='object')
    # Consider single phase and multi-phase matching cases separately
    if np.unique(z_matches[:, 0]).shape[0] == 1:
        # get best matching phase (there is only one here)
        results_array[0] = z_matches[0, 0]
        # get best matching orientation Euler angles
        results_array[1] = z_matches[0, 1]
        # get template matching metrics
        metrics = dict()
        metrics['correlation'] = z_matches[0, 2]
        metrics['orientation_reliability'] = 100 * (1 - z_matches[1, 2] / z_matches[0, 2]) if z_matches[0, 2] > 0 else 100
        results_array[2] = metrics
    else:
        # get best matching result
        index_best_match = np.argmax(z_matches[:, 2])
        # get best matching phase
        results_array[0] = z_matches[index_best_match, 0]
        # get best matching orientation Euler angles.
        results_array[1] = z_matches[index_best_match, 1]
        # get second highest correlation orientation for orientation_reliability
        z = z_matches[z_matches[:, 0] == results_array[0]]
        second_orientation = np.partition(z[:, 2], -2)[-2]
        # get second highest correlation phase for phase_reliability
        z = z_matches[z_matches[:, 0] != results_array[0]]
        second_phase = np.max(z[:, 2])
        # get template matching metrics
        metrics = dict()
        metrics['correlation'] = z_matches[index_best_match, 2]
        metrics['orientation_reliability'] = 100 * (1 - second_orientation / z_matches[index_best_match, 2])
        metrics['phase_reliability'] = 100 * (1 - second_phase / z_matches[index_best_match, 2])
        results_array[2] = metrics

    return results_array


def crystal_from_vector_matching(z_matches):
    """Takes vector matching results for a single navigation position and
    returns the best matching phase and orientation with correlation and
    reliability to define a crystallographic map.

    Parameters
    ----------
    z_matches : numpy.array
        Template matching results in an array of shape (m,5) sorted by
        total_error (ascending) within each phase, with entries
        [phase, R, match_rate, ehkls, total_error]

    Returns
    -------
    results_array : numpy.array
        Crystallographic mapping results in an array of shape (3) with entries
        [phase, np.array((z, x, z)), dict(metrics)]
    """
    # Create empty array for results.
    results_array = np.empty(3, dtype='object')
    # Consider single phase and multi-phase matching cases separately
    if np.unique(z_matches[:, 0]).shape[0] == 1:
        # get best matching phase (there is only one here)
        results_array[0] = z_matches[0, 0]
        # get best matching orientation Euler angles
        results_array[1] = np.rad2deg(mat2euler(z_matches[0, 1], 'rzxz'))
        # get vector matching metrics
        metrics = dict()
        metrics['match_rate'] = z_matches[0, 2]
        metrics['ehkls'] = z_matches[0, 3]
        metrics['total_error'] = z_matches[0, 4]
        metrics['orientation_reliability'] = 100 * (1 - z_matches[0, 4] / (z_matches[1, 4] or 1.0))
        results_array[2] = metrics

    else:
        # get best matching result, with minimal total_error
        index_best_match = np.argmin(z_matches[:, 4])
        # get best matching phase
        results_array[0] = z_matches[index_best_match, 0]
        # get best matching orientation Euler angles.
        results_array[1] = np.rad2deg(mat2euler(z_matches[index_best_match, 1], 'rzxz'))

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
