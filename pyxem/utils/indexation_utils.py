# -*- coding: utf-8 -*-
# Copyright 2016-2021 The pyXem developers
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


from itertools import combinations
from operator import attrgetter

import numpy as np

from pyxem.utils.expt_utils import _cart2polar
from pyxem.utils.vector_utils import get_rotation_matrix_between_vectors
from pyxem.utils.vector_utils import get_angle_cartesian

from transforms3d.euler import mat2euler

from collections import namedtuple


# container for OrientationResults
OrientationResult = namedtuple(
    "OrientationResult",
    "phase_index rotation_matrix match_rate error_hkls total_error scale center_x center_y".split(),
)


def get_nth_best_solution(
    single_match_result, mode, rank=0, key="match_rate", descending=True
):
    """Get the nth best solution by match_rate from a pool of solutions

    Parameters
    ----------
    single_match_result : VectorMatchingResults, TemplateMatchingResults
        Pool of solutions from the vector matching algorithm
    mode : str
        'vector' or 'template'
    rank : int
        The rank of the solution, i.e. rank=2 returns the third best solution
    key : str
        The key to sort the solutions by, default = match_rate
    descending : bool
        Rank the keys from large to small

    Returns
    -------
    VectorMatching:
        best_fit : `OrientationResult`
            Parameters for the best fitting orientation
            Library Number, rotation_matrix, match_rate, error_hkls, total_error
    TemplateMatching: np.array
            Parameters for the best fitting orientation
            Library Number , [z, x, z], Correlation Score
    """
    if mode == "vector":
        try:
            best_fit = sorted(
                single_match_result[0].tolist(), key=attrgetter(key), reverse=descending
            )[rank]
        except AttributeError:
            best_fit = sorted(
                single_match_result.tolist(), key=attrgetter(key), reverse=descending
            )[rank]
    if mode == "template":
        srt_idx = np.argsort(single_match_result[:, 2])[::-1][rank]
        best_fit = single_match_result[srt_idx]

    return best_fit


# Functions used in correlate_library.
def fast_correlation(image_intensities, int_local, pn_local, **kwargs):
    r"""Computes the correlation score between an image and a template

    Uses the formula

    .. math:: FastCorrelation
        \\frac{\\sum_{j=1}^m P(x_j, y_j) T(x_j, y_j)}{\\sqrt{\\sum_{j=1}^m T^2(x_j, y_j)}}

    Parameters
    ----------
    image_intensities: list
        list of intensity values in the image, for pixels where the template has a non-zero intensity
     int_local: list
        list of all non-zero intensities in the template
     pn_local: float
        pattern norm of the template

    Returns
    -------
    corr_local: float
        correlation score between template and image.

    See Also
    --------
    correlate_library, zero_mean_normalized_correlation

    """
    return (
        np.sum(np.multiply(image_intensities, int_local)) / pn_local
    )  # Correlation is the partially normalized dot product


def zero_mean_normalized_correlation(
    nb_pixels,
    image_std,
    average_image_intensity,
    image_intensities,
    int_local,
    **kwargs
):
    r"""Computes the correlation score between an image and a template.

    Uses the formula

    .. math:: zero_mean_normalized_correlation
        \\frac{\\sum_{j=1}^m P(x_j, y_j) T(x_j, y_j)- avg(P)avg(T)}{\\sqrt{\\sum_{j=1}^m (T(x_j, y_j)-avg(T))^2+\\sum_{Not {j}} avg(T)}}
        for a template T and an experimental pattern P.

    Parameters
    ----------
    nb_pixels: int
        total number of pixels in the image
    image_std: float
        Standard deviation of intensities in the image.
    average_image_intensity: float
        average intensity for the image
    image_intensities: list
        list of intensity values in the image, for pixels where the template has a non-zero intensity
     int_local: list
        list of all non-zero intensities in the template
     pn_local: float
        pattern norm of the template

    Returns
    -------
    corr_local: float
        correlation score between template and image.

    See Also
    --------
    correlate_library, fast_correlation

    """

    nb_pixels_star = len(int_local)
    average_pattern_intensity = nb_pixels_star * np.average(int_local) / nb_pixels

    match_numerator = (
        np.sum(np.multiply(image_intensities, int_local))
        - nb_pixels * average_image_intensity * average_pattern_intensity
    )
    match_denominator = image_std * (
        np.linalg.norm(int_local - average_pattern_intensity)
        + (nb_pixels - nb_pixels_star) * pow(average_pattern_intensity, 2)
    )

    if match_denominator == 0:
        corr_local = 0
    else:
        corr_local = (
            match_numerator / match_denominator
        )  # Correlation is the normalized dot product

    return corr_local


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
    return angles.argsort()[
        np.linspace(0, angles.shape[0] - 1, n_peaks_to_index, dtype=np.int)
    ]


def match_vectors(
    peaks, library, mag_tol, angle_tol, index_error_tol, n_peaks_to_index, n_best
):
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
        The maximum number of good solutions to be retained for each phase.

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
    top_matches = np.empty(len(library) * n_best, dtype="object")
    res_rhkls = []

    # Iterate over phases in DiffractionVectorLibrary and perform indexation
    # on each phase, storing the best results in top_matches.
    for phase_index, (phase, structure) in enumerate(
        zip(library.values(), library.structures)
    ):
        solutions = []
        lattice_recip = structure.lattice.reciprocal()
        phase_indices = phase["indices"]
        phase_measurements = phase["measurements"]

        if peaks.shape[0] < 2:  # pragma: no cover
            continue

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
        for vector_pair_index, peak_pair_indices in enumerate(
            list(combinations(unindexed_peak_ids, 2))
        ):
            # Consider a pair of experimental scattering vectors.
            q1, q2 = peaks[peak_pair_indices, :]
            q1_len, q2_len = np.linalg.norm(q1), np.linalg.norm(q2)

            # Ensure q1 is longer than q2 for consistent order.
            if q1_len < q2_len:  # pragma: no cover
                q1, q2 = q2, q1
                q1_len, q2_len = q2_len, q1_len

            # Calculate the angle between experimental scattering vectors.
            angle = get_angle_cartesian(q1, q2)

            # Get library indices for hkls matching peaks within tolerances.
            # TODO: phase are object arrays. Test performance of direct float arrays
            tolerance_mask = np.abs(phase_measurements[:, 0] - q1_len) < mag_tol
            tolerance_mask[tolerance_mask] &= (
                np.abs(phase_measurements[tolerance_mask, 1] - q2_len) < mag_tol
            )
            tolerance_mask[tolerance_mask] &= (
                np.abs(phase_measurements[tolerance_mask, 2] - angle) < angle_tol
            )

            # Iterate over matched library vectors determining the error in the
            # associated indexation.
            if np.count_nonzero(tolerance_mask) == 0:  # pragma: no cover
                continue

            # Reference vectors are cartesian coordinates of hkls
            reference_vectors = lattice_recip.cartesian(phase_indices[tolerance_mask])

            # Rotation from experimental to reference frame
            rotations = get_rotation_matrix_between_vectors(
                q1, q2, reference_vectors[:, 0], reference_vectors[:, 1]
            )

            # Index the peaks by rotating them to the reference coordinate
            # system. Use rotation directly since it is multiplied from the
            # right. Einsum gives list of peaks.dot(rotation).
            hklss = lattice_recip.fractional(np.einsum("ijk,lk->ilj", rotations, peaks))

            # Evaluate error of peak hkl indexation
            rhklss = np.rint(hklss)
            ehklss = np.abs(hklss - rhklss)
            valid_peak_mask = np.max(ehklss, axis=-1) < index_error_tol
            valid_peak_counts = np.count_nonzero(valid_peak_mask, axis=-1)
            error_means = ehklss.mean(axis=(1, 2))

            num_peaks = len(peaks)
            match_rates = (valid_peak_counts * (1 / num_peaks)) if num_peaks else 0

            possible_solution_mask = match_rates > 0
            solutions += [
                OrientationResult(
                    phase_index=phase_index,
                    rotation_matrix=R,
                    match_rate=match_rate,
                    error_hkls=ehkls,
                    total_error=error_mean,
                    scale=1.0,
                    center_x=0.0,
                    center_y=0.0,
                )
                for R, match_rate, ehkls, error_mean in zip(
                    rotations[possible_solution_mask],
                    match_rates[possible_solution_mask],
                    ehklss[possible_solution_mask],
                    error_means[possible_solution_mask],
                )
            ]

            res_rhkls += rhklss[possible_solution_mask].tolist()

        n_solutions = min(n_best, len(solutions))

        i = phase_index * n_best  # starting index in unfolded array

        if n_solutions > 0:
            top_n = sorted(solutions, key=attrgetter("match_rate"), reverse=True)[
                :n_solutions
            ]

            # Put the top n ranked solutions in the output array
            top_matches[i : i + n_solutions] = top_n

        if n_solutions < n_best:
            # Fill with dummy values
            top_matches[i + n_solutions : i + n_best] = [
                OrientationResult(
                    phase_index=0,
                    rotation_matrix=np.identity(3),
                    match_rate=0.0,
                    error_hkls=np.array([]),
                    total_error=1.0,
                    scale=1.0,
                    center_x=0.0,
                    center_y=0.0,
                )
                for x in range(n_best - n_solutions)
            ]

    # Because of a bug in numpy (https://github.com/numpy/numpy/issues/7453),
    # triggered by the way HyperSpy reads results (np.asarray(res), which fails
    # when the two tuple values have the same first dimension), we cannot
    # return a tuple directly, but instead have to format the result as an
    # array ourselves.
    res = np.empty(2, dtype=np.object)
    res[0] = top_matches
    res[1] = np.asarray(res_rhkls)
    return res
