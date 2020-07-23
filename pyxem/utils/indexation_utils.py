# -*- coding: utf-8 -*-
# Copyright 2016-2020 The pyXem developers
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
from operator import itemgetter, attrgetter

import numpy as np

from pyxem.utils.expt_utils import _cart2polar
from pyxem.utils.vector_utils import get_rotation_matrix_between_vectors
from pyxem.utils.vector_utils import get_angle_cartesian

from transforms3d.euler import mat2euler, euler2mat
from transforms3d.quaternions import mat2quat

from collections import namedtuple


# container for OrientationResults
OrientationResult = namedtuple(
    "OrientationResult",
    "phase_index rotation_matrix match_rate error_hkls total_error scale center_x center_y".split(),
)


def optimal_fft_size(target, real=False):
    """Wrapper around scipy function next_fast_len() for calculating optimal FFT padding.
    scipy.fft was only added in 1.4.0, so we fall back to scipy.fftpack
    if it is not available. The main difference is that next_fast_len()
    does not take a second argument in the older implementation.

    Parameters
    ----------
    target : int
        Length to start searching from. Must be a positive integer.
    real : bool, optional
        True if the FFT involves real input or output, only available
        for scipy > 1.4.0
    Returns
    -------
    int
        Optimal FFT size.
    """

    try:  # pragma: no cover
        from scipy.fft import next_fast_len

        support_real = True

    except ImportError:  # pragma: no cover
        from scipy.fftpack import next_fast_len

        support_real = False

    if support_real:  # pragma: no cover
        return next_fast_len(target, real)
    else:  # pragma: no cover
        return next_fast_len(target)


# Functions used in correlate_library.
def fast_correlation(image_intensities, int_local, pn_local, **kwargs):
    """
    Computes the correlation score between an image and a template, using the formula
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

    See also:
    ---------
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
    """
    Computes the correlation score between an image and a template, using the formula
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

    See also:
    ---------
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


def full_frame_correlation(image_FT, image_norm, pattern_FT, pattern_norm):
    """
    Computes the correlation score between an image and a template in Fourier space.

    Parameters:
    -----------
    image: numpy.ndarray
        Intensities of the image in fourier space, stored in a NxM numpy array
    image_norm: float
        The norm of the real space image, corresponding to image_FT
    fsize: numpy.ndarray
        The size of image_FT, for us in transform of template.
    template_coordinates: numpy array
        Array containing coordinates for non-zero intensities in the template
    template_intensities: list
        List of intensity values for the template.

    Returns:
    --------
    corr_local: float
        Correlation score between image and template.

    See also:
    ---------
    correlate_library, fast_correlation, zero_mean_normalized_correlation

    Reference:
    ----------
    A. Foden, D. M. Collins, A. J. Wilkinson and T. B. Britton "Indexing electron backscatter diffraction patterns with
     a refined template matching approach" doi: https://doi.org/10.1016/j.ultramic.2019.112845
    """

    fprod = pattern_FT * image_FT

    res_matrix = np.fft.ifftn(fprod)
    fsize = res_matrix.shape
    corr_local = np.max(
        np.real(
            res_matrix[
                max(fsize[0] // 2 - 3, 0) : min(fsize[0] // 2 + 3, fsize[0]),
                max(fsize[1] // 2 - 3, 0) : min(fsize[1] // 2 + 3, fsize[1]),
            ]
        )
    )
    if image_norm > 0 and pattern_norm > 0:
        corr_local = corr_local / (image_norm * pattern_norm)

    # Sub-pixel refinement can be done here - Equation (5) in reference article

    return corr_local


def correlate_library_from_dict(image, template_dict, n_largest, method, mask):
    """Correlates all simulated diffraction templates in a DiffractionLibrary
    with a particular experimental diffraction pattern (image).

    Parameters
    ----------
    image : numpy.array
        The experimental diffraction pattern of interest.
    template_dict : dict
        Dictionary containing orientations, fourier transform of templates and template norms for
        every phase.
    n_largest : int
        The number of well correlated simulations to be retained.
    method : str
        Name of method used to compute correlation between templates and diffraction patterns. Can be
         'full_frame_correlation'. (I believe angular decomposition can also fit this framework)
    mask : bool
        A mask for navigation axes. 1 indicates positions to be indexed.


    Returns
    -------
    top_matches : numpy.array
        Array of shape (<num phases>*n_largest, 3) containing the top n
        correlated simulations for the experimental pattern of interest, where
        each entry is on the form [phase index, [z, x, z], correlation].


    References
    ----------
    full_frame_correlation:
    A. Foden, D. M. Collins, A. J. Wilkinson and T. B. Britton "Indexing electron backscatter diffraction patterns with
     a refined template matching approach" doi: https://doi.org/10.1016/j.ultramic.2019.112845
    """

    top_matches = np.empty((len(template_dict), n_largest, 3), dtype="object")

    if method == "full_frame_correlation":
        size = 2 * np.array(image.shape) - 1
        fsize = [optimal_fft_size(a, real=True) for a in (size)]
        image_FT = np.fft.fftshift(np.fft.rfftn(image, fsize))
        image_norm = np.sqrt(full_frame_correlation(image_FT, 1, image_FT, 1))

    if mask == 1:
        for phase_index, library_entry in enumerate(template_dict.values()):
            orientations = library_entry["orientations"]
            patterns = library_entry["patterns"]
            pattern_norms = library_entry["pattern_norms"]

            zip_for_locals = zip(orientations, patterns, pattern_norms)

            or_saved, corr_saved = np.empty((n_largest, 3)), np.zeros((n_largest, 1))

            for (or_local, pat_local, pn_local) in zip_for_locals:

                if method == "full_frame_correlation":
                    corr_local = full_frame_correlation(
                        image_FT, image_norm, pat_local, pn_local
                    )

                if corr_local > np.min(corr_saved):
                    or_saved[np.argmin(corr_saved)] = or_local
                    corr_saved[np.argmin(corr_saved)] = corr_local

                combined_array = np.hstack((or_saved, corr_saved))
                combined_array = combined_array[
                    np.flip(combined_array[:, 3].argsort())
                ]  # see stackoverflow/2828059 for details
                top_matches[phase_index, :, 0] = phase_index
                top_matches[phase_index, :, 2] = combined_array[:, 3]  # correlation
                for i in np.arange(n_largest):
                    top_matches[phase_index, i, 1] = combined_array[
                        i, :3
                    ]  # orientation

    return top_matches.reshape(-1, 3)


def correlate_library(image, library, n_largest, method, mask):
    """Correlates all simulated diffraction templates in a DiffractionLibrary
    with a particular experimental diffraction pattern (image).

    Calculated using the normalised (see return type documentation) dot
    product, or cosine distance,

    .. math:: fast_correlation
        \\frac{\\sum_{j=1}^m P(x_j, y_j) T(x_j, y_j)}{\\sqrt{\\sum_{j=1}^m T^2(x_j, y_j)}}

    .. math:: zero_mean_normalized_correlation
        \\frac{\\sum_{j=1}^m P(x_j, y_j) T(x_j, y_j)- avg(P)avg(T)}{\\sqrt{\\sum_{j=1}^m (T(x_j, y_j)-avg(T))^2+\sum_{j=1}^m P(x_j,y_j)-avg(P)}}
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
    method : str
        Name of method used to compute correlation between templates and diffraction patterns. Can be
        'fast_correlation', 'full_frame_correlation' or 'zero_mean_normalized_correlation'. (ADDED in pyxem 0.11.0)
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

    A. Nakhmani and  A. Tannenbaum, "A New Distance Measure Based on Generalized Image Normalized Cross-Correlation
    for Robust Video Tracking and Image Recognition"
    Pattern Recognit Lett. 2013 Feb 1; 34(3): 315–321; doi: 10.1016/j.patrec.2012.10.025

    Discussion on Normalized cross correlation (xcdskd):
    https://xcdskd.readthedocs.io/en/latest/cross_correlation/cross_correlation_coefficient.html

    """

    top_matches = np.empty((len(library), n_largest, 3), dtype="object")

    if method == "zero_mean_normalized_correlation":
        nb_pixels = image.shape[0] * image.shape[1]
        average_image_intensity = np.average(image)
        image_std = np.linalg.norm(image - average_image_intensity)

    if mask == 1:
        for phase_index, library_entry in enumerate(library.values()):
            orientations = library_entry["orientations"]
            pixel_coords = library_entry["pixel_coords"]
            intensities = library_entry["intensities"]
            # TODO: This is only applicable some of the time, probably use an if + special_local in the for
            pattern_norms = library_entry["pattern_norms"]

            zip_for_locals = zip(orientations, pixel_coords, intensities, pattern_norms)

            or_saved, corr_saved = np.empty((n_largest, 3)), np.zeros((n_largest, 1))

            for (or_local, px_local, int_local, pn_local) in zip_for_locals:
                # TODO: Factorise out the generation of corr_local to a method='mthd' section
                # Extract experimental intensities from the diffraction image
                image_intensities = image[
                    px_local[:, 1], px_local[:, 0]
                ]  # Counter intuitive indexing? Why is it not px_local[:, 0], px_local[:, 1]?

                if method == "zero_mean_normalized_correlation":
                    corr_local = zero_mean_normalized_correlation(
                        nb_pixels,
                        image_std,
                        average_image_intensity,
                        image_intensities,
                        int_local,
                    )

                elif method == "fast_correlation":
                    corr_local = fast_correlation(
                        image_intensities, int_local, pn_local
                    )

                if corr_local > np.min(corr_saved):
                    or_saved[np.argmin(corr_saved)] = or_local
                    corr_saved[np.argmin(corr_saved)] = corr_local

                combined_array = np.hstack((or_saved, corr_saved))
                combined_array = combined_array[
                    np.flip(combined_array[:, 3].argsort())
                ]  # see stackoverflow/2828059 for details
                top_matches[phase_index, :, 0] = phase_index
                top_matches[phase_index, :, 2] = combined_array[:, 3]  # correlation
                for i in np.arange(n_largest):
                    top_matches[phase_index, i, 1] = combined_array[
                        i, :3
                    ]  # orientation

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
    return angles.argsort()[
        np.linspace(0, angles.shape[0] - 1, n_peaks_to_index, dtype=np.int)
    ]


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
            if q1_len < q2_len:
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
            if np.count_nonzero(tolerance_mask) == 0:
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
    results_array = np.empty(3, dtype="object")
    # Consider single phase and multi-phase matching cases separately
    if np.unique(z_matches[:, 0]).shape[0] == 1:
        # get best matching phase (there is only one here)
        results_array[0] = z_matches[0, 0]
        # get best matching orientation Euler angles
        results_array[1] = z_matches[0, 1]
        # get template matching metrics
        metrics = dict()
        metrics["correlation"] = z_matches[0, 2]
        metrics["orientation_reliability"] = (
            100 * (1 - z_matches[1, 2] / z_matches[0, 2])
            if z_matches[0, 2] > 0
            else 100
        )
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
        metrics["correlation"] = z_matches[index_best_match, 2]
        metrics["orientation_reliability"] = 100 * (
            1 - second_orientation / z_matches[index_best_match, 2]
        )
        metrics["phase_reliability"] = 100 * (
            1 - second_phase / z_matches[index_best_match, 2]
        )
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
    if z_matches.shape == (1,):  # pragma: no cover
        z_matches = z_matches[0]

    # Create empty array for results.
    results_array = np.empty(3, dtype="object")

    # get best matching phase
    best_match = get_nth_best_solution(
        z_matches, "vector", key="total_error", descending=False
    )
    results_array[0] = best_match.phase_index

    # get best matching orientation Euler angles
    results_array[1] = np.rad2deg(mat2euler(best_match.rotation_matrix, "rzxz"))

    # get vector matching metrics
    metrics = dict()
    metrics["match_rate"] = best_match.match_rate
    metrics["ehkls"] = best_match.error_hkls
    metrics["total_error"] = best_match.total_error

    # get second highest correlation phase for phase_reliability (if present)
    other_phase_matches = [
        match for match in z_matches if match.phase_index != best_match.phase_index
    ]

    if other_phase_matches:
        second_best_phase = sorted(
            other_phase_matches, key=attrgetter("total_error"), reverse=False
        )[0]

        metrics["phase_reliability"] = 100 * (
            1 - best_match.total_error / second_best_phase.total_error
        )

        # get second best matching orientation for orientation_reliability
        same_phase_matches = [
            match for match in z_matches if match.phase_index == best_match.phase_index
        ]
        second_match = sorted(
            same_phase_matches, key=attrgetter("total_error"), reverse=False
        )[1]
    else:
        # get second best matching orientation for orientation_reliability
        second_match = get_nth_best_solution(
            z_matches, "vector", rank=1, key="total_error", descending=False
        )

    metrics["orientation_reliability"] = 100 * (
        1 - best_match.total_error / (second_match.total_error or 1.0)
    )

    results_array[2] = metrics

    return results_array


def get_phase_name_and_index(library):

    """ Get a dictionary of phase names and its corresponding index value in library.keys().

     Parameters
    ----------
    library : DiffractionLibrary
        Diffraction library containing the phases and rotations

    Returns
    -------
    phase_name_index_dict : Dictionary {str : int}
    typically on the form {'phase_name 1' : 0, 'phase_name 2': 1, ...}
    """

    phase_name_index_dict = dict([(y, x) for x, y in enumerate(list(library.keys()))])
    return phase_name_index_dict


def peaks_from_best_template(single_match_result, library, rank=0):
    """ Takes a TemplateMatchingResults object and return the associated peaks,
    to be used in combination with map().

    Parameters
    ----------
    single_match_result : ndarray
        An entry in a TemplateMatchingResults.
    library : DiffractionLibrary
        Diffraction library containing the phases and rotations.
    rank : int
        Get peaks from nth best orientation (default: 0, best vector match)

    Returns
    -------
    peaks : array
        Coordinates of peaks in the matching results object in calibrated units.
    """
    best_fit = get_nth_best_solution(single_match_result, "template", rank=rank)

    phase_names = list(library.keys())
    phase_index = int(best_fit[0])
    phase = phase_names[phase_index]
    simulation = library.get_library_entry(phase=phase, angle=tuple(best_fit[1]))["Sim"]

    peaks = simulation.coordinates[:, :2]  # cut z
    return peaks


def peaks_from_best_vector_match(single_match_result, library, rank=0):
    """Takes a VectorMatchingResults object and return the associated peaks,
    to be used in combination with map().

    Parameters
    ----------
    single_match_result : ndarray
        An entry in a VectorMatchingResults
    library : DiffractionLibrary
        Diffraction library containing the phases and rotations
    rank : int
        Get peaks from nth best orientation (default: 0, best vector match)

    Returns
    -------
    peaks : ndarray
        Coordinates of peaks in the matching results object in calibrated units.
    """
    best_fit = get_nth_best_solution(single_match_result, "vector", rank=rank)
    phase_index = best_fit.phase_index

    rotation_orientation = mat2euler(best_fit.rotation_matrix)
    # Don't change the original
    structure = library.structures[phase_index]
    sim = library.diffraction_generator.calculate_ed_data(
        structure,
        reciprocal_radius=library.reciprocal_radius,
        rotation=rotation_orientation,
        with_direct_beam=False,
    )

    # Cut z
    return sim.coordinates[:, :2]
