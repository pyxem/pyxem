# -*- coding: utf-8 -*-
# Copyright 2016-2024 The pyXem developers
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

"""Utilities for indexing electron diffraction spot patterns."""

from collections import namedtuple
from itertools import combinations
from operator import attrgetter
import warnings

from dask.diagnostics import ProgressBar
from numba import njit, prange
import numpy as np
from orix.crystal_map import CrystalMap, PhaseList
from orix.quaternion import Rotation
import psutil
import scipy

from pyxem.utils.diffraction import _cart2polar
from pyxem.utils.vectors import get_rotation_matrix_between_vectors
from pyxem.utils.vectors import get_angle_cartesian
from pyxem.utils.cuda_utils import (
    is_cupy_array,
    get_array_module,
    _correlate_polar_image_to_library_gpu,
    TPB,
)
from pyxem.utils._dask import _get_dask_array
from pyxem.utils.polar_transform_utils import (
    _cartesian_positions_to_polar,
    get_polar_pattern_shape,
    image_to_polar,
    get_template_polar_coordinates,
    _warp_polar_custom,
)
from diffpy.structure import Atom, Structure, Lattice
from orix.crystal_map import Phase

try:
    import cupy as cp

    CUPY_INSTALLED = True
    import cupyx.scipy as spgpu
except ImportError:
    CUPY_INSTALLED = False


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
        indexation[i] = np.array((mags.data[i], indices), dtype=object)

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
        np.linspace(0, angles.shape[0] - 1, n_peaks_to_index, dtype=int)
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
    if peaks.shape == (1,) and peaks.dtype == object:
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
    res = np.empty(2, dtype=object)
    res[0] = top_matches
    res[1] = np.asarray(res_rhkls)
    return res


def _simulations_to_arrays(simulations, max_radius=None):
    """
    Convert simulation results to arrays of diffraction spots

    Parameters
    ----------
    simulations : list
        list of diffsims.sims.diffraction_simulation.DiffractionSimulation
        objects
    max_radius : float
        limit to g-vector length in pixel coordinates

    Returns
    -------
    positions : numpy.ndarray (N, 2, R)
        An array containing all (x,y) coordinates of reflections of N templates. R represents
        the maximum number of reflections; templates containing fewer
        reflections are padded with 0's at the end. In pixel units.
    intensities : numpy.ndarray (N, R)
        An array containing all intensities of reflections of N templates
    """
    num_spots = [i.intensities.shape[0] for i in simulations]
    max_spots = max(num_spots)
    positions = np.zeros((len(simulations), 2, max_spots), dtype=np.float64)
    intensities = np.zeros((len(simulations), max_spots), dtype=np.float64)
    for i, j in enumerate(simulations):
        x = j.calibrated_coordinates[:, 0]
        y = j.calibrated_coordinates[:, 1]
        intensity = j.intensities
        if max_radius is not None:
            condition = x**2 + y**2 < max_radius**2
            x = x[condition]
            y = y[condition]
            intensity = intensity[condition]
        positions[i, 0, : x.shape[0]] = x
        positions[i, 1, : y.shape[0]] = y
        intensities[i, : intensity.shape[0]] = intensity
    return positions, intensities


def _match_polar_to_polar_template(
    polar_image,
    r_template,
    theta_template,
    intensities,
):
    """
    Correlate a single polar template to a single polar image

    The template spots are shifted along the azimuthal axis by 1 pixel increments.
    A simple correlation index is calculated at each position.

    Parameters
    ----------
    polar_image : 2D ndarray
        the polar image
    r_template : 1D ndarray
        r coordinates of diffraction spots in template
    theta_template : 1D ndarray
        theta coordinates of diffraction spots in template
    intensities : 1D ndarray
        intensities of diffraction spots in template

    Returns
    -------
    correlation : 1D ndarray
        correlation index at each in-plane angle position
    """
    dispatcher = get_array_module(polar_image)
    sli = polar_image[:, r_template]
    rows, column_indices = dispatcher.ogrid[: sli.shape[0], : sli.shape[1]]
    rows = dispatcher.mod(rows + theta_template[None, :], polar_image.shape[0])
    extr = sli[rows, column_indices].astype(intensities.dtype)
    correlation = dispatcher.dot(extr, intensities)
    return correlation


@njit(parallel=True, nogil=True)
def _match_polar_to_polar_library_cpu(
    polar_image,
    r_templates,
    theta_templates,
    intensities_templates,
):
    """
    Correlates a polar pattern to all polar templates on CPU

    Parameters
    ----------
    polar_image : 2D numpy.ndarray
        The image converted to polar coordinates
    r_templates : 2D numpy.ndarray
        r-coordinates of diffraction spots in templates.
    theta_templates : 2D numpy ndarray
        theta-coordinates of diffraction spots in templates.
    intensities_templates : 2D numpy.ndarray
        intensities of the spots in each template

    Returns
    -------
    best_in_plane_shift : (N) 1D numpy.ndarray
        Shift for all templates that yields best correlation
    best_in_plane_corr : (N) 1D numpy.ndarray
        Correlation at best match for each template
    best_in_plane_shift_m : (N) 1D numpy.ndarray
        Shift for all mirrored templates that yields best correlation
    best_in_plane_corr_m : (N) 1D numpy.ndarray
        Correlation at best match for each mirrored template

    Notes
    -----
    The dimensions of r_templates and theta_templates should be (N, R) where
    N is the number of templates and R the number of spots in the template
    with the maximum number of spots
    """
    N = r_templates.shape[0]
    R = r_templates.shape[1]
    n_shifts = polar_image.shape[0]
    best_in_plane_shift = np.empty(N, dtype=np.int32)
    best_in_plane_shift_m = np.empty(N, dtype=np.int32)
    best_in_plane_corr = np.empty(N, dtype=polar_image.dtype)
    best_in_plane_corr_m = np.empty(N, dtype=polar_image.dtype)

    for template in prange(N):
        inplane_cor = np.zeros(n_shifts)
        inplane_cor_m = np.zeros(n_shifts)
        for spot in range(R):
            rsp = r_templates[template, spot]
            if rsp == 0:
                break
            tsp = theta_templates[template, spot]
            isp = intensities_templates[template, spot]
            split = n_shifts - tsp
            column = polar_image[:, rsp] * isp
            inplane_cor[:split] += column[tsp:]
            inplane_cor[split:] += column[:tsp]
            inplane_cor_m[:tsp] += column[split:]
            inplane_cor_m[tsp:] += column[:split]

        best_shift = np.argmax(inplane_cor)
        best_shift_m = np.argmax(inplane_cor_m)
        best_in_plane_shift[template] = best_shift
        best_in_plane_shift_m[template] = best_shift_m
        best_in_plane_corr[template] = inplane_cor[best_shift]
        best_in_plane_corr_m[template] = inplane_cor_m[best_shift_m]

    return (
        best_in_plane_shift,
        best_in_plane_corr,
        best_in_plane_shift_m,
        best_in_plane_corr_m,
    )


def _match_polar_to_polar_library_gpu(
    polar_image,
    r_templates,
    theta_templates,
    intensities_templates,
):
    """
    Correlates a polar pattern to all polar templates on GPU

    Parameters
    ----------
    polar_image : 2D cupy.ndarray
        The image converted to polar coordinates
    r_templates : 2D cupy.ndarray
        r-coordinates of diffraction spots in templates.
    theta_templates : 2D cupy ndarray
        theta-coordinates of diffraction spots in templates.
    intensities_templates : 2D cupy.ndarray
        intensities of the spots in each template

    Returns
    -------
    best_in_plane_shift : (N) 1D cupy.ndarray
        Shift for all templates that yields best correlation
    best_in_plane_corr : (N) 1D cupy.ndarray
        Correlation at best match for each template
    best_in_plane_shift_m : (N) 1D cupy.ndarray
        Shift for all mirrored templates that yields best correlation
    best_in_plane_corr_m : (N) 1D cupy.ndarray
        Correlation at best match for each mirrored template

    Notes
    -----
    The dimensions of r_templates and theta_templates should be (N, R) where
    N is the number of templates and R the number of spots in the template
    with the maximum number of spots
    """
    correlation = cp.empty(
        (r_templates.shape[0], polar_image.shape[0]), dtype=cp.float32
    )
    correlation_m = cp.empty(
        (r_templates.shape[0], polar_image.shape[0]), dtype=cp.float32
    )
    threadsperblock = (1, TPB)
    blockspergrid = (r_templates.shape[0], int(np.ceil(polar_image.shape[0] / TPB)))
    _correlate_polar_image_to_library_gpu[blockspergrid, threadsperblock](
        polar_image,
        r_templates,
        theta_templates,
        intensities_templates,
        correlation,
        correlation_m,
    )
    best_in_plane_shift = cp.argmax(correlation, axis=1).astype(np.int32)
    best_in_plane_shift_m = cp.argmax(correlation_m, axis=1).astype(np.int32)
    rows = cp.arange(correlation.shape[0], dtype=np.int32)
    best_in_plane_corr = correlation[rows, best_in_plane_shift]
    best_in_plane_corr_m = correlation_m[rows, best_in_plane_shift_m]
    return (
        best_in_plane_shift,
        best_in_plane_corr,
        best_in_plane_shift_m,
        best_in_plane_corr_m,
    )


def _get_row_norms(array):
    """Get the norm of all rows in a 2D array"""
    norms = ((array**2).sum(axis=1)) ** 0.5
    return norms


def _norm_rows(array):
    """Normalize all the rows in a 2D array"""
    norms = _get_row_norms(array)
    return array / norms[:, None]


def _get_integrated_polar_templates(
    r_max, r_templates, intensities_templates, normalize_templates
):
    """
    Get an azimuthally integrated representation of the templates.

    Parameters
    ----------
    r_max : float
        maximum radial distance to consider in pixel units. Typically the
        radial width of the polar images.
    r_templates : 2D numpy or cupy ndarray
        r-coordinate of all spots in the templates. Of shape (N, R) where
        N is the number of templates and R is the number of spots in the
        template with the maximum number of spots
    intensities_templates : 2D numpy or cupy ndarray
        intensities in all spots of the templates. Of shape (N, R)
    normalize_templates : bool
        Whether to normalize the integrated templates

    Returns
    -------
    integrated_templates : 2D numpy or cupy ndarray
        Templates integrated over the azimuthal axis of shape (N, r_max)
    """
    dispatcher = get_array_module(intensities_templates)
    data = intensities_templates.ravel()
    columns = r_templates.ravel()
    rows = dispatcher.arange(r_templates.shape[0]).repeat(r_templates.shape[1])
    out_shape = (r_templates.shape[0], r_max)
    if is_cupy_array(intensities_templates):
        integrated_templates = spgpu.sparse.coo_matrix(
            (data, (rows, columns)), shape=out_shape
        ).toarray()
    else:
        integrated_templates = scipy.sparse.coo_matrix(
            (data, (rows, columns)), shape=out_shape
        ).toarray()
    if normalize_templates:
        integrated_templates = _norm_rows(integrated_templates)
    return integrated_templates


def _match_library_to_polar_fast(polar_sum, integrated_templates):
    """
    Compare a polar image to azimuthally integrated templates

    Parameters
    ----------
    polar_sum : 1D numpy array or cupy array
        the image in polar coordinates integrated along the azimuthal axis
        (shape = (r_max,))
    integrated_templates : 2D numpy array or cupy array
        azimuthally integrated templates of shape (N, r_max) with N
        the number of templates and r_max the width of the polar image

    Returns
    -------
    correlations : 1D numpy array or cupy array
        the correlation between the integrated image and the integrated
        templates. (shape = (N,))
    """
    return (integrated_templates * polar_sum).sum(axis=1)


def _prepare_image_and_templates(
    image,
    simulations,
    delta_r,
    delta_theta,
    max_r,
    intensity_transform_function,
    find_direct_beam,
    direct_beam_position,
    normalize_image,
    normalize_templates,
):
    """
    Prepare a single cartesian coordinate image and a template library for comparison

    Parameters
    ----------
    image : 2D np or cp ndarray
        The diffraction pattern in cartesian coordinates
    simulations : list
        list of diffsims.sims.diffraction_simulation.DiffractionSimulation
    delta_r : float
        sampling interval for the r coordinate in the polar image in pixels
    delta_theta : float
        sampling interval for the theta coordinate in the polar image in degrees
    max_r : float
        maximum radius to consider in polar conversion, in pixels
    intensity_transform_function : Callable
        function to apply to both the image and template intensities. Must
        accept any dimensional numpy array as input and preferably operate
        independently on individual elements
    find_direct_beam : bool
        whether to refine the direct beam position in the image polar conversion
    direct_beam_position : 2-tuple of floats
        the (x, y) position of the direct beam in the image to override any
        defaults
    normalize_image : bool
        Whether to normalize the image
    normalize_templates : bool
        Whether to normalize the template intensities

    Returns
    -------
    polar_image : 2D np or cp ndarray
        The image in polar coordinates
    r : 2D np or cp ndarray
        The r coordinates in the polar image corresponding to template spots.
        shape = (N, R) with N the number of templates and R the number of spots
        contained by the template with the most spots
    theta : 2D np or cp ndarray
        The theta coordinates in the polar image corresponding to template spots.
        shape = (N, R) with N the number of templates and R the number of spots
        contained by the template with the most spots
    intensities :  2D np or cp ndarray
        The intensities corresponding to template spots.
        shape = (N, R) with N the number of templates and R the number of spots
        contained by the template with the most spots
    """
    polar_image = image_to_polar(
        image,
        delta_r,
        delta_theta,
        max_r=max_r,
        find_direct_beam=find_direct_beam,
        direct_beam_position=direct_beam_position,
    )
    dispatcher = get_array_module(polar_image)
    max_radius = polar_image.shape[1] * delta_r
    positions, intensities = _simulations_to_arrays(simulations, max_radius=max_radius)
    r, theta = _cartesian_positions_to_polar(
        positions[:, 0], positions[:, 1], delta_r, delta_theta
    )
    condition = r >= polar_image.shape[1]
    # we don't set r to 0 because it may quit the loop in the matching early
    r[condition] = polar_image.shape[1] - 1
    theta[condition] = 0
    intensities[condition] = 0.0
    if is_cupy_array(polar_image):
        # send data to GPU
        r = cp.asarray(r)
        theta = cp.asarray(theta)
        intensities = cp.asarray(intensities)
    if intensity_transform_function is not None:
        intensities = intensity_transform_function(intensities)
        polar_image = intensity_transform_function(polar_image)
    if normalize_image:
        polar_image = polar_image / dispatcher.linalg.norm(polar_image)
    if normalize_templates:
        intensities = _norm_rows(intensities)
    return (polar_image, r, theta, intensities)


def _mixed_matching_lib_to_polar(
    polar_image,
    integrated_templates,
    r_templates,
    theta_templates,
    intensities_templates,
    n_keep,
    frac_keep,
    n_best,
    transpose=False,
):
    """
    Match a polar image to a filtered subset of polar templates

    First does a fast matching basted on azimuthally integrated templates
    Then it takes the (1-fraction)*100% of patterns to do a full indexation on.
    Return the first n_best answers.

    Parameters
    ----------
    polar_image : 2D ndarray
        image in polar coordinates
    integrated_templates : 2D ndarray, (N, r_max)
        azimuthally integrated templates
    r_templates : 2D ndarray, (N, R)
        r coordinates of diffraction spots in all N templates
    theta_templates : 2D ndarray, (N, R)
        theta coordinates of diffraction spots in all N templates
    intensities_templates : 2D ndarray, (N, R)
        intensities of diffraction spots in all N templates
    frac_keep : float
        fraction of templates to pass on to the full indexation
    n_keep : float
        number of templates to pass to the full indexation
    n_best : int
        number of solutions to return in decending order of fit

    Return
    ------
    answer : 2D numpy array, (n_best, 4)
        in the colums are returned (template index, correlation, in-plane angle, factor)
        of the best fitting template, where factor is 1 if the direct template is
        matched and -1 if the mirror template is matched
    """
    if transpose:
        polar_image = polar_image.T
    polar_image = np.nan_to_num(polar_image)
    dispatcher = get_array_module(polar_image)
    # remove templates we don't care about with a fast match
    (
        template_indexes,
        r_templates,
        theta_templates,
        intensities_templates,
    ) = _prefilter_templates(
        polar_image,
        r_templates,
        theta_templates,
        intensities_templates,
        integrated_templates,
        frac_keep,
        n_keep,
    )
    # get a full match on the filtered data - we must branch for CPU/GPU
    (
        best_in_plane_shift,
        best_in_plane_corr,
        best_in_plane_shift_m,
        best_in_plane_corr_m,
    ) = _get_full_correlations(
        polar_image,
        r_templates,
        theta_templates,
        intensities_templates,
    )
    # compare positive and negative templates and combine
    positive_is_best = best_in_plane_corr >= best_in_plane_corr_m
    negative_is_best = ~positive_is_best
    # multiplication method is faster than dispatcher.choose
    best_sign = positive_is_best * 1 + negative_is_best * (-1)
    best_cors = (
        positive_is_best * best_in_plane_corr + negative_is_best * best_in_plane_corr_m
    )
    best_angles = (
        positive_is_best * best_in_plane_shift
        + negative_is_best * best_in_plane_shift_m
    )
    if n_best >= best_cors.shape[0]:
        n_best = best_cors.shape[0]
    if n_best < 1:
        nbest = 1
    answer = dispatcher.empty((n_best, 4), dtype=polar_image.dtype)
    if n_best == 1:
        max_index_filter = dispatcher.argmax(best_cors)
        max_cor = best_cors[max_index_filter]
        max_angle = best_angles[max_index_filter]
        max_index = template_indexes[max_index_filter]
        max_sign = best_sign[max_index_filter]
        answer[0] = dispatcher.array((max_index, max_cor, max_angle, max_sign))
    else:
        # a partial sort
        indices_nbest = dispatcher.argpartition(-best_cors, n_best - 1)[:n_best]
        nbest_cors = best_cors[indices_nbest]
        # a full sort on this subset
        indices_sorted = dispatcher.argsort(-nbest_cors)
        n_best_indices = indices_nbest[indices_sorted]
        answer[:, 0] = template_indexes[n_best_indices]
        answer[:, 1] = best_cors[n_best_indices]
        answer[:, 2] = best_angles[n_best_indices]
        answer[:, 3] = best_sign[n_best_indices]
    return answer


def _index_chunk(
    images,
    center,
    max_radius,
    output_shape,
    precision,
    integrated_templates,
    r_templates,
    theta_templates,
    intensities_templates,
    n_keep,
    frac_keep,
    n_best,
    norm_images,
    order=1,
):
    dispatcher = get_array_module(images)
    # prepare an empty results chunk
    indexation_result_chunk = dispatcher.empty(
        (images.shape[0], images.shape[1], n_best, 4),
        dtype=precision,
    )
    for index in np.ndindex(images.shape[:2]):
        polar_image = _warp_polar_custom(
            images[index],
            center,
            max_radius,
            output_shape,
            order=order,
            precision=precision,
        )
        if norm_images:
            polar_image = polar_image / dispatcher.linalg.norm(polar_image)
        indexation_result_chunk[index] = _mixed_matching_lib_to_polar(
            polar_image,
            integrated_templates,
            r_templates,
            theta_templates,
            intensities_templates,
            n_keep,
            frac_keep,
            n_best,
        )
    return indexation_result_chunk


def _index_chunk_gpu(images, *args, **kwargs):
    gpu_im = cp.asarray(images)
    indexed_chunk = _index_chunk(gpu_im, *args, **kwargs)
    return cp.asnumpy(indexed_chunk)


def get_in_plane_rotation_correlation(
    image,
    simulation,
    intensity_transform_function=None,
    delta_r=1,
    delta_theta=1,
    max_r=None,
    find_direct_beam=False,
    direct_beam_position=None,
    normalize_image=True,
    normalize_template=True,
):
    """
    Correlate a single image and simulation over the in-plane rotation angle

    Parameters
    ----------
    image : 2D numpy.ndarray
        The image of the diffraction pattern
    simulation : diffsims.sims.diffraction_simulation.DiffractionSimulation
        The diffraction pattern simulation
    intensity_transform_function : Callable, optional
        Function to apply to both image and template intensities on an
        element by element basis prior to comparison
    delta_r : float, optional
        The sampling interval of the radial coordinate in pixels
    delta_theta : float, optional
        The sampling interval of the azimuthal coordinate in degrees
    max_r : float, optional
        Maximum radius to consider in pixel units. By default it is the
        distance from the center of the image to a corner
    find_direct_beam : bool, optional
        Whether to optimize the direct beam, otherwise the center of the image
        is chosen
    direct_beam_position : 2-tuple of floats, optional
        (x, y) coordinate of the direct beam in pixel units. Overrides other
        settings for finding the direct beam
    normalize_image : bool, optional
        normalize the image to calculate the correlation coefficient
    normalize_template : bool, optional
        normalize the template to calculate the correlation coefficient

    Returns
    -------
    angle_array : 1D np.ndarray
        The in-plane angles at which the correlation is calculated in degrees
    correlation_array : 1D np.ndarray
        The correlation corresponding to these angles
    """
    polar_image = image_to_polar(
        image,
        delta_r,
        delta_theta,
        max_r=max_r,
        find_direct_beam=find_direct_beam,
        direct_beam_position=direct_beam_position,
    )
    r, theta, intensity = get_template_polar_coordinates(
        simulation,
        in_plane_angle=0.0,
        delta_r=delta_r,
        delta_theta=delta_theta,
        max_r=polar_image.shape[1],
    )
    if is_cupy_array(polar_image):
        dispatcher = cp
        r = cp.asarray(r)
        theta = cp.asarray(theta)
        intensity = cp.asarray(intensity)
    else:
        dispatcher = np
    if intensity_transform_function is not None:
        intensity = intensity_transform_function(intensity)
        polar_image = intensity_transform_function(polar_image)
    if normalize_image:
        polar_image = polar_image / dispatcher.linalg.norm(polar_image)
    if normalize_template:
        intensity = intensity / dispatcher.linalg.norm(intensity)
    correlation_array = _match_polar_to_polar_template(
        polar_image,
        r,
        theta,
        intensity,
    )
    angle_array = dispatcher.arange(correlation_array.shape[0]) * delta_theta
    return angle_array, correlation_array


def _get_fast_correlation_index(
    polar_image,
    r,
    intensities,
    normalize_image,
    normalize_templates,
):
    dispatcher = get_array_module(polar_image)
    integrated_polar = polar_image.sum(axis=0)
    rrr = dispatcher.arange(integrated_polar.shape[0]) / integrated_polar.shape[0]
    integrated_polar = integrated_polar * rrr
    integrated_templates = _get_integrated_polar_templates(
        integrated_polar.shape[0],
        r,
        intensities,
        normalize_templates,
    )
    if normalize_image:
        integrated_polar = integrated_polar / np.linalg.norm(integrated_polar)
    correlations = _match_library_to_polar_fast(
        integrated_polar,
        integrated_templates,
    )
    return correlations


def correlate_library_to_pattern_fast(
    image,
    simulations,
    delta_r=1,
    delta_theta=1,
    max_r=None,
    intensity_transform_function=None,
    find_direct_beam=False,
    direct_beam_position=None,
    normalize_image=True,
    normalize_templates=True,
):
    """
    Get the correlation between azimuthally integrated templates and patterns

    Parameters
    ----------
    image : 2D numpy.ndarray
        The image of the diffraction pattern
    simulations : list of diffsims.sims.diffraction_simulation.DiffractionSimulation
        The diffraction pattern simulation
    delta_r : float, optional
        The sampling interval of the radial coordinate in pixels
    delta_theta : float, optional
        The sampling interval of the azimuthal coordinate in degrees
    max_r : float, optional
        Maximum radius to consider in pixel units. By default it is the
        distance from the center of the image to a corner
    intensity_transform_function : Callable, optional
        Function to apply to both image and template intensities on an
        element by element basis prior to comparison
    find_direct_beam : bool, optional
        Whether to optimize the direct beam, otherwise the center of the image
        is chosen
    direct_beam_position : 2-tuple of floats, optional
        (x, y) coordinate of the direct beam in pixel units. Overrides other
        settings for finding the direct beam
    normalize_image : bool, optional
        normalize the image to calculate the correlation coefficient
    normalize_templates : bool, optional
        normalize the templates to calculate the correlation coefficient

    Returns
    -------
    correlations : 1D numpy.ndarray
        correlation between azimuthaly integrated template and each azimuthally integrated template

    Notes
    -----
    Mirrored templates have identical azimuthally integrated representations,
    so this only has to be done on the positive euler angle templates (0, Phi, phi2)
    """
    polar_image, r, theta, intensities = _prepare_image_and_templates(
        image,
        simulations,
        delta_r,
        delta_theta,
        max_r,
        intensity_transform_function,
        find_direct_beam,
        direct_beam_position,
        False,  # it is not necessary to normalize these here
        False,
    )
    return _get_fast_correlation_index(
        polar_image, r, intensities, normalize_image, normalize_templates
    )


def _get_max_n(N, n_keep, frac_keep):
    """
    Determine the number of templates to allow through
    """
    max_keep = N
    if frac_keep is not None:
        max_keep = max(round(frac_keep * N), 1)
    # n_keep overrides fraction
    if n_keep is not None:
        max_keep = max(n_keep, 1)
    return int(min(max_keep, N))


def _prefilter_templates(
    polar_image,
    r,
    theta,
    intensities,
    integrated_templates,
    frac_keep,
    n_keep,
):
    """
    Pre-filter the templates to reduce the number of templates to do a full match on

    Parameters
    ----------
    polar_image: 2D numpy.ndarray
        The image converted to polar coordinates in the form (theta, r)
    r:
    theta
    intensities
    integrated_templates
    frac_keep
    n_keep

    Returns
    -------

    """
    dispatcher = get_array_module(polar_image)
    max_keep = _get_max_n(r.shape[0], n_keep, frac_keep)
    template_indexes = dispatcher.arange(r.shape[0], dtype=np.int32)
    if max_keep != r.shape[0]:
        polar_sum = polar_image.sum(axis=0)
        rrr = dispatcher.arange(polar_sum.shape[0]) / polar_sum.shape[0]
        polar_sum = polar_sum * rrr
        correlations_fast = _match_library_to_polar_fast(
            polar_sum,
            integrated_templates,
        )
        sorted_cor_indx = dispatcher.argsort(-correlations_fast)[:max_keep]
        r = r[sorted_cor_indx]
        theta = theta[sorted_cor_indx]
        intensities = intensities[sorted_cor_indx]
        template_indexes = template_indexes[sorted_cor_indx]
    return template_indexes, r, theta, intensities


def _get_full_correlations(
    polar_image,
    r,
    theta,
    intensities,
):
    # get a full match on the filtered data - we must branch for CPU/GPU
    if is_cupy_array(polar_image):
        f = _match_polar_to_polar_library_gpu
    else:
        f = _match_polar_to_polar_library_cpu
    return f(polar_image, r, theta, intensities)


def correlate_library_to_pattern(
    image,
    simulations,
    frac_keep=1.0,
    n_keep=None,
    delta_r=1.0,
    delta_theta=1.0,
    max_r=None,
    intensity_transform_function=None,
    find_direct_beam=False,
    direct_beam_position=None,
    normalize_image=True,
    normalize_templates=True,
):
    """
    Get the best angle and associated correlation values, as well as the correlation with the inverted templates

    Parameters
    ----------
    image : 2D numpy.ndarray
        The image of the diffraction pattern
    simulations : list of diffsims.sims.diffraction_simulation.DiffractionSimulation
        The diffraction pattern simulation
    frac_keep : float
        Fraction (between 0-1) of templates to do a full matching on. By default
        all patterns are fully matched.
    n_keep : int
        Number of templates to do a full matching on. When set frac_keep will be
        ignored
    delta_r : float, optional
        The sampling interval of the radial coordinate in pixels
    delta_theta : float, optional
        The sampling interval of the azimuthal coordinate in degrees
    max_r : float, optional
        Maximum radius to consider in pixel units. By default it is the
        distance from the center of the image to a corner
    intensity_transform_function : Callable, optional
        Function to apply to both image and template intensities on an
        element by element basis prior to comparison
    find_direct_beam : bool, optional
        Whether to optimize the direct beam, otherwise the center of the image
        is chosen
    direct_beam_position : 2-tuple of floats, optional
        (x, y) coordinate of the direct beam in pixel units. Overrides other
        settings for finding the direct beam
    normalize_image : bool, optional
        normalize the image to calculate the correlation coefficient
    normalize_templates : bool, optional
        normalize the templates to calculate the correlation coefficient

    Returns
    -------
    indexes : 1D numpy.ndarray
        indexes of templates on which a full calculation has been performed
    angles : 1D numpy.ndarray
        best fit in-plane angle for the top "keep" templates
    correlations : 1D numpy.ndarray
        best correlation for the top "keep" templates
    angles_mirrored : 1D numpy.ndarray
        best fit in-plane angle for the top mirrored "keep" templates
    correlations_mirrored : 1D numpy.ndarray
        best correlation for the top mirrored "keep" templates

    Notes
    -----
    Mirrored refers to the templates corresponding to the inverted orientations
    (0, -Phi, -phi/2)
    """
    polar_image, r, theta, intensities = _prepare_image_and_templates(
        image,
        simulations,
        delta_r,
        delta_theta,
        max_r,
        intensity_transform_function,
        find_direct_beam,
        direct_beam_position,
        normalize_image,
        normalize_templates,
    )
    integrated_templates = _get_integrated_polar_templates(
        polar_image.shape[1],
        r,
        intensities,
        normalize_templates,
    )
    indexes, r, theta, intensities = _prefilter_templates(
        polar_image, r, theta, intensities, integrated_templates, frac_keep, n_keep
    )
    angles, cor, angles_m, cor_m = _get_full_correlations(
        polar_image,
        r,
        theta,
        intensities,
    )
    return (
        indexes,
        (angles * delta_theta).astype(polar_image.dtype),
        cor,
        (angles_m * delta_theta).astype(polar_image.dtype),
        cor_m,
    )


def get_n_best_matches(
    image,
    simulations,
    n_best=1,
    frac_keep=1.0,
    n_keep=None,
    delta_r=1.0,
    delta_theta=1.0,
    max_r=None,
    intensity_transform_function=None,
    find_direct_beam=False,
    direct_beam_position=None,
    normalize_image=True,
    normalize_templates=True,
):
    """
    Get the n templates best matching an image in descending order

    Parameters
    ----------
    image : 2D numpy or cupy ndarray
        The image of the diffraction pattern
    simulations : list of diffsims.sims.diffraction_simulation.DiffractionSimulation
        The diffraction pattern simulation
    n_best : int, optional
        Number of best solutions to return, in order of descending match
    n_keep : int, optional
        Number of templates to do a full matching on
    frac_keep : float, optional
        Fraction (between 0-1) of templates to do a full matching on. When set
        n_keep will be ignored
    delta_r : float, optional
        The sampling interval of the radial coordinate in pixels
    delta_theta : float, optional
        The sampling interval of the azimuthal coordinate in degrees
    max_r : float, optional
        Maximum radius to consider in pixel units. By default it is the
        distance from the center of the image to a corner
    intensity_transform_function : Callable, optional
        Function to apply to both image and template intensities on an
        element by element basis prior to comparison
    find_direct_beam : bool, optional
        Whether to optimize the direct beam, otherwise the center of the image
        is chosen
    direct_beam_position : 2-tuple of floats, optional
        (x, y) coordinate of the direct beam in pixel units. Overrides other
        settings for finding the direct beam
    normalize_image : bool, optional
        normalize the image to calculate the correlation coefficient
    normalize_templates : bool, optional
        normalize the templates to calculate the correlation coefficient

    Returns
    -------
    indexes : 1D numpy or cupy ndarray
        indexes of best fit templates
    angles : 1D numpy or cupy ndarray
        corresponding best fit in-plane angles
    correlations : 1D numpy or cupy ndarray
        corresponding correlation values
    signs : 1D numpy or cupy ndarray
        1 if the positive template (0, Phi, phi2) is best matched, -1 if
        the negative template (0, -Phi, -phi2) is best matched
    """
    polar_image, r, theta, intensities = _prepare_image_and_templates(
        image,
        simulations,
        delta_r,
        delta_theta,
        max_r,
        intensity_transform_function,
        find_direct_beam,
        direct_beam_position,
        normalize_image,
        normalize_templates,
    )
    integrated_templates = _get_integrated_polar_templates(
        polar_image.shape[1], r, intensities, normalize_templates
    )
    answer = _mixed_matching_lib_to_polar(
        polar_image,
        integrated_templates,
        r,
        theta,
        intensities,
        n_keep,
        frac_keep,
        n_best,
    )
    indices = answer[:, 0].astype(np.int32)
    cors = answer[:, 1]
    angles = (answer[:, 2] * delta_theta).astype(polar_image.dtype)
    sign = answer[:, 3]
    return indices, angles, cors, sign


def index_dataset_with_template_rotation(
    signal,
    library,
    phases=None,
    n_best=1,
    frac_keep=1.0,
    n_keep=None,
    delta_r=1.0,
    delta_theta=1.0,
    max_r=None,
    intensity_transform_function=None,
    normalize_images=False,
    normalize_templates=True,
    chunks="auto",
    parallel_workers="auto",
    target="cpu",
    scheduler="threads",
    precision=np.float64,
):
    """
    Index a dataset with template_matching while simultaneously optimizing in-plane rotation angle of the templates

    Parameters
    ----------
    signal : hyperspy.signals.Signal2D
        The 4D-STEM dataset.
    library : diffsims.libraries.diffraction_library.DiffractionLibrary
        The library of simulated diffraction patterns.
    phases : list, optional
        Names of phases in the library to do an indexation for. By default this is
        all phases in the library.
    n_best : int, optional
        Number of best solutions to return, in order of descending match.
    frac_keep : float, optional
        Fraction (between 0-1) of templates to do a full matching on. By default
        all templates will be fully matched. See notes for details.
    n_keep : int, optional
        Number of templates to do a full matching on. Overrides frac_keep.
    delta_r : float, optional
        The sampling interval of the radial coordinate in pixels.
    delta_theta : float, optional
        The sampling interval of the azimuthal coordinate in degrees. This will
        determine the maximum accuracy of the in-plane rotation angle.
    max_r : float, optional
        Maximum radius to consider in pixel units. By default it is the
        distance from the center of the patterns to a corner of the image.
    intensity_transform_function : Callable, optional
        Function to apply to both image and template intensities on an
        element by element basis prior to comparison. Note that the function
        is performed on the CPU.
    normalize_images : bool, optional
        Normalize the images in the correlation coefficient calculation
    normalize_templates : bool, optional
        Normalize the templates in the correlation coefficient calculation
    chunks : string or 4-tuple, optional
        Internally the work is done on dask arrays and this parameter determines
        the chunking of the original dataset. If set to None then no
        re-chunking will happen if the dataset was loaded lazily. If set to
        "auto" then dask attempts to find the optimal chunk size.
    parallel_workers: int, optional
        The number of workers to use in parallel. If set to "auto", the number
        of physical cores will be used when using the CPU. For GPU calculations
        the workers is determined based on the VRAM capacity, but it is probably
        better to choose a lower number.
    target: string, optional
        Use "cpu" or "gpu". If "gpu" is selected, the majority of the calculation
        intensive work will be performed on the CUDA enabled GPU. Fails if no
        such hardware is available.
    scheduler: string
        The scheduler used by dask to compute the result. "processes" is not
        recommended.
    precision: np.float32 or np.float64
        The level of precision to work with on internal calculations

    Returns
    -------
    result : dict
        Results dictionary containing keys: phase_index, template_index,
        orientation, correlation, and mirrored_template. phase_index is the
        phase map, with each unique integer representing a phase. template_index
        are the best matching templates for the respective phase. orientation is
        the best matching orientations expressed in Bunge convention Euler angles.
        Correlation is the matching correlation indices. mirrored template represents
        whether the original template best fits (False) or the mirror image (True).
        Each is a numpy array of shape (scan_y, scan_x, n_best) except orientation
        is of shape (scan_y, scan_x, n_best, 3).
    phase_key_dict: dictionary
        A small dictionary to translate the integers in the phase_index array
        to phase names in the original template library.


    Notes
    -----
    It is possible to run the indexation using a subset of the templates. This
    two-stage procedure is controlled through `n_keep` or `frac_keep`. If one
    of these parameters is set, the azimuthally integrated patterns are
    compared to azimuthally integrated templates in a first stage, which is
    very fast. The top matching patterns are passed to a second stage of
    full matching, whereby the in-plane angle is determined. Setting these
    parameters can usually achieve the same answer faster, but it is also
    possible an incorrect match is found.
    """
    if target == "gpu":
        # an error will be raised if cupy is not available
        if not CUPY_INSTALLED:
            raise ValueError(
                "There must be a CUDA enabled GPU and cupy must be installed."
            )
        dispatcher = cp
    else:
        dispatcher = np

    # get the dataset as a dask array
    data = _get_dask_array(signal)
    # check if we have a 4D dataset, and if not, make it
    navdim = signal.axes_manager.navigation_dimension
    if navdim == 0:
        # we assume we have a single image
        data = data[np.newaxis, np.newaxis, ...]
    elif navdim == 1:
        # we assume we have a line of images with the first dimension the line
        data = data[np.newaxis, ...]
    elif navdim == 2:
        # correct dimensions
        pass
    else:
        raise ValueError(f"Dataset has {navdim} navigation dimensions, max " "is 2")
    # change the chunking of the dataset if necessary
    if chunks is None:
        pass
    elif chunks == "auto":
        data = data.rechunk({0: "auto", 1: "auto", 2: None, 3: None})
    else:
        data = data.rechunk(chunks)

    # calculate the dimensions of the polar transform
    output_shape = get_polar_pattern_shape(
        data.shape[-2:], delta_r, delta_theta, max_r=max_r
    )
    theta_dim, r_dim = output_shape
    max_radius = r_dim * delta_r
    center = (data.shape[-2] / 2, data.shape[-1] / 2)
    # apply the intensity transform function to the images
    if intensity_transform_function is not None:
        data = data.map_blocks(intensity_transform_function)
    # combine the phases into a single library of templates to perform
    # indexation in a single step
    # TODO the library concatenation below is not memory efficient, some
    # lazy iterator would be better
    if phases is None:
        phases = library.keys()
    r_list = []  # list of all r arrays of each phase
    theta_list = []  # list of all theta arrays of each phase
    intensity_list = []  # list of all intensity arrays of each phase
    phase_index = []  # array to indicate the phase index of each template
    original_index = []  # index of the template in that phase library
    phase_key_dict = {}  # mapping the phase index to a phase name
    maximum_spot_number = 0  # to know the number of columns in template arrays
    total_template_number = 0  # to know number of rows in template arrays
    for index, phase_key in enumerate(phases):
        phase_library = library[phase_key]
        positions, intensities = _simulations_to_arrays(
            phase_library["simulations"], max_radius
        )
        r, theta = _cartesian_positions_to_polar(
            positions[:, 0], positions[:, 1], delta_r=delta_r, delta_theta=delta_theta
        )
        # ensure we don't have any out of bounds which could occur from rounding
        condition = r >= r_dim
        r[condition] = r_dim - 1
        theta[condition] = 0
        intensities[condition] = 0.0
        r_list.append(r)
        theta_list.append(theta)
        intensity_list.append(intensities.astype(precision))
        # for reconstructing the appropriate phase index and template number later
        phase_index.append(np.full((r.shape[0]), index, dtype=np.int8))
        original_index.append(np.arange(r.shape[0]))
        phase_key_dict[index] = phase_key
        # update number of spots
        if r.shape[1] > maximum_spot_number:
            maximum_spot_number = r.shape[1]
        # update number of templates
        total_template_number += r.shape[0]
    # allocate memory and concatenate arrays in list
    r = np.zeros((total_template_number, maximum_spot_number), dtype=np.int32)
    theta = np.zeros((total_template_number, maximum_spot_number), dtype=np.int32)
    intensities = np.zeros(
        (total_template_number, maximum_spot_number), dtype=precision
    )
    position = 0
    for rr, tt, ii in zip(r_list, theta_list, intensity_list):
        r[position : position + rr.shape[0], : rr.shape[1]] = rr
        theta[position : position + rr.shape[0], : rr.shape[1]] = tt
        intensities[position : position + rr.shape[0], : rr.shape[1]] = ii
        position += rr.shape[0]
    # phase_index and original index are 1D we can just concatenate
    phase_index = np.concatenate(phase_index)
    original_index = np.concatenate(original_index)

    if intensity_transform_function is not None:
        intensities = intensity_transform_function(intensities)
    # integrated intensity library for fast comparison
    integrated_templates = _get_integrated_polar_templates(
        r_dim, r, intensities, normalize_templates
    )
    # normalize the templates if required
    if normalize_templates:
        integrated_templates = _norm_rows(integrated_templates)
        intensities = _norm_rows(intensities)

    # put a limit on n_best
    max_n = _get_max_n(r.shape[0], n_keep, frac_keep)
    if n_best > max_n:
        n_best = max_n

    # copy relevant data to GPU memory if necessary
    if target == "gpu":
        integrated_templates = cp.asarray(integrated_templates)
        r = cp.asarray(r)
        theta = cp.asarray(theta)
        intensities = cp.asarray(intensities)
        indexation_function = _index_chunk_gpu
    else:
        indexation_function = _index_chunk

    indexation = data.map_blocks(
        indexation_function,
        center,
        max_radius,
        output_shape,
        precision,
        integrated_templates,
        r,
        theta,
        intensities,
        n_keep,
        frac_keep,
        n_best,
        normalize_images,
        dtype=precision,
        drop_axis=signal.axes_manager.signal_indices_in_array,
        chunks=(data.chunks[0], data.chunks[1], n_best, 4),
        new_axis=(2, 3),
    )

    # calculate number of workers
    if parallel_workers == "auto":
        # upper boundary if using CPU
        # only use number of physical cores, not logical
        parallel_workers = psutil.cpu_count(logical=False)
        if target == "gpu":
            # let's base it on the size of the free GPU memory and array blocksize
            # this is probably too many workers!
            max_gpu_mem = 0.6 * cp.cuda.Device().mem_info[0]
            blocksize = data.nbytes / data.npartitions
            max_workers = max_gpu_mem / blocksize
            if max_workers < parallel_workers:
                parallel_workers = max_workers

    with ProgressBar():
        res_index = indexation.compute(
            scheduler=scheduler, num_workers=parallel_workers, optimize_graph=True
        )

    # cupy retains memory on the GPU even after garbage collection
    # see https://docs.cupy.dev/en/stable/user_guide/memory.html
    # We manually clear the memory here to not give the user the impression that
    # their graphics card is going crazy
    if target == "gpu":
        cp.get_default_memory_pool().free_all_blocks()

    # wrangle data to results dictionary
    # we can't use a dask array to index into a dask array! Using a dask
    # array as indices for a numpy array computes the dask array automatically.
    result = {}
    result["phase_index"] = phase_index[res_index[:, :, :, 0].astype(np.int32)]
    result["template_index"] = original_index[res_index[:, :, :, 0].astype(np.int32)]
    # initialize orientations because we merge partially filled arrays by addition
    orients = 0
    for index, phase in phase_key_dict.items():
        oris = library[phase]["orientations"]
        phasemask = result["phase_index"] == index
        indices = (
            result["template_index"] * phasemask
        )  # everywhere false will get index 0 to ensure no out of bounds
        orimap = (
            oris[indices] * phasemask[..., np.newaxis]
        )  # everywhere false should not get any orientation
        # correct orientation maps with rescales and flips
        orimap[:, :, :, 1] = (
            orimap[:, :, :, 1] * res_index[:, :, :, 3]
        )  # multiply by the sign
        orimap[:, :, :, 2] = (
            orimap[:, :, :, 2] * res_index[:, :, :, 3]
        )  # multiply by the sign
        orimap[:, :, :, 0] = res_index[:, :, :, 2] * delta_theta * phasemask
        # add to orients, there should be no overlap on pixels or N
        orients = orients + orimap
    result["orientation"] = orients
    result["correlation"] = res_index[:, :, :, 1]
    result["mirrored_template"] = res_index[:, :, :, 3] == -1
    return result, phase_key_dict


def results_dict_to_crystal_map(
    results, phase_key_dict, diffraction_library=None, index=None
):
    """Export an indexation result from
    :func:`index_dataset_with_template_rotation` to a crystal map with
    `n_best` rotations, score, mirrors and one phase ID per data point.

    Parameters
    ----------
    results : dict
        Results dictionary obtained from
        :func:`index_dataset_with_template_rotation`.
    phase_key_dict : dict
        Dictionary mapping phase ID to phase name, obtained from
        :func:`index_dataset_with_template_rotation`.
    diffraction_library : diffsims.libraries.DiffractionLibrary, optional
        Used for the structures to be passed to
        :class:`orix.crystal_map.PhaseList`.
    index : int, optional
        Which of the `n_best` solutions (0-indexed) obtained from
        :func:`index_dataset_with_template_rotation` to get a crystal
        map from. Highest allowed value is `n_best` - 1. If not given,
        all solutions are used if `n_best` was more than one and
        `results["phase_index"]` only has one phase, otherwise, only the
        best solution is used.

    Returns
    -------
    orix.crystal_map.CrystalMap
        Crystal map containing `results`. The map has multiple rotations
        and properties ("correlation", "mirrored_template",
        "template_index") per point only if `n_best` passed to
        :func:`index_dataset_with_template_rotation` was more than one
        and "phase_index" only has one phase.

    Notes
    -----
    Phase's :attr:`~orix.crystal_map.Phase.point_group` must be set
    manually to the correct :class:`~orix.quaternion.Symmetry` after
    the crystal map is returned.

    Examples
    --------
    After getting `results` and `phase_key_dict` from template matching

    >>> xmap = results_dict_to_crystal_map(results, phase_key_dict)  # doctest: +SKIP
    >>> xmap.plot()  # Phase map  # doctest: +SKIP

    Getting the second best match if `n_best` passed to the template
    matching function is greater than one

    >>> xmap2 = results_dict_to_crystal_map(
    ...     results, phase_key_dict, index=1
    ... )  # doctest: +SKIP
    """
    ny, nx, n_best = results["phase_index"].shape
    if index is not None and index > n_best - 1:
        raise ValueError(f"`index` cannot be higher than {n_best - 1} (`n_best` - 1)")

    n_points = nx * ny

    # Phase ID (only one per point is allowed, always)
    if index is None:
        phase_id = results["phase_index"][:, :, 0].ravel()
    else:
        phase_id = results["phase_index"][:, :, index].ravel()
    n_phases = np.unique(phase_id).size

    x, y = np.indices((nx, ny)).reshape((2, n_points))

    if diffraction_library is not None:
        structures = diffraction_library.structures
    else:
        structures = None
    phase_list = PhaseList(names=phase_key_dict, structures=structures)

    euler = np.deg2rad(results["orientation"].reshape((n_points, n_best, 3)))
    if index is None and n_phases > 1:
        euler = euler[:, 0]  # Best match only
    elif index is not None:
        euler = euler[:, index]  # Desired match only
    euler = euler.squeeze()  # Remove singleton dimensions
    rotations = Rotation.from_euler(euler)

    props = {}
    for key in ("correlation", "mirrored_template", "template_index"):
        try:
            prop = results[key]
        except KeyError:
            warnings.warn(f"Property '{key}' was expected but not found in `results`")
            continue

        if index is None and n_phases > 1:
            prop = prop[:, :, 0].ravel()  # Best match only
        elif index is not None:
            prop = prop[:, :, index].ravel()  # Desired match only
        else:
            prop = prop.reshape((n_points, n_best))  # All
        props[key] = prop.squeeze()  # Remove singleton dimensions

    return CrystalMap(
        rotations=rotations,
        phase_id=phase_id,
        x=x,
        y=y,
        phase_list=phase_list,
        prop=props,
    )


def structure2dict(structure):
    atoms = structure.tolist()
    elements = [a.element for a in atoms]
    positions = structure.xyz
    lattice = [
        structure.lattice.a,
        structure.lattice.b,
        structure.lattice.c,
        structure.lattice.alpha,
        structure.lattice.beta,
        structure.lattice.gamma,
    ]
    return dict(
        elements=elements, positions=positions, lattice=lattice, title=structure.title
    )


def dict2structure(elements, positions, lattice, title):
    atoms = [Atom(atype=e, xyz=p) for e, p in zip(elements, positions)]
    lat = Lattice(*lattice)
    return Structure(atoms, lattice=lat, title=title)


def phase2dict(phase):
    structure_dict = structure2dict(phase.structure)
    return dict(
        name=phase.name,
        space_group=phase.space_group.number,
        point_group=phase.point_group.name,
        color=phase.color,
        structure=structure_dict,
    )


def dict2phase(name, space_group, point_group, color, structure):
    struct = dict2structure(**structure)
    return Phase(
        name=name,
        space_group=space_group,
        point_group=point_group,
        color=color,
        structure=struct,
    )
