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

from pyxem.utils.dask_tools import _get_dask_array
import os
from dask.diagnostics import ProgressBar
from numba import njit, objmode, prange, guvectorize
from skimage.filters import gaussian
from skimage.transform import warp_polar

from pyxem.utils.polar_transform_utils import (
    get_polar_pattern_shape,
    image_to_polar,
    get_template_polar_coordinates,
    chunk_to_polar,
)



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
    **kwargs,
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
            condition = x ** 2 + y ** 2 < max_radius ** 2
            x = x[condition]
            y = y[condition]
            intensity = intensity[condition]
        positions[i, 0, : x.shape[0]] = x
        positions[i, 1, : y.shape[0]] = y
        intensities[i, : intensity.shape[0]] = intensity
    return positions, intensities


def _cartesian_positions_to_polar(x, y, delta_r=1, delta_theta=1):
    """
    Convert 2D cartesian image coordinates to polar image coordinates

    Parameters
    ----------
    x : 1D numpy.ndarray
        x coordinates
    y : 1D numpy.ndarray
        y coordinates
    delta_r : float
        sampling interval in the r direction
    delta_theta : float
        sampling interval in the theta direction

    Returns
    -------
    r : 1D numpy.ndarray
        r coordinate or x coordinate in the polar image
    theta : 1D numpy.ndarray
        theta coordinate or y coordinate in the polar image
    """
    imag = (x) + 1j * (y)
    r = (np.abs(imag) / delta_r).astype(np.int32)
    angle = np.rad2deg(np.angle(imag))
    theta = (np.mod(angle, 360) / delta_theta).astype(np.int32)
    return r, theta


@njit
def _extract_pixel_intensities(image, x, y):
    experimental = np.zeros(x.shape, dtype=np.float64)
    for j in prange(x.shape[-1]):
        experimental[j] = image[y[j], x[j]]
    return experimental


@njit
def _simple_correlation(
    image_intensities, template_intensities, image_norm, template_norm
):
    """Simple correlation coefficient - sum of products divided by the norms"""
    return np.sum(np.multiply(image_intensities, template_intensities)) / (
        image_norm * template_norm
    )


@njit
def _match_polar_to_polar_template(
    polar_image, r_template, theta_template, intensities, image_norm, template_norm
):
    """
    Correlate a single polar template to a single polar image

    The template spots are shifted along the azimuthal axis by 1 pixel increments.
    A simple correlation index is calculated at each position.

    Parameters
    ----------
    polar_image : 2D numpy.ndarray
        the polar image
    r_template : 1D numpy.ndarray
        r coordinates of diffraction spots in template
    theta_template : 1D numpy.ndarray
        theta coordinates of diffraction spots in template
    intensities : 1D numpy.ndarray
        intensities of diffraction spots in template
    image_norm : float
        norm of the polar image
    template_norm : float
        norm of the template

    Returns
    -------
    correlation : 1D numpy.ndarray
        correlation index at each in-plane angle position
    """
    n = intensities.shape[0]  # number of reflections (maximum)
    match_matrix = np.empty((polar_image.shape[0], n), dtype=np.float64)
    theta_template = theta_template.astype(np.int64)
    for i in range(n):
        column = polar_image[:, r_template[i]].copy()  # extract column at r for each spot
        match_matrix[:, i] = np.roll(column, -theta_template[i])  # shift column over by theta and put in array
    correlation = np.dot(match_matrix, intensities.astype(np.float64)) / (image_norm * template_norm)  # all in-plane angles are encoded in the matrix
    return correlation


@njit
def _norm_array(ar):
    return ar / np.sqrt(np.sum(ar ** 2))


@njit(parallel=True)
def _correlate_polar_to_library_cpu(
    polar_image,
    r_templates,
    theta_templates,
    intensities_templates,
    polar_image_norm,
    template_norms,
    ):
    """
    Correlates a polar pattern to all polar templates at all in_plane angles

    Parameters
    ----------
    polar_image : (T, R) 2D numpy.ndarray
        The image converted to polar coordinates
    r_templates : (N, D) 2D numpy.ndarray
        r-coordinates of diffraction spots in templates.
    theta_templates : (N, D) 2D numpy ndarray
        theta-coordinates of diffraction spots in templates.
    intensities_templates : (N, D) 2D numpy.ndarray
        intensities of the spots in each template
    polar_image_norm : float
        norm of the polar image
    template_norms : (N), 1D numpy.ndarray
        norms of each template

    Returns
    -------
    correlations : (N, T) 2D numpy.ndarray
        the correlation index for each template at all in-plane angles with the image
    """
    correlation = np.empty((r_templates.shape[0], polar_image.shape[0]), dtype=np.float32)
    for template in prange(r_templates.shape[0]):
        for shift in range(polar_image.shape[0]):
            tmp = 0
            for spot in range(r_templates.shape[1]):
                tmp += (polar_image[(theta_templates[template, spot]+shift)%polar_image.shape[0],
                                    r_templates[template, spot]] * intensities_templates[template, spot])
            correlation[template, shift] = tmp / (polar_image_norm * template_norms[template])
    return correlation


@njit
def np_apply_along_axis(func1d, axis, arr):
    """
    Boilerplate from https://github.com/numba/numba/issues/1269 to gain
    access to the "axis" kwarg on numpy functions. Only works on 2D arrays
    """
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@njit(nogil=True)
def _match_polar_to_polar_library(
    polar_image,
    r_templates,
    theta_templates,
    intensities_templates,
    polar_image_norm,
    template_norms,
):
    """
    Correlates a polar pattern to all polar templates

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
    polar_image_norm : float
        norm of the polar image
    template_norms : 1D numpy.ndarray
        norms of each template

    Returns
    -------
    correlations : 1D numpy.ndarray
        the maximum correlation index for each template with the image
    angles : 1D numpy.ndarray
        the best fit in-plane angle for each template in degrees
    correlations_mirrored : 1D numpy.ndarray
        the maximum correlation index for the mirrored templates
    angles_mirrored : 1D numpy.ndarray
        the best fit in-plane angle for each mirrored template

    Notes
    -----
    The dimensions of r_templates and theta_templates should be (N, R) where
    N is the number of templates and R the number of spots in the template
    with the maximum number of spots
    """
    correlations = _correlate_polar_to_library_cpu(
            polar_image, r_templates, theta_templates, intensities_templates,
            polar_image_norm, template_norms)
    correlations_mirror = _correlate_polar_to_library_cpu(
            polar_image, r_templates, 359-theta_templates, intensities_templates,
            polar_image_norm, template_norms)
    best_angles = np_apply_along_axis(np.argmax, 1, correlations)
    best_correlations = np_apply_along_axis(np.amax, 1, correlations)
    best_angles_mirrored = np_apply_along_axis(np.argmax, 1, correlations_mirror)
    best_correlations_mirrored = np_apply_along_axis(np.amax, 1, correlations_mirror)
    return (best_correlations,
            best_angles,
            best_correlations_mirrored,
            best_angles_mirrored)


@njit
def _get_correlation_at_angle(
    polar_image,
    r_templates,
    theta_templates,
    intensities,
    angle_shifts,
    image_norm,
    template_norms,
):
    """
    Get the correlation between a polar image and the polar templates at specific in-plane angles

    Parameters
    ----------
    polar_image : 2D numpy.ndarray
        The image converted to polar coordinates
    r_templates : 2D numpy.ndarray
        r-coordinates of diffraction spots in templates.
    theta_templates : 2D numpy ndarray
        theta-coordinates of diffraction spots in templates.
    intensities : 2D numpy.ndarray
        intensities of the spots in each template
    angle_shifts : 1D numpy.ndarray
        first euler angle for each template
    polar_image_norm : float
        norm of the polar image
    template_norms : 1D numpy.ndarray
        norms of each template

    Returns
    -------
    correlations : 1D numpy.ndarray
        the maximum correlation index for each template with the image at the
        specified angle

    Notes
    -----
    The dimensions of r_templates, theta_templates, and intensities should be (N, R) where
    N is the number of templates and R the number of spots in the template
    with the maximum number of spots. All coordinates and angle_shifts should
    be in the same units as the axes of the polar image.
    """
    correlations = np.zeros(r_templates.shape[0], dtype=np.float64)
    for i in range(r_templates.shape[0]):
        angle = angle_shifts[i]
        r = r_templates[i]
        theta = np.mod(theta_templates[i] + angle, polar_image.shape[0])
        r = r.astype(np.int64)
        theta = theta.astype(np.int64)
        intensity = intensities[i]
        template_norm = template_norms[i]
        image_intensities = _extract_pixel_intensities(polar_image, r, theta)
        correlations[i] = _simple_correlation(
            image_intensities, intensities, image_norm, template_norm
        )
    return correlations


@njit
def _get_row_norms(array):
    """Get the norm of all rows in a 2D array"""
    norms = np.sqrt(np.sum(array ** 2, axis=1))
    return norms


@njit
def _norm_rows(array):
    """Normalize all the rows in a 2D array"""
    norms = _get_row_norms(array)
    array = (array.T / norms).T
    return array


@njit
def _get_integrated_polar_templates(r_max, r_templates, intensities_templates):
    """
    Get an azimuthally integrated representation of the templates.

    Parameters
    ----------
    r_max : float
        maximum radial distance to consider in pixel units. typically the
        radial width of the polar images.
    r_templates : 2D numpy array
        r-coordinate of all spots in the templates. Of shape (N, R) where
        N is the number of templates and R is the number of spots in the
        template with the maximum number of spots
    intensities_templates : 2D numpy array
        intensities in all spots of the templates. Of shape (N, R)

    Returns
    -------
    integrated_templates : 2D numpy array
        Templates integrated over the azimuthal axis of shape (N, r_max)
    """
    integrated_templates = np.zeros((r_templates.shape[0], r_max), dtype=np.float32)
    for i in range(intensities_templates.shape[0]):
        intensity = intensities_templates[i]
        r_template = r_templates[i]
        for j in range(intensity.shape[0]):
            inten = intensity[j]
            r_p = r_template[j]
            integrated_templates[i, r_p] = integrated_templates[i, r_p] + inten
    return integrated_templates


@njit
def _match_library_to_polar_fast(
    polar_sum, integrated_templates, polar_norm, template_norms
):
    """
    Compare a polar image to azimuthally integrated templates

    Parameters
    ----------
    polar_sum : 1D numpy array
        the image in polar coordinates integrated along the azimuthal axis
        (shape = (r_max,))
    integrated_templates : 2D numpy array
        azimuthally integrated templates of shape (N, r_max) with N
        the number of templates and r_max the width of the polar image
    polar_norm : float
        norm of the integrated polar template
    template_norm : 1D numpy array
        norms of all the integrated templates (shape = (N,))

    Returns
    -------
    correlations : 1D numpy array
        the correlation between the integrated image and the integrated
        templates. (shape = (N,))
    """
    coors = np.zeros(integrated_templates.shape[0], dtype=np.float32)
    for i in range(integrated_templates.shape[0]):
        intensity = integrated_templates[i]
        template_norm = template_norms[i]
        coors[i] = _simple_correlation(polar_sum, intensity, polar_norm, template_norm)
    return coors


def _prepare_image_and_templates(
    image,
    simulations,
    delta_r,
    delta_theta,
    max_r,
    intensity_transform_function,
    find_direct_beam,
    direct_beam_position,
):
    """
    Prepare a single cartesian coordinate image and a template library for comparison

    Parameters
    ----------
    image : 2D numpy array
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

    Returns
    -------
    polar_image : 2D numpy array
        The image in polar coordinates
    r : 2D numpy array
        The r coordinates in the polar image corresponding to template spots.
        shape = (N, R) with N the number of templates and R the number of spots
        contained by the template with the most spots
    theta : 2D numpy array
        The theta coordinates in the polar image corresponding to template spots.
        shape = (N, R) with N the number of templates and R the number of spots
        contained by the template with the most spots
    intensities :  2D numpy array
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
    max_radius = polar_image.shape[1] * delta_r
    positions, intensities = _simulations_to_arrays(simulations, max_radius=max_radius)
    if intensity_transform_function is not None:
        intensities = intensity_transform_function(intensities)
        polar_image = intensity_transform_function(polar_image)
    r, theta = _cartesian_positions_to_polar(
        positions[:, 0], positions[:, 1], delta_r, delta_theta
    )
    return (polar_image.astype(np.float32),
            r.astype(np.int32),
            theta.astype(np.int32),
            intensities.astype(np.float32))


@njit
def _mixed_matching_lib_to_polar(
    polar_image,
    polar_norm,
    polar_sum,
    polar_sum_norm,
    integrated_templates,
    integrated_template_norms,
    r_templates,
    theta_templates,
    intensities_templates,
    template_norms,
    fraction,
    n_best,
):
    """
    Match a polar image to a filtered subset of polar templates

    First does a fast matching basted on azimuthally integrated templates
    Then it takes the (1-fraction)*100% of patterns to do a full indexation on.
    Return the first n_best answers.

    Parameters
    ----------
    polar_image : 2D numpy array
        image in polar coordinates
    polar_norm : float
        norm of the polar image
    polar_sum : 1D numpy array
        azimuthally integrated polar image
    polar_sum_norm : float
        norm of the azimuthally integrated polar image
    integrated_templates : 2D numpy array, (N, r_max)
        azimuthally integrated templates
    integrated_template_norms : 1D numpy array, (N,)
        norms of azimuthally integrated templates
    r_templates : 2D numpy array, (N, R)
        r coordinates of diffraction spots in all N templates
    theta_templates : 2D numpy array, (N, R)
        theta coordinates of diffraction spots in all N templates
    intensities_templates : 2D numpy array, (N, R)
        intensities of diffraction spots in all N templates
    template_norms : 1D numpy array, (N,)
        Norms of templates
    fraction : float
        fraction of N templates to throw away
    n_best : int
        number of solutions to return in decending order of fit

    Return
    ------
    answer : 2D numpy array, (n_best, 4)
        in the colums are returned (template index, correlation, in-plane angle, factor)
        of the best fitting template, where factor is 1 if the direct template is
        matched and -1 if the mirror template is matched
    """
    coors = _match_library_to_polar_fast(
        polar_image, integrated_templates, polar_sum_norm, integrated_template_norms
    )
    template_indexes = np.arange(theta_templates.shape[0])
    lowest = np.percentile(coors, fraction * 100)
    condition = coors >= lowest
    r_templates_filter = r_templates[condition]
    theta_templates_filter = theta_templates[condition]
    intensities_templates_filter = intensities_templates[condition]
    template_indexes_filter = template_indexes[condition]
    template_norms_filter = template_norms[condition]
    full_cors, full_angles, full_cors_m, full_angles_m = _match_polar_to_polar_library(
        polar_image,
        r_templates_filter,
        theta_templates_filter,
        intensities_templates_filter,
        polar_norm,
        template_norms_filter,
    )
    # compare positive and negative templates and combine
    positive_is_best = full_cors > full_cors_m
    best_angles = positive_is_best*full_angles + np.invert(positive_is_best)*full_angles_m
    best_sign = positive_is_best*1 + np.invert(positive_is_best)*(-1)
    best_cors = positive_is_best*full_cors + np.invert(positive_is_best)*full_cors_m
    answer = np.empty((n_best, 4), dtype=np.float64)
    if n_best == 1:
        max_index_filter = np.argmax(best_cors)
        max_cor = best_cors[max_index_filter]
        max_angle = best_angles[max_index_filter]
        max_index = template_indexes_filter[max_index_filter]
        max_sign = best_sign[max_index_filter]
        answer[0] = np.array((max_index, max_cor, max_angle, max_sign))
    else:
        # at this time numba does not support np.argpartition which could speed up
        # and avoid full sort
        indices_sorted = np.argsort(-best_cors)
        n_best_indices = indices_sorted[:n_best]
        for i in range(n_best):
            answer[i, 0] = template_indexes_filter[n_best_indices[i]]
            answer[i, 1] = best_cors[n_best_indices[i]]
            answer[i, 2] = best_angles[n_best_indices[i]]
            answer[i, 3] = best_sign[n_best_indices[i]]
    return answer


@njit(nogil=True, parallel=True)
def _index_chunk(
    polar_images,
    integrated_templates,
    integrated_template_norms,
    r_templates,
    theta_templates,
    intensities_templates,
    template_norms,
    fraction,
    n_best,
    nim,
):
    """Function to map indexation over chunks"""
    indexation_result_chunk = np.empty(
        (polar_images.shape[0], polar_images.shape[1], n_best, 4), dtype=np.float32
    )
    for idx in prange(polar_images.shape[0]):
        for idy in prange(polar_images.shape[1]):
            pattern = polar_images[idx, idy]
            integrated_pattern = pattern.sum(axis=0)
            # calculate norms, if norm_images = True then it's the norm, else it's 1
            # this is to avoid if statement
            polar_norm = np.linalg.norm(pattern) * nim + (1.0 - nim)
            integrated_pattern_norm = np.linalg.norm(integrated_pattern) * nim + (
                1.0 - nim
            )
            indexresult = _mixed_matching_lib_to_polar(
                pattern,
                polar_norm,
                integrated_pattern,
                integrated_pattern_norm,
                integrated_templates,
                integrated_template_norms,
                r_templates,
                theta_templates,
                intensities_templates,
                template_norms,
                fraction,
                n_best,
            )
            indexation_result_chunk[idx, idy] = indexresult
    return indexation_result_chunk


@njit
def _renormalize_polar_block(polar_chunk):
    normed_polar = np.zeros_like(polar_chunk)
    for i in np.ndindex(polar_chunk.shape[:-2]):
        polar_image = polar_chunk[i]
        polar_image = polar_image - np.mean(polar_image)
        polar_image = _norm_array(polar_image)
        normed_polar[i] = polar_image
    return normed_polar


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
    r, theta, intensities = get_template_polar_coordinates(
        simulation,
        in_plane_angle=0.0,
        delta_r=delta_r,
        delta_theta=delta_theta,
        max_r=max_r,
    )
    r = np.rint(r).astype(np.int32)
    theta = np.rint(theta).astype(np.int32)
    condition = (r > 0) & (r < polar_image.shape[1])
    r = r[condition]
    theta = theta[condition]
    intensity = intensities[condition]
    if intensity_transform_function is not None:
        intensity = intensity_transform_function(intensity)
        polar_image = intensity_transform_function(polar_image)
    image_norm = 1.0 if not normalize_image else np.linalg.norm(polar_image)
    template_norm = 1.0 if not normalize_template else np.linalg.norm(intensity)
    correlation_array = _match_polar_to_polar_template(
        polar_image, r, theta, intensity, image_norm, template_norm
    )
    angle_array = np.arange(correlation_array.shape[0]) * delta_theta
    return angle_array, correlation_array


def correlate_library_to_pattern(
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
    Get the best angle and associated correlation values, as well as the correlation with the inverted templates

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
    angles : 1D numpy.ndarray
        best fit in-plane angle for each template of length N
    correlations : 1D numpy.ndarray
        best correlation for each template of length N
    angles_mirrored : 1D numpy.ndarray
        best fit in-plane angle for each mirrored template of length N
    correlations_mirrored : 1D numpy.ndarray
        best correlation for each mirrored template of length N

    Notes
    -----
    The mirrored templates correspond to inverted euler angle coordinates
    (0, -Phi, -phi2)
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
    )
    polar_image_norm = 1 if not normalize_image else np.linalg.norm(polar_image)
    template_norms = (
        np.ones(r.shape[0], dtype=np.float64)
        if not normalize_templates
        else _get_row_norms(intensities)
    )
    correlations, angles, correlations_mirrored, angles_mirrored = _match_polar_to_polar_library(
        polar_image, r, theta, intensities, polar_image_norm, template_norms
    )
    return angles, correlations, angles_mirrored, correlations_mirrored


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
    )
    integrated_polar = polar_image.sum(axis=0)
    integrated_templates = _get_integrated_polar_templates(
        integrated_polar.shape[0], r, intensities
    )
    polar_norm = 1 if not normalize_image else np.linalg.norm(integrated_polar)
    template_norms = (
        np.ones(r.shape[0], dtype=np.float64)
        if not normalize_templates
        else _get_row_norms(integrated_templates)
    )
    correlations = _match_library_to_polar_fast(
        integrated_polar, integrated_templates, polar_norm, template_norms
    )
    return correlations


def correlate_library_to_pattern_partial(
    image,
    simulations,
    n_keep=100,
    frac_keep=None,
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
    Get the best angle and associated correlation values, as well as the correlation with the inverted templates

    Parameters
    ----------
    image : 2D numpy.ndarray
        The image of the diffraction pattern
    simulations : list of diffsims.sims.diffraction_simulation.DiffractionSimulation
        The diffraction pattern simulation
    n_keep : int
        Number of templates to do a full matching on
    frac_keep : float
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
    )
    N = r.shape[0]
    fraction = max((N - abs(n_keep)) / N, 0.0)
    if frac_keep is not None:
        fraction = max(1.0 - abs(frac_keep), 0.0)
    # fast matching
    integrated_polar = polar_image.sum(axis=0)
    integrated_templates = _get_integrated_polar_templates(
        integrated_polar.shape[0], r, intensities
    )
    polar_norm = 1 if not normalize_image else np.linalg.norm(integrated_polar)
    template_norms = (
        np.ones(N, dtype=np.float64)
        if not normalize_templates
        else _get_row_norms(integrated_templates)
    )
    correlations_fast = _match_library_to_polar_fast(
        integrated_polar, integrated_templates, polar_norm, template_norms
    )
    # full matching
    template_indexes = np.arange(N)
    lowest = np.percentile(correlations_fast, fraction * 100)
    condition = correlations_fast >= lowest
    r_templates_filter = r[condition]
    theta_templates_filter = theta[condition]
    intensities_templates_filter = intensities[condition]
    template_indexes_filter = template_indexes[condition]
    polar_image_norm = 1 if not normalize_image else np.linalg.norm(polar_image)
    template_norms = (
        np.ones(intensities_templates_filter.shape[0], dtype=np.float64)
        if not normalize_templates
        else _get_row_norms(intensities_templates_filter)
    )
    full_cors, full_angles, full_cors_mirrored, full_angles_mirrored = _match_polar_to_polar_library(
        polar_image,
        r_templates_filter,
        theta_templates_filter,
        intensities_templates_filter,
        polar_image_norm,
        template_norms,
    )
    return template_indexes_filter, full_angles, full_cors, full_angles_mirrored, full_cors_mirrored


def get_n_best_matches(
    image,
    simulations,
    n_best=1,
    n_keep=100,
    frac_keep=None,
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
    Get the n templates best matching an image in descending order

    Parameters
    ----------
    image : 2D numpy.ndarray
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
    indexes : 1D numpy.ndarray
        indexes of best fit templates
    angles : 1D numpy.ndarray
        corresponding best fit in-plane angles
    correlations : 1D numpy.ndarray
        corresponding correlation values
    signs : 1D numpy.ndarray
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
    )
    N = r.shape[0]
    fraction = max((N - abs(n_keep)) / N, 0.0)
    if frac_keep is not None:
        fraction = max(1.0 - abs(frac_keep), 0.0)
    polar_norm = 1.0 if not normalize_image else np.linalg.norm(polar_image)
    integrated_polar = polar_image.sum(axis=0)
    polar_sum_norm = 1.0 if not normalize_image else np.linalg.norm(integrated_polar)
    integrated_templates = _get_integrated_polar_templates(
        integrated_polar.shape[0], r, intensities
    )
    integrated_template_norms = (
        np.ones(N, dtype=np.float64)
        if not normalize_templates
        else _get_row_norms(integrated_templates)
    )
    template_norms = (
        np.ones(N, dtype=np.float64)
        if not normalize_templates
        else _get_row_norms(intensities)
    )
    answer = _mixed_matching_lib_to_polar(
        polar_image,
        polar_norm,
        integrated_polar,
        polar_sum_norm,
        integrated_templates,
        integrated_template_norms,
        r,
        theta,
        intensities,
        template_norms,
        fraction,
        n_best,
    )
    indices = answer[:, 0].astype(np.int32)
    cors = answer[:, 1]
    angles = answer[:, 2]
    sign = answer[:,3]
    return indices, angles, cors, sign


def index_dataset_with_template_rotation(
    signal,
    library,
    phases=None,
    n_best=1,
    n_keep=100,
    frac_keep=None,
    delta_r=1.0,
    delta_theta=1.0,
    max_r=None,
    intensity_transform_function=None,
    find_direct_beam=False,
    direct_beam_positions=None,
    normalize_images=True,
    normalize_templates=True,
    parallelize_polar_conversion=False,
    chunks="auto",
    parallel_workers="auto",
):
    """
    Index a dataset with template_matching while simultaneously optimizing in-plane rotation angle of the templates

    Parameters
    ----------
    signal : hyperspy.signals.Signal2D
        The 4D-STEM dataset
    library : diffsims.libraries.diffraction_library.DiffractionLibrary
        The library of simulated diffraction patterns
    phases : list, optional
        Names of phases in the library to do an indexation for. By default this is
        all phases in the library.
    n_best : int, optional
        Number of best solutions to return, in order of descending match
    n_keep : int, optional
        Number of templates to do a full matching on in the second matching step
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
    direct_beam_positions : 2-tuple of floats or 3D numpy array of shape (scan_x, scan_y, 2), optional
        (x, y) coordinates of the direct beam in pixel units. Overrides other
        settings for finding the direct beam
    normalize_images : bool, optional
        normalize the images to calculate the correlation coefficient
    normalize_templates : bool, optional
        normalize the templates to calculate the correlation coefficient
    parallelize_polar_conversion : bool, optional
        use multiple workers for converting the dataset to polar coordinates. Overhead
        could make this slower on some hardware and for some datasets.
    chunks : string or 4-tuple, optional
        internally the work is done on dask datasets and this parameter determines
        the chunking. If set to None then no re-chunking will happen if the dataset
        was loaded lazily. If set to "auto" then dask attempts to find the optimal
        chunk size.
    parallel_workers: int, optional
        the number of workers to use in parallel. If set to "auto", the number
        will be determined from os.cpu_count()

    Returns
    -------
    result : dict
        Results dictionary containing keys [template_index, orientation], with
        values numpy arrays of shape (scan_x, scan_y, n_best) and (scan_x, scan_y, n_best, 3)
        respectively
    """
    result = {}
    # calculate number of workers
    if parallel_workers == "auto":
        parallel_workers = os.cpu_count()
    # get the dataset as a dask array
    data = _get_dask_array(signal)
    # check if we have a 4D dataset, and if not, make it
    navdim = signal.axes_manager.navigation_dimension
    if navdim == 0:
        data = data[np.newaxis, np.newaxis, ...]
    elif navdim == 1:
        data = data[np.newaxis, ...]
    elif navdim == 2:
        pass
    else:
        raise ValueError(f"Dataset has {navdim} navigation dimensions, max " "is 2")
    # change the chunking of the dataset
    if chunks is None:
        pass
    elif chunks == "auto":
        data = data.rechunk({0: "auto", 1: "auto", 2: None, 3: None})
    else:
        data = data.rechunk(chunks)
    # convert to polar dataset
    theta_dim, r_dim = get_polar_pattern_shape(
        data.shape[-2:], delta_r, delta_theta, max_r=max_r
    )
    polar_chunking = (data.chunks[0], data.chunks[1], theta_dim, r_dim)
    polar_data = data.map_blocks(
        chunk_to_polar,
        delta_r,
        delta_theta,
        max_r,
        find_direct_beam,
        direct_beam_positions,
        parallelize_polar_conversion,
        dtype=np.float32,
        drop_axis=signal.axes_manager.signal_indices_in_array,
        chunks=polar_chunking,
        new_axis=(2, 3),
    )
    # apply the intensity transform function to the images
    if intensity_transform_function is not None:
        polar_data = polar_data.map_blocks(intensity_transform_function)
    if phases is None:
        phases = library.keys()
    max_radius = r_dim * delta_r
    for phase_key in phases:
        phase_library = library[phase_key]
        positions, intensities = _simulations_to_arrays(
            phase_library["simulations"], max_radius
        )
        if intensity_transform_function is not None:
            intensities = intensity_transform_function(intensities)
        x = positions[:, 0]
        y = positions[:, 1]
        r, theta = _cartesian_positions_to_polar(
            x, y, delta_r=delta_r, delta_theta=delta_theta
        )
        # integrated intensity library for fast comparison
        integrated_templates = _get_integrated_polar_templates(r_dim, r, intensities)
        N = r.shape[0]
        fraction = max((N - abs(n_keep)) / N, 0.0)
        if frac_keep is not None:
            fraction = max(1.0 - abs(frac_keep), 0.0)
        # calculate the norms of the templates
        if normalize_templates:
            integrated_template_norms = _get_row_norms(integrated_templates)
            template_norms = _get_row_norms(intensities)
        else:
            integrated_template_norms = np.ones(N, dtype=np.float64)
            template_norms = np.ones(N, dtype=np.float64)
        # map the indexation to the blocks
        indexation = polar_data.map_blocks(
            _index_chunk,
            integrated_templates,
            integrated_template_norms,
            r,
            theta,
            intensities,
            template_norms,
            fraction,
            n_best,
            normalize_images,
            dtype=np.float32,
            drop_axis=signal.axes_manager.signal_indices_in_array,
            chunks=(polar_data.chunks[0], polar_data.chunks[1], n_best, 4),
            new_axis=(2, 3),
        )
        # wrangle data to (template_index), (orientation), (correlation)
        # TODO: there is some duplication here as the polar transform is re-calculated for each loop iteration
        # over the phases
        with ProgressBar():
            res_index = indexation.compute(
                scheduler="threads", num_workers=parallel_workers, optimize_graph=True
            )
        result[phase_key] = {}
        result[phase_key]["template_index"] = res_index[:, :, :, 0].astype(np.uint64)
        oris = phase_library["orientations"]
        orimap = oris[res_index[:, :, :, 0].astype(np.uint64)]
        orimap[:, :, :, 1] = orimap[:, :, :, 1] * res_index[:,:,:,3]  # multiply by the sign
        orimap[:, :, :, 2] = orimap[:, :, :, 2] * res_index[:,:,:,3]  # multiply by the sign
        orimap[:, :, :, 0] = res_index[:, :, :, 2] * delta_r
        result[phase_key]["orientation"] = orimap
        result[phase_key]["correlation"] = res_index[:, :, :, 1]
    return result
