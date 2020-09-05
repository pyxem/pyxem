# -*- coding: utf-8 -*-
# Copyright 2019 The pyXem developers
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

from diffsims.libraries.diffraction_library import DiffractionLibrary

from pyxem.generators.indexation_generator import get_library_FT_dict

from pyxem.utils.indexation_utils import (
    crystal_from_template_matching,
    crystal_from_vector_matching,
    match_vectors,
    zero_mean_normalized_correlation,
    fast_correlation,
    full_frame_correlation,
    optimal_fft_size,
    correlate_library_from_dict,
)


def test_zero_mean_normalized_correlation():
    np.testing.assert_approx_equal(
        zero_mean_normalized_correlation(
            3, np.linalg.norm([2 / 3, 1 / 3, 1 / 3]), 1 / 3, [1, 0, 0], [1, 0, 0]
        ),
        1,
    )
    # nb_pixels,image_std,average_image_intensity,image_intensities,int_local
    assert zero_mean_normalized_correlation(3, 0, 1, [1, 1, 1], [0, 0, 1]) == 0


def test_fast_correlation():
    np.testing.assert_approx_equal(
        fast_correlation([1, 1, 1], [1, 1, 1], np.sqrt(3)), np.sqrt(3)
    )
    np.testing.assert_approx_equal(fast_correlation([1, 1, 1], [1, 0, 0], 1), 1)


def test_full_frame_correlation():
    # Define testing parameters.
    in1 = np.zeros((10, 10))
    in2 = np.zeros((10, 10))
    in1[5, 5] = 1
    in1[7, 7] = 1
    in1[3, 7] = 1
    in1[7, 3] = 1
    in1_FT = np.fft.fftshift(np.fft.rfftn(in1, (20, 20)))
    norm_1 = np.sqrt(np.max(np.real(np.fft.ifftn(in1_FT ** 2))))
    in2_FT = np.fft.fftshift(np.fft.rfftn(in2, (20, 20)))
    norm_2 = np.sqrt(np.max(np.real(np.fft.ifftn(in2_FT ** 2))))
    np.testing.assert_approx_equal(
        full_frame_correlation(in1_FT, norm_1, in1_FT, norm_1), 1
    )
    np.testing.assert_approx_equal(
        full_frame_correlation(in1_FT, norm_1, in2_FT, norm_2), 0
    )


def test_optimal_fft_size():
    np.testing.assert_approx_equal(optimal_fft_size(8), 8)
    np.testing.assert_approx_equal(optimal_fft_size(20), 20)


def test_correlate_library_from_dict():
    new_template_library = DiffractionLibrary()
    new_template_library["GaSb"] = {
        "orientations": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        "pixel_coords": np.array([np.asarray([[1, 1]]), np.asarray([[2, 2], [1, 1]])]),
        "intensities": np.array([np.array([1,]), np.array([1, 1])]),
    }
    shape = (3, 3)
    fsize = (5, 5)
    new_template_dict = get_library_FT_dict(new_template_library, shape, fsize)
    image = np.zeros((3, 3))
    image[1, 1] = 1
    match_results = correlate_library_from_dict(
        image, new_template_dict, n_largest=1, method="full_frame_correlation", mask=1
    )
    np.testing.assert_approx_equal(match_results[0][1][1], 0.0)
    np.testing.assert_approx_equal(match_results[0][2], 1.0)


def test_crystal_from_template_matching_sp(sp_template_match_result):
    # branch single phase
    cmap = crystal_from_template_matching(sp_template_match_result)
    assert cmap[0] == 0
    np.testing.assert_allclose(cmap[1], [2, 3, 4])
    np.testing.assert_allclose(cmap[2]["correlation"], 0.7)
    np.testing.assert_allclose(
        cmap[2]["orientation_reliability"], 100 * (1 - (0.6 / 0.7))
    )


def test_crystal_from_template_matching_dp(dp_template_match_result):
    # branch double phase
    cmap = crystal_from_template_matching(dp_template_match_result)
    r_or = 100 * (1 - (0.7 / 0.8))
    r_ph = 100 * (1 - (0.5 / 0.8))
    assert cmap[0] == 0
    np.testing.assert_allclose(cmap[1], [2, 3, 5])
    np.testing.assert_allclose(cmap[2]["correlation"], 0.8)
    np.testing.assert_allclose(cmap[2]["orientation_reliability"], r_or)
    np.testing.assert_allclose(cmap[2]["phase_reliability"], r_ph)


def test_crystal_from_vector_matching_sp(sp_vector_match_result):
    # branch single phase
    cmap = crystal_from_vector_matching(sp_vector_match_result)
    assert cmap[0] == 0
    np.testing.assert_allclose(cmap[1], [0, 0, 90])
    np.testing.assert_allclose(cmap[2]["match_rate"], 0.5)
    np.testing.assert_allclose(cmap[2]["ehkls"], np.array([0.1, 0.05, 0.2]))
    np.testing.assert_allclose(cmap[2]["total_error"], 0.1)
    np.testing.assert_allclose(
        cmap[2]["orientation_reliability"], 100 * (1 - (0.1 / 0.2))
    )


def test_crystal_from_vector_matching_dp(dp_vector_match_result):
    # branch double phase
    cmap = crystal_from_vector_matching(dp_vector_match_result)
    r_or = 100 * (1 - (0.1 / 0.2))
    r_ph = 100 * (1 - (0.1 / 0.3))
    assert cmap[0] == 1
    np.testing.assert_allclose(cmap[1], [0, 45, 45])
    np.testing.assert_allclose(cmap[2]["match_rate"], 0.8)
    np.testing.assert_allclose(cmap[2]["ehkls"], np.array([0.1, 0.3, 0.2]))
    np.testing.assert_allclose(cmap[2]["total_error"], 0.1)
    np.testing.assert_allclose(cmap[2]["orientation_reliability"], r_or)
    np.testing.assert_allclose(cmap[2]["phase_reliability"], r_ph)


def test_match_vectors(vector_match_peaks, vector_library):
    # Wrap to test handling of ragged arrays
    peaks = np.empty(1, dtype="object")
    peaks[0] = vector_match_peaks
    matches, rhkls = match_vectors(
        peaks,
        vector_library,
        mag_tol=0.1,
        angle_tol=0.1,
        index_error_tol=0.3,
        n_peaks_to_index=2,
        n_best=1,
    )
    assert len(matches) == 1
    np.testing.assert_allclose(matches[0].match_rate, 1.0)
    np.testing.assert_allclose(matches[0].rotation_matrix, np.identity(3), atol=0.1)
    np.testing.assert_allclose(matches[0].total_error, 0.03, atol=0.01)

    np.testing.assert_allclose(rhkls[0][0], [1, 0, 0])
    np.testing.assert_allclose(rhkls[0][1], [0, 2, 0])
    np.testing.assert_allclose(rhkls[0][2], [1, 2, 3])


def test_match_vector_total_error_default(vector_match_peaks, vector_library):
    matches, rhkls = match_vectors(
        vector_match_peaks,
        vector_library,
        mag_tol=0.1,
        angle_tol=0.1,
        index_error_tol=0.0,
        n_peaks_to_index=2,
        n_best=5,
    )
    assert len(matches) == 5
    np.testing.assert_allclose(matches[0][2], 0.0)  # match rate
    np.testing.assert_allclose(matches[0][1], np.identity(3), atol=0.1)
    np.testing.assert_allclose(matches[0][4], 1.0)  # error mean

    assert len(rhkls) == 0
