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

from pyxem.utils.indexation_utils import (
    match_vectors,
    zero_mean_normalized_correlation,
    fast_correlation,
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
