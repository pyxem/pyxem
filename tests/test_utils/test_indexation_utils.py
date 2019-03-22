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
import pytest

from transforms3d.euler import euler2mat
import diffpy.structure

from pyxem.libraries.vector_library import DiffractionVectorLibrary
from pyxem.signals.diffraction_vectors import DiffractionVectors
from pyxem.utils.indexation_utils import crystal_from_template_matching, \
    crystal_from_vector_matching, match_vectors


@pytest.fixture
def sp_template_match_result():
    row_1 = np.array([0, np.array([2, 3, 4]), 0.7], dtype='object')
    row_2 = np.array([0, np.array([2, 3, 5]), 0.6], dtype='object')
    # note we require (correlation of row_1 > correlation row_2)
    return np.vstack((row_1, row_2))


@pytest.fixture
def dp_template_match_result():
    row_1 = np.array([0, np.array([2, 3, 4]), 0.7], dtype='object')
    row_2 = np.array([0, np.array([2, 3, 5]), 0.8], dtype='object')
    row_3 = np.array([1, np.array([2, 3, 4]), 0.5], dtype='object')
    row_4 = np.array([1, np.array([2, 3, 5]), 0.3], dtype='object')
    return np.vstack((row_1, row_2, row_3, row_4))


@pytest.fixture
def sp_vector_match_result():
    # We require (total_error of row_1 > correlation row_2)
    return np.array([
        # [phase, R, match_rate, ehkls, total_error]
        np.array([0, euler2mat(*np.deg2rad([0, 0, 90]), 'rzxz'), 0.5, np.array([0.1, 0.05, 0.2]), 0.1], dtype='object'),
        np.array([0, euler2mat(*np.deg2rad([0, 0, 90]), 'rzxz'), 0.6, np.array([0.1, 0.1, 0.2]), 0.2], dtype='object')
    ], dtype='object')


@pytest.fixture
def dp_vector_match_result():
    return np.array([
        # [phase, R, match_rate, ehkls, total_error]
        np.array([0, euler2mat(*np.deg2rad([90, 0, 0]), 'rzxz'), 0.6, np.array([0.1, 0.1, 0.2]), 0.3], dtype='object'),
        np.array([0, euler2mat(*np.deg2rad([0, 10, 20]), 'rzxz'), 0.5, np.array([0.1, 0.05, 0.2]), 0.4], dtype='object'),
        np.array([1, euler2mat(*np.deg2rad([0, 45, 45]), 'rzxz'), 0.8, np.array([0.1, 0.3, 0.2]), 0.1], dtype='object'),
        np.array([1, euler2mat(*np.deg2rad([0, 0, 90]), 'rzxz'), 0.7, np.array([0.1, 0.05, 0.1]), 0.2], dtype='object')
    ], dtype='object')


def test_crystal_from_template_matching_sp(sp_template_match_result):
    # branch single phase
    cmap = crystal_from_template_matching(sp_template_match_result)
    assert cmap[0] == 0
    np.testing.assert_allclose(cmap[1], [2, 3, 4])
    np.testing.assert_allclose(cmap[2]['correlation'], 0.7)
    np.testing.assert_allclose(cmap[2]['orientation_reliability'], 100 * (1 - (0.6 / 0.7)))


def test_crystal_from_template_matching_dp(dp_template_match_result):
    # branch double phase
    cmap = crystal_from_template_matching(dp_template_match_result)
    r_or = 100 * (1 - (0.7 / 0.8))
    r_ph = 100 * (1 - (0.5 / 0.8))
    assert cmap[0] == 0
    np.testing.assert_allclose(cmap[1], [2, 3, 5])
    np.testing.assert_allclose(cmap[2]['correlation'], 0.8)
    np.testing.assert_allclose(cmap[2]['orientation_reliability'], r_or)
    np.testing.assert_allclose(cmap[2]['phase_reliability'], r_ph)


def test_crystal_from_vector_matching_sp(sp_vector_match_result):
    # branch single phase
    cmap = crystal_from_vector_matching(sp_vector_match_result)
    assert cmap[0] == 0
    np.testing.assert_allclose(cmap[1], np.deg2rad([0, 0, 90]))
    np.testing.assert_allclose(cmap[2]['match_rate'], 0.5)
    np.testing.assert_allclose(cmap[2]['ehkls'], np.array([0.1, 0.05, 0.2]))
    np.testing.assert_allclose(cmap[2]['total_error'], 0.1)
    np.testing.assert_allclose(cmap[2]['orientation_reliability'], 100 * (1 - (0.1 / 0.2)))


def test_crystal_from_vector_matching_dp(dp_vector_match_result):
    # branch double phase
    cmap = crystal_from_vector_matching(dp_vector_match_result)
    r_or = 100 * (1 - (0.1 / 0.2))
    r_ph = 100 * (1 - (0.1 / 0.3))
    assert cmap[0] == 1
    np.testing.assert_allclose(cmap[1], np.deg2rad([0, 45, 45]))
    np.testing.assert_allclose(cmap[2]['match_rate'], 0.8)
    np.testing.assert_allclose(cmap[2]['ehkls'], np.array([0.1, 0.3, 0.2]))
    np.testing.assert_allclose(cmap[2]['total_error'], 0.1)
    np.testing.assert_allclose(cmap[2]['orientation_reliability'], r_or)
    np.testing.assert_allclose(cmap[2]['phase_reliability'], r_ph)


@pytest.fixture
def vector_match_peaks():
    return np.array([
        [1, 0.1, 0],
        [0, 2, 0],
        [1, 2, 3],
    ])


@pytest.fixture
def vector_library():
    library = DiffractionVectorLibrary()
    library['A'] = np.array([
        [np.array([1, 0, 0]), np.array([0, 2, 0]), 2, 1, np.pi / 2],
        [np.array([0, 0, 1]), np.array([2, 0, 0]), 1, 2, np.pi / 2],
    ])
    lattice = diffpy.structure.Lattice(1, 1, 1, 90, 90, 90)
    library.structures = [
        diffpy.structure.Structure(lattice=lattice)
    ]
    return library


def test_match_vectors(vector_match_peaks, vector_library):
    # Wrap to test handling of ragged arrays
    peaks = np.empty(1, dtype='object')
    peaks[0] = vector_match_peaks
    matches, rhkls = match_vectors(
        peaks,
        vector_library,
        mag_tol=0.1,
        angle_tol=0.1,
        index_error_tol=0.3,
        n_peaks_to_index=2,
        n_best=1)
    assert len(matches) == 1
    np.testing.assert_allclose(matches[0][2], 1.0)  # match rate
    np.testing.assert_allclose(matches[0][1], np.identity(3))
    np.testing.assert_allclose(matches[0][4], 0.01, atol=0.01)  # total error

    np.testing.assert_allclose(rhkls[0][0], [1, 0, 0])
    np.testing.assert_allclose(rhkls[0][1], [0, 2, 0])


def test_match_vector_total_error_default(vector_match_peaks, vector_library):
    matches, rhkls = match_vectors(
        vector_match_peaks,
        vector_library,
        mag_tol=0.1,
        angle_tol=0.1,
        index_error_tol=0.0,
        n_peaks_to_index=2,
        n_best=5)
    assert len(matches) == 5
    np.testing.assert_allclose(matches[0][2], 0.0)  # match rate
    np.testing.assert_allclose(matches[0][1], np.identity(3))
    np.testing.assert_allclose(matches[0][4], 1.0)  # total error

    np.testing.assert_allclose(rhkls[0][0], [1, 0, 0])
    np.testing.assert_allclose(rhkls[0][1], [0, 2, 0])
