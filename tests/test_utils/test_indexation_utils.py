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

from pyxem.utils.indexation_utils import crystal_from_template_matching, crystal_from_vector_matching


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
    # [phase, R, match_rate, ehkls, total_error]
    row_1 = np.array([0, euler2mat(*np.deg2rad([0, 0, 90]), 'rzxz'), 0.5, np.array([0.1, 0.05, 0.2]), 0.1], dtype='object')
    row_2 = np.array([0, euler2mat(*np.deg2rad([0, 0, 90]), 'rzxz'), 0.6, np.array([0.1, 0.1, 0.2]), 0.2], dtype='object')
    # We require (total_error of row_1 > correlation row_2)
    return np.vstack((row_1, row_2))


@pytest.fixture
def dp_vector_match_result():
    row_1 = np.array([0, euler2mat(*np.deg2rad([90, 0, 0]), 'rzxz'),  0.6, np.array([0.1, 0.1, 0.2]), 0.3], dtype='object')
    row_2 = np.array([0, euler2mat(*np.deg2rad([0, 10, 20]), 'rzxz'), 0.5, np.array([0.1, 0.05, 0.2]), 0.4], dtype='object')
    row_3 = np.array([1, euler2mat(*np.deg2rad([0, 45, 45]), 'rzxz'),  0.8, np.array([0.1, 0.3, 0.2]), 0.1], dtype='object')
    row_4 = np.array([1, euler2mat(*np.deg2rad([0, 0, 90]), 'rzxz'), 0.7, np.array([0.1, 0.05, 0.1]), 0.2], dtype='object')
    return np.vstack((row_1, row_2, row_3, row_4))


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
