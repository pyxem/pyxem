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

from pyxem.utils.vector_utils import calculate_norms
from pyxem.utils.vector_utils import calculate_norms_ragged
from pyxem.utils.vector_utils import detector_to_fourier
from pyxem.utils.vector_utils import get_rotation_matrix_between_vectors
from pyxem.utils.vector_utils import get_angle_cartesian
from pyxem.utils.vector_utils import get_angle_cartesian_vec


def test_calculate_norms():
    norms = calculate_norms([[3, 4], [6, 8]])
    assert np.allclose(norms, [5, 10])


def test_calculate_norms_ragged():
    norms = calculate_norms_ragged(np.array([[[3], [6, 8]]]))
    assert np.allclose(norms, [3, 10])


@pytest.mark.parametrize('wavelength, camera_length, detector_coords, k_expected', [
    (0.025, 0.2,
        np.array([
            [0, 0],
            [0, 1],
            [1, 1]
        ]),
        np.array([
            [0, 0, 1 / 0.025 - 40],
            [0, 1, np.sqrt(1 / (0.025**2) - 1) - 40],
            [1, 1, np.sqrt(1 / (0.025**2) - 1 - 1) - 40]
        ])
     )
])
def test_detector_to_fourier(wavelength,
                             camera_length,
                             detector_coords,
                             k_expected):
    k = detector_to_fourier(detector_coords, wavelength, camera_length)
    np.testing.assert_allclose(k, k_expected)


@pytest.mark.parametrize('from_v1, from_v2, to_v1, to_v2, expected_rotation', [
    # v2 from x to y
    ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], euler2mat(*np.deg2rad([90, 0, 0]), 'rzxz')),
    # Degenerate to-vectors gives half-way rotation (about y-axis)
    ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], euler2mat(*np.deg2rad([90, 45, -90]), 'rzxz')),
    # Edges to body diagonals
    ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.5, -0.5, 1 / np.sqrt(2)], [1 / np.sqrt(2), 1 / np.sqrt(2), 0],
        euler2mat(*np.deg2rad([45, 45, 0]), 'rzxz'))
])
def test_get_rotation_matrix_between_vectors(from_v1, from_v2, to_v1, to_v2, expected_rotation):
    rotation_matrix = get_rotation_matrix_between_vectors(
        np.array(from_v1), np.array(from_v2),
        np.array([to_v1]), np.array([to_v2]))
    np.testing.assert_allclose(rotation_matrix, np.array([expected_rotation]), atol=1e-15)


@pytest.mark.parametrize('vec_a, vec_b, expected_angle', [
    ([0, 0, 1], [0, 1, 0], np.deg2rad(90)),
    ([0, 0, 0], [0, 0, 1], 0)
])
def test_get_angle_cartesian(vec_a, vec_b, expected_angle):
    angle = get_angle_cartesian(vec_a, vec_b)
    np.testing.assert_allclose(angle, expected_angle)


@pytest.mark.parametrize('a, b, expected_angles', [
    (np.array([[0, 0, 1], [0, 0, 0]]), np.array([[0, 1, 0], [0, 0, 1]]), [np.deg2rad(90), 0])
])
def test_get_angle_cartesian_vec(a, b, expected_angles):
    angles = get_angle_cartesian_vec(a, b)
    np.testing.assert_allclose(angles, expected_angles)


@pytest.mark.xfail(raises=ValueError)
def test_get_angle_cartesian_vec_input_validation():
    get_angle_cartesian_vec(np.empty((2, 3)), np.empty((5, 3)))
