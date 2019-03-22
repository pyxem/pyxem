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

from pyxem.utils.vector_utils import calculate_norms, calculate_norms_ragged, \
    detector_to_fourier, get_rotation_matrix_between_vectors, \
    get_angle_cartesian


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


@pytest.mark.parametrize('k1, k2, ref_k1, ref_k2, expected_rotation', [
    ([0, 0, 1], [0, 0, 2], [1, 0, 0], [0, 1, 0], np.identity(3)),  # Degenerate
    ([0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0], euler2mat(np.deg2rad(90), 0, 0, 'rzxz')),
    ([0.5, -0.5, 1 / np.sqrt(2)], [1 / np.sqrt(2), 1 / np.sqrt(2), 0], [0, 0, 1], [1, 0, 0],
        euler2mat(np.deg2rad(45), np.deg2rad(45), 0, 'rzxz'))
])
def test_get_rotation_matrix_between_vectors(k1, k2, ref_k1, ref_k2,
                                             expected_rotation):
    rotation_matrix = get_rotation_matrix_between_vectors(k1, k2, ref_k1, ref_k2)
    assert np.allclose(rotation_matrix, expected_rotation)


@pytest.mark.parametrize('vec_a, vec_b, expected_angle', [
    ([0, 0, 1], [0, 1, 0], np.deg2rad(90)),
    ([0, 0, 0], [0, 0, 1], 0)
])
def test_get_angle_cartesian(vec_a, vec_b, expected_angle):
    angle = get_angle_cartesian(vec_a, vec_b)
    assert np.isclose(angle, expected_angle)
