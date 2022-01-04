# -*- coding: utf-8 -*-
# Copyright 2016-2022 The pyXem developers
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


from pyxem.utils import polar_transform_utils as ptu
import pytest
import numpy as np
from unittest.mock import Mock

try:
    import cupy as cp

    CUPY_INSTALLED = True
except ImportError:
    CUPY_INSTALLED = False
    cp = np


skip_cupy = pytest.mark.skipif(not CUPY_INSTALLED, reason="cupy is required")


@pytest.fixture()
def mock_simulation():
    mock_sim = Mock()
    mock_sim.calibrated_coordinates = np.array(
        [[3, 4, 0], [5, 12, 0], [8, 15, 0], [7, 24, 0]]  # 5  # 13  # 17
    )  # 25
    mock_sim.intensities = np.array([1, 2, 3, 4])
    return mock_sim


@skip_cupy
@pytest.fixture()
def mock_simulation_gpu():
    mock_sim = Mock()
    mock_sim.calibrated_coordinates = cp.array(
        [[3, 4, 0], [5, 12, 0], [8, 15, 0], [7, 24, 0]]  # 5  # 13  # 17
    )  # 25
    mock_sim.intensities = cp.array([1, 2, 3, 4])
    return mock_sim


@pytest.mark.parametrize(
    "max_r, expected_r",
    [
        (None, np.array([5, 13, 17, 25])),
        (15, np.array([5, 13])),
    ],
)
def test_template_to_polar(mock_simulation, max_r, expected_r):
    r, theta, _ = ptu.get_template_polar_coordinates(mock_simulation, max_r=max_r)
    np.testing.assert_array_almost_equal(r, expected_r)


@skip_cupy
@pytest.mark.parametrize(
    "max_r, expected_r",
    [
        (None, cp.array([5, 13, 17, 25])),
        (15, cp.array([5, 13])),
    ],
)
def test_template_to_polar_gpu(mock_simulation_gpu, max_r, expected_r):
    r, theta, _ = ptu.get_template_polar_coordinates(mock_simulation_gpu, max_r=max_r)
    cp.testing.assert_allclose(r, expected_r)


@pytest.mark.parametrize(
    "angle, window, expected_x, expected_y",
    [
        (0, None, np.array([3, 5, 8, 7]), np.array([4, 12, 15, 24])),
        (90, None, -np.array([4, 12, 15, 24]), np.array([3, 5, 8, 7])),
        (90, (7, 7), np.array([]), np.array([])),
    ],
)
def test_get_template_cartesian_coordinates(
    mock_simulation, angle, window, expected_x, expected_y
):
    x, y, _ = ptu.get_template_cartesian_coordinates(
        mock_simulation, in_plane_angle=angle, window_size=window
    )
    np.testing.assert_array_almost_equal(x, expected_x)
    np.testing.assert_array_almost_equal(y, expected_y)


@skip_cupy
@pytest.mark.parametrize(
    "angle, window, expected_x, expected_y",
    [
        (0, None, cp.array([3, 5, 8, 7]), cp.array([4, 12, 15, 24])),
        (90, None, -cp.array([4, 12, 15, 24]), cp.array([3, 5, 8, 7])),
        (90, (7, 7), cp.array([]), cp.array([])),
    ],
)
def test_get_template_cartesian_coordinates_gpu(
    mock_simulation_gpu, angle, window, expected_x, expected_y
):
    x, y, _ = ptu.get_template_cartesian_coordinates(
        mock_simulation_gpu, in_plane_angle=angle, window_size=window
    )
    cp.testing.assert_array_almost_equal(x, expected_x)
    cp.testing.assert_array_almost_equal(y, expected_y)


@pytest.mark.parametrize(
    "image_shape, max_r, expected",
    [
        ((20, 20), None, (360, 15)),
        ((20, 20), 10, (360, 10)),
    ],
)
def test_get_polar_pattern_shape(image_shape, max_r, expected):
    result = ptu.get_polar_pattern_shape(image_shape, 1, 1, max_r=max_r)
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize(
    "delta_r, delta_theta, max_r, fdb, db, expected_shape",
    [
        (1, 1, None, False, None, (360, 126)),
        (3, 2, None, False, (110, 80), (180, 42)),
        (1, 1, 200, True, None, (360, 200)),
    ],
)
def test_image_to_polar(delta_r, delta_theta, max_r, fdb, db, expected_shape):
    image = np.ones((200, 151))
    result = ptu.image_to_polar(
        image,
        delta_r=delta_r,
        delta_theta=delta_theta,
        max_r=max_r,
        find_direct_beam=fdb,
        direct_beam_position=db,
    )
    np.testing.assert_array_almost_equal(result.shape, expected_shape)


@skip_cupy
@pytest.mark.parametrize(
    "delta_r, delta_theta, max_r, fdb, db, expected_shape",
    [
        (1, 1, None, False, None, (360, 126)),
        (3, 2, None, False, (110, 80), (180, 42)),
        (1, 1, 200, True, None, (360, 200)),
    ],
)
def test_image_to_polar_gpu(delta_r, delta_theta, max_r, fdb, db, expected_shape):
    image = cp.ones((200, 151))
    result = ptu.image_to_polar(
        image,
        delta_r=delta_r,
        delta_theta=delta_theta,
        max_r=max_r,
        find_direct_beam=fdb,
        direct_beam_position=db,
    )
    cp.testing.assert_array_almost_equal(result.shape, expected_shape)


@pytest.mark.parametrize(
    "center, radius, output_shape, expected_shape",
    [
        ((1, 1), 1, (360, 7), (3, 2, 360, 7)),
        ((5, 8), 10, (10, 10), (3, 2, 10, 10)),
    ],
)
def test_chunk_polar(center, radius, output_shape, expected_shape):
    chunk = np.ones((3, 2, 10, 7))
    polar_chunk = ptu._chunk_to_polar(chunk, center, radius, output_shape)
    np.testing.assert_array_almost_equal(polar_chunk.shape, expected_shape)


@skip_cupy
@pytest.mark.parametrize(
    "center, radius, output_shape, expected_shape",
    [
        ((1, 1), 1, (360, 7), (3, 2, 360, 7)),
        ((5, 8), 10, (10, 10), (3, 2, 10, 10)),
    ],
)
def test_chunk_polar_gpu(center, radius, output_shape, expected_shape):
    chunk = cp.ones((3, 2, 10, 7))
    polar_chunk = ptu._chunk_to_polar(chunk, center, radius, output_shape)
    cp.testing.assert_array_almost_equal(polar_chunk.shape, expected_shape)
