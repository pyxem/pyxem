# -*- coding: utf-8 -*-
# Copyright 2016-2023 The pyXem developers
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

import pytest
import numpy as np

from pyxem.utils.correlation_utils import (
    _correlation,
    _get_interpolation_matrix,
    _wrap_set_float,
)


class TestCorrelations:
    @pytest.fixture
    def ones_array(self):
        return np.ones((10, 20))

    @pytest.fixture
    def ones_zero(self):
        ones = np.ones((10, 20))
        ones[0:20:2, :] = 0
        return ones

    @pytest.fixture
    def ones_hundred(self):
        ones = np.ones((10, 20))
        ones[0:20:2, :] = 100
        return ones

    def test_correlation_ones(self, ones_array):
        c = _correlation(ones_array)
        np.testing.assert_array_equal(c, np.zeros((10, 20)))

    def test_correlations_axis(self, ones_zero):
        c = _correlation(ones_zero, axis=0, normalize=True)
        result = np.ones((10, 20))
        result[1::2, :] = -1
        np.testing.assert_array_equal(c, result)
        c = _correlation(ones_zero, axis=0, normalize=False)
        result = np.zeros((10, 20))
        result[0::2, :] = 5
        np.testing.assert_array_almost_equal(c, result)

    def test_wrap_set_float(self):
        target = np.zeros(20)
        assert 2 == np.sum(_wrap_set_float(target, bottom=10.0, top=15.5, value=2))
        target = np.zeros(20)
        assert 2 == np.sum(_wrap_set_float(target, bottom=10.0, top=10.5, value=2))
        target = np.zeros(20)
        assert 2 == np.sum(_wrap_set_float(target, bottom=10.0, top=10.0, value=2))

    def test_correlations_normalization(self, ones_hundred):
        c = _correlation(ones_hundred, axis=0, normalize=True)
        result = np.zeros((10, 20))
        result[1::2, :] = -0.96078816
        result[0::2, :] = 0.96078816
        np.testing.assert_array_almost_equal(c, result)
        c = _correlation(ones_hundred, axis=1, normalize=True)
        result = np.zeros((10, 20))
        np.testing.assert_array_almost_equal(c, result)

    def test_correlations_mask(self, ones_hundred):
        m = np.zeros((10, 20))
        m[2:4, :] = 1
        c = _correlation(ones_hundred, axis=0, normalize=True, mask=m)
        print(c)
        result = np.zeros((10, 20))
        result[1::2, :] = -0.96078816
        result[0::2, :] = 0.96078816
        np.testing.assert_array_almost_equal(c, result)

    def test_correlations_wrapping(
        self, ones_hundred
    ):  # Need to do extra checks to assure this is correct
        m = np.zeros((10, 20))
        m[2:4, :] = 1
        c = _correlation(ones_hundred, axis=0, normalize=True, wrap=False)
        print(c)
        result = np.zeros((10, 20))
        result[0::2, :] = 2.26087665
        result[1::2, :] = -0.93478899
        np.testing.assert_array_almost_equal(c, result)

    @pytest.mark.parametrize(
        "angles",
        np.array(
            [
                [0.1],
                [
                    0.25,
                ],
                [
                    0.5,
                ],
                [
                    0.9,
                ],
            ]
        ),
    )
    @pytest.mark.parametrize("angular_range", [0, 4 * np.pi / 90])
    def test_get_interpolation_matrix(self, angles, angular_range):
        m = np.zeros((10, 20))
        m[2:4, :] = 1
        c = _get_interpolation_matrix(
            angles, num_points=90, angular_range=angular_range
        )
        com = np.average(range(90), weights=c)
        np.testing.assert_almost_equal(com, 90 * angles[0])
