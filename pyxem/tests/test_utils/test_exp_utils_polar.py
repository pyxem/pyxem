# -*- coding: utf-8 -*-
# Copyright 2017-2020 The pyXem developers
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

from pyxem.utils.exp_utils_polar import _correlation, angular_correlation, angular_power, _power


class TestCorrelations:
    @pytest.fixture
    def ones_array(self):
        return np.ones((10,20))
    @pytest.fixture
    def ones_zero(self):
        ones = np.ones((10,20))
        ones[0:20:2,:]=0
        return ones

    def test_correlation_ones(self, ones_array):
        c = _correlation(ones_array)
        np.testing.assert_array_equal(c,np.zeros((10,20)))

    def test_correlations_axis(self, ones_zero):
        c = _correlation(ones_zero, axis=0, normalize=True)
        result = np.ones((10,20))
        result[1::2,:]=-1
        np.testing.assert_array_equal(c, result)
        c = _correlation(ones_zero, axis=0, normalize=False)
        result = np.zeros((10,20))
        result[0::2,:]=5
        np.testing.assert_array_almost_equal(c, result)

    def test_correlations_(self, ones_zero):
        c = _correlation(ones_zero, axis=0, normalize=True)
        result = np.ones((10,20))
        result[1::2,:]=-1
        np.testing.assert_array_equal(c, result)
        c = _correlation(ones_zero, axis=0, normalize=False)
        result = np.zeros((10,20))
        result[0::2,:]=5
        np.testing.assert_array_almost_equal(c, result)


