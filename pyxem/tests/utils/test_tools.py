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


import pyxem.utils.pixelated_stem_tools as pst
import numpy as np


class TestGetLimitsFromArray:
    def test_simple(self):
        data_array0 = np.array((5, 10))
        clim0 = pst._get_limits_from_array(data_array0, sigma=4)
        assert (data_array0.min(), data_array0.max()) == clim0
        data_array1 = np.array((5, -10))
        clim1 = pst._get_limits_from_array(data_array1, sigma=4)
        assert (data_array1.min(), data_array1.max()) == clim1
        data_array2 = np.array((-5, 10))
        clim2 = pst._get_limits_from_array(data_array2, sigma=4)
        assert (data_array2.min(), data_array2.max()) == clim2
        data_array3 = np.array((-5, -10))
        clim3 = pst._get_limits_from_array(data_array3, sigma=4)
        assert (data_array3.min(), data_array3.max()) == clim3

    def test_simple_sigma(self):
        data_array0 = np.array((5, 10))
        clim0 = pst._get_limits_from_array(data_array0, sigma=0)
        assert (data_array0.mean(), data_array0.mean()) == clim0
        data_array1 = np.array((5, -10))
        clim1 = pst._get_limits_from_array(data_array1, sigma=0)
        assert (data_array1.mean(), data_array1.mean()) == clim1
        data_array2 = np.array((-5, 10))
        clim2 = pst._get_limits_from_array(data_array2, sigma=0)
        assert (data_array2.mean(), data_array2.mean()) == clim2
        data_array3 = np.array((-5, -10))
        clim3 = pst._get_limits_from_array(data_array3, sigma=0)
        assert (data_array3.mean(), data_array3.mean()) == clim3

    def test_ignore_zeros(self):
        data_array0 = np.zeros(shape=(100, 100))
        value = 50
        data_array0[:, 70:80] = value
        clim0_0 = pst._get_limits_from_array(data_array0, ignore_zeros=True)
        assert (value, value) == clim0_0
        clim0_1 = pst._get_limits_from_array(data_array0, sigma=0)
        assert (data_array0.mean(), data_array0.mean()) == clim0_1
        clim0_2 = pst._get_limits_from_array(data_array0, sigma=1)
        assert (0.0, 20.0) == clim0_2

    def test_ignore_edges(self):
        data_array = np.ones(shape=(100, 100)) * 5000
        value = 50
        data_array[1:-1, 1:-1] = value
        clim0 = pst._get_limits_from_array(data_array, ignore_edges=True)
        assert (value, value) == clim0
        clim1 = pst._get_limits_from_array(data_array, ignore_edges=False)
        assert not ((value, value) == clim1)
