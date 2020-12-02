# -*- coding: utf-8 -*-
# Copyright 2016-2020 The pyXem developers
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

import pyxem.dummy_data.dask_test_data as dtd


def test_get_hot_pixel_test_data_2d():
    data = dtd._get_hot_pixel_test_data_2d()
    assert len(data.shape) == 2


def test_get_hot_pixel_test_data_3d():
    data = dtd._get_hot_pixel_test_data_3d()
    assert len(data.shape) == 3


def test_get_hot_pixel_test_data_4d():
    data = dtd._get_hot_pixel_test_data_4d()
    assert len(data.shape) == 4


def test_get_dead_pixel_test_data_2d():
    data = dtd._get_dead_pixel_test_data_2d()
    assert len(data.shape) == 2


def test_get_dead_pixel_test_data_3d():
    data = dtd._get_dead_pixel_test_data_3d()
    assert len(data.shape) == 3


def test_get_dead_pixel_test_data_4d():
    data = dtd._get_dead_pixel_test_data_4d()
    assert len(data.shape) == 4
