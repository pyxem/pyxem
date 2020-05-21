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

import numpy as np
import dask.array as da


def _get_hot_pixel_test_data_2d():
    """Get artifical 2D dataset with hot pixels.

    Values are 50, except [21, 11] and [5, 38]
    being 50000 (to represent a "hot pixel").

    Examples
    --------
    >>> import pyxem.dummy_data.dask_test_data as dtd
    >>> data = dtd._get_hot_pixel_test_data_2d()

    """
    data = np.ones((40, 50)) * 50
    data[21, 11] = 50000
    data[5, 38] = 50000
    dask_array = da.from_array(data, chunks=(5, 5))
    return dask_array


def _get_hot_pixel_test_data_3d():
    """Get artifical 3D dataset with hot pixels.

    Values are 50, except [2, 21, 11] and [1, 5, 38]
    being 50000 (to represent a "hot pixel").

    Examples
    --------
    >>> import pyxem.dummy_data.dask_test_data as dtd
    >>> data = dtd._get_hot_pixel_test_data_3d()

    """
    data = np.ones((5, 40, 50)) * 50
    data[2, 21, 11] = 50000
    data[1, 5, 38] = 50000
    dask_array = da.from_array(data, chunks=(5, 5, 5))
    return dask_array


def _get_hot_pixel_test_data_4d():
    """Get artifical 4D dataset with hot pixels.

    Values are 50, except [4, 2, 21, 11] and [6, 1, 5, 38]
    being 50000 (to represent a "hot pixel").

    Examples
    --------
    >>> import pyxem.dummy_data.dask_test_data as dtd
    >>> data = dtd._get_hot_pixel_test_data_4d()

    """
    data = np.ones((10, 5, 40, 50)) * 50
    data[4, 2, 21, 11] = 50000
    data[6, 1, 5, 38] = 50000
    dask_array = da.from_array(data, chunks=(5, 5, 5, 5))
    return dask_array


def _get_dead_pixel_test_data_2d():
    """Get artifical 2D dataset with dead pixels.

    Values are 50, except [14, 42] and [2, 12]
    being 0 (to represent a "dead pixel").

    Examples
    --------
    >>> import pyxem.dummy_data.dask_test_data as dtd
    >>> data = dtd._get_dead_pixel_test_data_2d()

    """
    data = np.ones((40, 50)) * 50
    data[14, 42] = 0
    data[2, 12] = 0
    dask_array = da.from_array(data, chunks=(5, 5))
    return dask_array


def _get_dead_pixel_test_data_3d():
    """Get artifical 3D dataset with dead pixels.

    Values are 50, except [:, 14, 42] and [:, 2, 12]
    being 0 (to represent a "dead pixel").

    Examples
    --------
    >>> import pyxem.dummy_data.dask_test_data as dtd
    >>> data = dtd._get_dead_pixel_test_data_3d()

    """
    data = np.ones((5, 40, 50)) * 50
    data[:, 14, 42] = 0
    data[:, 2, 12] = 0
    dask_array = da.from_array(data, chunks=(5, 5, 5))
    return dask_array


def _get_dead_pixel_test_data_4d():
    """Get artifical 4D dataset with dead pixels.

    Values are 50, except [:, :, 14, 42] and [:, :, 2, 12]
    being 0 (to represent a "dead pixel").

    Examples
    --------
    >>> import pyxem.dummy_data.dask_test_data as dtd
    >>> data = dtd._get_dead_pixel_test_data_4d()

    """
    data = np.ones((10, 5, 40, 50)) * 50
    data[:, :, 14, 42] = 0
    data[:, :, 2, 12] = 0
    dask_array = da.from_array(data, chunks=(5, 5, 5, 5))
    return dask_array
