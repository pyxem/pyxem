# -*- coding: utf-8 -*-
# Copyright 2016-2025 The pyXem developers
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

"""Utils for Cuda."""

import numpy as np

from pyxem.common import CUPY_INSTALLED


def dask_array_to_gpu(dask_array):
    """
    Copy a dask array to the GPU chunk by chunk
    """
    if not CUPY_INSTALLED:
        raise BaseException("cupy is required")

    import cupy as cp

    return dask_array.map_blocks(cp.asarray)


def dask_array_from_gpu(dask_array):
    """
    Copy a dask array from the GPU chunk by chunk
    """
    if not CUPY_INSTALLED:
        raise BaseException("cupy is required")

    import cupy as cp

    return dask_array.map_blocks(cp.asnumpy)


def to_numpy(array):
    """
    Returns the array as an numpy array
    Parameters
    ----------
    array : numpy or cupy array
        Array to determine whether numpy or cupy should be used
    Returns
    -------
    array : numpy.ndarray
    """
    if is_cupy_array(array):
        import cupy as cp

        array = cp.asnumpy(array)
    return array


def get_array_module(array):
    """
    Returns the array module for the given array
    Parameters
    ----------
    array : numpy or cupy array
        Array to determine whether numpy or cupy should be used
    Returns
    -------
    module : module
    """
    module = np
    if CUPY_INSTALLED:
        import cupy as cp

        if is_cupy_array(array):
            module = cp

    return module


def is_cupy_array(array):
    """
    Convenience function to determine if an array is a cupy array

    Parameters
    ----------
    array : array
        The array to determine whether it is a cupy array or not.
    Returns
    -------
    bool
        True if it is cupy array, False otherwise.
    """
    if CUPY_INSTALLED:
        import cupy as cp

        return isinstance(array, cp.ndarray)
    else:
        return False
