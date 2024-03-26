# -*- coding: utf-8 -*-
# Copyright 2016-2024 The pyXem developers
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


from numba import cuda, int32, float32
import numpy as np

from pyxem.utils._deprecated import deprecated
from pyxem.utils._cuda import _is_cupy_installed

@deprecated(since="0.18.0", removal="0.20.0")
def dask_array_to_gpu(dask_array):
    """
    Copy a dask array to the GPU chunk by chunk
    """
    if _is_cupy_installed():
        return dask_array.map_blocks(cp.asarray)
    else:
        raise BaseException("cupy is required")

@deprecated(since="0.18.0", removal="0.20.0")
def dask_array_from_gpu(dask_array):
    """
    Copy a dask array from the GPU chunk by chunk
    """
    if _is_cupy_installed():
        return dask_array.map_blocks(cp.asnumpy)
    else:
        raise BaseException("cupy is required")

@deprecated(since="0.18.0", removal="0.20.0")
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

@deprecated(since="0.18.0", removal="0.20.0")
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
    try:
        import cupy as cp

        if isinstance(array, cp.ndarray):
            module = cp
    except ImportError:
        pass
    return module

@deprecated(since="0.18.0", removal="0.20.0")
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
    try:
        import cupy as cp

        return isinstance(array, cp.ndarray)
    except ImportError:
        return False