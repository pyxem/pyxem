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


from numba import cuda
import numpy as np

try:
    import cupy as cp

    CUPY_INSTALLED = True
except ImportError:
    CUPY_INSTALLED = False


def dask_array_to_gpu(dask_array):
    """
    Copy a dask array to the GPU chunk by chunk
    """
    if not CUPY_INSTALLED:
        raise BaseException("cupy is required")
    return dask_array.map_blocks(cp.asarray)


def dask_array_from_gpu(dask_array):
    """
    Copy a dask array from the GPU chunk by chunk
    """
    if not CUPY_INSTALLED:
        raise BaseException("cupy is required")
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
    try:
        import cupy as cp

        if isinstance(array, cp.ndarray):
            module = cp
    except ImportError:
        pass
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
    try:
        import cupy as cp

        return isinstance(array, cp.ndarray)
    except ImportError:
        return False


@cuda.jit
def _correlate_polar_image_to_library_gpu(
    polar_image, sim_r, sim_t, sim_i, correlation
):
    """
    Custom cuda kernel for calculating the correlation for each template
    in a library at all in-plane angles with a polar image

    Parameters
    ----------
    polar_image: 2D numpy.ndarray or cupy.ndarray
        The image in polar coordinates of shape (theta, R)
    sim_r: 2D numpy.ndarray or cupy.ndarray
        The r coordinates of all spots in all templates in the library.
        Spot number is the column, template is the row
    sim_t: 2D numpy.ndarray or cupy.ndarray
        The theta coordinates of all spots in all templates in the library.
        Spot number is the column, template is the row
    sim_i: 2D numpy.ndarray or cupy.ndarray
        The intensity of all spots in all templates in the library.
        Spot number is the column, template is the row
    correlation: 2D numpy.ndarray or cupy.ndarray
        The output correlation matrix of shape (template number, theta), giving
        the correlation of each template at each in-plane angle
    """
    # each cuda thread handles one element
    template, shift = cuda.grid(2)
    # don't calculate for grid positions outside
    if template >= correlation.shape[0] or shift >= correlation.shape[1]:
        return
    tmp = 0.0
    # add up all contributions to the correlation from spots
    for spot in range(sim_r.shape[1]):
        if sim_r[template, spot] == 0:
            break
        tmp += (
            polar_image[
                (sim_t[template, spot] + shift) % polar_image.shape[0],
                sim_r[template, spot],
            ]
            * sim_i[template, spot]
        )
    correlation[template, shift] = tmp
