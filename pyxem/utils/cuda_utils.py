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

"""Utils for Cuda."""

from numba import cuda, int32, float32
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


# must be defined outside the kernel
TPB = 256


@cuda.jit
def _correlate_polar_image_to_library_gpu(
    polar_image, sim_r, sim_t, sim_i, correlation, correlation_m
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
    correlation_m: 2D numpy.ndarray or cupy.ndarray
        The output correlation matrix of shape (template number, theta), giving
        the correlation of the mirror of each template at each in-plane angle
    """
    # each cuda thread handles one element
    # thread blocks should be shaped (1, TPB)
    template, shift = cuda.grid(2)

    # don't do anything for grid positions outside template number range
    # we still use threads with shift > correlation.shape[1] to load spot data
    # to shared memory
    if template >= correlation.shape[0]:
        return

    r_shared = cuda.shared.array(shape=(TPB,), dtype=int32)
    t_shared = cuda.shared.array(shape=(TPB,), dtype=int32)
    i_shared = cuda.shared.array(shape=(TPB,), dtype=float32)

    # thread index within block
    ty = cuda.threadIdx.y

    n_shifts = polar_image.shape[0]

    total_spots = sim_r.shape[1]

    # number of iterations necessary to process all the spots
    iters = total_spots // TPB + 1

    min_spot_iter = min((TPB, total_spots))

    tmp = 0.0
    tmp_m = 0.0
    for i in range(iters):
        # fill up the shared arrays with the template
        spot = ty + i * TPB
        if spot < total_spots:
            r_shared[ty] = sim_r[template, spot]
            t_shared[ty] = sim_t[template, spot]
            i_shared[ty] = sim_i[template, spot]
        else:
            r_shared[ty] = 0
            t_shared[ty] = 0
            i_shared[ty] = 0.0

        # wait for all the threads
        cuda.syncthreads()

        if shift < n_shifts:
            # use the shared arrays to update tmp
            for s in range(min_spot_iter):
                r_sp = r_shared[s]
                if r_sp == 0:
                    break
                t_sp = t_shared[s]
                i_sp = i_shared[s]
                t1 = t_sp + shift
                if t1 >= n_shifts:
                    t1 = t1 - n_shifts
                t2 = n_shifts - t_sp + shift
                if t2 >= n_shifts:
                    t2 = t2 - n_shifts
                tmp += polar_image[t1, r_sp] * i_sp
                tmp_m += polar_image[t2, r_sp] * i_sp

        # wait for all the threads
        cuda.syncthreads()

    # don't update out of range
    if shift >= n_shifts:
        return

    correlation[template, shift] = tmp
    correlation_m[template, shift] = tmp_m
