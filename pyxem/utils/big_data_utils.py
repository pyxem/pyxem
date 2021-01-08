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

import hyperspy.api as hs
import numpy as np


def _get_chunk_size(x_list, y_list):
    """Finds chunk size and validates list entries."""
    chunk_size = x_list[1] - x_list[0]
    if chunk_size != (y_list[1] - y_list[0]):
        raise ValueError("x_list and y_list need to have the same chunksize")
    epsilon = 1e-5  # np.arange returns excluding the endpoint argument
    if not np.allclose(x_list, np.arange(x_list[0], x_list[-1] + epsilon, chunk_size)):
        raise ValueError("There is a problem with your x_list")
    elif not np.allclose(
        y_list, np.arange(y_list[0], y_list[-1] + epsilon, chunk_size)
    ):
        raise ValueError("There is a problem with your y_list")

    return chunk_size


def _load_and_cast(filepath, x, y, chunk_size):
    """Loads a chunk of a larger diffraction pattern."""
    s = hs.load(filepath, lazy=True)
    s = s.inav[x : x + chunk_size, y : y + chunk_size]
    s.compute(close_file=True)
    s.set_signal_type("electron_diffraction")
    return s


def _factory(fp, x, y, chunk_size, function):
    """Loads a chunk of a signal, and applies the UDF function.

    See Also
    --------
    pxm.utils.big_data_utils.chunked_application_of_UDF
    """
    dp = _load_and_cast(fp, x, y, chunk_size)
    analysis_output = function(dp)
    return analysis_output


def _create_columns(results_list, left_index, right_index):
    """Provides the vstack for ._combine_list_into_navigation_space."""

    return np.vstack([x for x in results_list[left_index:right_index]])


def _combine_list_into_navigation_space(results_list, x_list, y_list):
    """Internal function that combines the results_list into a correctly shaped
    output object.

    See Also
    --------
    pyxem.utils.big_data_utils.chunked_application_of_UDF
    """
    vert_list = []
    num_cols = len(x_list)
    num_rows = len(y_list)
    for i in np.arange(0, num_cols):  # .arange doesn't include the endpoint
        left_index = i * num_rows
        right_index = (i + 1) * num_rows
        vert_list.append(_create_columns(results_list, left_index, right_index))

    np_output = np.hstack([x for x in vert_list])
    return np_output


def chunked_application_of_UDF(filepath, x_list, y_list, function):
    """Applies a user specificed function to a diffraction pattern object with
    chunking for memory.

    Parameters
    ----------
    filepath : str
        Path to the file contain the data to be investigated
    x_list : list or np.array
        Iterable running from the "start" index to the final start "index" with a fixed step
        size. ie) Data total is as with dp.inav[start:final+step_size]
    y_list : list or np.array
        Iterable running from the "start" index to the final start "index" with a fixed step
        size. Step size must be the same as in x_list.
    function : function
        A user defined function that take a ElectronDiffraction2D as an argument and returns the desired output

    Returns
    -------
    np_output : np.array
        The results, as a numpy array.
    """
    results_list = []
    chunk_size = _get_chunk_size(x_list, y_list)
    for x in x_list:
        for y in y_list:
            analysis_output = _factory(filepath, x, y, chunk_size, function)
            results_list.append(analysis_output.data)
    np_output = _combine_list_into_navigation_space(results_list, x_list, y_list)
    return np_output
