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

import os
import tempfile

import numpy as np
import pytest

import pyxem as pxm
from pyxem.utils.big_data_utils import chunked_application_of_UDF, _get_chunk_size


class Test_bad_xy_lists:
    def test_two_chunksizes(self):
        with pytest.raises(
            ValueError, match="x_list and y_list need to have the same chunksize"
        ):
            _get_chunk_size([0, 10], [0, 5])

    def test_bad_x_list(self):
        with pytest.raises(ValueError, match="There is a problem with your x_list"):
            _get_chunk_size([0, 2, 5], [0, 2])

    def test_bad_y_list(self):
        with pytest.raises(ValueError, match="There is a problem with your y_list"):
            _get_chunk_size([0, 2], [0, 2, 5])


"""
This runs a simple case of square rooting all elements of a pattern,
the chunked version is compared with doing the entire operation in memory.
"""


@pytest.fixture()
def big_electron_diffraction_pattern():
    z = np.arange(0, 160, step=1).reshape(4, 10, 2, 2)  # x_size=10, y_size=4 in hspy
    dp = pxm.signals.ElectronDiffraction2D(z)
    return dp


def single_navigation_square_root(z):
    return np.sqrt(z)


def dp_sqrt(dp):
    sqrts = dp.map(single_navigation_square_root, inplace=False)
    return sqrts


def test_core_big_data_functionality(big_electron_diffraction_pattern):
    expected_output = np.sqrt(big_electron_diffraction_pattern.data)
    with tempfile.TemporaryDirectory() as tmp:
        filepath = os.path.join(tmp, "tempfile_for_big_data_util_testing.hspy")
        big_electron_diffraction_pattern.save(filepath)

        x_list = [0, 2, 4, 6, 8]
        y_list = np.arange(0, 4, 2)  # [0,2] but as an array

        test_output = chunked_application_of_UDF(filepath, x_list, y_list, dp_sqrt)
        assert np.allclose(expected_output, test_output.data)
