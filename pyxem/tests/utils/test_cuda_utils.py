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

from pyxem.utils import cuda_utils as cutls
import dask.array as da
import numpy as np
import pytest
from unittest.mock import Mock


try:
    import cupy as cp

    CUPY_INSTALLED = True
except ImportError:
    CUPY_INSTALLED = False
    cp = np

skip_cupy = pytest.mark.skipif(not CUPY_INSTALLED, reason="cupy is required")


@skip_cupy
def test_dask_array_to_gpu():
    cutls.dask_array_to_gpu(da.array([1, 2, 3, 4]))


@skip_cupy
def test_dask_array_from_gpu():
    cutls.dask_array_from_gpu(da.array(cp.array([1, 2, 3, 4])))
