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

"""Cuda utils."""

from ._cuda import (
    dask_array_to_gpu,
    dask_array_from_gpu,
    to_numpy,
    is_cupy_array,
    get_array_module,
)
from ._cuda_kernels import (
    _correlate_polar_image_to_library_gpu,
    TPB,
)

__all__ = [
    "dask_array_to_gpu",
    "dask_array_from_gpu",
    "to_numpy",
    "is_cupy_array",
    "get_array_module",
    "_correlate_polar_image_to_library_gpu",
    "TPB",
]
