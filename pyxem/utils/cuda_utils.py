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
import importlib


__all__ = [
    "dask_array_to_gpu",
    "dask_array_from_gpu",
    "to_numpy",
    "is_cupy_array",
    "get_array_module",
    "_correlate_polar_image_to_library_gpu",
    "TPB",
]


def __dir__():
    return sorted(__all__)


_import_mapping = {
    "dask_array_to_gpu": "._cuda_utils",
    "dask_array_from_gpu": "._cuda_utils",
    "to_numpy": "._cuda_utils",
    "is_cupy_array": "._cuda_utils",
    "get_array_module": "._cuda_utils",
    "_correlate_polar_image_to_library_gpu": "._cuda_kernels",
    "TPB": "._cuda_kernels",
}


def __getattr__(name):
    if name in __all__:
        import_path = "pyxem.utils" + _import_mapping.get(name)
        return getattr(importlib.import_module(import_path), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
