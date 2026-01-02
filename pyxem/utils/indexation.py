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

"""Utilities for indexing electron diffraction spot patterns."""
import importlib


__all__ = [
    "get_nth_best_solution",
    "index_magnitudes",
    "match_vectors",
    "get_in_plane_rotation_correlation",
    "correlate_library_to_pattern_fast",
    "correlate_library_to_pattern",
    "get_n_best_matches",
    "index_dataset_with_template_rotation",
    "results_dict_to_crystal_map",
    "structure2dict",
    "dict2structure",
    "phase2dict",
    "dict2phase",
    "OrientationResult",
]


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    if name in __all__:
        return getattr(importlib.import_module("pyxem.utils._indexation"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
