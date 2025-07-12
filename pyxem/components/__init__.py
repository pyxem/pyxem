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

"""Classes for analyzing components in a material"""

import importlib


__all__ = [
    "ReducedIntensityCorrectionComponent",
    "ScatteringFitComponentLobato",
    "ScatteringFitComponentXTables",
]


def __dir__():
    return sorted(__all__)


_import_mapping = {
    "ReducedIntensityCorrectionComponent": ".reduced_intensity_correction_component",
    "ScatteringFitComponentLobato": ".scattering_fit_component_lobato",
    "ScatteringFitComponentXTables": ".scattering_fit_component_xtables",
}


def __getattr__(name):
    if name in __all__:
        import_path = "pyxem.components" + _import_mapping.get(name)
        return getattr(importlib.import_module(import_path), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
