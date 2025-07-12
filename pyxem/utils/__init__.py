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

"""Additional utility functions for processing signals.

.. currentmodule:: pyxem.utils

.. rubric:: Modules

.. autosummary::
    :toctree: ../generated/
    :template: custom-module-template.rst

    plotting
    ransac_ellipse_tools
    vectors
    diffraction
    calibration

"""
import importlib

__all__ = [
    "find_diffraction_calibration",
    "plot_template_over_pattern",
    "determine_ellipse",
    "ransac_ellipse_tools",
    "vectors",
    "calibration",
    "plotting",
    "diffraction",
]


def __dir__():
    return sorted(__all__)


_import_mapping = {
    "determine_ellipse": ".ransac_ellipse_tools",
    "find_diffraction_calibration": ".calibration",
    "plot_template_over_pattern": ".plotting",
}


def __getattr__(name):
    if name in __all__:
        if name in _import_mapping.keys():
            import_path = "pyxem.utils" + _import_mapping.get(name)
            return getattr(importlib.import_module(import_path), name)
        else:
            return importlib.import_module("." + name, "pyxem.utils")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
