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

import importlib
import warnings

from hyperspy.signals import Signal1D

from pyxem.common import VisibleDeprecationWarning
from pyxem.signals._common_diffraction import CommonDiffraction


class Diffraction1D(CommonDiffraction, Signal1D):
    """Signal class for Electron Diffraction radial profiles.

    Parameters
    ----------
    *args
        See :class:`hyperspy.api.signals.Signal1D`.
    **kwargs
        See :class:`hyperspy.api.signals.Signal1D`
    """

    _signal_type = "diffraction"

    pass


# ruff: noqa: F822

__all__ = [
    "Diffraction1D",
    "LazyDiffraction1D",
]


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    if "Lazy" in name:
        warnings.warn(
            f"Importing `{name}` from `{__name__}` is deprecated and will be "
            "removed in pyXem 1.0.0. Import it from "
            "`pyxem.signals` instead.",
            VisibleDeprecationWarning,
        )
        return getattr(importlib.import_module("pyxem.signals"), name)
    if name in __all__:
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
