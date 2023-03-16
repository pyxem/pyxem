# -*- coding: utf-8 -*-
# Copyright 2016-2023 The pyXem developers
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


from hyperspy.signals import Signal1D
from hyperspy._signals.lazy import LazySignal

from pyxem.signals.common_diffraction import CommonDiffraction


class Diffraction1D(CommonDiffraction, Signal1D):
    """Signal class for Electron Diffraction radial profiles."""

    _signal_type = "diffraction"

    pass


class LazyDiffraction1D(LazySignal, Diffraction1D):
    pass
