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

import warnings

from hyperspy.signals import LazySignal

from pyxem.signals.differential_phase_contrast import DPCSignal1D, DPCSignal2D


class LazyDPCSignal1D(LazySignal, DPCSignal1D):
    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            "DPCSignal2D is deprecated. The functionality has been moved to the "
            "BeamShift class. Use the `to_beamshift()` function to convert to "
            "the BeamShift class.",
            DeprecationWarning,
        )


class LazyDPCSignal2D(LazySignal, DPCSignal2D):
    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            "DPCSignal2D is deprecated. The functionality has been moved to the "
            "BeamShift class. Use the `to_beamshift()` function to convert to "
            "the BeamShift class.",
            DeprecationWarning,
        )
