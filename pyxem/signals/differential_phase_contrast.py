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

import warnings
from hyperspy.signals import Signal1D, Signal2D
from hyperspy._signals.lazy import LazySignal
from pyxem.utils._deprecated import deprecated


class DPCSignal1D(Signal1D):
    """
    Note: this signal will be removed in pyxem version 1.0.0.
    All the functionality has been moved to the BeamShift class.
    Convert this signal to the BeamShift class using the
    DPCSignal.to_beamshift() function.

    Signal for processing differential phase contrast (DPC) acquired using
    scanning transmission electron microscopy (STEM).

    The signal assumes the data is 2 dimensions, where the
    signal dimension is the probe position, and the navigation
    dimension is the x and y disk shifts.

    The first navigation index (s.inav[0]) is assumed to the be x-shift
    and the second navigation is the y-shift (s.inav[1]).

    """

    _signal_type = "dpc"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            "DPCSignal1D is deprecated. The functionality has been moved to the "
            "BeamShift class. Use the `to_beamshift()` function to convert to "
            "the BeamShift class.",
            DeprecationWarning,
        )

    def to_beamshift(self):
        """Get BeamShift signal from the DPCSignal.

        The BeamShift signal is a utility signal focused on correcting the shift of the
        center beam in the Diffraction2D signal.

        In practice, the signal and navigation dimensions are switched.

        """
        s_beam_shift = self.T
        s_beam_shift.set_signal_type("beam_shift")
        return s_beam_shift


class DPCSignal2D(Signal2D):
    """
    Note: this signal will be removed in pyxem version 1.0.0.
    All the functionality has been moved to the BeamShift class.
    Convert this signal to the BeamShift class using the
    DPCSignal.to_beamshift() function.

    Signal for processing differential phase contrast (DPC) acquired using
    scanning transmission electron microscopy (STEM).

    The signal assumes the data is 3 dimensions, where the two
    signal dimensions are the probe positions, and the navigation
    dimension is the x and y disk shifts.

    The first navigation index (s.inav[0]) is assumed to the be x-shift
    and the second navigation is the y-shift (s.inav[1]).

    """

    _signal_type = "dpc"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            "DPCSignal2D is deprecated. The functionality has been moved to the "
            "BeamShift class. Use the `to_beamshift()` function to convert to "
            "the BeamShift class.",
            DeprecationWarning,
        )

    def to_beamshift(self):
        """Get BeamShift signal from the DPCSignal.

        The BeamShift signal is a utility signal focused on correcting the shift of the
        center beam in the Diffraction2D signal.

        In practice, the signal and navigation dimensions are switched.

        """
        s_beam_shift = self.T
        s_beam_shift.set_signal_type("beam_shift")
        return s_beam_shift


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
