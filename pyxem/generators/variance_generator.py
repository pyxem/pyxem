# -*- coding: utf-8 -*-
# Copyright 2016-2020 The pyXem developers
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

"""Variance generators in real and reciprocal space for fluctuation electron microscopy."""

import numpy as np
from hyperspy.signals import Signal2D
from hyperspy.api import stack

from pyxem.signals import DiffractionVariance2D, ImageVariance
from pyxem.utils.signal import (
    transfer_navigation_axes_to_signal_axes,
    transfer_signal_axes,
)


class VarianceGenerator:
    """Generates variance images for a specified signal and set of aperture
    positions.

    Parameters
    ----------
    signal : ElectronDiffraction2D
        The signal of electron diffraction patterns to be indexed.

    """

    def __init__(self, signal, *args, **kwargs):
        self.signal = signal
        self.thickness_filter = None


        # add a check for calibration
