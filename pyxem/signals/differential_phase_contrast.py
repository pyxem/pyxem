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

import copy

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.ndimage import rotate, gaussian_filter

from hyperspy.signals import BaseSignal, Signal1D, Signal2D
from hyperspy._signals.lazy import LazySignal

import pyxem.utils._pixelated_stem_tools as pst




class DPCBaseSignal(BaseSignal):
    """
    Signal for processing differential phase contrast (DPC) acquired using
    scanning transmission electron microscopy (STEM).

    The signal assumes the data is 3 dimensions, where the two
    signal dimensions are the probe positions, and the navigation
    dimension is the x and y disk shifts.

    The first navigation index (s.inav[0]) is assumed to the be x-shift
    and the second navigation is the y-shift (s.inav[1]).

    """

    _signal_type = "dpc"

    pass

class DPCSignal1D(Signal1D):
    """
    Signal for processing differential phase contrast (DPC) acquired using
    scanning transmission electron microscopy (STEM).

    The signal assumes the data is 2 dimensions, where the
    signal dimension is the probe position, and the navigation
    dimension is the x and y disk shifts.

    The first navigation index (s.inav[0]) is assumed to the be x-shift
    and the second navigation is the y-shift (s.inav[1]).

    """

    _signal_type = "dpc"

    pass

class DPCSignal2D(Signal2D):
    """
    Signal for processing differential phase contrast (DPC) acquired using
    scanning transmission electron microscopy (STEM).

    The signal assumes the data is 3 dimensions, where the two
    signal dimensions are the probe positions, and the navigation
    dimension is the x and y disk shifts.

    The first navigation index (s.inav[0]) is assumed to the be x-shift
    and the second navigation is the y-shift (s.inav[1]).

    """

    _signal_type = "dpc"



class LazyDPCBaseSignal(LazySignal, DPCBaseSignal):
    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LazyDPCSignal1D(LazySignal, DPCSignal1D):
    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LazyDPCSignal2D(LazySignal, DPCSignal2D):
    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
