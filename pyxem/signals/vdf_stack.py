# -*- coding: utf-8 -*-
# Copyright 2017-2018 The pyXem developers
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
"""Signal class for a stack of virtual diffraction contrast images.

"""

from hyperspy._signals.lazy import LazySignal
from hyperspy.api import interactive, stack
from hyperspy.components1d import Voigt, Exponential, Polynomial
from hyperspy.signals import Signal1D, Signal2D, BaseSignal
from pyxem.signals.diffraction_profile import DiffractionProfile
from pyxem.signals.diffraction_vectors import DiffractionVectors
from pyxem.utils.expt_utils import *
from pyxem.utils.peakfinders2D import *


def peaks_as_gvectors(z, center, calibration):
    g = (z - center) * calibration
    return g[0]

class VDFStack(Signal2D):
    _signal_type = "vdf_stack"

    def __init__(self, *args, **kwargs):
        Signal2D.__init__(self, *args, **kwargs)
