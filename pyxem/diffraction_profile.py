# -*- coding: utf-8 -*-
# Copyright 2018 The pyXem developers
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
"""Signal class for Electron Diffraction radial profiles

"""

from hyperspy.components1d import Voigt, Exponential, Polynomial
from hyperspy.signals import Signal1D

from .utils.expt_utils import *

def peaks_as_gvectors(z, center, calibration):
    g = (z - center) * calibration
    return g[0]

class DiffractionProfile(Signal1D):
    _signal_type = "diffraction_profile"

    def __init__(self, *args, **kwargs):
        Signal1D.__init__(self, *args, **kwargs)

    def 
