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

"""Reduced intensity generator and associated tools.


"""
import numpy as np

from hyperspy.signals import Signal1D

from pyxem.signals.diffraction_profile import ElectronDiffractionProfile
from pyxem.signals.reduced_intensity_profile import ReducedIntensityProfile


class ReducedIntensityGenerator():
    """Generates a reduced intensity profile for a specified diffraction radial
    profile.


    Parameters
    ----------
    signal : ElectronDiffractionProfile
        An electron diffraction radial average profile.
    """
    def __init__(self, signal, *args, **kwargs):
        self.signal = signal
