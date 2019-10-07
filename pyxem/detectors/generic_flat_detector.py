# -*- coding: utf-8 -*-
# Copyright 2017-2019 The pyXem developers
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

"""
A generic flat detector class for azimuthal integration and other similar
operations using pyFAI azimuthalIntegrator.

?Maybe put in example text here?
"""

from pyFAI.detectors import Detector

class GenericFlatDetector(Detector):
    '''
    Flavour text
    '''
    IS_FLAT = True  # this detector is flat
    IS_CONTIGUOUS = True  # No gaps: all pixels are adjacents
    API_VERSION = "1.0"
    aliases = ["GenericFlatDetector"]
    def __init__(self, size_x, size_y):
        MAX_SHAPE = size_x,size_y
        Detector.__init__(self, pixel1=1, pixel2=1, max_shape=MAX_SHAPE)
