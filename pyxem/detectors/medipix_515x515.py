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
515x515 Medipix Direct Electron Detector class for azimuthal integration
and other similar operations using pyFAI azimuthalIntegrator.
The three central pixels in both orientations are blanks, so those and three
and the surrounding ones are "masked".
"""

from pyFAI.detectors import Detector

class Medipix515x515Detector(Detector):
    '''
    Flavour text
    '''
    IS_FLAT = False  # this detector is not flat
    IS_CONTIGUOUS = True  # No gaps: all pixels are adjacents
    API_VERSION = "1.0"
    aliases = ["Medipix256x256Detector"]
    MAX_SHAPE=515,515

    def __init__(self):
        pixel1=55e-6 #55 micron pixel size in x
        pixel2=55e-6 #55 micron pixel size in y
        raise NotImplementedError
        Detector.__init__(self, pixel1=pixel1, pixel2=pixel2)

    def calc_mask(self):
        #Overrides the Detector calc_mask() function to define a mask.
        #the missing segment is a 5-wide cross in the middle
        mask = np.zeros((515,515))
        mask[255:260,:] = 1
        mask[:,255:260] = 1
        return mask
