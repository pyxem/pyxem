# -*- coding: utf-8 -*-
# Copyright 2016-2021 The pyXem developers
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
import numpy as np


class Medipix515x515Detector(Detector):
    """
    A PyFAI Detector class for a 515x515 pixel Medipix Quad direct electron
    detector. A central 5x5 cross is not intepretable, and is stored as a
    calc_mask method.

    The detector class is used for get_azimuthal_integral in a Diffraction2D
    signal. The calibration is not assumed to be constant in scattering vector.

    Examples
    --------
    >>> from pyxem.detectors import Medipix515x515Detector
    >>> detector = Medipix515x515Detector()
    >>> detector
    Detector Medipix515x515Detector	 Spline= None
    PixelSize= 5.500e-05, 5.500e-05 m
    """

    IS_FLAT = False  # this detector is not flat
    IS_CONTIGUOUS = True  # No gaps: all pixels are adjacents
    API_VERSION = "1.0"
    aliases = ["Medipix515x515Detector"]
    MAX_SHAPE = 515, 515

    def __init__(self):
        pixel1 = 55e-6  # 55 micron pixel size in x
        pixel2 = 55e-6  # 55 micron pixel size in y
        Detector.__init__(self, pixel1=pixel1, pixel2=pixel2)

    def calc_mask(self):
        """Defines a function to define a mask of missing and uninterpretable
        pixels in the detector plane, following
        The missing segment is a 5-wide cross in the middle of the detector.
        """

        mask = np.zeros((515, 515))
        mask[255:260, :] = 1
        mask[:, 255:260] = 1
        return mask.astype(np.int8)
