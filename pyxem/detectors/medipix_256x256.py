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
256x256 Medipix Direct Electron Detector class for azimuthal integration
and other similar operations using pyFAI azimuthalIntegrator.
"""

from pyFAI.detectors import Detector


class Medipix256x256Detector(Detector):
    """
    A PyFAI Detector class for a 256256 pixel Medipix direct electron detector.
    The detector class is used for get_azimuthal_integral in a Diffraction2D
    signal. The calibration is not assumed to be constant in scattering vector.

    Examples
    --------
    >>> from pyxem.detectors import Medipix256x256Detector
    >>> detector = Medipix256x256Detector()
    >>> detector
    Detector Medipix256x256Detector	 Spline= None
    PixelSize= 5.500e-05, 5.500e-05 m
    """

    IS_FLAT = False  # this detector is not flat
    IS_CONTIGUOUS = True  # No gaps: all pixels are adjacents
    API_VERSION = "1.0"
    aliases = ["Medipix256x256Detector"]
    MAX_SHAPE = 256, 256

    def __init__(self):
        pixel1 = 55e-6  # 55 micron pixel size in x
        pixel2 = 55e-6  # 55 micron pixel size in y
        Detector.__init__(self, pixel1=pixel1, pixel2=pixel2)
