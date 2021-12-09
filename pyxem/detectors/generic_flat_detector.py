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
A generic flat detector class for azimuthal integration and other similar
operations using the pyFAI AzimuthalIntegrator for a

"""

from pyFAI.detectors import Detector


class GenericFlatDetector(Detector):
    """
    A PyFAI Detector class for an arbitrarily sized flat detector (i.e. the
    calibration is assumed to be constant across the detector plane)

    The detector class is used for get_azimuthal_integral in a Diffraction2D
    signal. The data is assumed to be flat in the small angle approximation.

    PyFAI works in real space coordinates, the pixel size is assumed to be 1 (m)
    and the remaining parameters, via knowing the calibration and wavelength,
    are calculated to have an appropriate physical setup.

    Parameters
    ----------
    size_x : int
        The size (in pixels) of the detector in the x coordinate.
    size_y : int
        The size (in pixels) of the detector in the y coordinate.

    Examples
    --------
    >>> from pyxem.detectors import GenericFlatDetector
    >>> detector = GenericFlatDetector(512,512)
    >>> detector
    Detector GenericFlatDetector	 Spline= None
    PixelSize= 1.000e+00, 1.000e+00 m
    """

    IS_FLAT = True  # this detector is flat
    IS_CONTIGUOUS = True  # No gaps: all pixels are adjacents
    API_VERSION = "1.0"
    aliases = ["GenericFlatDetector"]

    def __init__(self, size_x, size_y):
        MAX_SHAPE = size_x, size_y
        Detector.__init__(self, pixel1=1, pixel2=1, max_shape=MAX_SHAPE)
