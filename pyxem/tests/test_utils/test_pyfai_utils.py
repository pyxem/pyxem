# -*- coding: utf-8 -*-
# Copyright 2017-2020 The pyXem developers
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

import pytest
import numpy as np
from pyxem.utils.pyfai_utils import _get_radial_azim_extent
from pyFAI.detectors import Detector
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

class Test_PyFai_integration:

    def test_get_extent(self):
        dect = Detector(pixel1=1e-4, pixel2=1e-4)
        ai = AzimuthalIntegrator(detector=dect, dist=0.1)
        ai.setFit2D(directDist=1000, centerX=50.5, centerY=50.5)
        _get_radial_azim_extent(ai=ai,shape=(100,100), unit="2th_deg")
