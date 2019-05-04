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

import pytest
import pyxem as pxm
import os
import numpy as np

from hyperspy.signals import Signal2D

from pyxem.signals.electron_diffraction import ElectronDiffraction
from pyxem.libraries.calibration_library import CalibrationDataLibrary

@pytest.fixture
def library(diffraction_pattern):
    dp = diffraction_pattern.mean((0,1))
    im = Signal2D(np.ones((10,10)))
    cdl = CalibrationDataLibrary(au_x_grating_dp=dp,
                                 au_x_grating_im=im,
                                 moo3_dp=dp,
                                 moo3_im=im)
    return cdl

def test_initialization_dtype(library):
    assert isinstance(library.au_x_grating_dp, ElectronDiffraction)
