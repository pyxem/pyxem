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
import numpy as np

from hyperspy.signals import Signal2D
from hyperspy.roi import Line2DROI

from pyxem.signals.electron_diffraction import ElectronDiffraction
from pyxem.libraries.calibration_library import CalibrationDataLibrary

@pytest.fixture
def library(diffraction_pattern):
    dp = diffraction_pattern.mean((0,1))
    im = Signal2D(np.ones((10,10)))
    cdl = CalibrationDataLibrary(au_x_grating_dp=dp,
                                 au_x_grating_im=im)
    return cdl

def test_initialization_dtype(library):
    assert isinstance(library.au_x_grating_dp, ElectronDiffraction)


class TestPlotData:

    def test_plot_au_x_grating_dp(self, library):
        library.plot_calibration_data(data_to_plot='au_x_grating_dp')

    def test_plot_au_x_grating_im(self, library):
        library.plot_calibration_data(data_to_plot='au_x_grating_im')

    @pytest.mark.xfail(raises=ValueError)
    def test_plot_invalid(self, library):
        library.plot_calibration_data(data_to_plot='no_data')

    def test_plot_au_x_grating_dp_with_roi(self, library):
        line = Line2DROI(x1=1, y1=1, x2=3, y2=3, linewidth=1.)
        library.plot_calibration_data(data_to_plot='au_x_grating_dp',
                                      roi=line)
