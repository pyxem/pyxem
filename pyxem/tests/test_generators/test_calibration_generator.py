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
from pyxem.generators.calibration_generator import CalibrationGenerator
from pyxem.utils.calibration_utils import generate_ring_pattern
from pyxem.libraries.calibration_library import CalibrationDataLibrary
from pyxem.signals.electron_diffraction import ElectronDiffraction


@pytest.fixture
def input_parameters():
    x0 = [95, 1200, 2.8, 450, 1.5, 10]
    return x0

@pytest.fixture
def affine_answer():
    a = np.asarray([[1.06651526, 0.10258988, 0.],
                    [0.10258988, 1.15822961, 0.],
                    [0.        , 0.        , 1.]])
    return a

@pytest.fixture
def ring_pattern(input_parameters):
    x0 = input_parameters
    ring_data = generate_ring_pattern(image_size=256,
                                      mask=True,
                                      mask_radius=10,
                                      scale=x0[0],
                                      amplitude=x0[1],
                                      spread=x0[2],
                                      direct_beam_amplitude=x0[3],
                                      asymmetry=x0[4],
                                      rotation=x0[5])

    return ElectronDiffraction(ring_data)

@pytest.fixture
def calibration_library(request, ring_pattern):
    #  Create a dummy X-grating image
    data = np.zeros((200,200))
    data[:,10:20] = 100
    data[:,30:40] = 50
    data[:,150:160] = 50
    data[:,170:180] = 100
    im = Signal2D(data)
    return CalibrationDataLibrary(au_x_grating_dp=ring_pattern,
                                  au_x_grating_im=im)

@pytest.fixture
def calgen(request, calibration_library):
    return CalibrationGenerator(calibration_data=calibration_library)

@pytest.fixture
def cal_dist(request, calgen):
    calgen.get_elliptical_distortion(mask_radius=10,
                                     direct_beam_amplitude=450,
                                     scale=95, amplitude=1200,
                                     asymmetry=1.5,spread=2.8, rotation=10)
    return calgen


class TestCalibrationGenerator:

    def test_init(self, calgen):
        assert isinstance(calgen.calibration_data.au_x_grating_dp,
                          ElectronDiffraction)

    def test_get_elliptical_distortion(self, cal_dist,
                                       input_parameters, affine_answer):
        np.testing.assert_allclose(cal_dist.affine_matrix, affine_answer)
        np.testing.assert_allclose(cal_dist.ring_params, input_parameters)

    def test_get_distortion_residuals(self, cal_dist):
        residuals = cal_dist.get_distortion_residuals(mask_radius=10, spread=2)
        assert isinstance(residuals, ElectronDiffraction)

    def test_plot_corrected_diffraction_pattern(self, cal_dist):
        cal_dist.plot_corrected_diffraction_pattern()

    def test_get_diffraction_calibration(self, cal_dist):
        cal_dist.get_diffraction_calibration(mask_length=30,
                                             linewidth=5)
        np.testing.assert_almost_equal(cal_dist.diffraction_calibration,
                                       0.010648)

    def test_get_navigation_calibration(self, calgen):
        line = Line2DROI(x1=2.5, y1=13., x2=193., y2=12.5, linewidth=3.5)
        value = calgen.get_navigation_calibration(line_roi=line, x1=12.,x2=172.,
                                                  n=1, xspace=500.)
        np.testing.assert_almost_equal(calgen.navigation_calibration,
                                       value)

    def test_plot_calibrated_data_dp(self, cal_dist):
        cal_dist.get_diffraction_calibration(mask_length=30,
                                             linewidth=5)
        cal_dist.plot_calibrated_data(data_to_plot='au_x_grating_dp')

    def test_plot_calibrated_data_im(self, calgen):
        line = Line2DROI(x1=2.5, y1=13., x2=193., y2=12.5, linewidth=3.5)
        calgen.get_navigation_calibration(line_roi=line, x1=12.,x2=172.,
                                          n=1, xspace=500.)
        calgen.plot_calibrated_data(data_to_plot='au_x_grating_im')


@pytest.fixture
def empty_calibration_library(request):
    return CalibrationDataLibrary()

@pytest.fixture
def empty_calgen(request, empty_calibration_library):
    return CalibrationGenerator(calibration_data=empty_calibration_library)


@pytest.mark.xfail(raises=ValueError)
class TestEmptyCalibrationGenerator:

    def test_get_elliptical_distortion(self, empty_calgen,
                                       input_parameters, affine_answer):
        empty_calgen.get_elliptical_distortion(mask_radius=10,
                                               direct_beam_amplitude=450,
                                               scale=95, amplitude=1200,
                                               asymmetry=1.5,spread=2.8,
                                               rotation=10)

    def test_get_distortion_residuals_no_data(self, empty_calgen):
        empty_calgen.get_distortion_residuals(mask_radius=10, spread=2)

    def test_get_distortion_residuals_no_affine(self, calgen):
        calgen.get_distortion_residuals(mask_radius=10, spread=2)

    def test_plot_corrected_diffraction_pattern_no_data(self, empty_calgen):
        empty_calgen.plot_corrected_diffraction_pattern()

    def test_plot_corrected_diffraction_pattern_no_affine(self, calgen):
        calgen.plot_corrected_diffraction_pattern()

    def test_get_diffraction_calibration_no_data(self, empty_calgen):
        empty_calgen.get_diffraction_calibration(mask_length=30,
                                                 linewidth=5)

    def test_get_diffraction_calibration_no_affine(self, calgen):
        calgen.get_diffraction_calibration(mask_length=30,
                                           linewidth=5)

    def test_get_navigation_calibration_no_data(self, empty_calgen):
        line = Line2DROI(x1=2.5, y1=13., x2=193., y2=12.5, linewidth=3.5)
        empty_calgen.get_navigation_calibration(line_roi=line, x1=12.,x2=172.,
                                                n=1, xspace=500.)
