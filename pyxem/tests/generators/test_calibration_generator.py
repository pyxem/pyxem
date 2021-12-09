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

import pytest
import numpy as np

from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from hyperspy.signals import Signal2D
from hyperspy.roi import Line2DROI
from diffsims.utils.ring_pattern_utils import generate_ring_pattern

from pyxem.signals import ElectronDiffraction2D
from pyxem.generators import CalibrationGenerator
from pyxem.libraries import CalibrationDataLibrary


@pytest.fixture
def input_parameters():
    x0 = [95, 1200, 2.8, 450, 1.5, 10]
    return x0


@pytest.fixture
def affine_answer():
    a = np.asarray(
        [[1.06651526, 0.10258988, 0.0], [0.10258988, 1.15822961, 0.0], [0.0, 0.0, 1.0]]
    )
    return a


@pytest.fixture
def ring_pattern(input_parameters):
    x0 = input_parameters
    ring_data = generate_ring_pattern(
        image_size=256,
        mask=True,
        mask_radius=10,
        scale=x0[0],
        amplitude=x0[1],
        spread=x0[2],
        direct_beam_amplitude=x0[3],
        asymmetry=x0[4],
        rotation=x0[5],
    )

    return ElectronDiffraction2D(ring_data)


@pytest.fixture
def grating_image(request):
    #  Create a dummy X-grating image
    data = np.zeros((200, 200))
    data[:, 10:20] = 100
    data[:, 30:40] = 50
    data[:, 150:160] = 50
    data[:, 170:180] = 100
    im = Signal2D(data)
    return im


@pytest.fixture
def calgen(request, ring_pattern, grating_image):
    return CalibrationGenerator(
        diffraction_pattern=ring_pattern, grating_image=grating_image
    )


@pytest.fixture
def cal_dist(request, calgen):
    calgen.get_elliptical_distortion(
        mask_radius=10,
        direct_beam_amplitude=450,
        scale=95,
        amplitude=1200,
        asymmetry=1.5,
        spread=2.8,
        rotation=10,
    )
    return calgen


class TestCalibrationGenerator:
    def test_init(self, calgen):
        assert isinstance(calgen.diffraction_pattern, ElectronDiffraction2D)

    def test_str(self, calgen):
        print(calgen)

    def test_get_elliptical_distortion(self, cal_dist, input_parameters, affine_answer):
        np.testing.assert_allclose(cal_dist.affine_matrix, affine_answer, rtol=1e-3)
        np.testing.assert_allclose(cal_dist.ring_params, input_parameters, rtol=1e-3)

    def test_get_distortion_residuals(self, cal_dist):
        residuals = cal_dist.get_distortion_residuals(mask_radius=10, spread=2)
        assert isinstance(residuals, ElectronDiffraction2D)

    def test_plot_corrected_diffraction_pattern(self, cal_dist):
        cal_dist.plot_corrected_diffraction_pattern()

    def test_get_diffraction_calibration(self, cal_dist):
        cal_dist.get_diffraction_calibration(mask_length=30, linewidth=5)
        np.testing.assert_almost_equal(
            cal_dist.diffraction_calibration, 0.01061096, decimal=3
        )

    def test_get_navigation_calibration(self, calgen):
        line = Line2DROI(x1=2.5, y1=13.0, x2=193.0, y2=12.5, linewidth=3.5)
        value = calgen.get_navigation_calibration(
            line_roi=line, x1=12.0, x2=172.0, n=1, xspace=500.0
        )
        np.testing.assert_almost_equal(calgen.navigation_calibration, value, decimal=3)

    def test_get_rotation_calibration(self, calgen):
        real_line = Line2DROI(
            x1=2.69824, y1=81.4867, x2=229.155, y2=61.6898, linewidth=3
        )
        recip_line = Line2DROI(
            x1=-0.30367, y1=-1.21457, x2=0.344978, y2=1.24927, linewidth=0.115582
        )
        value = calgen.get_rotation_calibration(
            real_line=real_line, reciprocal_line=recip_line
        )
        np.testing.assert_almost_equal(value, -80.24669411537899, decimal=3)

    def test_plot_calibrated_data_dp(self, cal_dist):
        cal_dist.get_diffraction_calibration(mask_length=30, linewidth=5)
        cal_dist.plot_calibrated_data(data_to_plot="au_x_grating_dp")

    def test_plot_calibrated_data_im(self, calgen):
        line = Line2DROI(x1=2.5, y1=13.0, x2=193.0, y2=12.5, linewidth=3.5)
        calgen.get_navigation_calibration(
            line_roi=line, x1=12.0, x2=172.0, n=1, xspace=500.0
        )
        calgen.plot_calibrated_data(data_to_plot="au_x_grating_im")


class TestGetCorrectionMatrix:
    def test_get_correction_rotation_only(self, calgen):
        calgen.rotation_angle = -80.24669411537899
        corr = calgen.get_correction_matrix()
        np.testing.assert_almost_equal(
            corr,
            np.array(
                [
                    [0.16940637, 0.98554629, 0.0],
                    [-0.98554629, 0.16940637, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            decimal=3,
        )

    def test_get_correction_affine_only(self, calgen):
        affine = np.array(
            [
                [0.97579077, 0.01549655, 0.0],
                [0.01549655, 0.99008051, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        calgen.affine_matrix = affine
        corr = calgen.get_correction_matrix()
        np.testing.assert_almost_equal(corr, affine, decimal=3)

    def test_get_correction_affine_and_rotation(self, calgen):
        affine = np.array(
            [
                [0.97579077, 0.01549655, 0.0],
                [0.01549655, 0.99008051, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        calgen.affine_matrix = affine
        real_line = Line2DROI(
            x1=2.69824, y1=81.4867, x2=229.155, y2=61.6898, linewidth=3
        )
        recip_line = Line2DROI(
            x1=-0.30367, y1=-1.21457, x2=0.344978, y2=1.24927, linewidth=0.115582
        )
        value = calgen.get_rotation_calibration(
            real_line=real_line, reciprocal_line=recip_line
        )
        corr = calgen.get_correction_matrix()
        np.testing.assert_almost_equal(
            corr,
            np.array(
                [
                    [0.1805777, 0.9783954, 0.0],
                    [-0.9590618, 0.1524534, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            decimal=3,
        )

    def test_no_attributes_correction_matrix(self, calgen):
        with pytest.raises(
            ValueError,
            match="requires either an affine matrix to correct distortion",
        ):
            calgen.get_correction_matrix()


@pytest.fixture
def empty_calibration_library(request):
    return CalibrationDataLibrary()


@pytest.fixture
def empty_calgen(request, empty_calibration_library):
    return CalibrationGenerator()


class TestEmptyCalibrationGenerator:
    def test_get_elliptical_distortion(
        self, empty_calgen, input_parameters, affine_answer
    ):
        with pytest.raises(
            ValueError,
            match="This method requires a calibration diffraction"
            " pattern to be provided. Please set"
            " self.diffraction_pattern equal to some Signal2D.",
        ):
            empty_calgen.get_elliptical_distortion(
                mask_radius=10,
                direct_beam_amplitude=450,
                scale=95,
                amplitude=1200,
                asymmetry=1.5,
                spread=2.8,
                rotation=10,
            )

    def test_get_distortion_residuals_no_data(self, empty_calgen):
        with pytest.raises(ValueError, match="requires an Au X-grating diffraction"):
            empty_calgen.get_distortion_residuals(mask_radius=10, spread=2)

    def test_get_distortion_residuals_no_affine(self, calgen):
        with pytest.raises(ValueError, match="requires a distortion matrix"):
            calgen.get_distortion_residuals(mask_radius=10, spread=2)

    def test_plot_corrected_diffraction_pattern_no_data(self, empty_calgen):
        with pytest.raises(ValueError, match="requires an Au X-grating diffraction"):
            empty_calgen.plot_corrected_diffraction_pattern()

    def test_plot_corrected_diffraction_pattern_no_affine(self, calgen):
        with pytest.raises(ValueError, match="requires a distortion matrix"):
            calgen.plot_corrected_diffraction_pattern()

    def test_get_diffraction_calibration_no_data(self, empty_calgen):
        with pytest.raises(ValueError, match="requires an Au X-grating diffraction"):
            empty_calgen.get_diffraction_calibration(mask_length=30, linewidth=5)

    def test_get_diffraction_calibration_no_affine(self, calgen):
        with pytest.raises(ValueError, match="requires a distortion matrix"):
            calgen.get_diffraction_calibration(mask_length=30, linewidth=5)

    def test_get_navigation_calibration_no_data(self, empty_calgen):
        line = Line2DROI(x1=2.5, y1=13.0, x2=193.0, y2=12.5, linewidth=3.5)

        with pytest.raises(ValueError, match="requires an Au X-grating image"):
            empty_calgen.get_navigation_calibration(
                line_roi=line, x1=12.0, x2=172.0, n=1, xspace=500.0
            )

    def test_to_ai(self, calgen):
        calgen.get_elliptical_distortion(
            mask_radius=10,
            direct_beam_amplitude=450,
            scale=95,
            amplitude=1200,
            asymmetry=1.5,
            spread=2.8,
            rotation=10,
        )
        calgen.diffraction_calibration = (1, 1)
        ai = calgen.to_ai(wavelength=(2.53 * 10 ** -12))
        assert isinstance(ai, AzimuthalIntegrator)
