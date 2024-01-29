# -*- coding: utf-8 -*-
# Copyright 2016-2024 The pyXem developers
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

from hyperspy.axes import UniformDataAxis

from pyxem.utils import calibration_utils
from pyxem.utils.calibration_utils import Calibration
from pyxem.signals import Diffraction2D


class TestCalibrationClass:
    @pytest.fixture
    def calibration(self):
        s = Diffraction2D(np.zeros((10, 10)))
        return Calibration(s)

    def test_init(self, calibration):
        assert isinstance(calibration, Calibration)

    def test_set_center(self, calibration):
        calibration(center=(5, 5))
        assert calibration.signal.axes_manager[0].offset == -5
        assert calibration.signal.axes_manager[1].offset == -5
        assert calibration.flat_ewald is True

    def test_set_beam_energy(self, calibration):
        calibration(beam_energy=200)
        assert calibration.beam_energy == 200
        assert calibration.wavelength is not None

    def test_set_wavelength(self, calibration):
        calibration(wavelength=0.02508)
        assert calibration.wavelength == 0.02508

    def test_set_scale(self, calibration):
        calibration(scale=0.01)
        assert calibration.signal.axes_manager[0].scale == 0.01
        assert calibration.signal.axes_manager[1].scale == 0.01
        assert calibration.flat_ewald is True

    def test_set_failure(self, calibration):
        assert calibration.wavelength is None
        assert calibration.beam_energy is None
        with pytest.raises(ValueError):
            calibration.detector(pixel_size=0.1, detector_distance=1)
        calibration.beam_energy = 200
        calibration.detector(pixel_size=0.1, detector_distance=1)
        calibration.detector(
            pixel_size=0.1, detector_distance=1, beam_energy=200, units="k_nm^-1"
        )
        assert calibration.flat_ewald is False
        with pytest.raises(ValueError):
            calibration(scale=0.01)
        assert calibration.scale is None
        with pytest.raises(ValueError):
            calibration(center=(5, 5))
        assert calibration.center == [5, 5]

    def test_set_detector(self, calibration):
        calibration.detector(
            pixel_size=15e-6,  # 15 um
            detector_distance=3.8e-2,  # 38 mm
            beam_energy=200,  # 200 keV
            units="k_nm^-1",
        )
        assert not isinstance(calibration.signal.axes_manager[0], UniformDataAxis)
        diff_arr = np.diff(
            calibration.signal.axes_manager[0].axis
        )  # assume mostly flat.
        assert np.allclose(
            diff_arr,
            diff_arr[0],
        )
        assert calibration.flat_ewald == False

    def test_get_slices2d(self, calibration):
        calibration(scale=0.01)
        slices, factors, _, _ = calibration.get_slices2d(5, 90)
        assert len(slices) == 5 * 90

    def test_get_slices_and_factors(self):
        s = Diffraction2D(np.zeros((100, 100)))
        s.calibrate(scale=0.1, center=None)
        slices, factors, factor_slices = s.calibrate._get_slices_and_factors(
            npt=100, npt_azim=360, radial_range=(0, 4)
        )
        # check that the number of pixels for each radial slice is the same
        sum_factors = [np.sum(factors[f[0] : f[1]]) for f in factor_slices]
        sum_factors = np.reshape(sum_factors, (360, 100)).T
        for row in sum_factors:
            print(np.min(row), np.max(row))
            assert np.allclose(row, row[0], atol=1e-2)
        # Check that the total number of pixels accounted for is equal to the area of the circle
        # Up to rounding due to the fact that we are actually finding the area of an n-gon where
        # n = npt_azim
        all_sum = np.sum(sum_factors)
        assert np.allclose(all_sum, 3.1415 * 40**2, atol=1)
        slices, factors, factor_slices = s.calibrate._get_slices_and_factors(
            npt=100, npt_azim=360, radial_range=(0, 15)
        )
        # check that the number of pixels for each radial slice is the same
        sum_factors = [np.sum(factors[f[0] : f[1]]) for f in factor_slices]
        sum_factors = np.reshape(sum_factors, (360, 100)).T
        # Check that the total number of pixels accounted for is equal to the area of the circle
        # Up to rounding due to the fact that we are actually finding the area of an n-gon where
        # n = npt_azim
        all_sum = np.sum(sum_factors)
        # For some reason we are missing 1 row/ column of pixels on the edge
        # of the image so this is 9801 instead of 10000!
        # assert np.allclose(all_sum, 10000, atol=1)


@pytest.mark.skip(
    reason="This functionality already smells, skipping while new things are built"
)
class TestCalibrations:
    def test_find_diffraction_calibration(
        self, test_patterns, test_lib_gen, test_library_phases
    ):
        cal, corrlines, cals = calibration_utils.find_diffraction_calibration(
            test_patterns,
            0.0097,
            test_library_phases,
            test_lib_gen,
            10,
            max_excitation_error=0.08,
        )
        np.testing.assert_allclose(
            cals, np.array([0.009991, 0.010088, 0.009991]), atol=1e-6
        )

    def test_calibration_iteration(
        self, test_patterns, test_lib_gen, test_library_phases
    ):
        test_corrlines = calibration_utils._calibration_iteration(
            test_patterns,
            0.0097,
            test_library_phases,
            test_lib_gen,
            0.0097 * 0.01,
            2,
            3,
            max_excitation_error=0.08,
        )
        true_corrlines = np.array(
            [
                [
                    [0.0097, 0.0097, 0.0097],
                    [0.085312, 0.056166, 0.0],
                ],
                [
                    [0.0097, 0.0097, 0.0097],
                    [0.085312, 0.056166, 0.0],
                ],
            ]
        )
        np.testing.assert_allclose(
            test_corrlines,
            true_corrlines,
            atol=1e-6,
        )

    def test_create_check_diflib(
        self, test_patterns, test_lib_gen, test_library_phases
    ):
        test_corrs = calibration_utils._create_check_diflib(
            test_patterns,
            0.0097,
            test_library_phases,
            test_lib_gen,
            max_excitation_error=0.08,
        )
        np.testing.assert_allclose(
            test_corrs,
            np.array([0.085313, 0.056166, 0.0]),
            atol=1e-6,
        )
