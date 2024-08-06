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

from pyxem.utils import calibration
from pyxem.utils.calibration import Calibration
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

    def test_set_units(self, calibration):
        calibration(units="k_nm^-1")
        assert calibration.signal.axes_manager[0].units == "k_nm^-1"
        assert calibration.signal.axes_manager[1].units == "k_nm^-1"
        assert calibration.units == ["k_nm^-1", "k_nm^-1"]

    def test_calibrate_gain(self, calibration):
        calibration.signal.data[0, 0] = 1
        calibration.detector_gain = 2
        assert calibration.detector_gain == 2
        assert calibration.signal.data[0, 0] == 0.5

    def test_set_mask(self, calibration):
        calibration(mask=np.ones((10, 10)))
        assert calibration.mask is not None
        assert calibration.mask.shape == (10, 10)

    def test_set_affine(self, calibration):
        assert calibration.affine is None
        calibration(affine=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        np.testing.assert_array_equal(
            calibration.affine, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        )

    def test_set_beam_energy(self, calibration):
        assert calibration.beam_energy is None
        calibration(beam_energy=200)
        assert calibration.beam_energy == 200
        assert calibration.wavelength is not None

    def test_set_wavelength(self, calibration):
        assert calibration.wavelength is None
        calibration(wavelength=0.02508)
        assert calibration.wavelength == 0.02508

    def test_set_scale(self, calibration):
        calibration(scale=0.01)
        assert calibration.signal.axes_manager[0].scale == 0.01
        assert calibration.signal.axes_manager[1].scale == 0.01
        assert calibration.scale == [0.01, 0.01]
        assert calibration.flat_ewald is True

    def test_set_failure(self, calibration):
        assert calibration.wavelength is None
        assert calibration.beam_energy is None
        with pytest.raises(ValueError):
            calibration.detector(pixel_size=0.1, detector_distance=1)
        calibration.beam_energy = 200
        calibration.detector(pixel_size=0.1, detector_distance=1)
        # The center, in pixel coordinates, is (4.5, 4.5)
        # When using a detector, this gets rounded down to 4
        assert calibration.center == [4, 4]
        calibration.detector(
            pixel_size=0.1, detector_distance=1, beam_energy=200, units="k_nm^-1"
        )
        assert calibration.flat_ewald is False
        assert calibration.center == [4, 4]
        with pytest.raises(ValueError):
            calibration(scale=0.01)
        assert calibration.scale is None
        with pytest.raises(ValueError):
            calibration(center=(5, 5))
        assert calibration.center == [4, 4]

        with pytest.raises(ValueError):
            calibration.detector(pixel_size=0.1, detector_distance=1, units="nm^-1")

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
        s.calibration(scale=0.1, center=None)
        slices, factors, factor_slices = s.calibration._get_slices_and_factors(
            npt=100, npt_azim=360, radial_range=(0, 4), azimuthal_range=(0, 2 * np.pi)
        )
        # check that the number of pixels for each radial slice is the same
        sum_factors = [np.sum(factors[f[0] : f[1]]) for f in factor_slices]
        sum_factors = np.reshape(sum_factors, (360, 100))
        for row in sum_factors:
            print(np.min(row), np.max(row))
            assert np.allclose(row, row[0], atol=1e-2)
        # Check that the total number of pixels accounted for is equal to the area of the circle
        # Up to rounding due to the fact that we are actually finding the area of an n-gon where
        # n = npt_azim
        all_sum = np.sum(sum_factors)
        assert np.allclose(all_sum, 3.1415 * 40**2, atol=1)
        slices, factors, factor_slices = s.calibration._get_slices_and_factors(
            npt=100, npt_azim=360, radial_range=(0, 15), azimuthal_range=(0, 2 * np.pi)
        )
        # check that the number of pixels for each radial slice is the same
        sum_factors = [np.sum(factors[f[0] : f[1]]) for f in factor_slices]
        sum_factors = np.reshape(sum_factors, (360, 100))
        # Check that the total number of pixels accounted for is equal to the area of the circle
        # Up to rounding due to the fact that we are actually finding the area of an n-gon where
        # n = npt_azim
        all_sum = np.sum(sum_factors)
        assert np.allclose(all_sum, 10000, atol=1)

    def test_get_slices_and_factors1d(self):
        s = Diffraction2D(np.zeros((100, 100)))
        s.calibration(scale=0.1, center=None)
        slices, factors, factor_slices, _ = s.calibration.get_slices1d(
            100, radial_range=(0, 4)
        )
        # check that the number of pixels for each radial slice is the same
        for i in range(len(factor_slices) - 1):
            sl = factors[factor_slices[i] : factor_slices[i + 1]]
            print(np.sum(sl))
        np.testing.assert_almost_equal(np.sum(factors), 3.1415 * 40**2, decimal=0)

    def test_to_string(self, calibration):
        assert (
            str(calibration)
            == "Calibration for <Diffraction2D, title: , dimensions: (|10, 10)>, "
            "Ewald sphere: flat, shape: (10, 10), affine: False, mask: False"
        )
        calibration.detector(
            pixel_size=15e-6, detector_distance=3.8e-2, beam_energy=200, units="k_nm^-1"
        )
        assert (
            str(calibration)
            == "Calibration for <Diffraction2D, title: , dimensions: (|10, 10)>, "
            "Ewald sphere: curved, shape: (10, 10), affine: False, mask: False"
        )
