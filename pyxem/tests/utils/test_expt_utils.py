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
from scipy.ndimage.filters import gaussian_filter
from matplotlib import pyplot as plt

from pyFAI.detectors import Detector

from pyxem.utils.expt_utils import (
    _index_coords,
    _cart2polar,
    _polar2cart,
    remove_dead,
    find_beam_offset_cross_correlation,
    peaks_as_gvectors,
    investigate_dog_background_removal_interactive,
    find_beam_center_blur,
    find_beam_center_interpolate,
    azimuthal_integrate1d,
    azimuthal_integrate2d,
)


def test_index_coords(dp_single):
    x = np.array(
        [
            [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        ]
    )
    y = np.array(
        [
            [-4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0],
            [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0],
            [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        ]
    )
    xc, yc = _index_coords(dp_single.data)
    np.testing.assert_almost_equal(xc, x)
    np.testing.assert_almost_equal(yc, y)


def test_index_coords_non_centeral(dp_single):
    xc, yc = _index_coords(dp_single.data, origin=(0, 0))
    assert xc[0, 0] == 0
    assert yc[0, 0] == 0
    assert xc[0, 5] == 5
    assert yc[0, 5] == 0


@pytest.mark.parametrize(
    "x, y, r, theta",
    [
        (2, 2, 2.8284271247461903, -0.78539816339744828),
        (1, -2, 2.2360679774997898, 1.1071487177940904),
        (-3, 1, 3.1622776601683795, -2.81984209919315),
    ],
)
def test_cart2polar(x, y, r, theta):
    rc, thetac = _cart2polar(x=x, y=y)
    np.testing.assert_almost_equal(rc, r)
    np.testing.assert_almost_equal(thetac, theta)


@pytest.mark.parametrize(
    "r, theta, x, y",
    [
        (2.82842712, -0.78539816, 2, 2),
        (2.2360679774997898, 1.1071487177940904, 1, -2),
        (3.1622776601683795, -2.819842099193151, -3, 1),
    ],
)
def test_polar2cart(r, theta, x, y):
    xc, yc = _polar2cart(r=r, theta=theta)
    np.testing.assert_almost_equal(xc, x)
    np.testing.assert_almost_equal(yc, y)


@pytest.mark.parametrize(
    "z, center, calibration, g",
    [
        (
            np.array([[100, 100], [200, 200], [150, -150]]),
            np.array((127.5, 127.5)),
            0.0039,
            np.array([-0.10725, -0.10725]),
        ),
    ],
)
def test_peaks_as_gvectors(z, center, calibration, g):
    gc = peaks_as_gvectors(z=z, center=center, calibration=calibration)
    np.testing.assert_almost_equal(gc, g)


def test_remove_dead_pixels(dp_single):
    z = dp_single.data
    dead_removed = remove_dead(z, [[3, 3]])
    assert z[3, 3] != dead_removed[3, 3]


def test_dog_background_removal_interactive(dp_single):
    """ Test that this function runs without error """
    z = dp_single
    sigma_max_list = np.arange(10, 20, 4)
    sigma_min_list = np.arange(5, 15, 6)
    investigate_dog_background_removal_interactive(z, sigma_max_list, sigma_min_list)
    plt.close("all")
    assert True


class TestCenteringAlgorithm:
    @pytest.mark.parametrize("shifts_expected", [(0, 0)])
    def test_perfectly_centered_spot(self, shifts_expected):
        z = np.zeros((50, 50))
        z[24:26, 24:26] = 1
        z = gaussian_filter(z, sigma=2, truncate=3)
        shifts = find_beam_offset_cross_correlation(z, 1, 4)
        assert np.allclose(shifts, shifts_expected, atol=0.2)

    @pytest.mark.parametrize("shifts_expected", [(+0.5, -3.5)])
    @pytest.mark.parametrize("sigma", [1, 2, 3])
    def test_single_pixel_spot(self, shifts_expected, sigma):
        z = np.zeros((50, 50))
        z[28, 24] = 1
        z = gaussian_filter(z, sigma=sigma, truncate=3)
        shifts = find_beam_offset_cross_correlation(z, 1, 6)
        assert np.allclose(shifts, shifts_expected, atol=0.2)

    @pytest.mark.parametrize("shifts_expected", [(-0.5, -4.5)])
    def test_broader_starting_square_spot(self, shifts_expected):
        z = np.zeros((50, 50))
        z[28:31, 24:27] = 1
        z = gaussian_filter(z, sigma=2, truncate=3)
        shifts = find_beam_offset_cross_correlation(z, 1, 4)
        assert np.allclose(shifts, shifts_expected, atol=0.2)


@pytest.mark.parametrize("center_expected", [(25, 29)])
@pytest.mark.parametrize("sigma", [1, 2, 3])
def test_find_beam_center_blur(center_expected, sigma):
    z = np.zeros((50, 50))
    z[28:31, 24:27] = 1
    z = gaussian_filter(z, sigma=sigma)
    shifts = find_beam_center_blur(z, 10)
    assert np.allclose(shifts, center_expected, atol=0.2)


@pytest.mark.parametrize("center_expected", [(25.97, 29.52)])
@pytest.mark.parametrize("sigma", [1, 2, 3])
def test_find_beam_center_interpolate_1(center_expected, sigma):
    z = np.zeros((50, 50))
    z[28:31, 24:28] = 1
    z = gaussian_filter(z, sigma=sigma)
    centers = find_beam_center_interpolate(z, sigma=5, upsample_factor=100, kind=3)
    assert np.allclose(centers, center_expected, atol=0.2)


@pytest.mark.parametrize("center_expected", [(44, 9)])
@pytest.mark.parametrize("sigma", [2])
def test_find_beam_center_interpolate_2(center_expected, sigma):
    """Cover unlikely case when beam is close to the edge"""
    z = np.zeros((50, 50))
    z[5:15, 41:46] = 1
    z = gaussian_filter(z, sigma=sigma)
    centers = find_beam_center_interpolate(z, sigma=5, upsample_factor=100, kind=3)
    assert np.allclose(centers, center_expected, atol=0.2)


class TestAzimuthalIntegration:
    @pytest.fixture
    def radial_pattern(self):
        x, y = np.ogrid[-5:5, -5:5]
        radial = (x ** 2 + y ** 2) * np.pi
        radial[radial == 0] = 1
        return 100 / radial

    def test_1d_integrate(self, radial_pattern):
        from pyxem.utils.pyfai_utils import get_azimuthal_integrator

        dect = Detector(pixel1=1e-4, pixel2=1e-4)
        ai = get_azimuthal_integrator(
            detector=dect, detector_distance=1, shape=np.shape(radial_pattern)
        )
        integration = azimuthal_integrate1d(
            radial_pattern,
            ai,
            npt_rad=100,
            method="numpy",
            unit="2th_rad",
            correctSolidAngle=True,
        )

    def test_2d_integrate(self, radial_pattern):
        from pyxem.utils.pyfai_utils import get_azimuthal_integrator

        dect = Detector(pixel1=1e-4, pixel2=1e-4)
        ai = get_azimuthal_integrator(
            detector=dect, detector_distance=1, shape=np.shape(radial_pattern)
        )
        integration = azimuthal_integrate2d(
            radial_pattern,
            ai,
            npt_rad=100,
            npt_azim=100,
            method="numpy",
            unit="2th_rad",
            correctSolidAngle=True,
        )
