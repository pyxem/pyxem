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
from scipy.ndimage.filters import gaussian_filter
from matplotlib import pyplot as plt

from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D
from pyxem.utils.expt_utils import _index_coords, _cart2polar, _polar2cart, \
    remove_dead, find_beam_offset_cross_correlation, peaks_as_gvectors, \
    investigate_dog_background_removal_interactive, \
    find_beam_center_blur, find_beam_center_interpolate, \
    reproject_polar


def test_index_coords(dp_single):
    x = np.array([[-4., -3., -2., -1., 0., 1., 2., 3.],
                  [-4., -3., -2., -1., 0., 1., 2., 3.],
                  [-4., -3., -2., -1., 0., 1., 2., 3.],
                  [-4., -3., -2., -1., 0., 1., 2., 3.],
                  [-4., -3., -2., -1., 0., 1., 2., 3.],
                  [-4., -3., -2., -1., 0., 1., 2., 3.],
                  [-4., -3., -2., -1., 0., 1., 2., 3.],
                  [-4., -3., -2., -1., 0., 1., 2., 3.]])
    y = np.array([[-4., -4., -4., -4., -4., -4., -4., -4.],
                  [-3., -3., -3., -3., -3., -3., -3., -3.],
                  [-2., -2., -2., -2., -2., -2., -2., -2.],
                  [-1., -1., -1., -1., -1., -1., -1., -1.],
                  [0., 0., 0., 0., 0., 0., 0., 0.],
                  [1., 1., 1., 1., 1., 1., 1., 1.],
                  [2., 2., 2., 2., 2., 2., 2., 2.],
                  [3., 3., 3., 3., 3., 3., 3., 3.]])
    xc, yc = _index_coords(dp_single.data)
    np.testing.assert_almost_equal(xc, x)
    np.testing.assert_almost_equal(yc, y)


def test_index_coords_non_centeral(dp_single):
    xc, yc = _index_coords(dp_single.data, origin=(0, 0))
    assert xc[0, 0] == 0
    assert yc[0, 0] == 0
    assert xc[0, 5] == 5
    assert yc[0, 5] == 0


@pytest.mark.parametrize('x, y, r, theta', [
    (2, 2, 2.8284271247461903, -0.78539816339744828),
    (1, -2, 2.2360679774997898, 1.1071487177940904),
    (-3, 1, 3.1622776601683795, -2.81984209919315),
])
def test_cart2polar(x, y, r, theta):
    rc, thetac = _cart2polar(x=x, y=y)
    np.testing.assert_almost_equal(rc, r)
    np.testing.assert_almost_equal(thetac, theta)


@pytest.mark.parametrize('r, theta, x, y', [
    (2.82842712, -0.78539816, 2, 2),
    (2.2360679774997898, 1.1071487177940904, 1, -2),
    (3.1622776601683795, -2.819842099193151, -3, 1),
])
def test_polar2cart(r, theta, x, y):
    xc, yc = _polar2cart(r=r, theta=theta)
    np.testing.assert_almost_equal(xc, x)
    np.testing.assert_almost_equal(yc, y)


@pytest.mark.parametrize('z, center, calibration, g', [
    (np.array([[100, 100],
               [200, 200],
               [150, -150]]),
     np.array((127.5, 127.5)),
     0.0039,
     np.array([-0.10725, -0.10725])),
])
def test_peaks_as_gvectors(z, center, calibration, g):
    gc = peaks_as_gvectors(z=z, center=center, calibration=calibration)
    np.testing.assert_almost_equal(gc, g)


methods = ['average', 'nan']


@pytest.mark.parametrize('method', methods)
def test_remove_dead_pixels(dp_single, method):
    z = dp_single.data
    dead_removed = remove_dead(z, [[3, 3]], deadvalue=method)
    assert z[3, 3] != dead_removed[3, 3]

def test_dog_background_removal_interactive(dp_single):
    """ Test that this function runs without error """
    z = dp_single
    sigma_max_list = np.arange(10, 20, 4)
    sigma_min_list = np.arange(5, 15, 6)
    investigate_dog_background_removal_interactive(z, sigma_max_list,
                                                   sigma_min_list)
    plt.close('all')
    assert True


class TestReprojectPolar:

    def test_reproject_polar(self, dp_for_azimuthal):
        z = dp_for_azimuthal.data[0]
        polar = reproject_polar(z)
        answer = np.array([[5.02224257, 5.00103741, 5.02364932, 5.00011429,
                            5.02413059, 5.00011429, 5.02364932, 5.00103741],
                           [4.13276419, 3.74648023, 4.15075771, 3.72522511,
                            4.15671923, 3.72522511, 4.15075771, 3.74648023],
                           [3.14792522, 3.00206132, 3.12681647, 2.97425351,
                            3.11902093, 2.97425351, 3.12681647, 3.00206132],
                           [2.10499427, 2.65447878, 2.01885655, 2.70209801,
                            1.99006133, 2.70209801, 2.01885655, 2.65447878],
                           [0.        , 0.2113421 , 0.        , 0.2909519 ,
                            0.        , 0.2909519 , 0.        , 0.2113421 ]])
        assert np.allclose(polar, answer)

    def test_reproject_polar_wt_jacobian(self, dp_for_azimuthal):
        z = dp_for_azimuthal.data[0]
        polar = reproject_polar(z, dt=1)
        answer = np.array([[5.02224257, 5.0102774 , 5.00410641, 5.02413059,
                            5.00410641, 5.0102774 ],
                           [4.13276419, 3.93896298, 3.8137856 , 4.15671923,
                            3.8137856 , 3.93896298],
                           [3.14792522, 3.1880165 , 3.08781756, 3.11902093,
                            3.08781756, 3.1880165 ],
                           [2.10499427, 2.56387025, 2.57541611, 1.99006133,
                            2.57541611, 2.56387025],
                           [0.        , 0.        , 0.49589862, 0.        ,
                            0.49589862, 0.        ]])
        assert np.allclose(polar, answer)

    def test_reproject_polar_wt_dt(self, dp_for_azimuthal):
        z = dp_for_azimuthal.data[0]
        polar = reproject_polar(z, jacobian=True)
        answer = np.array([[3.55126178, 3.53626746, 3.5522565 , 3.53561472,
                            3.55259681, 3.53561472, 3.5522565 , 3.53626746],
                           [6.42907229, 5.82815547, 6.45706363, 5.79509026,
                            6.46633759, 5.79509026, 6.45706363, 5.82815547],
                           [7.56812551, 7.21744492, 7.51737665, 7.15059041,
                            7.49863489, 7.15059041, 7.51737665, 7.21744492],
                           [6.84689632, 8.63419974, 6.56671693, 8.78909041,
                            6.47305495, 8.78909041, 6.56671693, 8.63419974],
                           [0.        , 0.8667603 , 0.        , 1.19325755,
                            0.        , 1.19325755, 0.        , 0.8667603 ]])
        assert np.allclose(polar, answer)


class TestCenteringAlgorithm:

    @pytest.mark.parametrize("shifts_expected", [(0, 0)])
    def test_perfectly_centered_spot(self, shifts_expected):
        z = np.zeros((50, 50))
        z[24:26, 24:26] = 1
        z = gaussian_filter(z, sigma=2, truncate=3)
        shifts = find_beam_offset_cross_correlation(z, 1, 4)
        assert np.allclose(shifts, shifts_expected, atol=0.2)

    @pytest.mark.parametrize("shifts_expected", [(-3.5, +0.5)])
    @pytest.mark.parametrize("sigma", [1, 2, 3])
    def test_single_pixel_spot(self, shifts_expected, sigma):
        z = np.zeros((50, 50))
        z[28, 24] = 1
        z = gaussian_filter(z, sigma=sigma, truncate=3)
        shifts = find_beam_offset_cross_correlation(z, 1, 6)
        assert np.allclose(shifts, shifts_expected, atol=0.2)

    @pytest.mark.parametrize("shifts_expected", [(-4.5, -0.5)])
    def test_broader_starting_square_spot(self, shifts_expected):
        z = np.zeros((50, 50))
        z[28:31, 24:27] = 1
        z = gaussian_filter(z, sigma=2, truncate=3)
        shifts = find_beam_offset_cross_correlation(z, 1, 4)
        assert np.allclose(shifts, shifts_expected, atol=0.2)


@pytest.mark.parametrize("center_expected", [(29, 25)])
@pytest.mark.parametrize("sigma", [1, 2, 3])
def test_find_beam_center_blur(center_expected, sigma):
    z = np.zeros((50, 50))
    z[28:31, 24:27] = 1
    z = gaussian_filter(z, sigma=sigma)
    shifts = find_beam_center_blur(z, 10)
    assert np.allclose(shifts, center_expected, atol=0.2)


@pytest.mark.parametrize("center_expected", [(29.52, 25.97)])
@pytest.mark.parametrize("sigma", [1, 2, 3])
def test_find_beam_center_interpolate_1(center_expected, sigma):
    z = np.zeros((50, 50))
    z[28:31, 24:28] = 1
    z = gaussian_filter(z, sigma=sigma)
    centers = find_beam_center_interpolate(z, sigma=5,
                                           upsample_factor=100, kind=3)
    assert np.allclose(centers, center_expected, atol=0.2)


@pytest.mark.parametrize("center_expected", [(9, 44)])
@pytest.mark.parametrize("sigma", [2])
def test_find_beam_center_interpolate_2(center_expected, sigma):
    """Cover unlikely case when beam is close to the edge"""
    z = np.zeros((50, 50))
    z[5:15, 41:46] = 1
    z = gaussian_filter(z, sigma=sigma)
    centers = find_beam_center_interpolate(z, sigma=5,
                                           upsample_factor=100, kind=3)
    assert np.allclose(centers, center_expected, atol=0.2)
