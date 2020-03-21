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


@pytest.fixture(params=[
    np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 1., 0., 0., 0., 0., 0.],
              [0., 1., 2., 1., 0., 0., 0., 0.],
              [0., 0., 1., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 1., 0., 0.],
              [0., 0., 0., 0., 1., 2., 1., 0.],
              [0., 0., 0., 0., 0., 1., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0.]])
])
def diffraction_pattern_one_dimension(request):
    """
    1D (in navigation space) diffraction pattern <1|8,8>
    """
    return ElectronDiffraction2D(request.param)


def test_index_coords(diffraction_pattern_one_dimension):
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
    xc, yc = _index_coords(diffraction_pattern_one_dimension.data)
    np.testing.assert_almost_equal(xc, x)
    np.testing.assert_almost_equal(yc, y)


def test_index_coords_non_centeral(diffraction_pattern_one_dimension):
    xc, yc = _index_coords(diffraction_pattern_one_dimension.data, origin=(0, 0))
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
def test_remove_dead_pixels(diffraction_pattern_one_dimension, method):
    z = diffraction_pattern_one_dimension.data
    dead_removed = remove_dead(z, [[3, 3]], deadvalue=method)
    assert z[3, 3] != dead_removed[3, 3]

def test_investigate_dog_background_removal_interactive(diffraction_pattern_one_dimension):
    """ Test that this function runs without error """
    z = diffraction_pattern_one_dimension
    sigma_max_list = np.arange(10, 20, 4)
    sigma_min_list = np.arange(5, 15, 6)
    investigate_dog_background_removal_interactive(z, sigma_max_list, sigma_min_list)
    plt.close('all')
    assert True


class TestReprojectPolar:

    def test_reproject_polar(self, diffraction_pattern_one_dimension):
        z = diffraction_pattern_one_dimension.data
        polar = reproject_polar(z)
        correct_answer = np.array([[-4.76647043e-03, -9.53082887e-04,  2.29168886e-03,
                                    1.57668292e-04, -2.25377803e-04, -1.16148765e-04,
                                    -2.67402071e-03,  1.23632508e-03],
                                   [ 1.72265615e-01, -6.70150422e-02,  4.95300096e-01,
                                     1.51786794e+00,  3.74089740e-01, -6.61576106e-02,
                                     2.66020741e-01,  1.49057863e+00],
                                   [-3.31084317e-02,  2.00822510e-02, -8.11515633e-02,
                                     1.88403514e+00, -9.43454824e-02,  2.62205621e-02,
                                    -7.07187367e-02,  1.82428137e+00],
                                    [-1.66709278e-03, -9.04784390e-03, -4.37196718e-02,
                                      4.45232456e-01, -3.20271874e-02, -4.45885236e-03,
                                     -1.72254131e-02,  4.97117268e-01],
                                    [ 0.00000000e+00,  1.32550695e-03,  0.00000000e+00,
                                     -1.07289949e-01,  0.00000000e+00,  3.25581444e-03,
                                     0.00000000e+00, -5.43693326e-02]])
        assert np.allclose(polar, correct_answer)

    def test_reproject_polar_wt_jacobian(self,
                                         diffraction_pattern_one_dimension):
        pass

    def test_reproject_polar_wt_dt(self, diffraction_pattern_one_dimension):
        pass


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
    centers = find_beam_center_interpolate(z, sigma=5, upsample_factor=100, kind=3)
    assert np.allclose(centers, center_expected, atol=0.2)


@pytest.mark.parametrize("center_expected", [(9, 44)])
@pytest.mark.parametrize("sigma", [2])
def test_find_beam_center_interpolate_2(center_expected, sigma):
    """Cover unlikely case when beam is close to the edge"""
    z = np.zeros((50, 50))
    z[5:15, 41:46] = 1
    z = gaussian_filter(z, sigma=sigma)
    centers = find_beam_center_interpolate(z, sigma=5, upsample_factor=100, kind=3)
    assert np.allclose(centers, center_expected, atol=0.2)
