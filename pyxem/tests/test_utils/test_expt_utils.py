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
from scipy.ndimage.filters import gaussian_filter

from pyxem.signals.electron_diffraction import ElectronDiffraction
from pyxem.utils.expt_utils import _index_coords, _cart2polar, _polar2cart, \
    radial_average, gain_normalise, remove_dead, affine_transformation, \
    regional_filter, subtract_background_dog, subtract_background_median, \
    subtract_reference, circular_mask, reference_circle, \
    find_beam_offset_cross_correlation, peaks_as_gvectors


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
    return ElectronDiffraction(request.param)


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
