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

from pytest import approx, mark
import math
import numpy as np
from numpy.testing import assert_allclose
from scipy.signal import convolve2d
from skimage import morphology

from pyxem.dummy_data import make_diffraction_test_data as mdtd
from pyxem.signals import Diffraction2D
import pyxem.utils.ransac_ellipse_tools as ret


class TestIsEllipseGood:
    def test_simple(self):
        params = ret._make_ellipse_model_params_focus(30, 50, 30, 20, 0)
        model = ret.EllipseModel()
        model.params = params
        is_good = ret.is_ellipse_good(
            ellipse_model=model, data=None, xf=30, yf=50, rf_lim=5
        )
        assert is_good

    @mark.parametrize("xf,out", [(40, True), (46, False), (34, False)])
    def test_xf(self, xf, out):
        model = ret.EllipseModel()
        model.params = ret._make_ellipse_model_params_focus(40, 50, 10, 30, 0)
        is_good = ret.is_ellipse_good(
            ellipse_model=model, data=None, xf=xf, yf=50, rf_lim=5
        )
        assert is_good == out

    @mark.parametrize("yf,out", [(20, False), (10, True), (-9, False)])
    def test_yf(self, yf, out):
        model = ret.EllipseModel()
        model.params = ret._make_ellipse_model_params_focus(70, 10, 10, 30, 0)
        is_good = ret.is_ellipse_good(
            ellipse_model=model, data=None, xf=70, yf=yf, rf_lim=5
        )
        assert is_good == out

    @mark.parametrize("rf_lim,out", [(2, False), (4, True), (3.5, True)])
    def test_rf_lim(self, rf_lim, out):
        model = ret.EllipseModel()
        model.params = ret._make_ellipse_model_params_focus(47, 30, 10, 30, 0)
        is_good = ret.is_ellipse_good(
            ellipse_model=model, data=None, xf=50, yf=30, rf_lim=rf_lim
        )
        assert is_good == out

    @mark.parametrize("semi_len_min,out", [(4, True), (7, False), (6.5, True)])
    def test_semi_len_min(self, semi_len_min, out):
        model = ret.EllipseModel()
        model.params = ret._make_ellipse_model_params_focus(50, 30, 6.6, 10, 0)
        is_good = ret.is_ellipse_good(
            ellipse_model=model,
            data=None,
            xf=50,
            yf=30,
            rf_lim=5,
            semi_len_min=semi_len_min,
        )
        assert is_good == out

    @mark.parametrize(
        "a,b,semi_len_max,out",
        [
            (3, 6.5, 6.6, True),
            (3, 6.5, 4, False),
            (3, 6.5, 9, True),
            (6.5, 3, 4, False),
        ],
    )
    def test_semi_len_max(self, a, b, semi_len_max, out):
        model = ret.EllipseModel()
        model.params = ret._make_ellipse_model_params_focus(50, 30, a, b, 0)
        params = list(model.params)
        # Changing the params like this is necessary to test if b is larger
        # than semi_len_max. This is due to the content in model.params not being
        # the same as the input in ret._make_ellipse_model_params_focus.
        params[2:5] = a, b, 0.0
        model.params = tuple(params)
        is_good = ret.is_ellipse_good(
            ellipse_model=model,
            data=None,
            xf=50,
            yf=30,
            rf_lim=5,
            semi_len_max=semi_len_max,
        )
        assert is_good == out

    @mark.parametrize("semi_len_ratio_lim,out", [(2.1, True), (1.5, False), (3, True)])
    def test_semi_len_ratio_lim(self, semi_len_ratio_lim, out):
        model = ret.EllipseModel()
        model.params = ret._make_ellipse_model_params_focus(50, 30, 9, 4.5, 0)
        is_good = ret.is_ellipse_good(
            ellipse_model=model,
            data=None,
            xf=50,
            yf=30,
            rf_lim=5,
            semi_len_ratio_lim=semi_len_ratio_lim,
        )
        assert is_good == out

    @mark.parametrize("r", [0, 0.5, 0.9, 1.3, np.pi, 2 * np.pi, 9.9])
    def test_different_rotations(self, r):
        model = ret.EllipseModel()
        model.params = ret._make_ellipse_model_params_focus(50, 30, 3, 6.5, r)
        is_good = ret.is_ellipse_good(
            ellipse_model=model, data=None, xf=50, yf=30, rf_lim=5
        )
        assert is_good

    @mark.parametrize(
        "x, y, r, smin, smax, srl, out",
        [
            (5, 3, 1, 2, 7, 2, True),
            (3, 3, 1, 2, 7, 2, False),
            (5, 5, 1, 2, 7, 2, False),
            (5, 2, 3, 2, 7, 2, True),
            (5, 2, 3, 5, 7, 2, False),
            (5, 2, 3, 5, 6, 2, False),
            (5, 3, 1, 3, 7, 1.5, False),
            (1, 9, 1, 5, 6, 1.1, False),
        ],
    )
    def test_many_different_ones(self, x, y, r, smin, smax, srl, out):
        model = ret.EllipseModel()
        model.params = ret._make_ellipse_model_params_focus(5, 3, 4, 6.5, 0.1)
        is_good = ret.is_ellipse_good(
            ellipse_model=model,
            data=None,
            xf=x,
            yf=y,
            rf_lim=r,
            semi_len_min=smin,
            semi_len_max=smax,
            semi_len_ratio_lim=srl,
        )
        assert is_good == out


class TestMakeEllipseDataPoints:
    def test_simple(self):
        data = ret.make_ellipse_data_points(5, 2, 9, 5, 0)
        assert data.size > 0

    def test_nt(self):
        data0 = ret.make_ellipse_data_points(5, 2, 9, 5, 0, nt=10)
        assert data0.shape == (10, 2)
        data1 = ret.make_ellipse_data_points(5, 2, 9, 5, 0, nt=29)
        assert data1.shape == (29, 2)

    def test_use_focus(self):
        x, y, a, b, r = 5, 9, 10, 8, 0
        data0 = ret.make_ellipse_data_points(x, y, a, b, r, nt=99, use_focus=False)
        assert np.allclose(data0.mean(axis=0), (x, y))
        data1 = ret.make_ellipse_data_points(x, y, a, b, r, nt=99, use_focus=True)
        assert not np.allclose(data1.mean(axis=0), (x, y))

    def test_xy_use_focus_false(self):
        x0, y0, x1, y1 = 10, -5, -20, 30
        a, b, r = 10, 8, 0
        data0 = ret.make_ellipse_data_points(x0, y0, a, b, r, nt=99, use_focus=False)
        assert np.allclose(data0.mean(axis=0), (x0, y0))
        data1 = ret.make_ellipse_data_points(x1, y1, a, b, r, nt=99, use_focus=False)
        assert np.allclose(data1.mean(axis=0), (x1, y1))

    def test_xy_use_focus_true(self):
        x0, y0, x1, y1 = 10, -5, -20, 30
        a, b, r = 10, 8, 0
        data0 = ret.make_ellipse_data_points(x0, y0, a, b, r, nt=99, use_focus=True)
        xc0, yc0 = data0.mean(axis=0)
        f0 = ret._get_closest_focus(x0, y0, xc0, yc0, a, b, r)
        assert approx(f0) == (x0, y0)
        data1 = ret.make_ellipse_data_points(x1, y1, a, b, r, nt=99, use_focus=True)
        xc1, yc1 = data1.mean(axis=0)
        f1 = ret._get_closest_focus(x1, y1, xc1, yc1, a, b, r)
        assert approx(f1) == (x1, y1)

    def test_ab(self):
        x, y, r, nt = 10, 20, 0, 9999
        a0, b0, a1, b1 = 10, 5, 8, 15
        data0 = ret.make_ellipse_data_points(x, y, a0, b0, r, nt=nt, use_focus=False)
        assert np.allclose(data0.min(axis=0), (x - a0, y - b0), atol=10e-5)
        data1 = ret.make_ellipse_data_points(x, y, a1, b1, r, nt=nt, use_focus=False)
        assert np.allclose(data1.min(axis=0), (x - a1, y - b1), atol=10e-5)

    def test_r(self):
        x, y, a, b, nt = 10, 20, 15, 5, 9999
        r0, r1 = 0, math.pi / 2
        data0 = ret.make_ellipse_data_points(x, y, a, b, r0, nt=nt, use_focus=False)
        assert np.allclose(data0.min(axis=0), (x - a, y - b), atol=10e-5)
        data1 = ret.make_ellipse_data_points(x, y, a, b, r1, nt=nt, use_focus=False)
        assert np.allclose(data1.min(axis=0), (x - b, y - a), atol=10e-5)


class TestGetClosestFocus:
    def test_circle(self):
        xc, yc = 10, 20
        x, y, a, b, r = 10, 20, 10, 10, 0
        xf, yf = ret._get_closest_focus(xc, yc, x, y, a, b, r)
        assert (xf, yf) == (10, 20)

    def test_horizontal_ellipse(self):
        x, y, a, b, r = 10, 20, 20, 10, 0
        c = math.sqrt(a ** 2 - b ** 2)
        xf0, yf0 = ret._get_closest_focus(20, 20, x, y, a, b, r)
        assert (xf0, yf0) == (x + c, y)
        xf1, yf1 = ret._get_closest_focus(5, 20, x, y, a, b, r)
        assert (xf1, yf1) == (x - c, y)

    def test_vertical_ellipse(self):
        x, y, a, b, r = 10, 20, 15, 10, math.pi / 2
        c = math.sqrt(a ** 2 - b ** 2)
        xf0, yf0 = ret._get_closest_focus(10, 30, x, y, a, b, r)
        assert (xf0, yf0) == (x, y + c)
        xf1, yf1 = ret._get_closest_focus(10, 10, x, y, a, b, r)
        assert (xf1, yf1) == (x, y - c)


class TestEllipseCentreToFocus:
    def test_circle(self):
        x, y, a, b, r = 10, 20, 10, 10, 0
        f0, f1 = ret._ellipse_centre_to_focus(x, y, a, b, r)
        assert f0 == (x, y)
        assert f1 == (x, y)

    def test_horizontal_ellipse(self):
        x, y, a, b, r = 10, 20, 20, 10, 0
        f0, f1 = ret._ellipse_centre_to_focus(x, y, a, b, r)
        c = math.sqrt(a ** 2 - b ** 2)
        assert f0 == (x + c, y)
        assert f1 == (x - c, y)

    def test_vertical_ellipse(self):
        x, y, a, b, r = 10, 20, 10, 5, math.pi / 2
        f0, f1 = ret._ellipse_centre_to_focus(x, y, a, b, r)
        c = math.sqrt(a ** 2 - b ** 2)
        assert f0 == (x, y + c)
        assert f1 == (x, y - c)

    def test_rotated45_ellipse(self):
        x, y, a, b, r = 10, 20, 10, 5, math.pi / 4
        f0, f1 = ret._ellipse_centre_to_focus(x, y, a, b, r)
        c = math.sqrt(a ** 2 - b ** 2)
        assert approx(f0) == (x + c * math.sin(r), y + c * math.cos(r))
        assert approx(f1) == (x - c * math.sin(r), y - c * math.cos(r))

    def test_rotated_negative45_ellipse(self):
        x, y, a, b, r = 10, 20, 10, 5, -math.pi / 4
        f0, f1 = ret._ellipse_centre_to_focus(x, y, a, b, r)
        c = math.sqrt(a ** 2 - b ** 2)
        assert approx(f0) == (x - c * math.sin(r), y - c * math.cos(r))
        assert approx(f1) == (x + c * math.sin(r), y + c * math.cos(r))

    def test_horizontal_negative(self):
        x, y, a, b, r = 5, 20, 20, 10, 0
        f0, f1 = ret._ellipse_centre_to_focus(x, y, a, b, r)
        c = math.sqrt(a ** 2 - b ** 2)
        assert approx(f0) == (x + c, y)
        assert approx(f1) == (x - c, y)

    def test_vertical_negative(self):
        x, y, a, b, r = 10, 5, 10, 20, 0
        f0, f1 = ret._ellipse_centre_to_focus(x, y, a, b, r)
        c = math.sqrt(b ** 2 - a ** 2)
        assert approx(f0) == (x, y + c)
        assert approx(f1) == (x, y - c)


def compare_model_params(params0, params1, rel=None, abs=None):
    xf0, yf0, a0, b0, r0 = params0
    xf1, yf1, a1, b1, r1 = params1
    # only case in the params
    if a1 < b1:
        r1 += math.pi / 2
        a1, b1 = b1, a1
    r0 = r0 % math.pi
    r1 = r1 % math.pi
    assert approx((xf0, yf0), rel=rel, abs=abs) == (xf1, yf1)
    assert approx((a0, b0), rel=rel, abs=abs) == (a1, b1)
    assert approx(r0, rel=rel, abs=abs) == r1


class TestGetEllipseModelRansacSingleFrame:
    def test_simple(self):
        data = ret.make_ellipse_data_points(50, 55, 20, 16, 2, nt=15)
        ret.get_ellipse_model_ransac_single_frame(data)

    def test_min_samples(self):
        xf, yf, a, b, r = 50, 55, 21, 20, 0
        data = ret.make_ellipse_data_points(xf, yf, a, b, r, nt=20)
        data = np.vstack((data, [70, 55]))
        model_ransac0, inliers0 = ret.get_ellipse_model_ransac_single_frame(
            data,
            xf=50,
            yf=55,
            rf_lim=5,
            semi_len_min=15,
            semi_len_max=None,
            semi_len_ratio_lim=None,
            min_samples=20,
            residual_threshold=3,
            max_trails=100,
        )
        assert inliers0[:-1].all()
        assert not inliers0[-1]
        model_ransac1, inliers1 = ret.get_ellipse_model_ransac_single_frame(
            data,
            xf=50,
            yf=55,
            rf_lim=5,
            semi_len_min=15,
            semi_len_max=None,
            semi_len_ratio_lim=None,
            min_samples=20,
            residual_threshold=6,
            max_trails=100,
        )
        assert inliers1.all()

    def test_min_samples_smaller_than_data(self):
        data = ret.make_ellipse_data_points(50, 55, 20, 16, 2, nt=15)
        ret.get_ellipse_model_ransac_single_frame(data, min_samples=25)

    def test_all_inliers(self):
        xf, yf, a, b, r = 50, 55, 20, 16, 2
        data = ret.make_ellipse_data_points(xf, yf, a, b, r, nt=25)
        model_ransac, inliers = ret.get_ellipse_model_ransac_single_frame(
            data,
            xf=50,
            yf=55,
            rf_lim=2,
            semi_len_min=10,
            semi_len_max=25,
            semi_len_ratio_lim=2,
            min_samples=6,
            residual_threshold=10,
            max_trails=100,
        )
        params0f = (xf, yf, a, b, r)
        params1f = ret._ellipse_model_centre_to_focus(*model_ransac.params, xf, yf)
        assert inliers.all()
        compare_model_params(params0f, params1f, abs=0.01)

    def test_xf(self):
        xf0, xf1, yf, a, b, r = 50, 120, 55, 21, 20, 2
        data0 = ret.make_ellipse_data_points(xf0, yf, a, b, r, nt=25)
        data1 = ret.make_ellipse_data_points(xf1, yf, a, b, r, nt=25)
        data = np.vstack((data0, data1))
        model_ransac0, inliers0 = ret.get_ellipse_model_ransac_single_frame(
            data,
            xf=xf0,
            yf=55,
            rf_lim=2,
            semi_len_min=19,
            semi_len_max=None,
            semi_len_ratio_lim=2,
            min_samples=5,
            residual_threshold=10,
            max_trails=300,
        )
        params00f = (xf0, yf, a, b, r)
        params01f = ret._ellipse_model_centre_to_focus(*model_ransac0.params, xf0, yf)
        compare_model_params(params00f, params01f, abs=0.1)

        model_ransac1, inliers1 = ret.get_ellipse_model_ransac_single_frame(
            data,
            xf=xf1,
            yf=55,
            rf_lim=2,
            semi_len_min=19,
            semi_len_max=None,
            semi_len_ratio_lim=2,
            min_samples=5,
            residual_threshold=10,
            max_trails=300,
        )
        params10f = (xf1, yf, a, b, r)
        params11f = ret._ellipse_model_centre_to_focus(*model_ransac1.params, xf1, yf)
        compare_model_params(params10f, params11f, abs=0.1)

    def test_yf(self):
        xf, yf0, yf1, a, b, r = 50, 55, 140, 21, 20, 2
        data0 = ret.make_ellipse_data_points(xf, yf0, a, b, r, nt=25)
        data1 = ret.make_ellipse_data_points(xf, yf1, a, b, r, nt=25)
        data = np.vstack((data0, data1))
        model_ransac0, inliers0 = ret.get_ellipse_model_ransac_single_frame(
            data,
            xf=50,
            yf=yf0,
            rf_lim=2,
            semi_len_min=19,
            semi_len_max=None,
            semi_len_ratio_lim=2,
            min_samples=5,
            residual_threshold=10,
            max_trails=300,
        )
        params00f = (xf, yf0, a, b, r)
        params01f = ret._ellipse_model_centre_to_focus(*model_ransac0.params, xf, yf0)
        compare_model_params(params00f, params01f, abs=0.1)

        model_ransac1, inliers1 = ret.get_ellipse_model_ransac_single_frame(
            data,
            xf=50,
            yf=yf1,
            rf_lim=2,
            semi_len_min=19,
            semi_len_max=None,
            semi_len_ratio_lim=2,
            min_samples=5,
            residual_threshold=10,
            max_trails=300,
        )
        params10f = (xf, yf1, a, b, r)
        params11f = ret._ellipse_model_centre_to_focus(*model_ransac1.params, xf, yf1)
        compare_model_params(params10f, params11f, abs=0.1)

    def test_rf_lim(self):
        xf, yf, a, b, r = 50, 55, 21, 20, 2
        data = ret.make_ellipse_data_points(xf, yf, a, b, r, nt=25)
        model_ransac0, inliers0 = ret.get_ellipse_model_ransac_single_frame(
            data,
            xf=50,
            yf=50,
            rf_lim=6,
            semi_len_min=15,
            semi_len_max=23,
            semi_len_ratio_lim=1.2,
            min_samples=5,
            residual_threshold=10,
            max_trails=100,
        )
        params0f = (xf, yf, a, b, r)
        params1f = ret._ellipse_model_centre_to_focus(*model_ransac0.params, xf, yf)
        assert inliers0.all()
        compare_model_params(params0f, params1f, abs=0.1)

        model_ransac1, inliers1 = ret.get_ellipse_model_ransac_single_frame(
            data,
            xf=50,
            yf=50,
            rf_lim=3,
            semi_len_min=15,
            semi_len_max=23,
            semi_len_ratio_lim=1.2,
            min_samples=5,
            residual_threshold=10,
            max_trails=100,
        )
        assert model_ransac1 is None

    def test_semi_len_min(self):
        xf, yf, a, b, r = 50, 55, 21, 20, 2
        data = ret.make_ellipse_data_points(xf, yf, a, b, r, nt=25)
        model_ransac0, inliers0 = ret.get_ellipse_model_ransac_single_frame(
            data,
            xf=50,
            yf=55,
            rf_lim=2,
            semi_len_min=19,
            semi_len_max=None,
            semi_len_ratio_lim=2,
            min_samples=6,
            residual_threshold=10,
            max_trails=100,
        )
        params0f = (xf, yf, a, b, r)
        params1f = ret._ellipse_model_centre_to_focus(*model_ransac0.params, xf, yf)
        assert inliers0.all()
        compare_model_params(params0f, params1f, abs=0.1)

        model_ransac1, inliers1 = ret.get_ellipse_model_ransac_single_frame(
            data,
            xf=50,
            yf=55,
            rf_lim=2,
            semi_len_min=25,
            semi_len_max=None,
            semi_len_ratio_lim=2,
            min_samples=6,
            residual_threshold=10,
            max_trails=100,
        )
        assert model_ransac1 is None

    def test_semi_len_max(self):
        xf, yf, a, b, r = 50, 55, 21, 20, 2
        data = ret.make_ellipse_data_points(xf, yf, a, b, r, nt=25)
        model_ransac0, inliers0 = ret.get_ellipse_model_ransac_single_frame(
            data,
            xf=50,
            yf=55,
            rf_lim=2,
            semi_len_min=None,
            semi_len_max=25,
            semi_len_ratio_lim=2,
            min_samples=6,
            residual_threshold=10,
            max_trails=100,
        )
        params0f = (xf, yf, a, b, r)
        params1f = ret._ellipse_model_centre_to_focus(*model_ransac0.params, xf, yf)
        assert inliers0.all()
        compare_model_params(params0f, params1f, abs=0.1)

        model_ransac1, inliers1 = ret.get_ellipse_model_ransac_single_frame(
            data,
            xf=50,
            yf=55,
            rf_lim=2,
            semi_len_min=None,
            semi_len_max=15,
            semi_len_ratio_lim=2,
            min_samples=6,
            residual_threshold=10,
            max_trails=100,
        )
        assert model_ransac1 is None


class TestGetEllipseModelRansac:
    def test_simple(self):
        xc, yc = np.ones((2, 3)), np.ones((2, 3))
        semi0, semi1 = np.ones((2, 3)), np.ones((2, 3))
        rot = np.zeros((2, 3))
        peak_array = mdtd._make_4d_peak_array_test_data(xc, yc, semi0, semi1, rot)
        ellipse_array, inlier_array = ret.get_ellipse_model_ransac(
            peak_array, max_trails=50
        )
        assert ellipse_array.shape == xc.shape
        assert inlier_array.shape == xc.shape

    def test_xc_yc(self):
        np.random.seed(7)
        xf = np.random.randint(90, 100, size=(2, 3))
        yf = np.random.randint(110, 120, size=(2, 3))
        semi0, semi1 = np.ones((2, 3)) * 60, np.ones((2, 3)) * 60
        rot = np.zeros((2, 3))
        peak_array = mdtd._make_4d_peak_array_test_data(
            xf, yf, semi0, semi1, rot, nt=20
        )
        ellipse_array0, inlier_array0 = ret.get_ellipse_model_ransac(
            peak_array,
            xf=95,
            yf=115,
            rf_lim=20,
            semi_len_min=50,
            semi_len_max=65,
            semi_len_ratio_lim=1.2,
            min_samples=15,
            max_trails=20,
        )
        ellipse_array1, inlier_array1 = ret.get_ellipse_model_ransac(
            peak_array,
            xf=10,
            yf=12,
            rf_lim=20,
            semi_len_min=50,
            semi_len_max=65,
            semi_len_ratio_lim=1.2,
            min_samples=15,
            max_trails=20,
        )

        for iy, ix in np.ndindex(xf.shape):
            assert approx(xf[iy, ix]) == ellipse_array0[iy, ix][1]
            assert approx(yf[iy, ix]) == ellipse_array0[iy, ix][0]
            assert inlier_array0[iy, ix].all()
            assert ellipse_array1[iy, ix] is None
            assert inlier_array1[iy, ix] is None

    def test_semi_lengths(self):
        xf, yf = np.ones((2, 3)) * 200, np.ones((2, 3)) * 210
        np.random.seed(7)
        semi0 = np.random.randint(90, 110, size=(2, 3))
        semi1 = np.random.randint(130, 140, size=(2, 3))
        rot = np.zeros((2, 3))
        peak_array = mdtd._make_4d_peak_array_test_data(
            xf, yf, semi0, semi1, rot, nt=20
        )
        ellipse_array, inlier_array = ret.get_ellipse_model_ransac(
            peak_array,
            xf=200,
            yf=210,
            rf_lim=20,
            semi_len_min=80,
            semi_len_max=150,
            semi_len_ratio_lim=1.7,
            min_samples=15,
            max_trails=20,
        )
        for iy, ix in np.ndindex(xf.shape):
            semi_min = min(ellipse_array[iy, ix][2], ellipse_array[iy, ix][3])
            semi_max = max(ellipse_array[iy, ix][2], ellipse_array[iy, ix][3])
            assert approx(semi_min, abs=0.01) == semi0[iy, ix]
            assert approx(semi_max, abs=0.01) == semi1[iy, ix]

    def test_rf_lim(self):
        xf, yf = np.ones((2, 3)) * 200, np.ones((2, 3)) * 210
        np.random.seed(7)
        semi0 = np.random.randint(90, 110, size=(2, 3))
        semi1 = np.random.randint(130, 140, size=(2, 3))
        rot = np.zeros((2, 3))
        peak_array = mdtd._make_4d_peak_array_test_data(
            xf, yf, semi0, semi1, rot, nt=20
        )
        ellipse_array0, inlier_array0 = ret.get_ellipse_model_ransac(
            peak_array,
            xf=200,
            yf=210,
            rf_lim=20,
            semi_len_min=80,
            semi_len_max=150,
            semi_len_ratio_lim=1.7,
            min_samples=15,
            max_trails=20,
        )
        ellipse_array1, inlier_array1 = ret.get_ellipse_model_ransac(
            peak_array,
            xf=200,
            yf=210,
            rf_lim=20,
            semi_len_min=80,
            semi_len_max=150,
            semi_len_ratio_lim=1.01,
            min_samples=15,
            max_trails=20,
        )
        for iy, ix in np.ndindex(xf.shape):
            ellipse_params0 = ellipse_array0[iy, ix]
            ellipse_params1 = ellipse_array1[iy, ix]
            assert ellipse_params0 != ellipse_params1

    def test_semi_len_ratio_lim(self):
        xf, yf = np.ones((2, 3)) * 200, np.ones((2, 3)) * 210
        semi00, semi01 = np.ones((2, 3)) * 100, np.ones((2, 3)) * 100
        semi10, semi11 = np.ones((2, 3)) * 100, np.ones((2, 3)) * 190
        rot = np.zeros((2, 3))
        peak_array = mdtd._make_4d_peak_array_test_data(
            xf, yf, semi00, semi01, rot, nt=20
        )
        peak_array1 = mdtd._make_4d_peak_array_test_data(
            xf, yf, semi10, semi11, rot, nt=20
        )
        for iy, ix in np.ndindex(xf.shape):
            peak_array[iy, ix] = np.vstack((peak_array[iy, ix], peak_array1[iy, ix]))
        ellipse_array0, inlier_array0 = ret.get_ellipse_model_ransac(
            peak_array,
            xf=200,
            yf=210,
            rf_lim=20,
            semi_len_min=95,
            semi_len_max=195,
            semi_len_ratio_lim=1.11,
            min_samples=15,
            max_trails=200,
        )
        ellipse_array1, inlier_array1 = ret.get_ellipse_model_ransac(
            peak_array,
            xf=200,
            yf=210,
            rf_lim=20,
            semi_len_min=95,
            semi_len_max=195,
            semi_len_ratio_lim=2.0,
            min_samples=15,
            max_trails=200,
        )
        semi_len_ratio_list = []
        for iy, ix in np.ndindex(xf.shape):
            if ellipse_array1[iy, ix] is not None:
                semi0, semi1 = ellipse_array1[iy, ix][2:4]
                semi_len_ratio = max(semi0, semi1) / min(semi0, semi1)
                semi_len_ratio_list.append(semi_len_ratio)
        semi_len_ratio_list = np.array(semi_len_ratio_list)
        assert (semi_len_ratio_list > 1.8).any()

    def test_residual_threshold(self):
        xyf, semi = np.ones((2, 3)) * 100, np.ones((2, 3)) * 90
        rot = np.zeros((2, 3))
        peak_array = mdtd._make_4d_peak_array_test_data(
            xyf, xyf, semi, semi, rot, nt=20
        )
        for iy, ix in np.ndindex(xyf.shape):
            peak_array[iy, ix] = np.vstack((peak_array[iy, ix], [100, 5]))
        ellipse_array0, inlier_array0 = ret.get_ellipse_model_ransac(
            peak_array,
            xf=100,
            yf=100,
            semi_len_min=85,
            semi_len_max=95,
            semi_len_ratio_lim=1.1,
            residual_threshold=1,
            max_trails=100,
            min_samples=15,
        )
        ellipse_array1, inlier_array1 = ret.get_ellipse_model_ransac(
            peak_array,
            xf=100,
            yf=100,
            semi_len_min=85,
            semi_len_max=95,
            semi_len_ratio_lim=1.1,
            residual_threshold=5,
            max_trails=100,
            min_samples=15,
        )

        for iy, ix in np.ndindex(xyf.shape):
            inlier0 = inlier_array0[iy, ix]
            inlier1 = inlier_array1[iy, ix]
            assert inlier0.sum() == 20
            assert inlier1.sum() == 21

    def test_no_values_out_of_bounds(self):
        xf, yf, rf_lim = 100, 100, 20
        semi_len_min, semi_len_max = 50, 100
        semi_len_ratio_lim = 1.15
        peak_array = np.random.randint(0, 200, size=(10, 11, 200, 2))
        ellipse_array, inlier_array = ret.get_ellipse_model_ransac(
            peak_array,
            xf=xf,
            yf=yf,
            rf_lim=rf_lim,
            semi_len_min=semi_len_min,
            semi_len_max=semi_len_max,
            semi_len_ratio_lim=semi_len_ratio_lim,
            min_samples=15,
            max_trails=5,
        )
        for iy, ix in np.ndindex(peak_array.shape[:2]):
            if ellipse_array[iy, ix] is not None:
                xc, yc, semi0, semi1, r = ellipse_array[iy, ix]
                x, y = ret._get_closest_focus(xf, yf, xc, yc, semi0, semi1, r)
                rf = math.hypot(x - xf, y - yf)
                semi_ratio = max(semi0, semi1) / min(semi0, semi1)
                assert rf < rf_lim
                assert semi_ratio < semi_len_ratio_lim
                assert semi0 > semi_len_min
                assert semi1 > semi_len_min
                assert semi0 < semi_len_max
                assert semi1 < semi_len_max


class TestGetInlierOutlierPeakArrays:
    def test_simple(self):
        x, y, n = 2, 3, 10
        peak_array = np.arange(y * x * n * 2).reshape((y, x, n, 2))
        inlier_array = np.ones((y, x, n), dtype=bool)
        inlier_parray0, outlier_parray0 = ret._get_inlier_outlier_peak_arrays(
            peak_array, inlier_array
        )
        inlier_parray1, outlier_parray1 = ret._get_inlier_outlier_peak_arrays(
            peak_array, ~inlier_array
        )
        for iy, ix in np.ndindex(peak_array.shape[:2]):
            assert len(inlier_parray0[iy, ix]) == n
            assert len(outlier_parray0[iy, ix]) == 0
            assert len(inlier_parray1[iy, ix]) == 0
            assert len(outlier_parray1[iy, ix]) == n

    def test_some_true_some_false(self):
        x, y, n = 2, 3, 10
        peak_array = np.arange(y * x * n * 2).reshape((y, x, n, 2))
        inlier_array = np.ones((y, x, n), dtype=bool)
        inlier_array[:, :, 4:] = False
        inlier_parray, outlier_parray = ret._get_inlier_outlier_peak_arrays(
            peak_array, inlier_array
        )
        for iy, ix in np.ndindex(peak_array.shape[:2]):
            assert len(inlier_parray[iy, ix]) == 4
            assert len(outlier_parray[iy, ix]) == 6
            assert (peak_array[iy, ix][:4] == inlier_parray[iy, ix]).all()
            assert (peak_array[iy, ix][4:] == outlier_parray[iy, ix]).all()

    def test_inlier_none(self):
        x, y, n = 2, 3, 10
        peak_array = np.arange(y * x * n * 2).reshape((y, x, n, 2))
        inlier_array = np.empty((y, x), dtype=object)
        for iy, ix in np.ndindex(inlier_array.shape):
            inlier_array[iy, ix] = None

        inlier_parray, outlier_parray = ret._get_inlier_outlier_peak_arrays(
            peak_array, inlier_array
        )

        for iy, ix in np.ndindex(inlier_array.shape):
            assert inlier_parray[iy, ix] is None
            assert len(outlier_parray[iy, ix]) == n


class TestGetLinesListFromEllipseParams:
    def test_nr(self):
        lines_list0 = ret._get_lines_list_from_ellipse_params((5, 5, 10, 15, 0), nr=5)
        lines_list1 = ret._get_lines_list_from_ellipse_params((5, 5, 10, 15, 0), nr=9)
        assert len(lines_list0) == 5
        assert len(lines_list1) == 9

    def test_correct_values(self):
        y, x, sy, sx, r = 10.0, 20.0, 6.0, 5.0, 0.0
        lines_list = ret._get_lines_list_from_ellipse_params((y, x, sy, sx, r), nr=4)
        assert approx(lines_list[0]) == [y + sy, x, y, x + sx]
        assert approx(lines_list[1]) == [y, x + sx, y - sy, x]
        assert approx(lines_list[2]) == [y - sy, x, y, x - sx]
        assert approx(lines_list[3]) == [y, x - sx, y + sy, x]


class TestGetLinesArrayFromEllipseArray:
    def test_correct_values(self):
        xc_array = np.random.randint(10, 20, size=(2, 3))
        yc_array = np.random.randint(30, 40, size=(2, 3))
        sx_array = np.random.randint(70, 80, size=(2, 3))
        sy_array = np.random.randint(89, 99, size=(2, 3))
        ro_array = np.zeros((2, 3))
        ellipse_array = np.empty((2, 3), dtype=object)
        for iy, ix in np.ndindex(ellipse_array.shape):
            xc, yc = xc_array[iy, ix], yc_array[iy, ix]
            sx, sy = sx_array[iy, ix], sy_array[iy, ix]
            ro = ro_array[iy, ix]
            ellipse_array[iy, ix] = (yc, xc, sy, sx, ro)

        lines_array = ret._get_lines_array_from_ellipse_array(ellipse_array, nr=4)
        for iy, ix in np.ndindex(ellipse_array.shape):
            xc, yc = xc_array[iy, ix], yc_array[iy, ix]
            sx, sy = sx_array[iy, ix], sy_array[iy, ix]
            ro = ro_array[iy, ix]
            lines_list = lines_array[iy, ix]
            assert approx(lines_list[0]) == [yc + sy, xc, yc, xc + sx]
            assert approx(lines_list[1]) == [yc, xc + sx, yc - sy, xc]
            assert approx(lines_list[2]) == [yc - sy, xc, yc, xc - sx]
            assert approx(lines_list[3]) == [yc, xc - sx, yc + sy, xc]

    def test_ellipse_array_none(self):
        ellipse_array = np.empty(shape=(2, 3), dtype=object)
        for ix, iy in np.ndindex(ellipse_array.shape):
            ellipse_array[ix, iy] = None
        lines_array = ret._get_lines_array_from_ellipse_array(ellipse_array)
        for ix, iy in np.ndindex(lines_array.shape):
            assert lines_array[ix, iy] is None

    def test_nr(self):
        nr0, nr1 = 5, 9
        ellipse_array = np.random.randint(10, 100, size=(2, 3, 5))
        lines_array0 = ret._get_lines_array_from_ellipse_array(ellipse_array, nr=nr0)
        lines_array1 = ret._get_lines_array_from_ellipse_array(ellipse_array, nr=nr1)
        for iy, ix in np.ndindex(lines_array0.shape):
            assert len(lines_array0[iy, ix]) == nr0
            assert len(lines_array1[iy, ix]) == nr1


class TestGetEllipseMarkerListFromEllipseArray:
    def test_nr(self):
        nr0, nr1 = 5, 9
        ellipse_array = np.random.randint(10, 100, size=(2, 3, 5))
        marker_list0 = ret._get_ellipse_marker_list_from_ellipse_array(
            ellipse_array, nr=nr0
        )
        marker_list1 = ret._get_ellipse_marker_list_from_ellipse_array(
            ellipse_array, nr=nr1
        )
        assert len(marker_list0) == nr0
        assert len(marker_list1) == nr1

    def test_color(self):
        color0, color1 = "blue", "green"
        ellipse_array = np.random.randint(10, 100, size=(2, 3, 5))
        marker_list0 = ret._get_ellipse_marker_list_from_ellipse_array(
            ellipse_array, color=color0
        )
        marker_list1 = ret._get_ellipse_marker_list_from_ellipse_array(
            ellipse_array, color=color1
        )
        for marker0, marker1 in zip(marker_list0, marker_list1):
            assert marker0.marker_properties["color"] == color0
            assert marker1.marker_properties["color"] == color1

    def test_linestyle_linewidth(self):
        linewidth0, linewidth1 = 12, 32
        linestyle0, linestyle1 = "solid", "dashed"
        ellipse_array = np.random.randint(10, 100, size=(2, 3, 5))
        marker_list0 = ret._get_ellipse_marker_list_from_ellipse_array(
            ellipse_array, linestyle=linestyle0, linewidth=linewidth0
        )
        marker_list1 = ret._get_ellipse_marker_list_from_ellipse_array(
            ellipse_array, linestyle=linestyle1, linewidth=linewidth1
        )
        for marker0, marker1 in zip(marker_list0, marker_list1):
            assert marker0.marker_properties["linestyle"] == linestyle0
            assert marker1.marker_properties["linestyle"] == linestyle1
            assert marker0.marker_properties["linewidth"] == linewidth0
            assert marker1.marker_properties["linewidth"] == linewidth1

    def test_correct_values(self):
        xc_array = np.random.randint(120, 130, size=(2, 3))
        yc_array = np.random.randint(130, 140, size=(2, 3))
        sx_array = np.random.randint(70, 80, size=(2, 3))
        sy_array = np.random.randint(89, 99, size=(2, 3))
        ro_array = np.zeros((2, 3))
        ellipse_array = np.empty((2, 3), dtype=object)
        for iy, ix in np.ndindex(ellipse_array.shape):
            xc, yc = xc_array[iy, ix], yc_array[iy, ix]
            sx, sy = sx_array[iy, ix], sy_array[iy, ix]
            ro = ro_array[iy, ix]
            ellipse_array[iy, ix] = (yc, xc, sy, sx, ro)

        nr = 4
        marker_list = ret._get_ellipse_marker_list_from_ellipse_array(
            ellipse_array, nr=nr
        )
        assert len(marker_list) == nr
        for marker in marker_list:
            assert marker.data["x1"][()].shape == xc_array.shape

        m0, m1, m2, m3 = marker_list
        assert_allclose(m0.data["x1"][()], xc_array)
        assert_allclose(m0.data["y1"][()], yc_array + sy_array)
        assert_allclose(m0.data["x2"][()], xc_array + sx_array)
        assert_allclose(m0.data["y2"][()], yc_array)

        assert_allclose(m1.data["x1"][()], xc_array + sx_array)
        assert_allclose(m1.data["y1"][()], yc_array)
        assert_allclose(m1.data["x2"][()], xc_array)
        assert_allclose(m1.data["y2"][()], yc_array - sy_array)

        assert_allclose(m2.data["x1"][()], xc_array)
        assert_allclose(m2.data["y1"][()], yc_array - sy_array)
        assert_allclose(m2.data["x2"][()], xc_array - sx_array)
        assert_allclose(m2.data["y2"][()], yc_array)

        assert_allclose(m3.data["x1"][()], xc_array - sx_array)
        assert_allclose(m3.data["y1"][()], yc_array)
        assert_allclose(m3.data["x2"][()], xc_array)
        assert_allclose(m3.data["y2"][()], yc_array + sy_array)


def test_full_ellipse_ransac_processing():
    xf, yf, a, b, r, nt = 100, 115, 45, 35, 0, 15
    data_points = ret.make_ellipse_data_points(xf, yf, a, b, r, nt)
    image = np.zeros(shape=(200, 210), dtype=np.float32)
    for x, y in data_points:
        image[int(round(x)), int(round(y))] = 100
    disk = morphology.disk(5, np.uint16)
    image = convolve2d(image, disk, mode="same")
    data = np.zeros((2, 3, 210, 200), dtype=np.float32)
    data[:, :] = image.T

    s = Diffraction2D(data)
    s_t = s.template_match_disk(disk_r=5)
    peak_array = s_t.find_peaks_lazy(lazy_result=False)

    c = math.sqrt(math.pow(a, 2) - math.pow(b, 2))
    xc, yc = xf - c * math.cos(r), yf - c * math.sin(r)

    for iy, ix in np.ndindex(peak_array.shape):
        peaks = peak_array[iy, ix]
        assert len(peaks) == 15
        assert approx(peaks[:, 1].mean(), abs=2) == xc
        assert approx(peaks[:, 0].mean(), abs=2) == yc
        assert approx(peaks[:, 1].max(), abs=2) == xc + a
        assert approx(peaks[:, 0].max(), abs=2) == yc + b
        assert approx(peaks[:, 1].min(), abs=2) == xc - a
        assert approx(peaks[:, 0].min(), abs=2) == yc - b

    ellipse_array, inlier_array = ret.get_ellipse_model_ransac(
        peak_array,
        yf=yf,
        xf=xf,
        rf_lim=15,
        semi_len_min=min(a, b) - 5,
        semi_len_max=max(a, b) + 5,
        semi_len_ratio_lim=5,
        max_trails=50,
        min_samples=10,
    )
    s.add_ellipse_array_as_markers(ellipse_array)

    for iy, ix in np.ndindex(ellipse_array.shape):
        ycf, xcf, bf, af, rf = ellipse_array[iy, ix]
        assert approx((xcf, ycf, af, bf, rf), abs=0.1) == [xc, yc, a, b, r]
        assert inlier_array[iy, ix].all()

    s.add_ellipse_array_as_markers(ellipse_array)
    x_list, y_list = [], []
    for _, marker in list(s.metadata.Markers):
        x_list.append(marker.data["x1"][()][0][0])
        y_list.append(marker.data["y1"][()][0][0])
    assert approx(np.mean(x_list), abs=1) == xc
    assert approx(np.mean(y_list), abs=1) == yc
    assert approx(np.max(x_list), abs=1) == xc + a
    assert approx(np.max(y_list), abs=1) == yc + b
    assert approx(np.min(x_list), abs=1) == xc - a
    assert approx(np.min(y_list), abs=1) == yc - b
