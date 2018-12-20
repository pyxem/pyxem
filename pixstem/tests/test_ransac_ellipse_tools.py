import pytest
from pytest import approx
import math
import numpy as np
from numpy.testing import assert_allclose
from scipy.signal import convolve2d
from skimage import morphology
import pixstem.make_diffraction_test_data as mdtd
import pixstem.ransac_ellipse_tools as ret
from pixstem.pixelated_stem_class import PixelatedSTEM


class TestIsEllipseGood:

    def test_simple(self):
        ellipse_model = ret.EllipseModel()
        ellipse_model.params = (30, 50, 100, 120, 0)
        is_good = ret.is_ellipse_good(
                ellipse_model=ellipse_model, data=None,
                xc=50, yc=30, r_elli_lim=5,
                semi_len_min=90, semi_len_max=130, semi_len_ratio_lim=1.5)
        assert is_good

    def test_xc(self):
        ellipse_model = ret.EllipseModel()
        ellipse_model.params = (30, 50, 100, 120, 0)
        is_good0 = ret.is_ellipse_good(
                ellipse_model=ellipse_model, data=None,
                xc=56, yc=30, r_elli_lim=5,
                semi_len_min=90, semi_len_max=130, semi_len_ratio_lim=1.5)
        assert not is_good0
        is_good1 = ret.is_ellipse_good(
                ellipse_model=ellipse_model, data=None,
                xc=44, yc=30, r_elli_lim=5,
                semi_len_min=90, semi_len_max=130, semi_len_ratio_lim=1.5)
        assert not is_good1

    def test_yc(self):
        ellipse_model = ret.EllipseModel()
        ellipse_model.params = (30, 50, 100, 120, 0)
        is_good0 = ret.is_ellipse_good(
                ellipse_model=ellipse_model, data=None,
                xc=50, yc=36, r_elli_lim=5,
                semi_len_min=90, semi_len_max=130, semi_len_ratio_lim=1.5)
        assert not is_good0
        is_good1 = ret.is_ellipse_good(
                ellipse_model=ellipse_model, data=None,
                xc=50, yc=24, r_elli_lim=5,
                semi_len_min=90, semi_len_max=130, semi_len_ratio_lim=1.5)
        assert not is_good1

    def test_r_elli_lim(self):
        ellipse_model = ret.EllipseModel()
        ellipse_model.params = (30, 50, 100, 120, 0)
        is_good0 = ret.is_ellipse_good(
                ellipse_model=ellipse_model, data=None,
                xc=50, yc=33, r_elli_lim=5,
                semi_len_min=90, semi_len_max=130, semi_len_ratio_lim=1.5)
        assert is_good0
        is_good1 = ret.is_ellipse_good(
                ellipse_model=ellipse_model, data=None,
                xc=50, yc=33, r_elli_lim=2,
                semi_len_min=90, semi_len_max=130, semi_len_ratio_lim=1.5)
        assert not is_good1

    def test_semi_len_min(self):
        ellipse_model = ret.EllipseModel()
        ellipse_model.params = (30, 50, 100, 120, 0)
        is_good = ret.is_ellipse_good(
                ellipse_model=ellipse_model, data=None,
                xc=50, yc=30, r_elli_lim=5,
                semi_len_min=105, semi_len_max=130, semi_len_ratio_lim=1.5)
        assert not is_good

    def test_semi_len_max(self):
        ellipse_model = ret.EllipseModel()
        ellipse_model.params = (30, 50, 100, 120, 0)
        is_good = ret.is_ellipse_good(
                ellipse_model=ellipse_model, data=None,
                xc=50, yc=30, r_elli_lim=5,
                semi_len_min=90, semi_len_max=115, semi_len_ratio_lim=1.5)
        assert not is_good

    def test_semi_len_ratio_lim(self):
        ellipse_model = ret.EllipseModel()
        ellipse_model.params = (30, 50, 100, 120, 0)
        is_good = ret.is_ellipse_good(
                ellipse_model=ellipse_model, data=None,
                xc=50, yc=30, r_elli_lim=5,
                semi_len_min=90, semi_len_max=130, semi_len_ratio_lim=1.01)
        assert not is_good
        with pytest.raises(ValueError):
            ret.is_ellipse_good(
                    ellipse_model=ellipse_model, data=None,
                    xc=50, yc=30, r_elli_lim=5,
                    semi_len_min=90, semi_len_max=130, semi_len_ratio_lim=0.9)


class TestIsDataGood:

    def test_simple(self):
        data = np.array([[10, 20], [11, 20]])
        assert ret.is_data_good(data, xc=0, yc=0, r_peak_lim=2)

    def test_large_array(self):
        data = np.random.randint(50, 100, size=(1000, 2))
        assert ret.is_data_good(data, xc=30, yc=30, r_peak_lim=5)

    def test_xc(self):
        data = np.array([[10, 20], [41, 30], [132, 31]])
        assert ret.is_data_good(data, xc=17, yc=10, r_peak_lim=2)
        assert not ret.is_data_good(data, xc=19, yc=10, r_peak_lim=2)

    def test_yc(self):
        data = np.array([[10, 20], [41, 30], [132, 31]])
        assert ret.is_data_good(data, xc=30, yc=38, r_peak_lim=2)
        assert not ret.is_data_good(data, xc=30, yc=40, r_peak_lim=2)

    def test_r_peak_lim(self):
        data = np.array([[10, 20], [41, 30], [132, 31]])
        assert ret.is_data_good(data, xc=30, yc=38, r_peak_lim=2)
        assert not ret.is_data_good(data, xc=30, yc=38, r_peak_lim=5)


class TestGetEllipseModelRansacSingleFrame:

    def test_simple(self):
        x, y, r = 160, 70, 60
        data = ret.EllipseModel().predict_xy(
                np.arange(0, 2*np.pi, 0.5), params=(y, x, r, r, 0))
        ellipse_model, inliers = ret.get_ellipse_model_ransac_single_frame(
                data, xc=x, yc=y, r_elli_lim=30, r_peak_lim=40,
                semi_len_min=r-10, semi_len_max=r+10, max_trails=100)
        assert inliers.all()
        assert approx(ellipse_model.params[1], abs=0.01) == x
        assert approx(ellipse_model.params[0], abs=0.01) == y
        assert approx(ellipse_model.params[2], abs=0.01) == r
        assert approx(ellipse_model.params[3], abs=0.01) == r

    def test_outlier(self):
        x, y, r = 160, 70, 60
        data = ret.EllipseModel().predict_xy(
                np.arange(0, 2*np.pi, 0.5), params=(y, x, r, r, 0))
        data = np.append(data, np.array([[10, 10], [250, 250]]), axis=0)
        ellipse_model, inliers = ret.get_ellipse_model_ransac_single_frame(
                data, xc=x, yc=y, r_elli_lim=30, r_peak_lim=40,
                semi_len_min=r-10, semi_len_max=r+10, max_trails=100)
        assert inliers[:-2].all()
        assert (np.invert(inliers[-2:])).all()
        assert approx(ellipse_model.params[1], abs=0.01) == x
        assert approx(ellipse_model.params[0], abs=0.01) == y
        assert approx(ellipse_model.params[2], abs=0.01) == r
        assert approx(ellipse_model.params[3], abs=0.01) == r

    def test_semi_lengths(self):
        x, y, r0, r1 = 160, 70, 40, 60
        data0 = ret.EllipseModel().predict_xy(
                np.arange(0, 2*np.pi, 0.5), params=(y, x, r0, r0, 0))
        data1 = ret.EllipseModel().predict_xy(
                np.arange(0, 2*np.pi, 0.5), params=(y, x, r1, r1, 0))
        data = np.append(data0, data1, axis=0)
        ellipse_model0, inliers0 = ret.get_ellipse_model_ransac_single_frame(
                data, xc=x, yc=y, r_elli_lim=10, r_peak_lim=10,
                semi_len_min=55, semi_len_max=65, semi_len_ratio_lim=1.01,
                max_trails=400)
        assert inliers0[len(data0):].all()
        assert np.invert(inliers0[:len(data0)]).all()
        assert approx(ellipse_model0.params[1], abs=0.01) == x
        assert approx(ellipse_model0.params[0], abs=0.01) == y
        assert approx(ellipse_model0.params[2], abs=0.01) == r1
        assert approx(ellipse_model0.params[3], abs=0.01) == r1

        ellipse_model1, inliers1 = ret.get_ellipse_model_ransac_single_frame(
                data, xc=x, yc=y, r_elli_lim=10, r_peak_lim=10,
                semi_len_min=35, semi_len_max=45, semi_len_ratio_lim=1.01,
                max_trails=400)
        assert inliers1[:-len(data0)].all()
        assert np.invert(inliers1[len(data0):]).all()
        assert approx(ellipse_model1.params[1], abs=0.01) == x
        assert approx(ellipse_model1.params[0], abs=0.01) == y
        assert approx(ellipse_model1.params[2], abs=0.01) == r0
        assert approx(ellipse_model1.params[3], abs=0.01) == r0

        ellipse_model2, inliers2 = ret.get_ellipse_model_ransac_single_frame(
                data, xc=x, yc=y, r_elli_lim=10, r_peak_lim=10,
                semi_len_min=80, semi_len_max=90, semi_len_ratio_lim=1.01,
                max_trails=400)
        assert ellipse_model2 is None
        assert inliers2 is None

    def test_xc_yc(self):
        x0, y0, x1, y1, r = 160, 70, 50, 200, 40,
        data0 = ret.EllipseModel().predict_xy(
                np.arange(0, 2*np.pi, 0.5), params=(y0, x0, r, r, 0))
        data1 = ret.EllipseModel().predict_xy(
                np.arange(0, 2*np.pi, 0.5), params=(y1, x1, r, r, 0))
        data = np.append(data0, data1, axis=0)
        ellipse_model0, inliers = ret.get_ellipse_model_ransac_single_frame(
                data, xc=x0, yc=y0, r_elli_lim=10, r_peak_lim=10,
                semi_len_min=35, semi_len_max=45, semi_len_ratio_lim=1.01,
                max_trails=400)
        assert approx(ellipse_model0.params[1], abs=0.01) == x0
        assert approx(ellipse_model0.params[0], abs=0.01) == y0

        ellipse_model1, inliers = ret.get_ellipse_model_ransac_single_frame(
                data, xc=x1, yc=y1, r_elli_lim=10, r_peak_lim=10,
                semi_len_min=35, semi_len_max=45, semi_len_ratio_lim=1.01,
                max_trails=400)
        assert approx(ellipse_model1.params[1], abs=0.01) == x1
        assert approx(ellipse_model1.params[0], abs=0.01) == y1

    def test_semi_len_ratio_lim(self):
        x, y, r0, r1 = 160, 70, 30, 60
        data = ret.EllipseModel().predict_xy(
                np.arange(0, 2*np.pi, 0.5), params=(y, x, r0, r1, 0.2))
        ellipse_model, inliers = ret.get_ellipse_model_ransac_single_frame(
                data, xc=x, yc=y, r_elli_lim=10, r_peak_lim=10,
                semi_len_min=25, semi_len_max=65, semi_len_ratio_lim=2.5,
                max_trails=400)
        semi0, semi1 = ellipse_model.params[2], ellipse_model.params[3]
        assert inliers.all()
        assert approx(ellipse_model.params[1], abs=0.01) == x
        assert approx(ellipse_model.params[0], abs=0.01) == y
        assert approx(min(semi0, semi1), abs=0.01) == r0
        assert approx(max(semi0, semi1), abs=0.01) == r1

        ellipse_model1, inliers1 = ret.get_ellipse_model_ransac_single_frame(
                data, xc=x, yc=y, r_elli_lim=10, r_peak_lim=10,
                semi_len_min=25, semi_len_max=65, semi_len_ratio_lim=1.01,
                max_trails=400)
        assert ellipse_model1 is None


class TestGetEllipseModelRansac:

    def test_simple(self):
        xc, yc = np.ones((2, 3)), np.ones((2, 3))
        semi0, semi1 = np.ones((2, 3)), np.ones((2, 3))
        rot = np.zeros((2, 3))
        peak_array = mdtd._make_4d_peak_array_test_data(
                xc, yc, semi0, semi1, rot)
        ellipse_array, inlier_array = ret.get_ellipse_model_ransac(
                peak_array, max_trails=50)
        assert ellipse_array.shape == xc.shape
        assert inlier_array.shape == xc.shape

    def test_xc_yc(self):
        np.random.seed(7)
        xc = np.random.randint(90, 100, size=(2, 3))
        yc = np.random.randint(110, 120, size=(2, 3))
        semi0, semi1 = np.ones((2, 3))*60, np.ones((2, 3))*65
        rot = np.zeros((2, 3))
        peak_array = mdtd._make_4d_peak_array_test_data(
                xc, yc, semi0, semi1, rot, nr=20)
        ellipse_array0, inlier_array0 = ret.get_ellipse_model_ransac(
                peak_array, xc=95, yc=115, r_elli_lim=20, r_peak_lim=10,
                semi_len_min=55, semi_len_max=76, semi_len_ratio_lim=1.2,
                min_samples=15, max_trails=20)
        ellipse_array1, inlier_array1 = ret.get_ellipse_model_ransac(
                peak_array, xc=10, yc=12, r_elli_lim=20, r_peak_lim=10,
                semi_len_min=55, semi_len_max=76, semi_len_ratio_lim=1.2,
                min_samples=15, max_trails=20)
        for iy, ix in np.ndindex(xc.shape):
            assert approx(xc[iy, ix]) == ellipse_array0[iy, ix][1]
            assert approx(yc[iy, ix]) == ellipse_array0[iy, ix][0]
            assert inlier_array0[iy, ix].all()
            assert ellipse_array1[iy, ix] is None
            assert inlier_array1[iy, ix] is None

    def test_semi_lengths(self):
        xc, yc = np.ones((2, 3)) * 200, np.ones((2, 3)) * 210
        np.random.seed(7)
        semi0 = np.random.randint(90, 110, size=(2, 3))
        semi1 = np.random.randint(130, 140, size=(2, 3))
        rot = np.zeros((2, 3))
        peak_array = mdtd._make_4d_peak_array_test_data(
                xc, yc, semi0, semi1, rot, nr=20)
        ellipse_array, inlier_array = ret.get_ellipse_model_ransac(
                peak_array, xc=200, yc=210, r_elli_lim=20, r_peak_lim=10,
                semi_len_min=80, semi_len_max=150, semi_len_ratio_lim=1.7,
                min_samples=15, max_trails=20)
        for iy, ix in np.ndindex(xc.shape):
            semi_min = min(ellipse_array[iy, ix][2], ellipse_array[iy, ix][3])
            semi_max = max(ellipse_array[iy, ix][2], ellipse_array[iy, ix][3])
            assert approx(semi_min, abs=0.01) == semi0[iy, ix]
            assert approx(semi_max, abs=0.01) == semi1[iy, ix]

    def test_elli_lim(self):
        xc, yc = np.ones((2, 3)) * 200, np.ones((2, 3)) * 210
        np.random.seed(7)
        semi0 = np.random.randint(90, 110, size=(2, 3))
        semi1 = np.random.randint(130, 140, size=(2, 3))
        rot = np.zeros((2, 3))
        peak_array = mdtd._make_4d_peak_array_test_data(
                xc, yc, semi0, semi1, rot, nr=20)
        ellipse_array0, inlier_array0 = ret.get_ellipse_model_ransac(
                peak_array, xc=200, yc=210, r_elli_lim=20, r_peak_lim=10,
                semi_len_min=80, semi_len_max=150, semi_len_ratio_lim=1.7,
                min_samples=15, max_trails=20)
        ellipse_array1, inlier_array1 = ret.get_ellipse_model_ransac(
                peak_array, xc=200, yc=210, r_elli_lim=20, r_peak_lim=10,
                semi_len_min=80, semi_len_max=150, semi_len_ratio_lim=1.01,
                min_samples=15, max_trails=20)
        for iy, ix in np.ndindex(xc.shape):
            ellipse_params0 = ellipse_array0[iy, ix]
            ellipse_params1 = ellipse_array1[iy, ix]
            assert ellipse_params0 != ellipse_params1

    def test_r_peak_lim(self):
        np.random.seed(7)
        xc, yc = np.ones((4, 6)) * 200, np.ones((4, 6)) * 210
        semi0, semi1 = np.ones((4, 6)) * 100, np.ones((4, 6)) * 100
        rot = np.zeros((4, 6))
        peak_array = mdtd._make_4d_peak_array_test_data(
                xc, yc, semi0, semi1, rot, nr=10)
        for iy, ix in np.ndindex(xc.shape):
            centre_array = np.random.randint(190, 220, size=(10, 2))
            peak_array[iy, ix] = np.vstack((peak_array[iy, ix], centre_array))
        ellipse_array0, inlier_array0 = ret.get_ellipse_model_ransac(
                peak_array, xc=200, yc=210, r_elli_lim=200, r_peak_lim=60,
                semi_len_min=5, semi_len_max=250, semi_len_ratio_lim=14.01,
                min_samples=10, max_trails=30)
        ellipse_array1, inlier_array1 = ret.get_ellipse_model_ransac(
                peak_array, xc=200, yc=210, r_elli_lim=200, r_peak_lim=0.5,
                semi_len_min=5, semi_len_max=250, semi_len_ratio_lim=14.01,
                min_samples=10, max_trails=30)
        for iy, ix in np.ndindex(xc.shape):
            inlier0, inlier1 = inlier_array0[iy, ix], inlier_array1[iy, ix]
            if inlier0 is not None:
                assert not inlier0[10:].any()
            if inlier1 is not None:
                assert inlier1[10:].any()

    def test_semi_len_ratio_lim(self):
        xc, yc = np.ones((2, 3)) * 200, np.ones((2, 3)) * 210
        semi00, semi01 = np.ones((2, 3)) * 100, np.ones((2, 3)) * 100
        semi10, semi11 = np.ones((2, 3)) * 100, np.ones((2, 3)) * 190
        rot = np.zeros((2, 3))
        peak_array = mdtd._make_4d_peak_array_test_data(
                xc, yc, semi00, semi01, rot, nr=20)
        peak_array1 = mdtd._make_4d_peak_array_test_data(
                xc, yc, semi10, semi11, rot, nr=20)
        for iy, ix in np.ndindex(xc.shape):
            peak_array[iy, ix] = np.vstack((peak_array[iy, ix],
                                            peak_array1[iy, ix]))
        ellipse_array0, inlier_array0 = ret.get_ellipse_model_ransac(
                peak_array, xc=200, yc=210, r_elli_lim=20, r_peak_lim=20,
                semi_len_min=95, semi_len_max=195, semi_len_ratio_lim=1.11,
                min_samples=15, max_trails=200)
        ellipse_array1, inlier_array1 = ret.get_ellipse_model_ransac(
                peak_array, xc=200, yc=210, r_elli_lim=20, r_peak_lim=20,
                semi_len_min=95, semi_len_max=195, semi_len_ratio_lim=2.,
                min_samples=15, max_trails=200)
        for iy, ix in np.ndindex(xc.shape):
            if ellipse_array0[iy, ix] is not None:
                semi0, semi1 = ellipse_array0[iy, ix][2:4]
                assert max(semi0, semi1) / min(semi0, semi1) < 1.12
        semi_len_ratio_list = []
        for iy, ix in np.ndindex(xc.shape):
            if ellipse_array1[iy, ix] is not None:
                semi0, semi1 = ellipse_array1[iy, ix][2:4]
                semi_len_ratio = max(semi0, semi1) / min(semi0, semi1)
                semi_len_ratio_list.append(semi_len_ratio)
        semi_len_ratio_list = np.array(semi_len_ratio_list)
        assert (semi_len_ratio_list > 1.8).any()

    def test_residual_threshold(self):
        xyc, semi = np.ones((2, 3)) * 100, np.ones((2, 3)) * 90
        rot = np.zeros((2, 3))
        peak_array = mdtd._make_4d_peak_array_test_data(
                xyc, xyc, semi, semi, rot, nr=20)
        for iy, ix in np.ndindex(xyc.shape):
            peak_array[iy, ix] = np.vstack((peak_array[iy, ix], [100, 5]))
        ellipse_array0, inlier_array0 = ret.get_ellipse_model_ransac(
                peak_array, xc=100, yc=100, semi_len_min=85, semi_len_max=95,
                semi_len_ratio_lim=1.1, residual_threshold=1, max_trails=100,
                min_samples=15)
        ellipse_array1, inlier_array1 = ret.get_ellipse_model_ransac(
                peak_array, xc=100, yc=100, semi_len_min=85, semi_len_max=95,
                semi_len_ratio_lim=1.1, residual_threshold=5, max_trails=100,
                min_samples=15)

        for iy, ix in np.ndindex(xyc.shape):
            inlier0 = inlier_array0[iy, ix]
            inlier1 = inlier_array1[iy, ix]
            assert inlier0.sum() == 20
            assert inlier1.sum() == 21

    def test_no_values_out_of_bounds(self):
        xc, yc, r_elli_lim = 100, 100, 20
        semi_len_min, semi_len_max = 50, 100
        semi_len_ratio_lim = 1.15
        peak_array = np.random.randint(0, 200, size=(10, 11, 200, 2))
        ellipse_array, inlier_array = ret.get_ellipse_model_ransac(
                peak_array, xc=xc, yc=yc, r_elli_lim=r_elli_lim, r_peak_lim=20,
                semi_len_min=semi_len_min, semi_len_max=semi_len_max,
                semi_len_ratio_lim=semi_len_ratio_lim,
                min_samples=15, max_trails=5)
        for iy, ix in np.ndindex(peak_array.shape[:2]):
            if ellipse_array[iy, ix] is not None:
                x, y, semi0, semi1, rot = ellipse_array[iy, ix]
                rc = math.hypot(x - xc, y - yc)
                semi_ratio = max(semi0, semi1)/min(semi0, semi1)
                assert rc < r_elli_lim
                assert semi_ratio < semi_len_ratio_lim
                assert semi0 > semi_len_min
                assert semi1 > semi_len_min
                assert semi0 < semi_len_max
                assert semi1 < semi_len_max


class TestGetEllipseModelData:

    def test_simple(self):
        y, x, sy, sx, r = 10., 20., 5., 5., 0.
        ellipse_data = ret._get_ellipse_model_data((y, x, sy, sx, r), nr=4)
        assert approx(ellipse_data[0]) == [y + sy, x]
        assert approx(ellipse_data[1]) == [y, x + sx]
        assert approx(ellipse_data[2]) == [y - sy, x]
        assert approx(ellipse_data[3]) == [y, x - sx]

    def test_nr(self):
        ellipse_data0 = ret._get_ellipse_model_data((5, 5, 9, 8, 1), nr=5)
        ellipse_data1 = ret._get_ellipse_model_data((5, 5, 9, 8, 1), nr=9)
        assert len(ellipse_data0) == 5
        assert len(ellipse_data1) == 9


class TestGetInlierOutlierPeakArrays:

    def test_simple(self):
        x, y, n = 2, 3, 10
        peak_array = np.arange(y*x*n*2).reshape((y, x, n, 2))
        inlier_array = np.ones((y, x, n), dtype=np.bool)
        inlier_parray0, outlier_parray0 = ret._get_inlier_outlier_peak_arrays(
                peak_array, inlier_array)
        inlier_parray1, outlier_parray1 = ret._get_inlier_outlier_peak_arrays(
                peak_array, ~inlier_array)
        for iy, ix in np.ndindex(peak_array.shape[:2]):
            assert len(inlier_parray0[iy, ix]) == n
            assert len(outlier_parray0[iy, ix]) == 0
            assert len(inlier_parray1[iy, ix]) == 0
            assert len(outlier_parray1[iy, ix]) == n

    def test_some_true_some_false(self):
        x, y, n = 2, 3, 10
        peak_array = np.arange(y*x*n*2).reshape((y, x, n, 2))
        inlier_array = np.ones((y, x, n), dtype=np.bool)
        inlier_array[:, :, 4:] = False
        inlier_parray, outlier_parray = ret._get_inlier_outlier_peak_arrays(
                peak_array, inlier_array)
        for iy, ix in np.ndindex(peak_array.shape[:2]):
            assert len(inlier_parray[iy, ix]) == 4
            assert len(outlier_parray[iy, ix]) == 6
            assert (peak_array[iy, ix][:4] == inlier_parray[iy, ix]).all()
            assert (peak_array[iy, ix][4:] == outlier_parray[iy, ix]).all()


class TestGetLinesListFromEllipseParams:

    def test_nr(self):
        lines_list0 = ret._get_lines_list_from_ellipse_params(
                (5, 5, 10, 15, 0), nr=5)
        lines_list1 = ret._get_lines_list_from_ellipse_params(
                (5, 5, 10, 15, 0), nr=9)
        assert len(lines_list0) == 5
        assert len(lines_list1) == 9

    def test_correct_values(self):
        y, x, sy, sx, r = 10., 20., 6., 5., 0.
        lines_list = ret._get_lines_list_from_ellipse_params(
                (y, x, sy, sx, r), nr=4)
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
        ellipse_array = np.empty((2, 3), dtype=np.object)
        for iy, ix in np.ndindex(ellipse_array.shape):
            xc, yc = xc_array[iy, ix], yc_array[iy, ix]
            sx, sy = sx_array[iy, ix], sy_array[iy, ix]
            ro = ro_array[iy, ix]
            ellipse_array[iy, ix] = (yc, xc, sy, sx, ro)

        lines_array = ret._get_lines_array_from_ellipse_array(
                ellipse_array, nr=4)
        for iy, ix in np.ndindex(ellipse_array.shape):
            xc, yc = xc_array[iy, ix], yc_array[iy, ix]
            sx, sy = sx_array[iy, ix], sy_array[iy, ix]
            ro = ro_array[iy, ix]
            lines_list = lines_array[iy, ix]
            assert approx(lines_list[0]) == [yc + sy, xc, yc, xc + sx]
            assert approx(lines_list[1]) == [yc, xc + sx, yc - sy, xc]
            assert approx(lines_list[2]) == [yc - sy, xc, yc, xc - sx]
            assert approx(lines_list[3]) == [yc, xc - sx, yc + sy, xc]

    def test_nr(self):
        nr0, nr1 = 5, 9
        ellipse_array = np.random.randint(10, 100, size=(2, 3, 5))
        lines_array0 = ret._get_lines_array_from_ellipse_array(
                ellipse_array, nr=nr0)
        lines_array1 = ret._get_lines_array_from_ellipse_array(
                ellipse_array, nr=nr1)
        for iy, ix in np.ndindex(lines_array0.shape):
            assert len(lines_array0[iy, ix]) == nr0
            assert len(lines_array1[iy, ix]) == nr1


class TestGetEllipseMarkerListFromEllipseArray:

    def test_nr(self):
        nr0, nr1 = 5, 9
        ellipse_array = np.random.randint(10, 100, size=(2, 3, 5))
        marker_list0 = ret._get_ellipse_marker_list_from_ellipse_array(
                ellipse_array, nr=nr0)
        marker_list1 = ret._get_ellipse_marker_list_from_ellipse_array(
                ellipse_array, nr=nr1)
        assert len(marker_list0) == nr0
        assert len(marker_list1) == nr1

    def test_color(self):
        color0, color1 = 'blue', 'green'
        ellipse_array = np.random.randint(10, 100, size=(2, 3, 5))
        marker_list0 = ret._get_ellipse_marker_list_from_ellipse_array(
                ellipse_array, color=color0)
        marker_list1 = ret._get_ellipse_marker_list_from_ellipse_array(
                ellipse_array, color=color1)
        for marker0, marker1 in zip(marker_list0, marker_list1):
            assert marker0.marker_properties['color'] == color0
            assert marker1.marker_properties['color'] == color1

    def test_linestyle_linewidth(self):
        linewidth0, linewidth1 = 12, 32
        linestyle0, linestyle1 = 'solid', 'dashed'
        ellipse_array = np.random.randint(10, 100, size=(2, 3, 5))
        marker_list0 = ret._get_ellipse_marker_list_from_ellipse_array(
                ellipse_array, linestyle=linestyle0, linewidth=linewidth0)
        marker_list1 = ret._get_ellipse_marker_list_from_ellipse_array(
                ellipse_array, linestyle=linestyle1, linewidth=linewidth1)
        for marker0, marker1 in zip(marker_list0, marker_list1):
            assert marker0.marker_properties['linestyle'] == linestyle0
            assert marker1.marker_properties['linestyle'] == linestyle1
            assert marker0.marker_properties['linewidth'] == linewidth0
            assert marker1.marker_properties['linewidth'] == linewidth1

    def test_correct_values(self):
        xc_array = np.random.randint(120, 130, size=(2, 3))
        yc_array = np.random.randint(130, 140, size=(2, 3))
        sx_array = np.random.randint(70, 80, size=(2, 3))
        sy_array = np.random.randint(89, 99, size=(2, 3))
        ro_array = np.zeros((2, 3))
        ellipse_array = np.empty((2, 3), dtype=np.object)
        for iy, ix in np.ndindex(ellipse_array.shape):
            xc, yc = xc_array[iy, ix], yc_array[iy, ix]
            sx, sy = sx_array[iy, ix], sy_array[iy, ix]
            ro = ro_array[iy, ix]
            ellipse_array[iy, ix] = (yc, xc, sy, sx, ro)

        nr = 4
        marker_list = ret._get_ellipse_marker_list_from_ellipse_array(
                ellipse_array, nr=nr)
        assert len(marker_list) == nr
        for marker in marker_list:
            assert marker.data['x1'][()].shape == xc_array.shape

        m0, m1, m2, m3 = marker_list
        assert_allclose(m0.data['x1'][()], xc_array)
        assert_allclose(m0.data['y1'][()], yc_array + sy_array)
        assert_allclose(m0.data['x2'][()], xc_array + sx_array)
        assert_allclose(m0.data['y2'][()], yc_array)

        assert_allclose(m1.data['x1'][()], xc_array + sx_array)
        assert_allclose(m1.data['y1'][()], yc_array)
        assert_allclose(m1.data['x2'][()], xc_array)
        assert_allclose(m1.data['y2'][()], yc_array - sy_array)

        assert_allclose(m2.data['x1'][()], xc_array)
        assert_allclose(m2.data['y1'][()], yc_array - sy_array)
        assert_allclose(m2.data['x2'][()], xc_array - sx_array)
        assert_allclose(m2.data['y2'][()], yc_array)

        assert_allclose(m3.data['x1'][()], xc_array - sx_array)
        assert_allclose(m3.data['y1'][()], yc_array)
        assert_allclose(m3.data['x2'][()], xc_array)
        assert_allclose(m3.data['y2'][()], yc_array + sy_array)


def test_full_ellipse_ransac_processing():
    xc, yc, sx, sy, rot, nr = 100, 115, 55, 50, 0, 15
    ellipse_data = ret._get_ellipse_model_data((yc, xc, sy, sx, rot), nr=nr)
    image = np.zeros(shape=(200, 210), dtype=np.float32)
    for y, x in ellipse_data:
        image[int(round(y)), int(round(x))] = 100
    disk = morphology.disk(5, np.uint16)
    image = convolve2d(image, disk, mode='same')

    data = np.zeros((2, 3, 200, 210), dtype=np.float32)
    data[:, :] = image
    s = PixelatedSTEM(data)
    s_t = s.template_match_disk(disk_r=5)
    peak_array = s_t.find_peaks(lazy_result=False)

    for iy, ix in np.ndindex(peak_array.shape):
        peaks = peak_array[iy, ix]
        assert len(peaks) == 15
        assert approx(peaks[:, 1].mean(), abs=1) == xc
        assert approx(peaks[:, 0].mean(), abs=1) == yc
        assert approx(peaks[:, 1].max(), abs=1) == xc + sx
        assert approx(peaks[:, 0].max(), abs=1) == yc + sy
        assert approx(peaks[:, 1].min(), abs=1) == xc - sx
        assert approx(peaks[:, 0].min(), abs=1) == yc - sy

    ellipse_array, inlier_array = ret.get_ellipse_model_ransac(
            peak_array, xc=xc, yc=yc, r_elli_lim=15, r_peak_lim=15,
            semi_len_min=min(sx, sy) - 5, semi_len_max=max(sx, sy) + 5,
            semi_len_ratio_lim=5, max_trails=50,
            min_samples=10)

    for iy, ix in np.ndindex(ellipse_array.shape):
        assert approx(ellipse_array[iy, ix], abs=0.1) == [yc, xc, sy, sx, rot]
        assert inlier_array[iy, ix].all()

    s.add_ellipse_array_as_markers(ellipse_array)
    x_list, y_list = [], []
    for _, marker in list(s.metadata.Markers):
        x_list.append(marker.data['x1'][()][0][0])
        y_list.append(marker.data['y1'][()][0][0])
    assert approx(np.mean(x_list), abs=1) == xc
    assert approx(np.mean(y_list), abs=1) == yc
    assert approx(np.max(x_list), abs=1) == xc + sx
    assert approx(np.max(y_list), abs=1) == yc + sy
    assert approx(np.min(x_list), abs=1) == xc - sx
    assert approx(np.min(y_list), abs=1) == yc - sy
