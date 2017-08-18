import unittest
from fpd_data_processing.pixelated_stem_class import DPCSignal
import numpy as np


class test_dpc_signal_create(unittest.TestCase):

    def test_create(self):
        data = np.ones(shape=(2, 10, 10))
        s = DPCSignal(data)
        with self.assertRaises(ValueError):
            DPCSignal(np.zeros(10))


class test_dpc_signal_correct_ramp(unittest.TestCase):

    def test_correct_ramp_flat(self):
        data0 = np.ones(shape=(2, 64, 64))
        s0 = DPCSignal(data0)
        s0_corr = s0.correct_ramp(corner_size=0.05)
        self.assertTrue((s0.data==data0).all())
        np.testing.assert_allclose(
                s0_corr.data, np.zeros_like(data0), atol=1e-8)

        s0.correct_ramp(corner_size=0.05, out=s0)
        np.testing.assert_allclose(
                s0.data, np.zeros_like(data0), atol=1e-8)

    def test_correct_ramp_x_y(self):
        array_x, array_y = np.meshgrid(range(64), range(64))
        data_x = np.swapaxes(
                np.dstack((array_x, array_x)), 0, 2).astype('float64')
        data_y = np.swapaxes(
                np.dstack((array_y, array_y)), 0, 2).astype('float64')
        s_x = DPCSignal(data_x)
        s_y = DPCSignal(data_y)
        s_x_corr = s_x.correct_ramp(corner_size=0.05)
        s_y_corr = s_y.correct_ramp(corner_size=0.05)
        np.testing.assert_allclose(
                s_x_corr.data, np.zeros_like(data_x), atol=1e-8)
        np.testing.assert_allclose(
                s_y_corr.data, np.zeros_like(data_y), atol=1e-8)

        data_xy = np.swapaxes(
                np.dstack((array_x, array_y)), 0, 2).astype('float64')
        data_yx = np.swapaxes(
                np.dstack((array_y, array_x)), 0, 2).astype('float64')
        s_xy = DPCSignal(data_xy)
        s_yx = DPCSignal(data_yx)
        s_xy_corr = s_xy.correct_ramp(corner_size=0.05)
        s_yx_corr = s_yx.correct_ramp(corner_size=0.05)
        np.testing.assert_allclose(
                s_xy_corr.data, np.zeros_like(data_xy), atol=1e-8)
        np.testing.assert_allclose(
                s_yx_corr.data, np.zeros_like(data_yx), atol=1e-8)

        data_tilt = np.swapaxes(np.dstack((
            array_x+array_y,
            np.fliplr(array_x)+array_y)), 0, 2).astype('float64')
        s_tilt = DPCSignal(data_tilt)
        s_tilt_corr = s_tilt.correct_ramp()
        np.testing.assert_allclose(
                s_tilt_corr.data, np.zeros_like(data_tilt), atol=1e-8)
        s_tilt.correct_ramp(out=s_tilt)
        np.testing.assert_allclose(
                s_tilt.data, np.zeros_like(data_tilt), atol=1e-8)

    def test_correct_ramp_random(self):
        array_x, array_y = np.meshgrid(range(64), range(64))
        data_tilt = np.swapaxes(np.dstack((
            array_x+array_y,
            np.fliplr(array_x)+array_y)), 0, 2).astype('float64')
        data_random = data_tilt + np.random.random(size=(2, 64, 64))*10
        s_random = DPCSignal(data_random)
        s_random_corr = s_random.correct_ramp()
        np.testing.assert_allclose(
                s_random_corr.data, np.zeros_like(data_random), atol=10)
        s_random.correct_ramp(out=s_random)
        np.testing.assert_allclose(
                s_random.data, np.zeros_like(data_random), atol=10)

    def test_correct_ramp_one_large_value(self):
        array_x, array_y = np.meshgrid(range(64), range(64))
        data = np.swapaxes(np.dstack((
            array_x+array_y,
            np.fliplr(array_x)+array_y)), 0, 2).astype('float64')
        data[:, 20:30, 30:40] += 1000
        s = DPCSignal(data)
        s_corr = s.correct_ramp()
        s_corr.data[:, 20:30, 30:40] -= 1000
        print(s_corr.data.max())
        np.testing.assert_allclose(
                s_corr.data, np.zeros_like(data), atol=1e-8)


class test_dpc_signal_color_signal(unittest.TestCase):

    def test_get_color_signal(self):
        array_x, array_y = np.meshgrid(range(64), range(64))
        data_tilt = np.swapaxes(np.dstack((
            array_x+array_y,
            np.fliplr(array_x)+array_y)), 0, 2).astype('float64')
        data_random = data_tilt + np.random.random(size=(2, 64, 64))*10
        s_random = DPCSignal(data_random)
        s_random.get_color_signal()


class test_dpc_signal_bivariate_histogram(unittest.TestCase):

    def test_get_bivariate_histogram(self):
        array_x, array_y = np.meshgrid(range(64), range(64))
        data_tilt = np.swapaxes(np.dstack((
            array_x+array_y,
            np.fliplr(array_x)+array_y)), 0, 2).astype('float64')
        data_random = data_tilt + np.random.random(size=(2, 64, 64))*10
        s_random = DPCSignal(data_random)
        s_random.get_bivariate_histogram()

