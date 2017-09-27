import unittest
import fpd_data_processing.diffraction_test_data_generator as TestData
import numpy as np


class test_diffraction_test_data_generator(unittest.TestCase):

    def test_init(self):
        TestData.TestData()
        TestData.TestData(size_x=1, size_y=10, scale=0.05)
        TestData.TestData(
                size_x=200, size_y=100, scale=2, default=False, blur=False,
                sigma_blur=3., downscale=False)

    def test_zero_signal(self):
        test_data_1 = TestData.TestData(default=False)
        self.assertTrue((test_data_1.signal.data == 0.).all())

        test_data_2 = TestData.TestData()
        test_data_2.set_signal_zero()
        self.assertTrue((test_data_2.signal.data == 0.).all())

    def test_simple_disks(self):
        test0 = TestData.TestData(
                size_x=100, size_y=100,
                default=False, blur=False, downscale=False)
        test0.add_disk(50, 50, 1, 20)

        self.assertTrue((test0.signal.isig[49:52, 49:52].data == 20).all())
        test0.signal.data[49:52, 49:52] = 0
        self.assertFalse(test0.signal.data.any())

        test1 = TestData.TestData(
                size_x=100, size_y=100,
                default=False, blur=False, downscale=False)
        test1.add_disk(20, 40, 2, 30)
        self.assertTrue((test1.signal.isig[18:23, 38:43].data == 30).all())
        test1.signal.data[38:43, 18:23] = 0
        self.assertFalse(test1.signal.data.any())

        test2 = TestData.TestData(
                size_x=80, size_y=120,
                default=False, blur=False, downscale=False)
        test2.add_disk(50, 40, 1, 10)
        test2.add_disk(20, 30, 2, 30)
        self.assertTrue((test2.signal.isig[49:52, 39:42].data == 10).all())
        self.assertTrue((test2.signal.isig[19:22, 29:32].data == 30).all())
        test2.signal.data[39:42, 49:52] = 0
        test2.signal.data[28:33, 18:23] = 0
        self.assertFalse(test2.signal.data.any())

        test3 = TestData.TestData(size_x=150, size_y=50,
                default=False, blur=False, downscale=False)
        test3.add_disk(200, 400, 2, 30)
        self.assertFalse(test3.signal.data.any())

        test4 = TestData.TestData(size_x=150, size_y=50,
                default=False, blur=False, downscale=False)
        test4.add_disk(50, 50, 500, 500)
        self.assertTrue((test4.signal.data==500).all())

        test5 = TestData.TestData(
                size_x=100, size_y=200,
                default=False, blur=False, downscale=True)
        test5.add_disk(50, 50, 500, 10)
        self.assertTrue((test5.signal.data==10).all())

        test6 = TestData.TestData(
                size_x=100, size_y=200,
                default=False, blur=True, downscale=False)
        test6.add_disk(50, 30, 500, 10)
        self.assertTrue((test6.signal.data==10).all())

    def test_large_disk(self):
        test_data_1 = TestData.TestData(
                size_x=10, size_y=10,
                scale=0.01, default=False)
        test_data_1.add_disk(x0=5, y0=5, r=20, I=100)
        self.assertTrue((test_data_1.signal.data > 0.).all())

    def test_radius(self):
        r = 20
        x0, y0 = 0, 0
        ring_1 = TestData.TestData(
                size_x=100, size_y=100,
                default=False, blur=False, downscale=False)
        ring_1.add_ring(x0=x0, y0=y0, r=r, I=10)
        slice_y = ring_1.signal.data[:, 0]
        slice_x = ring_1.signal.data[0, :]
        r_x_idx = np.where(slice_x > 0)[0][-1]
        r_y_idx = np.where(slice_y > 0)[0][-1]
        self.assertTrue(r_x_idx == r)
        self.assertTrue(r_y_idx == r)

        x0, y0 = 100, 100
        ring_2 = TestData.TestData(
                size_x=100, size_y=100,
                default=False, blur=False, downscale=False)
        ring_2.add_ring(x0=x0, y0=y0, r=r, I=10)
        slice_y_2 = ring_2.signal.data[:, -1]
        slice_x_2 = ring_2.signal.data[-1, :]
        r_x_idx_2 = np.where(slice_x_2 > 0)[0][0]
        r_y_idx_2 = np.where(slice_y_2 > 0)[0][0]
        self.assertTrue(r_x_idx_2 == (100 - r))
        self.assertTrue(r_y_idx_2 == (100 - r))
