import unittest
import numpy as np
from fpd_data_processing.make_diffraction_test_data import (
        TestData, generate_4d_disk_data)


class test_make_diffraction_test_data(unittest.TestCase):

    def test_init(self):
        TestData()
        TestData(size_x=1, size_y=10, scale=0.05)
        TestData(
                size_x=200, size_y=100, scale=2, default=False, blur=False,
                blur_sigma=3., downscale=False)

    def test_zero_signal(self):
        test_data_1 = TestData(default=False)
        self.assertTrue((test_data_1.signal.data == 0.).all())

        test_data_2 = TestData()
        test_data_2.set_signal_zero()
        self.assertTrue((test_data_2.signal.data == 0.).all())

    def test_simple_disks(self):
        test0 = TestData(
                size_x=100, size_y=100,
                default=False, blur=False, downscale=False)
        test0.add_disk(50, 50, 1, 20)

        self.assertTrue((test0.signal.isig[49:52, 49:52].data == 20).all())
        test0.signal.data[49:52, 49:52] = 0
        self.assertFalse(test0.signal.data.any())

        test1 = TestData(
                size_x=100, size_y=100,
                default=False, blur=False, downscale=False)
        test1.add_disk(20, 40, 2, 30)
        self.assertTrue((test1.signal.isig[18:23, 38:43].data == 30).all())
        test1.signal.data[38:43, 18:23] = 0
        self.assertFalse(test1.signal.data.any())

        test2 = TestData(
                size_x=80, size_y=120,
                default=False, blur=False, downscale=False)
        test2.add_disk(50, 40, 1, 10)
        test2.add_disk(20, 30, 2, 30)
        self.assertTrue((test2.signal.isig[49:52, 39:42].data == 10).all())
        self.assertTrue((test2.signal.isig[19:22, 29:32].data == 30).all())
        test2.signal.data[39:42, 49:52] = 0
        test2.signal.data[28:33, 18:23] = 0
        self.assertFalse(test2.signal.data.any())

        test3 = TestData(size_x=150, size_y=50,
                default=False, blur=False, downscale=False)
        test3.add_disk(200, 400, 2, 30)
        self.assertFalse(test3.signal.data.any())

        test4 = TestData(size_x=150, size_y=50,
                default=False, blur=False, downscale=False)
        test4.add_disk(50, 50, 500, 500)
        self.assertTrue((test4.signal.data==500).all())

        test5 = TestData(
                size_x=100, size_y=200,
                default=False, blur=False, downscale=True)
        test5.add_disk(50, 50, 500, 10)
        self.assertTrue((test5.signal.data==10).all())

        test6 = TestData(
                size_x=100, size_y=200,
                default=False, blur=True, downscale=False)
        test6.add_disk(50, 30, 500, 10)
        test6_ref = np.full_like(test6.signal.data, 10.)
        np.testing.assert_allclose(test6.signal.data, test6_ref, rtol=1e-05)

    def test_large_disk(self):
        test_data_1 = TestData(
                size_x=10, size_y=10,
                scale=0.01, default=False)
        test_data_1.add_disk(x0=5, y0=5, r=20, I=100)
        self.assertTrue((test_data_1.signal.data > 0.).all())

    def test_radius(self):
        r = 20
        x0, y0 = 0, 0
        ring_1 = TestData(
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
        ring_2 = TestData(
                size_x=100, size_y=100,
                default=False, blur=False, downscale=False)
        ring_2.add_ring(x0=x0, y0=y0, r=r, I=10)
        slice_y_2 = ring_2.signal.data[:, -1]
        slice_x_2 = ring_2.signal.data[-1, :]
        r_x_idx_2 = np.where(slice_x_2 > 0)[0][0]
        r_y_idx_2 = np.where(slice_y_2 > 0)[0][0]
        self.assertTrue(r_x_idx_2 == (100 - r))
        self.assertTrue(r_y_idx_2 == (100 - r))


class test_generate_4d_disk_data(unittest.TestCase):

    def test_simple0(self):
        generate_4d_disk_data()

    def test_all_arguments(self):
        s = generate_4d_disk_data(
                probe_size_x=10, probe_size_y=10,
                image_size_x=50, image_size_y=50,
                disk_x=20, disk_y=20, disk_r=5, I=30,
                blur=True, blur_sigma=1,
                downscale=True)

    def test_different_size(self):
        s = generate_4d_disk_data(
                probe_size_x=5, probe_size_y=7,
                image_size_x=30, image_size_y=50)
        ax = s.axes_manager
        self.assertEqual(ax.navigation_dimension, 2)
        self.assertEqual(ax.signal_dimension, 2)
        self.assertEqual(ax.navigation_shape, (5, 7))
        self.assertEqual(ax.signal_shape, (30, 50))

    def test_disk_outside_image(self):
        s = generate_4d_disk_data(
                probe_size_x=6, probe_size_y=4,
                image_size_x=40, image_size_y=40,
                disk_x=1000, disk_y=1000, disk_r=5)
        self.assertTrue((s.data == 0).all())

    def test_disk_cover_whole_image(self):
        s = generate_4d_disk_data(
                probe_size_x=6, probe_size_y=4,
                image_size_x=20, image_size_y=20,
                disk_x=10, disk_y=10, disk_r=40, I=50,
                blur=False, downscale=False)
        self.assertTrue((s.data == 50).all())

    def test_disk_cover_whole_image(self):
        s = generate_4d_disk_data(
                probe_size_x=6, probe_size_y=4,
                image_size_x=20, image_size_y=20,
                disk_x=10, disk_y=10, disk_r=40, I=50,
                blur=False, downscale=False)
        self.assertTrue((s.data == 50).all())

    def test_disk_position_array(self):
        ps_x, ps_y, I = 4, 7, 30
        disk_x = np.random.randint(5, 35, size=(ps_y, ps_x))
        disk_y = np.random.randint(5, 45, size=(ps_y, ps_x))
        s = generate_4d_disk_data(
                probe_size_x=ps_x, probe_size_y=ps_y,
                image_size_x=40, image_size_y=50,
                disk_x=disk_x, disk_y=disk_y, disk_r = 1, I=I,
                blur=False, downscale=False)
        for x in range(ps_x):
            for y in range(ps_y):
                cX, cY = disk_x[y, x], disk_y[y, x]
                sl = np.s_[cY-1:cY+2, cX-1:cX+2]
                im = s.inav[x, y].data[:]
                self.assertTrue((im[sl] == I).all())
                im[sl] = 0
                self.assertFalse(im.any())
