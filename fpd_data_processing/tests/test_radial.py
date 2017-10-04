import os
import unittest
import numpy as np
import fpd_data_processing.api as fp
import fpd_data_processing.radial as ra
import fpd_data_processing.make_diffraction_test_data as mdtd


class test_radial_module(unittest.TestCase):

    def test_centre_comparison(self):
        s = fp.PixelatedSTEM(np.ones((20, 20)))
        s_list = ra._centre_comparison(s, 1, 1)
        self.assertEqual(len(s_list), 9)

        s1 = fp.PixelatedSTEM(np.ones((5, 20, 20)))
        with self.assertRaises(ValueError):
            ra._centre_comparison(s1, 1, 1)

        s2 = fp.PixelatedSTEM(np.ones((30, 30)))
        s2_list = ra._centre_comparison(s2, 1, 1, angleN=20)
        for temp_s in s2_list:
            self.assertEqual(temp_s.axes_manager.navigation_shape, (20, ))

        s3 = fp.PixelatedSTEM(np.ones((40, 40)))
        s3_list = ra._centre_comparison(
                s3, 1, 1, angleN=10,
                crop_radial_signal=(3, 8))
        for temp_s in s3_list:
            self.assertEqual(temp_s.axes_manager.signal_shape, (5, ))
            self.assertEqual(temp_s.axes_manager.navigation_shape, (10, ))

    def test_get_coordinate_of_min(self):
        array = np.ones((10, 10))*10
        array[5, 7] = 1
        s = fp.PixelatedSTEM(array)
        s.axes_manager[0].offset = 55
        s.axes_manager[1].offset = 50
        s.axes_manager[0].scale = 0.5
        s.axes_manager[1].scale = 0.4
        min_pos = ra.get_coordinate_of_min(s)
        self.assertEqual(min_pos, (58.5, 57.5))

    def test_find_centre(self):
        x0, y0 = 300., 300.
        test_data = mdtd.MakeTestData(size_x=600, size_y=600, default=False)
        test_data.add_ring(x0=x0, y0=y0, r=200, I=10, lw_pix=2)
        s = test_data.signal
        s.axes_manager[0].offset = -301.
        s.axes_manager[1].offset = -301.
        s_centre_position = fp.radial.get_optimal_centre_position(
                s, radial_signal_span=(180, 210), steps=2, step_size=1)
        x,y = fp.radial.get_coordinate_of_min(s_centre_position)
        self.assertTrue((x0 - 0.5) <= x and x <= (x0 + 0.5))
        self.assertTrue((x0 - 0.5) <= x and x <= (x0 + 0.5))
