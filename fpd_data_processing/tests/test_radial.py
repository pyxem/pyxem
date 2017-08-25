import os
import unittest
import numpy as np
import fpd_data_processing.api as fdp
from fpd_data_processing.radial import centre_comparison


class test_radial_module(unittest.TestCase):

    def test_centre_comparison(self):
        s = fdp.PixelatedSTEM(np.ones((20, 20)))
        centre_x_list, centre_y_list = [9, 10, 11], [9, 10, 11]
        s_list = centre_comparison(s, centre_x_list, centre_y_list)
        self.assertEqual(len(s_list), 3)

        s1 = fdp.PixelatedSTEM(np.ones((5, 20, 20)))
        with self.assertRaises(ValueError):
            centre_comparison(s1, centre_x_list, centre_y_list)

        s2 = fdp.PixelatedSTEM(np.ones((30, 30)))
        s2_list = centre_comparison(s2, centre_x_list, centre_y_list, angleN=20)
        for temp_s in s2_list:
            self.assertEqual(temp_s.axes_manager.navigation_shape, (20, ))

        s3 = fdp.PixelatedSTEM(np.ones((40, 40)))
        s3_list = centre_comparison(
                s3, centre_x_list, centre_y_list, angleN=10,
                crop_radial_signal=(3, 8))
        for temp_s in s3_list:
            self.assertEqual(temp_s.axes_manager.signal_shape, (5, ))
            self.assertEqual(temp_s.axes_manager.navigation_shape, (10, ))
