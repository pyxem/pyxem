import unittest
import fpd_data_processing.make_diffraction_test_data as TestData
import fpd_data_processing.api as fp
import numpy as np


class test_center_finder(unittest.TestCase):

    def test_same_pixel(self):
        x0, y0 = 500., 500.
        test_data = TestData.TestData(size_x=1000, size_y=1000, default=False)
        test_data.add_ring(x0=x0, y0=y0, r=300, I=10, lw_pix=2)
        s = test_data.signal
        s.axes_manager[0].offset = -501.
        s.axes_manager[1].offset = -501.
        s_centre_position = fp.radial.get_optimal_centre_position(
                s, radial_signal_span=(280,310), steps=2, step_size=1)
        x,y = fp.radial.get_coordinate_of_min(s_centre_position)
        self.assertTrue( (x0-0.5) <= x and x <= (x0+0.5) )
        self.assertTrue( (x0-0.5) <= x and x <= (x0+0.5) )
