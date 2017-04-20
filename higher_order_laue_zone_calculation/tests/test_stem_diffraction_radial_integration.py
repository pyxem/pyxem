import matplotlib
matplotlib.use('Agg')
import unittest
import numpy as np
from hyperspy.signals import Signal2D

import higher_order_laue_zone_calculation.stem_diffraction_radial_integration as sdri


class testGetDiskCentreFromSignal(unittest.TestCase):

    def setUp(self):
        data_shape = 10, 10, 20, 20
        data = np.zeros(data_shape)
        self.pixel_x, self.pixel_y = 15, 7
        data[:, :, self.pixel_y, self.pixel_x] = 2
        self.signal = Signal2D(data)
    
    def test_get_disk_centre_from_signal(self):
        signal = self.signal
        comx, comy = sdri._get_disk_centre_from_signal(signal)
        self.assertTrue(np.all(comx==self.pixel_x))
        self.assertTrue(np.all(comy==self.pixel_y))
        self.assertFalse(np.all(comx==(self.pixel_x+1)))
        self.assertFalse(np.all(comy==(self.pixel_y-1)))
