import matplotlib
matplotlib.use('Agg')
import unittest
import numpy as np
from hyperspy.signals import Signal2D

import higher_order_laue_zone_calculation.stem_diffraction_radial_integration as sdri


#class testGetDiskCentreFromSignal(unittest.TestCase):
#
#    def setUp(self):
#        data_shape = 10, 10, 20, 20
#        data = np.zeros(data_shape)
#        self.pixel_x, self.pixel_y = 15, 7
#        data[:, :, self.pixel_y, self.pixel_x] = 2
#        self.signal = Signal2D(data)
#    
#    def test_get_disk_centre_from_signal(self):
#        signal = self.signal
#        comx, comy = sdri._get_disk_centre_from_signal(signal)
#        self.assertTrue(np.all(comx==self.pixel_x))
#        self.assertTrue(np.all(comy==self.pixel_y))
#        self.assertFalse(np.all(comx==(self.pixel_x+1)))
#        self.assertFalse(np.all(comy==(self.pixel_y-1)))


class testGetRadialProfileOfDiffImage(unittest.TestCase):

    def test_get_radial_profile_of_diff_image_simple(self):
        data_shape = 11, 11
        data = np.ones(data_shape)
        radial_profile = sdri._get_radial_profile_of_diff_image(data, 5, 5)
        self.assertTrue(np.all(radial_profile==1.))

    def test_get_radial_profile_of_diff_image_simple2(self):
        data_shape = 11, 11
        radial_result = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        data = np.zeros(data_shape)
        data[5, 5] = 1
        radial_profile = sdri._get_radial_profile_of_diff_image(data, 5, 5)
        self.assertTrue(np.all(radial_profile==radial_result))


class testGetAngleSectorMask(unittest.TestCase):

    def test_get_angle_sector_mask_simple(self):
        data = np.zeros((10, 10))
        data[0:5, 0:5] = 1
        signal = Signal2D(data)
        signal.axes_manager[0].offset = -4.5
        signal.axes_manager[1].offset = -4.5
        mask = sdri._get_angle_sector_mask(signal, 0.0, 0.5*np.pi)
        self.assertTrue(mask[0:5, 0:5].all())
        self.assertFalse(mask[5:,:].any())
        self.assertFalse(mask[:,5:].any())

    def test_get_angle_sector_mask_radial_integration1(self):
        x, y = -4.5, -4.5
        data = np.zeros((10, 10))
        data[0:5, 0:5] = 1
        signal = Signal2D(data)
        signal.axes_manager[0].offset = x
        signal.axes_manager[1].offset = y
        mask = sdri._get_angle_sector_mask(signal, 0.0, 0.5*np.pi)
        radial_profile = sdri._get_radial_profile_of_diff_image(
                signal.data, -x, -y, mask)
        self.assertTrue(np.all(radial_profile==1.))
        mask = sdri._get_angle_sector_mask(signal, 0.0, np.pi)
        radial_profile = sdri._get_radial_profile_of_diff_image(
                signal.data, -x, -y, mask)
        self.assertTrue(np.all(radial_profile==0.5))
        mask = sdri._get_angle_sector_mask(signal, 0.0, 2*np.pi)
        radial_profile = sdri._get_radial_profile_of_diff_image(
                signal.data, -x, -y, mask)
        self.assertTrue(np.all(radial_profile==0.25))
        mask = sdri._get_angle_sector_mask(signal, np.pi, 2*np.pi)
        radial_profile = sdri._get_radial_profile_of_diff_image(
                signal.data, -x, -y, mask)
        self.assertTrue(np.all(radial_profile==0.0))
