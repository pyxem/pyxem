import unittest
from fpd_data_processing.pixelated_stem_class import PixelatedSTEM
import numpy as np


class test_pixelated_stem(unittest.TestCase):

    def test_create(self):
        array0 = np.zeros(shape=(10, 10, 10, 10))
        s0 = PixelatedSTEM(array0)
        self.assertEqual(array0.shape, s0.axes_manager.shape)

        # This should fail due to PixelatedSTEM inheriting
        # signal2D, i.e. the data has to be at least
        # 2-dimensions
        with self.assertRaises(ValueError):
            PixelatedSTEM(np.zeros(10))

        array1 = np.zeros(shape=(10, 10))
        s1 = PixelatedSTEM(array1)
        self.assertEqual(array1.shape, s1.axes_manager.shape)

    def test_center_of_mass(self):
        x0, y0 = 5, 7
        array0 = np.zeros(shape=(10, 10, 10, 10))
        array0[:, :, y0, x0] = 1
        s0 = PixelatedSTEM(array0)
        s_com0 = s0.center_of_mass()
        self.assertTrue((s_com0.inav[0].data == x0).all())
        self.assertTrue((s_com0.inav[1].data == y0).all())
        
        array1 = np.zeros(shape=(10, 10, 10, 10))
        x1_array = np.random.randint(0, 10, size=(10, 10))
        y1_array = np.random.randint(0, 10, size=(10, 10))
        for i in range(10):
            for j in range(10):
                array1[i, j, y1_array[i, j], x1_array[i, j]] = 1
        s1 = PixelatedSTEM(array1)
        s_com1 = s1.center_of_mass()
        self.assertTrue((s_com1.data[0] == x1_array).all())
        self.assertTrue((s_com1.data[1] == y1_array).all())

    def test_radial_integration(self):
        array0 = np.ones(shape=(10, 10, 40, 40))
        s0 = PixelatedSTEM(array0)
        s0_r = s0.radial_integration()
        self.assertTrue((s0_r.data[:, :, :-1] == 1).all())

        data_shape = 2, 2, 11, 11
        array1 = np.zeros(data_shape)
        array1[: ,: , 5, 5] = 1
        s1 = PixelatedSTEM(array1)
        s1.axes_manager.signal_axes[0].offset = -5
        s1.axes_manager.signal_axes[1].offset = -5
        s1_r = s1.radial_integration()
        self.assertTrue(np.all(s1_r.data[:,:,0]==1))
        self.assertTrue(np.all(s1_r.data[:,:,1:]==0))

    def test_radial_integration_nav_0(self):
        data_shape = (40, 40)
        array0 = np.ones(shape=data_shape)
        s0 = PixelatedSTEM(array0)
        s0_r = s0.radial_integration()
        self.assertEqual(s0_r.axes_manager.navigation_dimension, 0)
        self.assertTrue((s0_r.data[:-1] == 1).all())

    def test_radial_integration_nav_1(self):
        data_shape = (5, 40, 40)
        array0 = np.ones(shape=data_shape)
        s0 = PixelatedSTEM(array0)
        s0_r = s0.radial_integration()
        self.assertEqual(s0_r.axes_manager.navigation_shape, data_shape[:1])
        self.assertTrue((s0_r.data[:,:-1] == 1).all())

    def test_radial_integration_big_value(self):
        data_shape = (5, 40, 40)
        big_value = 50000000
        array0 = np.ones(shape=data_shape)*big_value
        s0 = PixelatedSTEM(array0)
        s0_r = s0.radial_integration()
        self.assertEqual(s0_r.axes_manager.navigation_shape, data_shape[:1])
        self.assertTrue((s0_r.data[:,:-1] == big_value).all())


    def test_get_angle_sector_mask_simple(self):
        array = np.zeros((10, 10, 10, 10))
        array[:, :, 0:5, 0:5] = 1
        s = PixelatedSTEM(array)
        s.axes_manager.signal_axes[0].offset = -4.5
        s.axes_manager.signal_axes[1].offset = -4.5
        mask = s.angular_mask(0.0, 0.5*np.pi)
        self.assertTrue(mask[:, :, 0:5, 0:5].all())
        self.assertFalse(mask[:, :, 5:,:].any())
        self.assertFalse(mask[:, :, :,5:].any())

    def test_get_angle_sector_mask_radial_integration1(self):
        x, y = 4.5, 4.5
        array = np.zeros((10, 10, 10, 10))
        array[:, :, 0:5, 0:5] = 1
        centre_x_array = np.ones_like(array)*x
        centre_y_array = np.ones_like(array)*y
        s = PixelatedSTEM(array)
        s.axes_manager.signal_axes[0].offset = -x
        s.axes_manager.signal_axes[1].offset = -y
        mask0 = s.angular_mask(0.0, 0.5*np.pi)
        s_r0 = s.radial_integration(
                centre_x_array=centre_x_array, centre_y_array=centre_y_array,
                mask_array=mask0)
        self.assertTrue(np.all(s_r0.isig[0:6].data==1.))

        mask1 = s.angular_mask(0, np.pi)
        s_r1 = s.radial_integration(
                centre_x_array=centre_x_array, centre_y_array=centre_y_array,
                mask_array=mask1)
        self.assertTrue(np.all(s_r1.isig[0:6].data==0.5))

        mask2 = s.angular_mask(0.0, 2*np.pi)
        s_r2 = s.radial_integration(
                centre_x_array=centre_x_array, centre_y_array=centre_y_array,
                mask_array=mask2)
        self.assertTrue(np.all(s_r2.isig[0:6].data==0.25))

        mask3 = s.angular_mask(np.pi, 2*np.pi)
        s_r3 = s.radial_integration(
                centre_x_array=centre_x_array, centre_y_array=centre_y_array,
                mask_array=mask3)
        self.assertTrue(np.all(s_r3.data==0.0))
