import unittest
import numpy as np
from fpd_data_processing.pixelated_stem_class import PixelatedSTEM
import fpd_data_processing.make_diffraction_test_data as mdtd


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


class test_pixelated_stem_center_of_mass(unittest.TestCase):

    def test_center_of_mass_0d(self):
        x0, y0 = 2, 3
        array0 = np.zeros(shape=(7, 9))
        array0[y0, x0] = 1
        s0 = PixelatedSTEM(array0)
        s_com0 = s0.center_of_mass()
        self.assertTrue((s_com0.inav[0].data == x0).all())
        self.assertTrue((s_com0.inav[1].data == y0).all())
        self.assertEqual(s_com0.axes_manager.navigation_shape, (2, ))
        self.assertEqual(s_com0.axes_manager.signal_shape, ())

    def test_center_of_mass_1d(self):
        x0, y0 = 2, 3
        array0 = np.zeros(shape=(5, 6, 4))
        array0[:, y0, x0] = 1
        s0 = PixelatedSTEM(array0)
        s_com0 = s0.center_of_mass()
        self.assertTrue((s_com0.inav[0].data == x0).all())
        self.assertTrue((s_com0.inav[1].data == y0).all())
        self.assertEqual(s_com0.axes_manager.navigation_shape, (2, ))
        self.assertEqual(s_com0.axes_manager.signal_shape, (5, ))

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

    def test_center_of_mass_different_shapes(self):
        array1 = np.zeros(shape=(5, 10, 15, 8))
        x1_array = np.random.randint(1, 7, size=(5, 10))
        y1_array = np.random.randint(1, 14, size=(5, 10))
        for i in range(5):
            for j in range(10):
                array1[i, j, y1_array[i, j], x1_array[i, j]] = 1
        s1 = PixelatedSTEM(array1)
        s_com1 = s1.center_of_mass()
        self.assertTrue((s_com1.inav[0].data == x1_array).all())
        self.assertTrue((s_com1.inav[1].data == y1_array).all())

    def test_center_of_mass_different_shapes2(self):
        psX, psY = 11, 9
        s = mdtd.generate_4d_disk_data(probe_size_x=psX, probe_size_y=psY)
        s_com = s.center_of_mass()
        self.assertEqual(s_com.axes_manager.shape, (2, psX, psY))

    def test_different_shape_no_blur_no_downscale(self):
        y, x = np.mgrid[75:83:9j, 85:95:11j]
        s = mdtd.generate_4d_disk_data(
                probe_size_x=11, probe_size_y=9,
                image_size_x=160, image_size_y=140, disk_x=x, disk_y=y,
                disk_r=40, I=20, blur=False, blur_sigma=1, downscale=False)
        s_com = s.center_of_mass()
        self.assertTrue((s_com.inav[0].data == x).all())
        self.assertTrue((s_com.inav[1].data == y).all())

    def test_different_shape_no_downscale(self):
        y, x = np.mgrid[75:83:9j, 85:95:11j]
        s = mdtd.generate_4d_disk_data(
                probe_size_x=11, probe_size_y=9,
                image_size_x=160, image_size_y=140, disk_x=x, disk_y=y,
                disk_r=40, I=20, blur=True, blur_sigma=1, downscale=False)
        s_com = s.center_of_mass()
        np.testing.assert_allclose(s_com.inav[0].data, x)
        np.testing.assert_allclose(s_com.inav[1].data, y)

    def test_mask(self):
        y, x = np.mgrid[75:83:9j, 85:95:11j]
        s = mdtd.generate_4d_disk_data(
                probe_size_x=11, probe_size_y=9,
                image_size_x=160, image_size_y=140, disk_x=x, disk_y=y,
                disk_r=40, I=20, blur=False, blur_sigma=1, downscale=False)
        s.data[:, :, 15, 10] = 1000000
        s_com0 = s.center_of_mass()
        s_com1 = s.center_of_mass(mask=(90, 79, 60))
        self.assertFalse((s_com0.inav[0].data == x).all())
        self.assertFalse((s_com0.inav[1].data == y).all())
        self.assertTrue((s_com1.inav[0].data == x).all())
        self.assertTrue((s_com1.inav[1].data == y).all())

    def test_mask_2(self):
        x, y = 60, 50
        s = mdtd.generate_4d_disk_data(
                probe_size_x=5, probe_size_y=5,
                image_size_x=120, image_size_y=100, disk_x=x, disk_y=y,
                disk_r=20, I=20, blur=False, downscale=False)
        # Add one large value
        s.data[:, :, 50, 30] = 200000  # Large value to the left of the disk

        # Center of mass should not be in center of the disk, due to the
        # large value.
        s_com0 = s.center_of_mass()
        self.assertFalse((s_com0.inav[0].data == x).all())
        self.assertTrue((s_com0.inav[1].data == y).all())

        # Here, the large value is masked
        s_com1 = s.center_of_mass(mask=(60, 50, 25))
        self.assertTrue((s_com1.inav[0].data == x).all())
        self.assertTrue((s_com1.inav[1].data == y).all())

        # Here, the large value is right inside the edge of the mask
        s_com3 = s.center_of_mass(mask=(60, 50, 31))
        self.assertFalse((s_com3.inav[0].data == x).all())
        self.assertTrue((s_com3.inav[1].data == y).all())

        # Here, the large value is right inside the edge of the mask
        s_com4 = s.center_of_mass(mask=(59, 50, 30))
        self.assertFalse((s_com4.inav[0].data == x).all())
        self.assertTrue((s_com4.inav[1].data == y).all())

        s.data[:, :, 50, 30] = 0
        s.data[:, :, 80, 60] = 200000  # Large value under the disk

        # The large value is masked
        s_com5 = s.center_of_mass(mask=(60, 50, 25))
        self.assertTrue((s_com5.inav[0].data == x).all())
        self.assertTrue((s_com5.inav[1].data == y).all())

        # The large value just not masked
        s_com6 = s.center_of_mass(mask=(60, 50, 31))
        self.assertTrue((s_com6.inav[0].data == x).all())
        self.assertFalse((s_com6.inav[1].data == y).all())

        # The large value just not masked
        s_com7 = s.center_of_mass(mask=(60, 55, 25))
        self.assertTrue((s_com7.inav[0].data == x).all())
        self.assertFalse((s_com7.inav[1].data == y).all())


class test_pixelated_stem_radial_integration(unittest.TestCase):

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

    def test_radial_integration_different_shape(self):
        array = np.ones(shape=(7, 9, 30, 40))
        s = PixelatedSTEM(array)
        s_r = s.radial_integration()
        self.assertTrue((s_r.data[:, :, :-2] == 1).all())

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


class test_pixelated_stem_angle_sector(unittest.TestCase):

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
                centre_x=centre_x_array, centre_y=centre_y_array,
                mask_array=mask0)
        self.assertTrue(np.all(s_r0.isig[0:6].data==1.))

        mask1 = s.angular_mask(0, np.pi)
        s_r1 = s.radial_integration(
                centre_x=centre_x_array, centre_y=centre_y_array,
                mask_array=mask1)
        self.assertTrue(np.all(s_r1.isig[0:6].data==0.5))

        mask2 = s.angular_mask(0.0, 2*np.pi)
        s_r2 = s.radial_integration(
                centre_x=centre_x_array, centre_y=centre_y_array,
                mask_array=mask2)
        self.assertTrue(np.all(s_r2.isig[0:6].data==0.25))

        mask3 = s.angular_mask(np.pi, 2*np.pi)
        s_r3 = s.radial_integration(
                centre_x=centre_x_array, centre_y=centre_y_array,
                mask_array=mask3)
        self.assertTrue(np.all(s_r3.data==0.0))

    def test_com_angle_sector_mask(self):
        x, y = 4, 7
        array = np.zeros((5, 4, 10, 20))
        array[:, :, y, x] = 1
        s = PixelatedSTEM(array)
        s_com = s.center_of_mass()
        mask0 = s.angular_mask(
                0.0, 0.5*np.pi,
                centre_x_array=s_com.inav[0].data,
                centre_y_array=s_com.inav[1].data)
