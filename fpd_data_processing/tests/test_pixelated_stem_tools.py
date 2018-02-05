import unittest
import pytest
from fpd_data_processing.pixelated_stem_class import PixelatedSTEM
import fpd_data_processing.pixelated_stem_tools as pst
import numpy as np
from hyperspy.signals import Signal2D


class test_pixelated_tools(unittest.TestCase):

    def test_find_longest_distance_manual(self):
        # These values are tested manually, against knowns results,
        # to make sure everything works fine.
        imX, imY = 10, 10
        centre_list = ((0, 0), (imX, 0), (0, imY), (imX, imY))
        distance = 14
        for cX, cY in centre_list:
            dist = pst._find_longest_distance(imX, imY, cX, cY, cX, cY)
            self.assertEqual(dist, distance)

        centre_list = ((1, 1), (imX-1, 1), (1, imY-1), (imX-1, imY-1))
        distance = 12
        for cX, cY in centre_list:
            dist = pst._find_longest_distance(imX, imY, cX, cY, cX, cY)
            self.assertEqual(dist, distance)

        imX, imY = 10, 5
        centre_list = ((0, 0), (imX, 0), (0, imY), (imX, imY))
        distance = 11
        for cX, cY in centre_list:
            dist = pst._find_longest_distance(imX, imY, cX, cY, cX, cY)
            self.assertEqual(dist, distance)

        imX, imY = 10, 10
        cX_min, cX_max, cY_min, cY_max = 1, 2, 2, 3
        distance = 12
        dist = pst._find_longest_distance(
                imX, imY, cX_min, cX_max, cY_min, cY_max)
        self.assertEqual(dist, distance)

    def test_find_longest_distance_all(self):
        imX, imY = 100, 100
        x_array, y_array = np.mgrid[0:10, 0:10]
        for x, y in zip(x_array.flatten(), y_array.flatten()):
            distance = int(((imX-x)**2+(imY-y)**2)**0.5)
            dist = pst._find_longest_distance(
                    imX, imY, x, y, x, y)
            self.assertEqual(dist, distance)

        x_array, y_array = np.mgrid[90:100, 90:100]
        for x, y in zip(x_array.flatten(), y_array.flatten()):
            distance = int((x**2+y**2)**0.5)
            dist = pst._find_longest_distance(
                    imX, imY, x, y, x, y)
            self.assertEqual(dist, distance)

        x_array, y_array = np.mgrid[0:10, 90:100]
        for x, y in zip(x_array.flatten(), y_array.flatten()):
            distance = int(((imX-x)**2+y**2)**0.5)
            dist = pst._find_longest_distance(
                    imX, imY, x, y, x, y)
            self.assertEqual(dist, distance)

        x_array, y_array = np.mgrid[90:100, 0:10]
        for x, y in zip(x_array.flatten(), y_array.flatten()):
            distance = int((x**2+(imY-y)**2)**0.5)
            dist = pst._find_longest_distance(
                    imX, imY, x, y, x, y)
            self.assertEqual(dist, distance)

    def test_make_centre_array_from_signal(self):
        s = Signal2D(np.ones((5, 10, 20, 7)))
        sa = s.axes_manager.signal_axes
        offset_x = sa[0].offset
        offset_y = sa[1].offset
        mask = pst._make_centre_array_from_signal(s)
        self.assertEqual(mask[0].shape[::-1], s.axes_manager.navigation_shape)
        self.assertEqual(mask[1].shape[::-1], s.axes_manager.navigation_shape)
        self.assertTrue((offset_x == mask[0]).all())
        self.assertTrue((offset_y == mask[1]).all())

        offset0_x, offset0_y = -3, -2
        sa[0].offset, sa[1].offset = offset0_x, offset0_y
        mask = pst._make_centre_array_from_signal(s)
        self.assertTrue((-offset0_x == mask[0]).all())
        self.assertTrue((-offset0_y == mask[1]).all())


class TestGetAngleSectorMask(unittest.TestCase):

    def test_0d(self):
        data_shape = (10, 8)
        data = np.ones(data_shape)*100.
        s = Signal2D(data)
        mask0 = pst._get_angle_sector_mask(s, angle0=0.0, angle1=np.pi/2)
        np.testing.assert_array_equal(
                mask0, np.zeros_like(mask0, dtype=np.bool))
        self.assertEqual(mask0.shape, data_shape)

        s.axes_manager.signal_axes[0].offset = -5
        s.axes_manager.signal_axes[1].offset = -5
        mask1 = pst._get_angle_sector_mask(s, angle0=0.0, angle1=np.pi/2)
        self.assertTrue(mask1[:5, :5].all())
        mask1[:5, :5] = False
        np.testing.assert_array_equal(
                mask1, np.zeros_like(mask1, dtype=np.bool))

        s.axes_manager.signal_axes[0].offset = -15
        s.axes_manager.signal_axes[1].offset = -15
        mask2 = pst._get_angle_sector_mask(s, angle0=0.0, angle1=np.pi/2)
        self.assertTrue(mask2.all())

        s.axes_manager.signal_axes[0].offset = -15
        s.axes_manager.signal_axes[1].offset = -15
        mask3 = pst._get_angle_sector_mask(s, angle0=np.pi*3/2, angle1=np.pi*2)
        self.assertFalse(mask3.any())

    def test_0d_com(self):
        data_shape = (10, 8)
        data = np.zeros(data_shape)
        s = PixelatedSTEM(data)
        s.data[4, 5] = 5.
        s_com = s.center_of_mass()
        pst._get_angle_sector_mask(
                s,
                centre_x_array=s_com.inav[0].data,
                centre_y_array=s_com.inav[1].data,
                angle0=np.pi*3/2, angle1=np.pi*2)

    def test_1d(self):
        data_shape = (5, 7, 10)
        data = np.ones(data_shape)*100.
        s = Signal2D(data)
        mask0 = pst._get_angle_sector_mask(s, angle0=0.0, angle1=np.pi/2)
        np.testing.assert_array_equal(
                mask0, np.zeros_like(mask0, dtype=np.bool))
        self.assertEqual(mask0.shape, data_shape)

        s.axes_manager.signal_axes[0].offset = -5
        s.axes_manager.signal_axes[1].offset = -4
        mask1 = pst._get_angle_sector_mask(s, angle0=0.0, angle1=np.pi/2)
        self.assertTrue(mask1[:, :4, :5].all())
        mask1[:, :4, :5] = False
        np.testing.assert_array_equal(
                mask1, np.zeros_like(mask1, dtype=np.bool))

    def test_2d(self):
        data_shape = (3, 5, 7, 10)
        data = np.ones(data_shape)*100.
        s = Signal2D(data)
        mask0 = pst._get_angle_sector_mask(s, angle0=np.pi, angle1=2*np.pi)
        self.assertEqual(mask0.shape, data_shape)

    def test_3d(self):
        data_shape = (5, 3, 5, 7, 10)
        data = np.ones(data_shape)*100.
        s = Signal2D(data)
        mask0 = pst._get_angle_sector_mask(s, angle0=np.pi, angle1=2*np.pi)
        self.assertEqual(mask0.shape, data_shape)

    def test_centre_xy(self):
        data_shape = (3, 5, 7, 10)
        data = np.ones(data_shape)*100.
        s = Signal2D(data)
        nav_shape = s.axes_manager.navigation_shape
        centre_x, centre_y = np.ones(
                nav_shape[::-1])*5, np.ones(nav_shape[::-1])*9
        mask0 = pst._get_angle_sector_mask(
            s, angle0=np.pi, angle1=2*np.pi,
            centre_x_array=centre_x, centre_y_array=centre_y)
        self.assertEqual(mask0.shape, data_shape)

    def test_bad_angles(self):
        s = Signal2D(np.zeros((100, 100)))
        with pytest.raises(ValueError):
            pst._get_angle_sector_mask(s, angle0=2, angle1=-1)

    def test_angles_across_pi(self):
        s = Signal2D(np.zeros((100, 100)))
        s.axes_manager[0].offset, s.axes_manager[1].offset = -49.5, -49.5
        mask0 = pst._get_angle_sector_mask(s, 0.5*np.pi, 1.5*np.pi)
        assert np.invert(mask0[:, 0:50]).all()
        assert mask0[:, 50:].all()

        mask1 = pst._get_angle_sector_mask(s, 2.5*np.pi, 3.5*np.pi)
        assert np.invert(mask1[:, 0:50]).all()
        assert mask1[:, 50:].all()

        mask2 = pst._get_angle_sector_mask(s, 4.5*np.pi, 5.5*np.pi)
        assert np.invert(mask2[:, 0:50]).all()
        assert mask2[:, 50:].all()

        mask3 = pst._get_angle_sector_mask(s, -3.5*np.pi, -2.5*np.pi)
        assert np.invert(mask3[:, 0:50]).all()
        assert mask3[:, 50:].all()

    def test_angles_across_zero(self):
        s = Signal2D(np.zeros((100, 100)))
        s.axes_manager[0].offset, s.axes_manager[1].offset = -50, -50
        mask0 = pst._get_angle_sector_mask(s, -0.5*np.pi, 0.5*np.pi)
        assert mask0[:, 0:50].all()
        assert np.invert(mask0[:, 50:]).all()

        mask1 = pst._get_angle_sector_mask(s, 1.5*np.pi, 2.5*np.pi)
        assert mask1[:, 0:50].all()
        assert np.invert(mask1[:, 50:]).all()

        mask2 = pst._get_angle_sector_mask(s, 3.5*np.pi, 4.5*np.pi)
        assert mask2[:, 0:50].all()
        assert np.invert(mask2[:, 50:]).all()

        mask3 = pst._get_angle_sector_mask(s, -4.5*np.pi, -3.5*np.pi)
        assert mask3[:, 0:50].all()
        assert np.invert(mask3[:, 50:]).all()

    def test_angles_more_than_2pi(self):
        s = Signal2D(np.zeros((100, 100)))
        s.axes_manager[0].offset, s.axes_manager[1].offset = -50, -50
        mask0 = pst._get_angle_sector_mask(s, 0.1*np.pi, 4*np.pi)
        assert mask0.all()

        mask1 = pst._get_angle_sector_mask(s, -1*np.pi, 2.1*np.pi)
        assert mask1.all()


class test_dpcsignal_tools(unittest.TestCase):

    def test_get_corner_value(self):
        corner_size = 0.1
        image_size = 100
        s = Signal2D(np.ones(shape=(image_size, image_size)))
        corner_list = pst._get_corner_value(s, corner_size=0.1)
        corner0, corner1 = corner_list[:, 0], corner_list[:, 1]
        corner2, corner3 = corner_list[:, 2], corner_list[:, 3]

        pos = image_size*corner_size*0.5
        high_value = s.axes_manager[0].high_value
        self.assertTrue(((pos, pos, 1) == corner0).all())
        self.assertTrue(((pos, high_value-pos, 1) == corner1).all())
        self.assertTrue(((high_value-pos, pos, 1) == corner2).all())
        self.assertTrue(((high_value-pos, high_value-pos, 1) == corner3).all())


class test_copy_signal2d_axes_manager_metadata(unittest.TestCase):

    def setUp(self):
        s = Signal2D(np.zeros((50, 50)))
        s.axes_manager.signal_axes[0].offset = 10
        s.axes_manager.signal_axes[1].offset = 20
        s.axes_manager.signal_axes[0].scale = 0.5
        s.axes_manager.signal_axes[1].scale = 0.3
        self.s = s

    def test_simple(self):
        s = self.s
        s_new = Signal2D(np.zeros((50, 50)))
        pst._copy_signal2d_axes_manager_metadata(s, s_new)
        self.assertEqual(
                s.axes_manager.signal_axes[0].offset,
                s_new.axes_manager.signal_axes[0].offset)
        self.assertEqual(
                s.axes_manager.signal_axes[1].offset,
                s_new.axes_manager.signal_axes[1].offset)
        self.assertEqual(
                s.axes_manager.signal_axes[0].scale,
                s_new.axes_manager.signal_axes[0].scale)
        self.assertEqual(
                s.axes_manager.signal_axes[1].scale,
                s_new.axes_manager.signal_axes[1].scale)
