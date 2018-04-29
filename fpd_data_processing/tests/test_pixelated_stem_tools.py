import unittest
import pytest
import numpy as np
import dask.array as da
from fpd_data_processing.pixelated_stem_class import PixelatedSTEM
import fpd_data_processing.pixelated_stem_tools as pst
from hyperspy.signals import Signal2D


class TestCenterOfMassDaskArray:

    def test_simple(self):
        numpy_array = np.zeros((10, 10, 50, 50))
        numpy_array[:, :, 25, 25] = 1
        dask_array = da.from_array(numpy_array, chunks=(5, 5, 5, 5))
        data = pst._center_of_mass_dask_array(
                dask_array, show_progressbar=False)
        assert data.shape == (10, 10, 2)
        assert (data == np.ones((10, 10, 2))*25).all()

    def test_mask(self):
        numpy_array = np.zeros((10, 10, 50, 50))
        numpy_array[:, :, 25, 25] = 1
        numpy_array[:, :, 1, 1] = 100000000
        dask_array = da.from_array(numpy_array, chunks=(5, 5, 5, 5))
        data0 = pst._center_of_mass_dask_array(
                dask_array, show_progressbar=False)
        np.testing.assert_allclose(data0, np.ones((10, 10, 2)), rtol=1e-05)
        mask = pst._make_circular_mask(25, 25, 50, 50, 10)
        data1 = pst._center_of_mass_dask_array(
                dask_array, mask=mask, show_progressbar=False)
        assert (data1 == np.ones((10, 10, 2))*25).all()

    def test_threshold(self):
        numpy_array = np.zeros((10, 10, 50, 50))
        numpy_array[:, :, 25, 25] = 1
        numpy_array[:, :, 1, 1] = 100000000
        dask_array = da.from_array(numpy_array, chunks=(5, 5, 5, 5))
        data0 = pst._center_of_mass_dask_array(
                dask_array, show_progressbar=False)
        np.testing.assert_allclose(data0, np.ones((10, 10, 2)), rtol=1e-05)
        data1 = pst._center_of_mass_dask_array(
                dask_array, threshold=1, show_progressbar=False)
        assert (data1 == np.ones((10, 10, 2))).all()


class TestRadialIntegrationDaskArray:

    def test_simple(self):
        dask_array = da.zeros((10, 10, 15, 15), chunks=(5, 5, 5, 5))
        centre_x, centre_y = np.ones((2, 100))*7.5
        data = pst._radial_integration_dask_array(
                dask_array, return_sig_size=11,
                centre_x=centre_x, centre_y=centre_y, show_progressbar=False)
        assert data.shape == (10, 10, 11)
        assert (data == 0.0).all()

    def test_different_size(self):
        dask_array = da.zeros((5, 10, 12, 15), chunks=(5, 5, 5, 5))
        centre_x, centre_y = np.ones((2, 100))*7.5
        data = pst._radial_integration_dask_array(
                dask_array, return_sig_size=11,
                centre_x=centre_x, centre_y=centre_y, show_progressbar=False)
        assert data.shape == (5, 10, 11)
        assert (data == 0.0).all()

    def test_mask(self):
        numpy_array = np.zeros((10, 10, 30, 30))
        numpy_array[:, :, 0, 0] = 1000
        numpy_array[:, :, -1, -1] = 1
        dask_array = da.from_array(numpy_array, chunks=(5, 5, 5, 5))
        centre_x, centre_y = np.ones((2, 100))*15
        data = pst._radial_integration_dask_array(
                dask_array, return_sig_size=22,
                centre_x=centre_x, centre_y=centre_y, show_progressbar=False)
        assert data.shape == (10, 10, 22)
        assert (data != 0.0).any()
        mask = pst._make_circular_mask(15, 15, 30, 30, 15)
        data = pst._radial_integration_dask_array(
                dask_array, return_sig_size=22,
                centre_x=centre_x, centre_y=centre_y, mask_array=mask,
                show_progressbar=False)
        assert data.shape == (10, 10, 22)
        assert (data == 0.0).all()


class TestThresholdAndMaskSingleFrame:

    def test_no_data_change(self):
        im = np.random.random((52, 43))
        im_out = pst._threshold_and_mask_single_frame(im)
        assert (im == im_out).all()

    def test_mask_all_zeros(self):
        im = np.zeros((12, 45))
        im[2, 10] = 10
        mask = pst._make_circular_mask(6, 31, 45, 12, 2)
        im_out = pst._threshold_and_mask_single_frame(im, mask=mask)
        assert (im_out == 0).all()

    def test_mask_not_all_zeros(self):
        im = np.zeros((12, 45))
        im[7, 31] = 10
        im[2, 43] = 5
        mask = pst._make_circular_mask(32, 7, 45, 12, 2)
        im_out = pst._threshold_and_mask_single_frame(im, mask=mask)
        assert im_out[7, 31] == 10
        im_out[7, 31] = 0
        assert (im_out == 0).all()

    def test_threshold(self):
        im = np.ones((12, 45))
        im[7, 12] = 10000000
        im_out = pst._threshold_and_mask_single_frame(im, threshold=2)
        assert im_out[7, 12] == 1
        im_out[7, 12] = 0
        assert (im_out == 0).all()


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


class TestShiftSingleFrame:

    @pytest.mark.parametrize("x,y", [(1, 1), (-2, 2), (-2, 5), (-4, -3)])
    def test_simple_shift(self, x, y):
        im = np.zeros((20, 20))
        x0, y0 = 10, 12
        im[x0, y0] = 1
        # Note that the shifts are switched when calling the function,
        # to stay consistent with the HyperSpy axis ordering
        im_shift = pst._shift_single_frame(im=im, shift_x=y, shift_y=x)
        assert im_shift[x0 - x, y0 - y] == 1
        assert im_shift.sum() == 1
        im_shift[x0 - x, y0 - y] = 0
        assert (im_shift == 0.0).all()

    @pytest.mark.parametrize(
            "x,y", [(0.5, 1), (-4, 2.5), (-6, 1.5), (-3.5, -4)])
    def test_single_half_shift(self, x, y):
        im = np.zeros((20, 20))
        x0, y0 = 10, 12
        im[x0, y0] = 1
        # Note that the shifts are switched when calling the function,
        # to stay consistent with the HyperSpy axis ordering
        im_shift = pst._shift_single_frame(im=im, shift_x=y, shift_y=x)
        assert im_shift[x0 - int(x), y0 - int(y)] == 0.5
        assert im_shift.max() == 0.5
        assert im_shift.sum() == 1

    @pytest.mark.parametrize(
            "x,y", [(0.5, 1.5), (-4.5, 2.5), (-6.5, 1.5), (-3.5, -4.5)])
    def test_two_half_shift(self, x, y):
        im = np.zeros((20, 20))
        x0, y0 = 10, 12
        im[x0, y0] = 1
        # Note that the shifts are switched when calling the function,
        # to stay consistent with the HyperSpy axis ordering
        im_shift = pst._shift_single_frame(im=im, shift_x=y, shift_y=x)
        assert im_shift[x0 - int(x), y0 - int(y)] == 0.25
        assert im_shift.max() == 0.25
        assert im_shift.sum() == 1


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


class AxesManagerMetadataCopying:

    def setup_method(self):
        s = Signal2D(np.zeros((50, 50)))
        s.axes_manager.signal_axes[0].offset = 10
        s.axes_manager.signal_axes[1].offset = 20
        s.axes_manager.signal_axes[0].scale = 0.5
        s.axes_manager.signal_axes[1].scale = 0.3
        s.axes_manager.signal_axes[0].name = 'axes0'
        s.axes_manager.signal_axes[1].name = 'axes1'
        s.axes_manager.signal_axes[0].units = 'unit0'
        s.axes_manager.signal_axes[1].units = 'unit1'
        self.s = s

    def test_copy_axes_manager(self):
        s = self.s
        s_new = Signal2D(np.zeros((50, 50)))
        pst._copy_signal2d_axes_manager_metadata(s, s_new)
        sa_ori = s.axes_manager.signal_axes
        sa_new = s_new.axes_manager.signal_axes
        assert sa_ori[0].offset == sa_new[0].offset
        assert sa_ori[1].offset == sa_new[1].offset
        assert sa_ori[0].scale == sa_new[0].scale
        assert sa_ori[1].scale == sa_new[1].scale
        assert sa_ori[0].name == sa_new[0].name
        assert sa_ori[1].name == sa_new[1].name
        assert sa_ori[0].units == sa_new[0].units
        assert sa_ori[1].units == sa_new[1].units

    def test_copy_axes_object(self):
        s = self.s
        s_new = Signal2D(np.zeros((50, 50)))
        ax_o = s.axes_manager[0]
        ax_n = s_new.axes_manager[0]
        pst._copy_axes_object_metadata(ax_o, ax_n)
        assert ax_o.offset == ax_n.offset
        assert ax_o.scale == ax_n.scale
        assert ax_o.name == ax_n.name
        assert ax_o.units == ax_n.units
