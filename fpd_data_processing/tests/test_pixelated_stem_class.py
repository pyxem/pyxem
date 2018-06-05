import pytest
import unittest
import numpy as np
from numpy.random import randint
import dask.array as da
from hyperspy.signals import Signal2D
from fpd_data_processing.pixelated_stem_class import PixelatedSTEM
from fpd_data_processing.pixelated_stem_class import LazyPixelatedSTEM
import fpd_data_processing.make_diffraction_test_data as mdtd
import fpd_data_processing.dummy_data as dd


class TestPixelatedStem:

    def test_create(self):
        array0 = np.zeros(shape=(10, 10, 10, 10))
        s0 = PixelatedSTEM(array0)
        assert array0.shape == s0.axes_manager.shape

        # This should fail due to PixelatedSTEM inheriting
        # signal2D, i.e. the data has to be at least
        # 2-dimensions
        with pytest.raises(ValueError):
            PixelatedSTEM(np.zeros(10))

        array1 = np.zeros(shape=(10, 10))
        s1 = PixelatedSTEM(array1)
        assert array1.shape == s1.axes_manager.shape


class TestPlotting:

    def test_simple_plot(self):
        s = PixelatedSTEM(np.zeros(shape=(3, 4, 6, 10)))
        s.plot()

    def test_navigation_signal_plot(self):
        s = PixelatedSTEM(np.zeros(shape=(3, 4, 6, 10)))
        s_nav = Signal2D(np.zeros((3, 4)))
        s.navigation_signal = s_nav
        s.plot()

    def test_navigation_signal_plot_argument(self):
        s = PixelatedSTEM(np.zeros(shape=(3, 4, 6, 10)))
        s_nav = Signal2D(np.zeros((3, 4)))
        s.plot(navigator=s_nav)


class TestPixelatedStemFlipDiffraction:

    def test_flip_x(self):
        array = np.zeros(shape=(3, 4, 6, 10))
        array[:, :, :, 5:] = 1
        s = PixelatedSTEM(array)
        assert (s.data[:, :, :, 5:] == 1).all()
        s_flip = s.flip_diffraction_x()
        assert (s_flip.data[:, :, :, 5:] == 0).all()
        assert (s_flip.data[:, :, :, :5] == 1).all()

    def test_flip_y(self):
        array = np.zeros(shape=(3, 4, 6, 10))
        array[:, :, 3:, :] = 1
        s = PixelatedSTEM(array)
        assert (s.data[:, :, 3:, :] == 1).all()
        s_flip = s.flip_diffraction_y()
        assert (s_flip.data[:, :, 3:, :] == 0).all()
        assert (s_flip.data[:, :, :3, :] == 1).all()


class TestPixelatedSTEMThresholdAndMask:

    @pytest.mark.parametrize(
            "shape", [(9, 9, 9, 9), (4, 8, 6, 3), (6, 3, 2, 5)])
    def test_no_change(self, shape):
        s = PixelatedSTEM(np.zeros(shape))
        s1 = s.threshold_and_mask()
        assert (s.data == s1.data).all()

    @pytest.mark.parametrize("mask", [(3, 5, 1), (6, 8, 1), (4, 6, 1)])
    def test_mask(self, mask):
        s = PixelatedSTEM(np.ones((10, 10, 10, 10)))
        s1 = s.threshold_and_mask(mask=mask)
        slice0 = np.s_[:, :, mask[1]-mask[2]:mask[1]+mask[2]+1, mask[0]]
        assert (s1.data[slice0] == 1.).all()
        slice1 = np.s_[:, :, mask[1], mask[0]-mask[2]:mask[0]+mask[2]+1]
        assert (s1.data[slice1] == 1.).all()
        s1.data[slice0] = 0
        s1.data[slice1] = 0
        assert (s1.data == 0).all()

    @pytest.mark.parametrize("x, y", [(3, 5), (7, 5), (5, 2)])
    def test_threshold(self, x, y):
        s = PixelatedSTEM(np.random.randint(0, 10, size=(10, 10, 10, 10)))
        s.data[:, :, x, y] = 1000000
        s1 = s.threshold_and_mask(threshold=1)
        assert (s1.data[:, :, x, y] == 1.).all()
        s1.data[:, :, x, y] = 0
        assert (s1.data == 0).all()

    def test_threshold_mask(self):
        s = PixelatedSTEM(np.zeros((12, 11, 13, 10)))
        s.data[:, :, 1, 2] = 1000000
        s.data[:, :, 8, 6] = 10
        s1 = s.threshold_and_mask(threshold=1)
        assert (s1.data[:, :, 1, 2] == 1.).all()
        s1.data[:, :, 1, 2] = 0
        assert (s1.data == 0).all()

        s2 = s.threshold_and_mask(threshold=1, mask=(6, 8, 1))
        assert (s2.data[:, :, 8, 6] == 1.).all()
        s2.data[:, :, 8, 6] = 0
        assert (s2.data == 0).all()

    def test_lazy_exception(self):
        s = dd.get_disk_shift_simple_test_signal(lazy=True)
        with pytest.raises(NotImplementedError):
            s.threshold_and_mask()


class TestPixelatedStemCenterOfMass:

    def test_center_of_mass_0d(self):
        x0, y0 = 2, 3
        array0 = np.zeros(shape=(7, 9))
        array0[y0, x0] = 1
        s0 = PixelatedSTEM(array0)
        s_com0 = s0.center_of_mass()
        assert (s_com0.inav[0].data == x0).all()
        assert (s_com0.inav[1].data == y0).all()
        assert s_com0.axes_manager.navigation_shape == (2, )
        assert s_com0.axes_manager.signal_shape == ()

    def test_center_of_mass_1d(self):
        x0, y0 = 2, 3
        array0 = np.zeros(shape=(5, 6, 4))
        array0[:, y0, x0] = 1
        s0 = PixelatedSTEM(array0)
        s_com0 = s0.center_of_mass()
        assert (s_com0.inav[0].data == x0).all()
        assert (s_com0.inav[1].data == y0).all()
        assert s_com0.axes_manager.navigation_shape == (2, )
        assert s_com0.axes_manager.signal_shape == (5, )

    def test_center_of_mass(self):
        x0, y0 = 5, 7
        array0 = np.zeros(shape=(10, 10, 10, 10))
        array0[:, :, y0, x0] = 1
        s0 = PixelatedSTEM(array0)
        s_com0 = s0.center_of_mass()
        assert (s_com0.inav[0].data == x0).all()
        assert (s_com0.inav[1].data == y0).all()

        array1 = np.zeros(shape=(10, 10, 10, 10))
        x1_array = np.random.randint(0, 10, size=(10, 10))
        y1_array = np.random.randint(0, 10, size=(10, 10))
        for i in range(10):
            for j in range(10):
                array1[i, j, y1_array[i, j], x1_array[i, j]] = 1
        s1 = PixelatedSTEM(array1)
        s_com1 = s1.center_of_mass()
        assert (s_com1.data[0] == x1_array).all()
        assert (s_com1.data[1] == y1_array).all()

    def test_center_of_mass_different_shapes(self):
        array1 = np.zeros(shape=(5, 10, 15, 8))
        x1_array = np.random.randint(1, 7, size=(5, 10))
        y1_array = np.random.randint(1, 14, size=(5, 10))
        for i in range(5):
            for j in range(10):
                array1[i, j, y1_array[i, j], x1_array[i, j]] = 1
        s1 = PixelatedSTEM(array1)
        s_com1 = s1.center_of_mass()
        assert (s_com1.inav[0].data == x1_array).all()
        assert (s_com1.inav[1].data == y1_array).all()

    def test_center_of_mass_different_shapes2(self):
        psX, psY = 11, 9
        s = mdtd.generate_4d_data(
                probe_size_x=psX, probe_size_y=psY, ring_x=None)
        s_com = s.center_of_mass()
        assert s_com.axes_manager.shape == (2, psX, psY)

    def test_different_shape_no_blur_no_downscale(self):
        y, x = np.mgrid[75:83:9j, 85:95:11j]
        s = mdtd.generate_4d_data(
                probe_size_x=11, probe_size_y=9, ring_x=None,
                image_size_x=160, image_size_y=140, disk_x=x, disk_y=y,
                disk_r=40, disk_I=20, blur=False, blur_sigma=1,
                downscale=False)
        s_com = s.center_of_mass()
        assert (s_com.inav[0].data == x).all()
        assert (s_com.inav[1].data == y).all()

    def test_different_shape_no_downscale(self):
        y, x = np.mgrid[75:83:9j, 85:95:11j]
        s = mdtd.generate_4d_data(
                probe_size_x=11, probe_size_y=9, ring_x=None,
                image_size_x=160, image_size_y=140, disk_x=x, disk_y=y,
                disk_r=40, disk_I=20, blur=True, blur_sigma=1,
                downscale=False)
        s_com = s.center_of_mass()
        np.testing.assert_allclose(s_com.inav[0].data, x)
        np.testing.assert_allclose(s_com.inav[1].data, y)

    def test_mask(self):
        y, x = np.mgrid[75:83:9j, 85:95:11j]
        s = mdtd.generate_4d_data(
                probe_size_x=11, probe_size_y=9, ring_x=None,
                image_size_x=160, image_size_y=140, disk_x=x, disk_y=y,
                disk_r=40, disk_I=20, blur=False, blur_sigma=1,
                downscale=False)
        s.data[:, :, 15, 10] = 1000000
        s_com0 = s.center_of_mass()
        s_com1 = s.center_of_mass(mask=(90, 79, 60))
        assert not (s_com0.inav[0].data == x).all()
        assert not (s_com0.inav[1].data == y).all()
        assert (s_com1.inav[0].data == x).all()
        assert (s_com1.inav[1].data == y).all()

    def test_mask_2(self):
        x, y = 60, 50
        s = mdtd.generate_4d_data(
                probe_size_x=5, probe_size_y=5, ring_x=None,
                image_size_x=120, image_size_y=100, disk_x=x, disk_y=y,
                disk_r=20, disk_I=20, blur=False, downscale=False)
        # Add one large value
        s.data[:, :, 50, 30] = 200000  # Large value to the left of the disk

        # Center of mass should not be in center of the disk, due to the
        # large value.
        s_com0 = s.center_of_mass()
        assert not (s_com0.inav[0].data == x).all()
        assert (s_com0.inav[1].data == y).all()

        # Here, the large value is masked
        s_com1 = s.center_of_mass(mask=(60, 50, 25))
        assert (s_com1.inav[0].data == x).all()
        assert (s_com1.inav[1].data == y).all()

        # Here, the large value is right inside the edge of the mask
        s_com3 = s.center_of_mass(mask=(60, 50, 31))
        assert not (s_com3.inav[0].data == x).all()
        assert (s_com3.inav[1].data == y).all()

        # Here, the large value is right inside the edge of the mask
        s_com4 = s.center_of_mass(mask=(59, 50, 30))
        assert not (s_com4.inav[0].data == x).all()
        assert (s_com4.inav[1].data == y).all()

        s.data[:, :, 50, 30] = 0
        s.data[:, :, 80, 60] = 200000  # Large value under the disk

        # The large value is masked
        s_com5 = s.center_of_mass(mask=(60, 50, 25))
        assert (s_com5.inav[0].data == x).all()
        assert (s_com5.inav[1].data == y).all()

        # The large value just not masked
        s_com6 = s.center_of_mass(mask=(60, 50, 31))
        assert (s_com6.inav[0].data == x).all()
        assert not (s_com6.inav[1].data == y).all()

        # The large value just not masked
        s_com7 = s.center_of_mass(mask=(60, 55, 25))
        assert (s_com7.inav[0].data == x).all()
        assert not (s_com7.inav[1].data == y).all()

    def test_threshold(self):
        x, y = 60, 50
        s = mdtd.generate_4d_data(
                probe_size_x=4, probe_size_y=3, ring_x=None,
                image_size_x=120, image_size_y=100, disk_x=x, disk_y=y,
                disk_r=20, disk_I=20, blur=False, blur_sigma=1,
                downscale=False)
        s.data[:, :, 0:30, 0:30] = 5

        # The extra values are ignored due to thresholding
        s_com0 = s.center_of_mass(threshold=2)
        assert (s_com0.inav[0].data == x).all()
        assert (s_com0.inav[1].data == y).all()

        # The extra values are not ignored
        s_com1 = s.center_of_mass(threshold=1)
        assert not (s_com1.inav[0].data == x).all()
        assert not (s_com1.inav[1].data == y).all()

        # The extra values are not ignored
        s_com2 = s.center_of_mass()
        assert not (s_com2.inav[0].data == x).all()
        assert not (s_com2.inav[1].data == y).all()

    def test_threshold_and_mask(self):
        x, y = 60, 50
        s = mdtd.generate_4d_data(
                probe_size_x=4, probe_size_y=3, ring_x=None,
                image_size_x=120, image_size_y=100, disk_x=x, disk_y=y,
                disk_r=20, disk_I=20, blur=False, blur_sigma=1,
                downscale=False)
        s.data[:, :, 0:30, 0:30] = 5
        s.data[:, :, 1, -2] = 60

        # The extra values are ignored due to thresholding and mask
        s_com0 = s.center_of_mass(threshold=2, mask=(60, 50, 50))
        assert (s_com0.inav[0].data == x).all()
        assert (s_com0.inav[1].data == y).all()

        # The extra values are not ignored
        s_com1 = s.center_of_mass(mask=(60, 50, 50))
        assert not (s_com1.inav[0].data == x).all()
        assert not (s_com1.inav[1].data == y).all()

        # The extra values are not ignored
        s_com3 = s.center_of_mass(threshold=2)
        assert not (s_com3.inav[0].data == x).all()
        assert not (s_com3.inav[1].data == y).all()

        # The extra values are not ignored
        s_com4 = s.center_of_mass()
        assert not (s_com4.inav[0].data == x).all()
        assert not (s_com4.inav[1].data == y).all()

    def test_1d_signal(self):
        x = np.arange(45, 45+9).reshape((1, 9))
        y = np.arange(55, 55+9).reshape((1, 9))
        s = mdtd.generate_4d_data(
                probe_size_x=9, probe_size_y=1, ring_x=None,
                image_size_x=120, image_size_y=100, disk_x=x, disk_y=y,
                disk_r=20, disk_I=20, blur=False, blur_sigma=1,
                downscale=False)
        s_com = s.inav[:, 0].center_of_mass()
        assert (s_com.inav[0].data == x).all()
        assert (s_com.inav[1].data == y).all()

    def test_0d_signal(self):
        x, y = 40, 51
        s = mdtd.generate_4d_data(
                probe_size_x=1, probe_size_y=1, ring_x=None,
                image_size_x=120, image_size_y=100, disk_x=x, disk_y=y,
                disk_r=20, disk_I=20, blur=False, blur_sigma=1,
                downscale=False)
        s_com = s.inav[0, 0].center_of_mass()
        assert (s_com.inav[0].data == x).all()
        assert (s_com.inav[1].data == y).all()

    def test_lazy(self):
        y, x = np.mgrid[75:83:9j, 85:95:11j]
        s = mdtd.generate_4d_data(
                probe_size_x=11, probe_size_y=9, ring_x=None,
                image_size_x=160, image_size_y=140, disk_x=x, disk_y=y,
                disk_r=40, disk_I=20, blur=True, blur_sigma=1,
                downscale=False)
        s_lazy = LazyPixelatedSTEM(
                da.from_array(s.data, chunks=(1, 1, 140, 160)))
        s_lazy_com = s_lazy.center_of_mass()
        np.testing.assert_allclose(s_lazy_com.inav[0].data, x)
        np.testing.assert_allclose(s_lazy_com.inav[1].data, y)

    def test_compare_lazy_and_nonlazy(self):
        y, x = np.mgrid[75:83:9j, 85:95:11j]
        s = mdtd.generate_4d_data(
                probe_size_x=11, probe_size_y=9, ring_x=None,
                image_size_x=160, image_size_y=140, disk_x=x, disk_y=y,
                disk_r=40, disk_I=20, blur=True, blur_sigma=1,
                downscale=False)
        s_lazy = LazyPixelatedSTEM(
                da.from_array(s.data, chunks=(1, 1, 140, 160)))
        s_com = s.center_of_mass()
        s_lazy_com = s_lazy.center_of_mass()
        np.testing.assert_equal(s_com.data, s_lazy_com.data)

        com_nav_extent = s_com.axes_manager.navigation_extent
        lazy_com_nav_extent = s_lazy_com.axes_manager.navigation_extent
        assert com_nav_extent == lazy_com_nav_extent

        com_sig_extent = s_com.axes_manager.signal_extent
        lazy_com_sig_extent = s_lazy_com.axes_manager.signal_extent
        assert com_sig_extent == lazy_com_sig_extent

    def test_lazy_result(self):
        data = da.ones((10, 10, 20, 20), chunks=(10, 10, 10, 10))
        s_lazy = LazyPixelatedSTEM(data)
        s_lazy_com = s_lazy.center_of_mass(lazy_result=True)
        assert s_lazy_com._lazy


class TestPixelatedStemRadialIntegration:

    def test_simple(self):
        array0 = np.ones(shape=(10, 10, 40, 40))
        s0 = PixelatedSTEM(array0)
        s0_r = s0.radial_integration()
        assert (s0_r.data[:, :, :-1] == 1).all()

        data_shape = 2, 2, 11, 11
        array1 = np.zeros(data_shape)
        array1[:, :, 5, 5] = 1
        s1 = PixelatedSTEM(array1)
        s1.axes_manager.signal_axes[0].offset = -5
        s1.axes_manager.signal_axes[1].offset = -5
        s1_r = s1.radial_integration()
        assert np.all(s1_r.data[:, :, 0] == 1)
        assert np.all(s1_r.data[:, :, 1:] == 0)

    def test_different_shape(self):
        array = np.ones(shape=(7, 9, 30, 40))
        s = PixelatedSTEM(array)
        s_r = s.radial_integration()
        assert (s_r.data[:, :, :-2] == 1).all()

    def test_nav_0(self):
        data_shape = (40, 40)
        array0 = np.ones(shape=data_shape)
        s0 = PixelatedSTEM(array0)
        s0_r = s0.radial_integration()
        assert s0_r.axes_manager.navigation_dimension == 0
        assert (s0_r.data[:-1] == 1).all()

    def test_nav_1(self):
        data_shape = (5, 40, 40)
        array0 = np.ones(shape=data_shape)
        s0 = PixelatedSTEM(array0)
        s0_r = s0.radial_integration()
        assert s0_r.axes_manager.navigation_shape == data_shape[:1]
        assert (s0_r.data[:, :-1] == 1).all()

    def test_big_value(self):
        data_shape = (5, 40, 40)
        big_value = 50000000
        array0 = np.ones(shape=data_shape)*big_value
        s0 = PixelatedSTEM(array0)
        s0_r = s0.radial_integration()
        assert s0_r.axes_manager.navigation_shape == data_shape[:1]
        assert (s0_r.data[:, :-1] == big_value).all()

    def test_correct_radius_simple(self):
        x, y, r, px, py = 40, 51, 30, 4, 5
        s = mdtd.generate_4d_data(
                probe_size_x=px, probe_size_y=py,
                image_size_x=120, image_size_y=100,
                disk_I=0, ring_x=x, ring_y=y, ring_r=r, ring_I=5,
                blur=True, downscale=False)
        s.axes_manager.signal_axes[0].offset = -x
        s.axes_manager.signal_axes[1].offset = -y
        s_r = s.radial_integration()
        assert s_r.axes_manager.navigation_shape == (px, py)
        assert (s_r.data.argmax(axis=-1) == 30).all()

    def test_correct_radius_random(self):
        x, y, px, py = 56, 48, 4, 5
        r = np.random.randint(20, 40, size=(py, px))
        s = mdtd.generate_4d_data(
                probe_size_x=px, probe_size_y=py,
                image_size_x=120, image_size_y=100,
                disk_I=0, ring_x=x, ring_y=y, ring_r=r, ring_I=5,
                blur=True, downscale=False)
        s.axes_manager.signal_axes[0].offset = -x
        s.axes_manager.signal_axes[1].offset = -y
        s_r = s.radial_integration()
        assert (s_r.data.argmax(axis=-1) == r).all()

    def test_correct_disk_x_y_and_radius_random(self):
        x, y, px, py = 56, 48, 4, 5
        x, y = randint(45, 55, size=(py, px)), randint(45, 55, size=(py, px))
        r = randint(20, 40, size=(py, px))
        s = mdtd.generate_4d_data(
                probe_size_x=px, probe_size_y=py,
                image_size_x=120, image_size_y=100,
                disk_x=x, disk_y=y, disk_r=5, disk_I=20,
                ring_x=x, ring_y=y, ring_r=r, ring_I=5,
                blur=True, downscale=False)
        s_com = s.center_of_mass()
        s_r = s.radial_integration(
                centre_x=s_com.inav[0].data, centre_y=s_com.inav[1].data)
        s_r = s_r.isig[15:]  # Do not include the disk
        r -= 15  # Need to shift the radius, due to not including the disk
        assert (s_r.data.argmax(axis=-1) == r).all()


class TestPixelatedStemRadialIntegrationLazy(unittest.TestCase):

    def test_simple(self):
        array0 = da.ones(shape=(10, 10, 40, 40), chunks=(5, 5, 5, 5))
        s0 = LazyPixelatedSTEM(array0)
        s0_r = s0.radial_integration()
        self.assertTrue((s0_r.data[:, :, :-1] == 1).all())

        data_shape = 2, 2, 11, 11
        array1 = np.zeros(data_shape)
        array1[:, :, 5, 5] = 1
        dask_array = da.from_array(array1, chunks=(1, 1, 1, 1))
        s1 = LazyPixelatedSTEM(dask_array)
        s1.axes_manager.signal_axes[0].offset = -5
        s1.axes_manager.signal_axes[1].offset = -5
        s1_r = s1.radial_integration()
        self.assertTrue(np.all(s1_r.data[:, :, 0] == 1))
        self.assertTrue(np.all(s1_r.data[:, :, 1:] == 0))

    def test_different_shape(self):
        array = da.ones(shape=(7, 9, 30, 40), chunks=(3, 3, 5, 5))
        s = LazyPixelatedSTEM(array)
        s_r = s.radial_integration()
        self.assertTrue((s_r.data[:, :, :-2] == 1).all())

    def test_nav_1(self):
        data_shape = (5, 40, 40)
        array0 = da.ones(shape=data_shape, chunks=(5, 5, 5))
        s0 = LazyPixelatedSTEM(array0)
        s0_r = s0.radial_integration()
        self.assertEqual(s0_r.axes_manager.navigation_shape, data_shape[:1])
        self.assertTrue((s0_r.data[:, :-1] == 1).all())

    def test_big_value(self):
        data_shape = (5, 40, 40)
        big_value = 50000000
        array0 = np.ones(shape=data_shape)*big_value
        dask_array = da.from_array(array0, chunks=(2, 10, 10))
        s0 = LazyPixelatedSTEM(dask_array)
        s0_r = s0.radial_integration()
        self.assertEqual(s0_r.axes_manager.navigation_shape, data_shape[:1])
        self.assertTrue((s0_r.data[:, :-1] == big_value).all())

    def test_correct_radius_simple(self):
        x, y, r, px, py = 40, 51, 30, 4, 5
        s = mdtd.generate_4d_data(
                probe_size_x=px, probe_size_y=py,
                image_size_x=120, image_size_y=100,
                disk_I=0, ring_x=x, ring_y=y, ring_r=r, ring_I=5,
                blur=True, downscale=False)
        dask_array = da.from_array(s.data, chunks=(4, 4, 50, 50))
        s = LazyPixelatedSTEM(dask_array)
        s.axes_manager.signal_axes[0].offset = -x
        s.axes_manager.signal_axes[1].offset = -y
        s_r = s.radial_integration()
        self.assertEqual(s_r.axes_manager.navigation_shape, (px, py))
        self.assertTrue((s_r.data.argmax(axis=-1) == 30).all())

    def test_correct_radius_random(self):
        x, y, px, py = 56, 48, 4, 5
        r = np.random.randint(20, 40, size=(py, px))
        s = mdtd.generate_4d_data(
                probe_size_x=px, probe_size_y=py,
                image_size_x=120, image_size_y=100,
                disk_I=0, ring_x=x, ring_y=y, ring_r=r, ring_I=5,
                blur=True, downscale=False)
        dask_array = da.from_array(s.data, chunks=(4, 4, 50, 50))
        s = LazyPixelatedSTEM(dask_array)
        s.axes_manager.signal_axes[0].offset = -x
        s.axes_manager.signal_axes[1].offset = -y
        s_r = s.radial_integration()
        self.assertTrue((s_r.data.argmax(axis=-1) == r).all())

    def test_correct_disk_x_y_and_radius_random(self):
        x, y, px, py = 56, 48, 4, 5
        x, y = randint(45, 55, size=(py, px)), randint(45, 55, size=(py, px))
        r = randint(20, 40, size=(py, px))
        s = mdtd.generate_4d_data(
                probe_size_x=px, probe_size_y=py,
                image_size_x=120, image_size_y=100,
                disk_x=x, disk_y=y, disk_r=5, disk_I=20,
                ring_x=x, ring_y=y, ring_r=r, ring_I=5,
                blur=True, downscale=False)
        dask_array = da.from_array(s.data, chunks=(4, 4, 50, 50))
        s = LazyPixelatedSTEM(dask_array)
        s_com = s.center_of_mass()
        s_r = s.radial_integration(
                centre_x=s_com.inav[0].data, centre_y=s_com.inav[1].data)
        s_r = s_r.isig[15:]  # Do not include the disk
        r -= 15  # Need to shift the radius, due to not including the disk
        self.assertTrue((s_r.data.argmax(axis=-1) == r).all())


class test_pixelated_stem_angle_sector(unittest.TestCase):

    def test_get_angle_sector_mask_simple(self):
        array = np.zeros((10, 10, 10, 10))
        array[:, :, 0:5, 0:5] = 1
        s = PixelatedSTEM(array)
        s.axes_manager.signal_axes[0].offset = -4.5
        s.axes_manager.signal_axes[1].offset = -4.5
        mask = s.angular_mask(0.0, 0.5*np.pi)
        self.assertTrue(mask[:, :, 0:5, 0:5].all())
        self.assertFalse(mask[:, :, 5:, :].any())
        self.assertFalse(mask[:, :, :, 5:].any())

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
        self.assertTrue(np.all(s_r0.isig[0:6].data == 1.))

        mask1 = s.angular_mask(0, np.pi)
        s_r1 = s.radial_integration(
                centre_x=centre_x_array, centre_y=centre_y_array,
                mask_array=mask1)
        self.assertTrue(np.all(s_r1.isig[0:6].data == 0.5))

        mask2 = s.angular_mask(0.0, 2*np.pi)
        s_r2 = s.radial_integration(
                centre_x=centre_x_array, centre_y=centre_y_array,
                mask_array=mask2)
        self.assertTrue(np.all(s_r2.isig[0:6].data == 0.25))

        mask3 = s.angular_mask(np.pi, 2*np.pi)
        s_r3 = s.radial_integration(
                centre_x=centre_x_array, centre_y=centre_y_array,
                mask_array=mask3)
        self.assertTrue(np.all(s_r3.data == 0.0))

    def test_com_angle_sector_mask(self):
        x, y = 4, 7
        array = np.zeros((5, 4, 10, 20))
        array[:, :, y, x] = 1
        s = PixelatedSTEM(array)
        s_com = s.center_of_mass()
        s.angular_mask(
                0.0, 0.5*np.pi,
                centre_x_array=s_com.inav[0].data,
                centre_y_array=s_com.inav[1].data)


class test_angular_slice_radial_integration(unittest.TestCase):

    def test_same_radius(self):
        x, y, r, px, py, angleN = 56, 48, 20, 4, 5, 20
        s = mdtd.generate_4d_data(
                probe_size_x=px, probe_size_y=py,
                image_size_x=120, image_size_y=100,
                disk_I=0, ring_x=x, ring_y=y, ring_r=r, ring_I=5,
                blur=True, downscale=False)
        s_ar = s.angular_slice_radial_integration(
                centre_x=x, centre_y=y, angleN=20)
        self.assertTrue(s_ar.axes_manager.navigation_shape, (x, y, angleN))
        self.assertTrue((s_ar.data.argmax(-1) == r).all())

    def test_different_radius(self):
        x, y, r, px, py, iX, iY = 50, 50, 20, 6, 5, 100, 100
        kwrds = {
                'probe_size_x': px, 'probe_size_y': py,
                'image_size_x': iX, 'image_size_y': iY, 'disk_I': 0,
                'ring_x': x, 'ring_y': y, 'ring_r': r, 'ring_I': 5,
                'blur': True, 'downscale': False}
        r0, r1, r2, r3 = 20, 25, 30, 27
        kwrds['ring_r'] = r0
        s = mdtd.generate_4d_data(**kwrds)
        kwrds['ring_r'] = r1
        s1 = mdtd.generate_4d_data(**kwrds)
        kwrds['ring_r'] = r2
        s2 = mdtd.generate_4d_data(**kwrds)
        kwrds['ring_r'] = r3
        s3 = mdtd.generate_4d_data(**kwrds)

        s.data[:, :, :y, x:] = s1.data[:, :, :y, x:]
        s.data[:, :, y:, x:] = s2.data[:, :, y:, x:]
        s.data[:, :, y:, :x] = s3.data[:, :, y:, :x]

        s_ar = s.angular_slice_radial_integration(
                centre_x=x, centre_y=y, angleN=4)
        self.assertTrue((s_ar.inav[:, :, 0].data.argmax(axis=-1) == r0).all())
        self.assertTrue((s_ar.inav[:, :, 1].data.argmax(axis=-1) == r1).all())
        self.assertTrue((s_ar.inav[:, :, 2].data.argmax(axis=-1) == r2).all())
        self.assertTrue((s_ar.inav[:, :, 3].data.argmax(axis=-1) == r3).all())

    def test_different_radius_not_square_image(self):
        x, y, r, px, py, iX, iY = 40, 55, 20, 6, 5, 120, 100
        kwrds = {
                'probe_size_x': px, 'probe_size_y': py,
                'image_size_x': iX, 'image_size_y': iY, 'disk_I': 0,
                'ring_x': x, 'ring_y': y, 'ring_r': r, 'ring_I': 5,
                'blur': True, 'downscale': False}
        r0, r1, r2, r3 = 20, 25, 30, 27
        kwrds['ring_r'] = r0
        s = mdtd.generate_4d_data(**kwrds)
        kwrds['ring_r'] = r1
        s1 = mdtd.generate_4d_data(**kwrds)
        kwrds['ring_r'] = r2
        s2 = mdtd.generate_4d_data(**kwrds)
        kwrds['ring_r'] = r3
        s3 = mdtd.generate_4d_data(**kwrds)

        s.data[:, :, :y, x:] = s1.data[:, :, :y, x:]
        s.data[:, :, y:, x:] = s2.data[:, :, y:, x:]
        s.data[:, :, y:, :x] = s3.data[:, :, y:, :x]

        s_ar = s.angular_slice_radial_integration(
                centre_x=x, centre_y=y, angleN=4)
        self.assertTrue((s_ar.inav[:, :, 0].data.argmax(axis=-1) == r0).all())
        self.assertTrue((s_ar.inav[:, :, 1].data.argmax(axis=-1) == r1).all())
        self.assertTrue((s_ar.inav[:, :, 2].data.argmax(axis=-1) == r2).all())
        self.assertTrue((s_ar.inav[:, :, 3].data.argmax(axis=-1) == r3).all())

    def test_slice_overlap(self):
        x, y, r, px, py, iX, iY = 40, 55, 20, 6, 5, 120, 100
        kwrds = {
                'probe_size_x': px, 'probe_size_y': py,
                'image_size_x': iX, 'image_size_y': iY, 'disk_I': 0,
                'ring_x': x, 'ring_y': y, 'ring_r': r, 'ring_I': 5,
                'blur': True, 'downscale': False}
        r0, r1 = 20, 30
        kwrds['ring_r'] = r0
        s = mdtd.generate_4d_data(**kwrds)
        kwrds['ring_r'] = r1
        kwrds['ring_I'] = 500
        s1 = mdtd.generate_4d_data(**kwrds)

        s.data[:, :, y:, :] = s1.data[:, :, y:, :]

        s_ar = s.angular_slice_radial_integration(
                centre_x=x, centre_y=y, angleN=2)
        self.assertTrue((s_ar.inav[:, :, 0].data.argmax(axis=-1) == r0).all())
        self.assertTrue((s_ar.inav[:, :, 1].data.argmax(axis=-1) == r1).all())

        s_ar1 = s.angular_slice_radial_integration(
                centre_x=x, centre_y=y, angleN=2, slice_overlap=0.1)
        self.assertTrue((s_ar1.inav[:, :, 0].data.argmax(axis=-1) == r1).all())
        self.assertTrue((s_ar1.inav[:, :, 1].data.argmax(axis=-1) == r1).all())

        with pytest.raises(ValueError):
                s.angular_slice_radial_integration(slice_overlap=1.2)
        with pytest.raises(ValueError):
                s.angular_slice_radial_integration(slice_overlap=-0.2)


class test_pixelated_stem_virtual_annular_dark_field(unittest.TestCase):

    def test_simple(self):
        shape = (5, 9, 12, 14)
        s = PixelatedSTEM(np.zeros(shape))
        s1 = s.virtual_annular_dark_field(cx=6, cy=6, r_inner=2, r=5)
        self.assertEqual(s1.axes_manager.signal_shape, (shape[1], shape[0]))
        self.assertEqual(s1.data.sum(), 0.)

    def test_one_value(self):
        shape = (5, 9, 12, 14)
        s = PixelatedSTEM(np.zeros(shape))
        s.data[:, :, 9, 9] = 1
        s1 = s.virtual_annular_dark_field(cx=6, cy=6, r_inner=2, r=5)
        self.assertEqual(s1.axes_manager.signal_shape, (shape[1], shape[0]))
        self.assertTrue((s1.data == 1.).all())

    def test_lazy(self):
        shape = (5, 9, 12, 14)
        data = da.zeros((5, 9, 12, 14), chunks=(10, 10, 10, 10))
        s = LazyPixelatedSTEM(data)
        s1 = s.virtual_annular_dark_field(cx=6, cy=6, r_inner=2, r=5)
        self.assertEqual(s1.axes_manager.signal_shape, (shape[1], shape[0]))


class test_pixelated_stem_virtual_bright_field(unittest.TestCase):

    def test_simple(self):
        shape = (5, 9, 12, 14)
        s = PixelatedSTEM(np.zeros(shape))
        s1 = s.virtual_bright_field()
        self.assertEqual(s1.axes_manager.signal_shape, (shape[1], shape[0]))
        self.assertEqual(s1.data.sum(), 0.)

    def test_one_value(self):
        shape = (5, 9, 12, 14)
        s = PixelatedSTEM(np.zeros(shape))
        s.data[:, :, 10, 13] = 1
        s1 = s.virtual_bright_field()
        self.assertEqual(s1.axes_manager.signal_shape, (shape[1], shape[0]))
        self.assertTrue((s1.data == 1.).all())

        s2 = s.virtual_bright_field(6, 6, 2)
        self.assertEqual(s2.axes_manager.signal_shape, (shape[1], shape[0]))
        self.assertEqual(s2.data.sum(), 0)

    def test_lazy(self):
        shape = (5, 9, 12, 14)
        data = da.zeros((5, 9, 12, 14), chunks=(10, 10, 10, 10))
        s = LazyPixelatedSTEM(data)
        s1 = s.virtual_bright_field(cx=6, cy=6, r=5)
        self.assertEqual(s1.axes_manager.signal_shape, (shape[1], shape[0]))

    def test_lazy_result(self):
        data = da.ones((10, 10, 20, 20), chunks=(10, 10, 10, 10))
        s = LazyPixelatedSTEM(data)
        s_out = s.virtual_bright_field(lazy_result=True)
        assert s_out._lazy


class test_pixelated_stem_rotate_diffraction(unittest.TestCase):

    def test_rotate_diffraction_keep_shape(self):
        shape = (7, 5, 4, 15)
        s = PixelatedSTEM(np.zeros(shape))
        s_rot = s.rotate_diffraction(angle=45)
        assert s.axes_manager.shape == s_rot.axes_manager.shape

        s_lazy = LazyPixelatedSTEM(da.zeros(shape, chunks=(1, 1, 1, 1)))
        s_rot_lazy = s_lazy.rotate_diffraction(angle=45)
        assert s_lazy.axes_manager.shape == s_rot_lazy.axes_manager.shape

    def test_rotate_diffraction_values(self):
        data = np.zeros((10, 5, 12, 14))
        data[:, :, 6:, 7:] = 1
        s = PixelatedSTEM(data)
        s_rot = s.rotate_diffraction(angle=180)
        np.testing.assert_almost_equal(
                s.data[0, 0, :6, :7], np.zeros_like(s.data[0, 0, :6, :7]))
        np.testing.assert_almost_equal(
                s_rot.data[0, 0, :6, :7],
                np.ones_like(s_rot.data[0, 0, :6, :7]))
        s_rot.data[:, :, :6, :7] = 0
        np.testing.assert_almost_equal(s_rot.data, np.zeros_like(s.data))


class TestPixelatedStemShiftDiffraction:

    @pytest.mark.parametrize("shift_x,shift_y", [(2, 5), (-6, -1), (2, -4)])
    def test_single_shift(self, shift_x, shift_y):
        s = PixelatedSTEM(np.zeros((10, 10, 30, 40)))
        x, y = 20, 10
        s.data[:, :, y, x] = 1
        s_shift = s.shift_diffraction(shift_x=shift_x, shift_y=shift_y)
        assert s_shift.data[0, 0, y - shift_y, x - shift_x] == 1
        s_shift.data[:, :, y - shift_y, x - shift_x] = 0
        assert s_shift.data.sum() == 0

    @pytest.mark.parametrize("centre_x,centre_y", [(25, 25), (30, 20)])
    def test_random_shifts(self, centre_x, centre_y):
        y, x = np.mgrid[20:30:7j, 20:30:5j]
        s = mdtd.generate_4d_data(
                probe_size_x=5, probe_size_y=7,
                disk_x=x, disk_y=y, disk_r=1, blur=True, ring_x=None)
        s_com = s.center_of_mass()
        s_com.data[0] -= centre_x
        s_com.data[1] -= centre_y
        s_shift = s.shift_diffraction(
                shift_x=s_com.inav[0].data, shift_y=s_com.inav[1].data)
        s_shift_c = s_shift.center_of_mass()
        np.testing.assert_allclose(
                s_shift_c.data[0], np.ones_like(s_shift_c.data[0])*centre_x)
        np.testing.assert_allclose(
                s_shift_c.data[1], np.ones_like(s_shift_c.data[1])*centre_y)

    def test_inplace(self):
        s = PixelatedSTEM(np.zeros((10, 10, 30, 40)))
        x, y, shift_x, shift_y = 20, 10, 4, -3
        s.data[:, :, y, x] = 1
        s.shift_diffraction(shift_x=shift_x, shift_y=shift_y, inplace=True)
        assert s.data[0, 0, y - shift_y, x - shift_x] == 1
        s.data[:, :, y - shift_y, x - shift_x] = 0
        assert s.data.sum() == 0

    def test_lazy(self):
        data = np.zeros((10, 10, 30, 30))
        x, y = 20, 10
        data[:, :, y, x] += 1
        data = da.from_array(data, chunks=(5, 5, 5, 5))
        s = LazyPixelatedSTEM(data)
        shift_x, shift_y = 4, 3
        s_shift = s.shift_diffraction(shift_x=shift_x, shift_y=shift_y)
        s_shift.compute()
        assert s_shift.data[0, 0, y - shift_y, x - shift_x] == 1
        s_shift.data[:, :, y - shift_y, x - shift_x] = 0
        assert s_shift.data.sum() == 0
