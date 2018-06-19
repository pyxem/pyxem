import pytest
import numpy as np
import dask.array as da
import pixstem.dask_tools as dt
import pixstem.pixelated_stem_tools as pst


class TestCenterOfMassArray:

    def test_simple(self):
        numpy_array = np.zeros((10, 10, 50, 50))
        numpy_array[:, :, 25, 25] = 1
        dask_array = da.from_array(numpy_array, chunks=(5, 5, 5, 5))

        data = dt._center_of_mass_array(dask_array)
        data = data.compute()
        assert data.shape == (2, 10, 10)
        assert (data == np.ones((2, 10, 10))*25).all()

    def test_mask(self):
        numpy_array = np.zeros((10, 10, 50, 50))
        numpy_array[:, :, 25, 25] = 1
        numpy_array[:, :, 1, 1] = 100000000
        dask_array = da.from_array(numpy_array, chunks=(5, 5, 5, 5))
        data0 = dt._center_of_mass_array(dask_array)
        data0 = data0.compute()
        np.testing.assert_allclose(data0, np.ones((2, 10, 10)), rtol=1e-05)
        mask_array = pst._make_circular_mask(25, 25, 50, 50, 10)
        mask_array = np.invert(mask_array)
        data1 = dt._center_of_mass_array(dask_array, mask_array=mask_array)
        data1 = data1.compute()
        assert (data1 == np.ones((2, 10, 10))*25).all()

    def test_threshold(self):
        numpy_array = np.zeros((10, 10, 50, 50))
        numpy_array[:, :, 25, 25] = 1
        numpy_array[:, :, 1, 1] = 100000000
        dask_array = da.from_array(numpy_array, chunks=(5, 5, 5, 5))
        data0 = dt._center_of_mass_array(dask_array)
        data0 = data0.compute()
        np.testing.assert_allclose(data0, np.ones((2, 10, 10)), rtol=1e-05)
        data1 = dt._center_of_mass_array(dask_array, threshold_value=1)
        data1 = data1.compute()
        assert (data1 == np.ones((2, 10, 10))).all()


class TestMaskArray:

    def test_simple(self):
        numpy_array = np.zeros((11, 10, 40, 50))
        numpy_array[:, :, 25, 25] = 1
        dask_array = da.from_array(numpy_array, chunks=(5, 5, 5, 5))
        mask_array = pst._make_circular_mask(25, 25, 50, 40, 10)
        data = dt._mask_array(dask_array, mask_array=mask_array)
        data = data.compute()
        assert data.shape == (11, 10, 40, 50)
        assert (data == np.zeros((11, 10, 40, 50))).all()

    def test_wrong_mask_array_shape(self):
        dask_array = da.zeros((10, 12, 23, 8), chunks=(2, 2, 2, 2))
        mask_array = pst._make_circular_mask(7, 4, 22, 9, 5)
        with pytest.raises(ValueError):
            dt._mask_array(dask_array, mask_array=mask_array)


class TestThresholdArray:

    def test_simple(self):
        numpy_array = np.ones((10, 12, 20, 15))
        numpy_array[:, :, 10, 5] = 10000000
        dask_array = da.from_array(numpy_array, chunks=(5, 5, 5, 5))
        data = dt._threshold_array(dask_array, threshold_value=1)
        data = data.compute()
        assert data.shape == (10, 12, 20, 15)
        assert data.dtype == np.bool
        assert (data[:, :, 10, 5] == np.ones((10, 12), dtype=np.bool)).all()
        data[:, :, 10, 5] = False
        assert (data == np.zeros((10, 12, 20, 15))).all()


class TestFindDeadPixels:

    def test_simple(self):
        data = np.random.randint(10, 100, size=(20, 30))
        data[5, 9] = 0
        data[2, 1] = 0
        dask_array = da.from_array(data, chunks=(5, 5))
        dead_pixels = dt._find_dead_pixels(dask_array)
        assert data.shape == dead_pixels.shape
        dead_pixels = dead_pixels.compute()
        assert dead_pixels[5, 9]
        assert dead_pixels[2, 1]
        dead_pixels[5, 9] = False
        dead_pixels[2, 1] = False
        assert not dead_pixels.any()

    def test_3d(self):
        data = np.random.randint(10, 100, size=(5, 20, 30))
        data[:, 5, 9] = 0
        data[:, 2, 1] = 0
        data[3, 9, 3] = 0
        dask_array = da.from_array(data, chunks=(5, 5, 5))
        dead_pixels = dt._find_dead_pixels(dask_array)
        assert data.shape[-2:] == dead_pixels.shape
        dead_pixels = dead_pixels.compute()
        assert (dead_pixels[5, 9]).all()
        assert (dead_pixels[2, 1]).all()
        dead_pixels[5, 9] = False
        dead_pixels[2, 1] = False
        assert not dead_pixels.any()

    def test_4d(self):
        data = np.random.randint(10, 100, size=(8, 5, 20, 30))
        data[:, :, 5, 9] = 0
        data[:, :, 2, 1] = 0
        data[6, :, 6, 8] = 0
        dask_array = da.from_array(data, chunks=(5, 5, 5, 5))
        dead_pixels = dt._find_dead_pixels(dask_array)
        assert data.shape[-2:] == dead_pixels.shape
        dead_pixels = dead_pixels.compute()
        assert (dead_pixels[5, 9]).all()
        assert (dead_pixels[2, 1]).all()
        dead_pixels[5, 9] = False
        dead_pixels[2, 1] = False
        assert not dead_pixels.any()

    def test_dead_pixel_value(self):
        data = np.random.randint(10, 100, size=(20, 30))
        data[5, 9] = 3
        data[2, 1] = 3
        dask_array = da.from_array(data, chunks=(5, 5))
        dead_pixels = dt._find_dead_pixels(dask_array, dead_pixel_value=3)
        assert data.shape == dead_pixels.shape
        dead_pixels = dead_pixels.compute()
        assert dead_pixels[5, 9]
        assert dead_pixels[2, 1]
        dead_pixels[5, 9] = False
        dead_pixels[2, 1] = False
        assert not dead_pixels.any()

    def test_mask_array(self):
        data = np.random.randint(10, 100, size=(20, 30))
        data[16, 25] = 0
        data[2, 1] = 0
        dask_array = da.from_array(data, chunks=(5, 5))
        mask_array = np.zeros((20, 30), dtype=np.bool)
        mask_array[:10, :] = True
        dead_pixels = dt._find_dead_pixels(dask_array, mask_array=mask_array)
        dead_pixels = dead_pixels.compute()
        assert dead_pixels[16, 25]
        dead_pixels[16, 25] = False
        assert not dead_pixels.any()

    def test_1d_wrong_shape(self):
        data = np.random.randint(10, 100, size=40)
        dask_array = da.from_array(data, chunks=(5))
        with pytest.raises(ValueError):
            dt._find_dead_pixels(dask_array)
