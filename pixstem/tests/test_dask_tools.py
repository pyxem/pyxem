import pytest
import numpy as np
import dask.array as da
import pixstem.dask_tools as dat
import pixstem.pixelated_stem_tools as pst


class TestCenterOfMassArray:

    def test_simple(self):
        numpy_array = np.zeros((10, 10, 50, 50))
        numpy_array[:, :, 25, 25] = 1
        dask_array = da.from_array(numpy_array, chunks=(5, 5, 5, 5))

        data = dat._center_of_mass_array(dask_array)
        data = data.compute()
        assert data.shape == (2, 10, 10)
        assert (data == np.ones((2, 10, 10))*25).all()

    def test_mask(self):
        numpy_array = np.zeros((10, 10, 50, 50))
        numpy_array[:, :, 25, 25] = 1
        numpy_array[:, :, 1, 1] = 100000000
        dask_array = da.from_array(numpy_array, chunks=(5, 5, 5, 5))
        data0 = dat._center_of_mass_array(dask_array)
        data0 = data0.compute()
        np.testing.assert_allclose(data0, np.ones((2, 10, 10)), rtol=1e-05)
        mask_array = pst._make_circular_mask(25, 25, 50, 50, 10)
        mask_array = np.invert(mask_array)
        data1 = dat._center_of_mass_array(dask_array, mask_array=mask_array)
        data1 = data1.compute()
        assert (data1 == np.ones((2, 10, 10))*25).all()

    def test_threshold(self):
        numpy_array = np.zeros((10, 10, 50, 50))
        numpy_array[:, :, 25, 25] = 1
        numpy_array[:, :, 1, 1] = 100000000
        dask_array = da.from_array(numpy_array, chunks=(5, 5, 5, 5))
        data0 = dat._center_of_mass_array(dask_array)
        data0 = data0.compute()
        np.testing.assert_allclose(data0, np.ones((2, 10, 10)), rtol=1e-05)
        data1 = dat._center_of_mass_array(dask_array, threshold_value=1)
        data1 = data1.compute()
        assert (data1 == np.ones((2, 10, 10))).all()


class TestMaskArray:

    def test_simple(self):
        numpy_array = np.zeros((11, 10, 40, 50))
        numpy_array[:, :, 25, 25] = 1
        dask_array = da.from_array(numpy_array, chunks=(5, 5, 5, 5))
        mask_array = pst._make_circular_mask(25, 25, 50, 40, 10)
        data = dat._mask_array(dask_array, mask_array=mask_array)
        data = data.compute()
        assert data.shape == (11, 10, 40, 50)
        assert (data == np.zeros((11, 10, 40, 50))).all()

    def test_wrong_mask_array_shape(self):
        dask_array = da.zeros((10, 12, 23, 8), chunks=(2, 2, 2, 2))
        mask_array = pst._make_circular_mask(7, 4, 22, 9, 5)
        with pytest.raises(ValueError):
            dat._mask_array(dask_array, mask_array=mask_array)


class TestThresholdArray:

    def test_simple(self):
        numpy_array = np.ones((10, 12, 20, 15))
        numpy_array[:, :, 10, 5] = 10000000
        dask_array = da.from_array(numpy_array, chunks=(5, 5, 5, 5))
        data = dat._threshold_array(dask_array, threshold_value=1)
        data = data.compute()
        assert data.shape == (10, 12, 20, 15)
        assert data.dtype == np.bool
        assert (data[:, :, 10, 5] == np.ones((10, 12), dtype=np.bool)).all()
        data[:, :, 10, 5] = False
        assert (data == np.zeros((10, 12, 20, 15))).all()
