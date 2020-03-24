import pytest
import numpy as np
import dask.array as da
import pixstem.lazy_tools as lt


class TestGetDaskChunkSliceList:

    def test_simple(self):
        dask_array = da.zeros((10, 10, 50, 50), chunks=(5, 5, 25, 25))
        slice_list = lt._get_dask_chunk_slice_list(dask_array)
        assert len(slice_list) == 4
        assert slice_list[0] == np.s_[0:5, 0:5, :, :]
        assert slice_list[1] == np.s_[0:5, 5:10, :, :]
        assert slice_list[2] == np.s_[5:10, 0:5, :, :]
        assert slice_list[3] == np.s_[5:10, 5:10, :, :]

    def test_1d_nav(self):
        dask_array = da.zeros((10, 50, 50), chunks=(5, 25, 25))
        slice_list = lt._get_dask_chunk_slice_list(dask_array)
        assert len(slice_list) == 2
        assert slice_list[0] == np.s_[0:5, :, :]
        assert slice_list[1] == np.s_[5:10, :, :]

    def test_nav_chunk_1(self):
        dask_array = da.zeros((2, 2, 10, 10), chunks=(1, 1, 10, 10))
        slice_list = lt._get_dask_chunk_slice_list(dask_array)
        assert len(slice_list) == 4
        assert slice_list[0] == np.s_[0:1, 0:1, :, :]
        assert slice_list[1] == np.s_[0:1, 1:2, :, :]
        assert slice_list[2] == np.s_[1:2, 0:1, :, :]
        assert slice_list[3] == np.s_[1:2, 1:2, :, :]

    def test_nav_chunk_full(self):
        dask_array = da.zeros((2, 2, 10, 10), chunks=(2, 2, 10, 10))
        slice_list = lt._get_dask_chunk_slice_list(dask_array)
        assert len(slice_list) == 1
        assert slice_list[0] == np.s_[0:2, 0:2, :, :]

    def test_2_dim_error(self):
        dask_array = da.zeros((50, 50), chunks=(5, 5))
        with pytest.raises(NotImplementedError):
            lt._get_dask_chunk_slice_list(dask_array)

    def test_5_dim_error(self):
        dask_array = da.zeros((5, 10, 15, 50, 50), chunks=(5, 5, 5, 5, 5))
        with pytest.raises(NotImplementedError):
            lt._get_dask_chunk_slice_list(dask_array)

    def test_6_dim_error(self):
        dask_array = da.zeros((5, 6, 7, 8, 9, 9), chunks=(2, 2, 2, 2, 2, 2))
        with pytest.raises(NotImplementedError):
            lt._get_dask_chunk_slice_list(dask_array)


def sum_frame(image, multiplier=1):
    data = image.sum()*multiplier
    return data


def return_two_value(image):
    data = image[0:2, 0]
    return data


class TestCalculateFunctionOnDaskArray:

    def test_simple(self):
        dask_array = da.ones((10, 10, 50, 50), chunks=(5, 5, 25, 25))
        data = lt._calculate_function_on_dask_array(
                dask_array, sum_frame, show_progressbar=False)
        assert data.shape == (10, 10)
        assert (data == (np.ones((10, 10))*50*50)).all()

    def test_1d_nav(self):
        dask_array = da.ones((10, 50, 50), chunks=(5, 25, 25))
        data = lt._calculate_function_on_dask_array(
                dask_array, sum_frame, show_progressbar=False)
        assert data.shape == (10, )
        assert (data == (np.ones((10, ))*50*50)).all()

    def test_2_dim_error(self):
        dask_array = da.ones((50, 50), chunks=(25, 25))
        with pytest.raises(NotImplementedError):
            lt._calculate_function_on_dask_array(
                    dask_array, sum_frame, show_progressbar=False)

    def test_5_dim_error(self):
        dask_array = da.ones((4, 4, 6, 50, 50), chunks=(2, 2, 2, 25, 25))
        with pytest.raises(NotImplementedError):
            lt._calculate_function_on_dask_array(
                    dask_array, sum_frame, show_progressbar=False)

    def test_nav_size(self):
        dask_array = da.ones((6, 9, 10, 10), chunks=(3, 3, 10, 10))
        data = lt._calculate_function_on_dask_array(
                dask_array, sum_frame, show_progressbar=False)
        assert data.shape == (6, 9)

    def test_value(self):
        numpy_array = np.zeros((6, 9, 10, 10))
        numpy_array[0, 0, :, :] += 1
        dask_array = da.from_array(numpy_array, chunks=(3, 3, 10, 10))
        data = lt._calculate_function_on_dask_array(
                dask_array, sum_frame, show_progressbar=False)
        assert data.shape == (6, 9)
        assert data[0, 0] == 100
        data[0, 0] = 0
        assert not data.any()

    def test_return_sig_size(self):
        dask_array = da.ones((10, 10, 50, 50), chunks=(5, 5, 25, 25))
        data0 = lt._calculate_function_on_dask_array(
                dask_array, return_two_value, return_sig_size=2,
                show_progressbar=False)
        assert data0.shape == (10, 10, 2)
        data1 = lt._calculate_function_on_dask_array(
                dask_array, sum_frame, show_progressbar=False)
        assert data1.shape == (10, 10)

    def test_return_sig_size_wrong(self):
        dask_array = da.ones((10, 10, 50, 50), chunks=(5, 5, 25, 25))
        with pytest.raises(ValueError):
            lt._calculate_function_on_dask_array(
                    dask_array, return_two_value, return_sig_size=1,
                    show_progressbar=False)
