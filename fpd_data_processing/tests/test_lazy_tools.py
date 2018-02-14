import pytest
import numpy as np
import dask.array as da
import fpd_data_processing.lazy_tools as lt


class TestGetDaskChunkSliceList:

    def test_simple(self):
        dask_array = da.zeros((10, 10, 50, 50), chunks=(5, 5, 25, 25))
        slice_list = lt._get_dask_chunk_slice_list(dask_array)
        assert len(slice_list) == 4
        assert slice_list[0] == np.s_[0:5, 0:5, :, :]
        assert slice_list[1] == np.s_[0:5, 5:10, :, :]
        assert slice_list[2] == np.s_[5:10, 0:5, :, :]
        assert slice_list[3] == np.s_[5:10, 5:10, :, :]

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
