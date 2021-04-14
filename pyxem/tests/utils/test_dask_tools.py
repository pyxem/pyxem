# -*- coding: utf-8 -*-
# Copyright 2016-2021 The pyXem developers
#
# This file is part of pyXem.
#
# pyXem is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyXem is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyXem.  If not, see <http://www.gnu.org/licenses/>.

import pytest
import numpy as np
import dask.array as da
import skimage.morphology as sm

from pyxem.signals import Diffraction2D, LazyDiffraction2D
import pyxem.utils.dask_tools as dt
import pyxem.utils.pixelated_stem_tools as pst


class TestSignalDimensionGetChunkSliceList:
    @pytest.mark.parametrize(
        "sig_chunks",
        [(10, 10), (5, 10), (5, 10), (5, 5), (20, 10), (20, 20)],
    )
    def test_chunksizes(self, sig_chunks):
        xchunk, ychunk = sig_chunks
        data = da.zeros((20, 20, 20, 20), chunks=(10, 10, ychunk, xchunk))
        chunk_slice_list = dt.get_signal_dimension_chunk_slice_list(data.chunks)
        assert len(data.chunks[-1]) * len(data.chunks[-2]) == len(chunk_slice_list)
        for chunk_slice in chunk_slice_list:
            xsize = chunk_slice[0].stop - chunk_slice[0].start
            ysize = chunk_slice[1].stop - chunk_slice[1].start
            assert xsize == xchunk
            assert ysize == ychunk

    def test_non_square_chunks(self):
        data = da.zeros((2, 2, 20, 20), chunks=(2, 2, 15, 15))
        chunk_slice_list = dt.get_signal_dimension_chunk_slice_list(data.chunks)
        assert chunk_slice_list[0] == (slice(0, 15, None), slice(0, 15, None))
        assert chunk_slice_list[1] == (slice(15, 20, None), slice(0, 15, None))
        assert chunk_slice_list[2] == (slice(0, 15, None), slice(15, 20, None))
        assert chunk_slice_list[3] == (slice(15, 20, None), slice(15, 20, None))

    def test_one_signal_chunk(self):
        data = da.zeros((2, 2, 20, 20), chunks=(1, 1, 20, 20))
        chunk_slice_list = dt.get_signal_dimension_chunk_slice_list(data.chunks)
        assert len(chunk_slice_list) == 1
        assert chunk_slice_list[0] == np.s_[0:20, 0:20]

    def test_rechunk(self):
        data = da.zeros((2, 2, 20, 20), chunks=(1, 1, 20, 20))
        data1 = data.rechunk((2, 2, 10, 10))
        chunk_slice_list = dt.get_signal_dimension_chunk_slice_list(data1.chunks)
        assert len(chunk_slice_list) == 4

    def test_slice_navigation(self):
        data = da.zeros((2, 2, 20, 20), chunks=(1, 1, 20, 20))
        data1 = data[0, 1]
        chunk_slice_list = dt.get_signal_dimension_chunk_slice_list(data1.chunks)
        assert len(chunk_slice_list) == 1
        assert chunk_slice_list[0] == np.s_[0:20, 0:20]


@pytest.mark.parametrize(
    "input_shape,iter_shape",
    [
        [(9, 8, 6, 6), (9, 8, 2)],
        [(9, 8, 6, 6), (9, 8, 2, 2)],
        [(9, 8, 6), (9, 8)],
        [(9, 8, 6, 20, 20), (9, 8)],
    ],
)
def test_expand_iter_dimensions(input_shape, iter_shape):
    data_dask = da.zeros(input_shape, chunks=[2] * len(input_shape))
    iter_dask = da.zeros(iter_shape, chunks=[2] * len(iter_shape))
    output_array = dt._expand_iter_dimensions(iter_dask, len(data_dask.shape))
    assert len(data_dask.shape) == len(output_array.shape)


class TestGetSignalDimensionHostChunkSlice:
    @pytest.mark.parametrize(
        "xy, sig_slice, xchunk, ychunk",
        [
            ((0, 0), np.s_[0:10, 0:10], 10, 10),
            ((5, 5), np.s_[0:10, 0:10], 10, 10),
            ((15, 5), np.s_[10:20, 0:10], 10, 10),
            ((5, 15), np.s_[0:10, 10:20], 10, 10),
            ((19, 19), np.s_[10:20, 10:20], 10, 10),
            ((15, 9), np.s_[0:20, 0:20], 20, 20),
            ((17, 16), np.s_[15:20, 15:20], 15, 15),
            ((17, 16), np.s_[15:20, 10:20], 15, 10),
            ((7, 16), np.s_[0:15, 10:20], 15, 10),
            ((25, 2), False, 10, 10),
            ((2, 25), False, 10, 10),
            ((22, 25), False, 10, 10),
            ((-5, 10), False, 10, 10),
            ((5, -1), False, 10, 10),
        ],
    )
    def test_simple(self, xy, sig_slice, xchunk, ychunk):
        x, y = xy
        data = da.zeros((2, 2, 20, 20), chunks=(1, 1, ychunk, xchunk))
        chunk_slice = dt.get_signal_dimension_host_chunk_slice(x, y, data.chunks)
        assert chunk_slice == sig_slice

    @pytest.mark.parametrize(
        "xy, sig_slice",
        [
            ((0, 0), np.s_[0:10, 0:10]),
            ((28, 5), False),
            ((5, 28), np.s_[0:10, 20:30]),
            ((5, 32), False),
            ((12, 18), np.s_[10:20, 10:20]),
        ],
    )
    def test_non_square(self, xy, sig_slice):
        x, y = xy
        data = da.zeros((2, 2, 30, 20), chunks=(1, 1, 10, 10))
        chunk_slice = dt.get_signal_dimension_host_chunk_slice(x, y, data.chunks)
        assert chunk_slice == sig_slice


class TestProcessChunk:
    def test_simple(self):
        dtype = np.int16
        chunk_input = np.zeros((3, 4, 10, 8), dtype=dtype)
        block_info = {None: {"dtype": dtype}}
        iter_array = None

        def test_function(image):
            return image + 1

        chunk_output = dt._process_chunk(
            chunk_input, iter_array, test_function, block_info=block_info
        )
        assert chunk_input.shape == chunk_output.shape
        assert chunk_output.dtype == dtype
        assert np.all(chunk_output == 1)

    def test_output_signal_size(self):
        dtype = np.int16
        chunk_input = np.zeros((3, 4, 10, 8), dtype=dtype)
        block_info = {None: {"dtype": dtype}}
        iter_array = None

        def test_function(image):
            return (5, 2)

        chunk_output = dt._process_chunk(
            chunk_input,
            iter_array,
            test_function,
            output_signal_size=(2,),
            block_info=block_info,
        )
        output_shape = chunk_input.shape[:-2] + (2,)
        assert output_shape == chunk_output.shape
        assert chunk_output.dtype == dtype
        assert np.all(chunk_output == [5, 2])

    @pytest.mark.parametrize(
        "dtype",
        [np.float16, np.float32, np.uint8, np.uint32, np.int8, np.int32, bool],
    )
    def test_dtype(self, dtype):
        chunk_input = np.zeros((3, 4, 10, 8), dtype=np.int16)
        block_info = {None: {"dtype": dtype}}
        iter_array = None

        def test_function(image):
            return 4.8

        chunk_output = dt._process_chunk(
            chunk_input,
            iter_array,
            test_function,
            output_signal_size=(1,),
            block_info=block_info,
        )
        output_shape = chunk_input.shape[:-2] + (1,)
        assert output_shape == chunk_output.shape
        assert chunk_output.dtype == dtype

    def test_args_process(self):
        dtype = np.int16
        chunk_input = np.zeros((3, 4, 10, 8), dtype=dtype)
        block_info = {None: {"dtype": dtype}}
        iter_array = None

        def test_function(image, value1, value2):
            return (image + value1) / value2

        value1, value2 = 24, 4
        chunk_output = dt._process_chunk(
            chunk_input,
            iter_array,
            test_function,
            args_process=[value1, value2],
            block_info=block_info,
        )
        assert np.all(chunk_output == 6)

    def test_kwargs_process(self):
        dtype = np.int16
        chunk_input = np.zeros((3, 4, 10, 8), dtype=dtype)
        block_info = {None: {"dtype": dtype}}
        iter_array = None

        def test_function(image, value1=2, value2=2):
            return (image + value1) / value2

        value1, value2 = 15, 3
        chunk_output1 = dt._process_chunk(
            chunk_input,
            iter_array,
            test_function,
            kwargs_process={"value1": value1, "value2": value2},
            block_info=block_info,
        )
        assert np.all(chunk_output1 == 5)

        chunk_output2 = dt._process_chunk(
            chunk_input, iter_array, test_function, block_info=block_info
        )
        assert np.all(chunk_output2 == 1)

    @pytest.mark.parametrize(
        "shape", [(6, 9), (5, 3, 4), (3, 2, 9, 8), (5, 4, 2, 8, 7)]
    )
    def test_input_dimensions(self, shape):
        dtype = np.int16
        chunk_input = np.zeros(shape, dtype=dtype)
        block_info = {None: {"dtype": dtype}}
        iter_array = None

        def test_function(image):
            return image

        chunk_output = dt._process_chunk(
            chunk_input, iter_array, test_function, block_info=block_info
        )
        assert chunk_input.shape == chunk_output.shape

    def test_iter_array(self):
        dtype = np.int16
        chunk_input = np.zeros((3, 4, 10, 8), dtype=dtype)
        iter_array = np.random.randint(0, 256, (3, 4, 1, 1))
        block_info = {None: {"dtype": dtype}}

        def test_function(image, value):
            return value

        chunk_output = dt._process_chunk(
            chunk_input,
            iter_array,
            test_function,
            output_signal_size=(1,),
            block_info=block_info,
        )
        assert (chunk_output.squeeze() == iter_array.squeeze()).all()

    def test_iter_array_wrong_shape(self):
        dtype = np.int16
        chunk_input = np.zeros((3, 4, 10, 8), dtype=dtype)
        iter_array = np.random.randint(0, 256, (3, 5, 1, 1))
        block_info = {None: {"dtype": dtype}}
        test_function = lambda a: 1
        with pytest.raises(ValueError):
            chunk_output = dt._process_chunk(
                chunk_input,
                iter_array,
                test_function,
                output_signal_size=(1,),
                block_info=block_info,
            )


class TestProcessDaskArray:
    def test_simple(self):
        dask_input = da.zeros((4, 6, 8, 10), chunks=(2, 2, 2, 2), dtype=np.uint16)

        def test_function(image):
            return image

        dask_output = dt._process_dask_array(dask_input, test_function)
        output_chunks = dask_input.chunksize[:-2] + dask_input.shape[-2:]
        assert dask_output.chunksize == output_chunks
        array_output = dask_output.compute()
        assert dask_input.shape == array_output.shape
        assert np.all(array_output == 0)

    @pytest.mark.parametrize(
        "dtype",
        [np.float16, np.float32, np.uint8, np.uint32, np.int8, np.int32, bool],
    )
    def test_dtype(self, dtype):
        dask_input = da.zeros((4, 6, 8, 10), chunks=(2, 2, 2, 2), dtype=np.uint16)

        def test_function(image):
            return image

        dask_output = dt._process_dask_array(
            dask_input,
            test_function,
            dtype=dtype,
        )
        array_output = dask_output.compute()
        assert array_output.dtype == dtype

    def test_reduced_signal_shape(self):
        dask_input = da.zeros((4, 6, 8, 10), chunks=(2, 2, 2, 2), dtype=np.uint16)

        def test_function(image):
            return [2, 6]

        dask_output = dt._process_dask_array(
            dask_input,
            test_function,
            chunks=(2, 2, 2),
            drop_axis=(2, 3),
            new_axis=2,
            output_signal_size=(2,),
        )
        assert dask_output.shape == (4, 6, 2)
        assert dask_output.chunksize == (2, 2, 2)
        array_output = dask_output.compute()
        assert array_output.shape == (4, 6, 2)
        assert array_output.dtype == dask_input.dtype
        assert np.all(array_output == [2, 6])

    def test_reduced_signal_shape_object_dtype(self):
        dask_input = da.zeros((4, 6, 8, 10), chunks=(2, 2, 2, 2), dtype=np.uint16)

        def test_function(image):
            return list(range(11))

        dask_output = dt._process_dask_array(
            dask_input,
            test_function,
            chunks=(2, 2),
            drop_axis=(2, 3),
            dtype=object,
            new_axis=None,
            output_signal_size=(),
        )
        array_output = dask_output.compute()
        assert array_output.dtype == object
        for iy, ix in np.ndindex(array_output.shape):
            assert array_output[iy, ix] == list(range(11))

    def test_args(self):
        dask_input = da.ones((4, 6, 8, 10), chunks=(2, 2, 2, 2), dtype=np.uint16)

        def test_function(image, value1, value2):
            return (image + value1) / value2

        value1, value2 = 9, 2
        dask_output = dt._process_dask_array(
            dask_input,
            test_function,
            value1=value1,
            value2=value2,
        )
        array_output = dask_output.compute()
        assert dask_input.shape == array_output.shape
        assert np.all(array_output == 5)

    def test_kwargs(self):
        dask_input = da.ones((4, 6, 8, 10), chunks=(2, 2, 2, 2), dtype=np.uint16)

        def test_function(image, value1=7, value2=2):
            return (image + value1) / value2

        value1, value2 = 9, 2
        dask_output1 = dt._process_dask_array(
            dask_input,
            test_function,
            value1=value1,
            value2=value2,
        )
        dask_output2 = dt._process_dask_array(
            dask_input,
            test_function,
        )
        array_output1 = dask_output1.compute()
        array_output2 = dask_output2.compute()
        assert np.all(array_output1 == 5)
        assert np.all(array_output2 == 4)

    @pytest.mark.parametrize(
        "shape", [(6, 9), (5, 3, 4), (3, 2, 9, 8), (5, 4, 2, 8, 7)]
    )
    def test_input_dimensions(self, shape):
        chunks = [2] * len(shape)
        dask_input = da.zeros(shape, chunks=chunks, dtype=np.uint16)

        def test_function(image):
            return image

        dask_output = dt._process_dask_array(dask_input, test_function)
        array_output = dask_output.compute()
        assert dask_input.shape == array_output.shape

    def test_dask_array_wrong_type(self):
        array_input = np.zeros((4, 6, 10, 10))
        test_function = lambda a: 1
        with pytest.raises(AttributeError):
            dt._process_dask_array(array_input, test_function)

    @pytest.mark.parametrize(
        "dask_shape,iter_shape",
        [
            [(8, 10), ()],
            [(6, 8, 10), (6,)],
            [(4, 6, 8, 10), (4, 6)],
            [(2, 4, 6, 8, 10), (2, 4, 6)],
            [(2, 2, 4, 6, 8, 10), (2, 2, 4, 6)],
            [(8, 10), (10,)],
            [(6, 8, 10), (6, 10)],
            [(4, 6, 8, 10), (4, 6, 10)],
            [(2, 4, 6, 8, 10), (2, 4, 6, 10)],
            [(2, 2, 4, 6, 8, 10), (2, 2, 4, 6, 10)],
            [(8, 10), (8, 10)],
            [(6, 8, 10), (6, 8, 10)],
            [(4, 6, 8, 10), (4, 6, 8, 10)],
            [(2, 4, 6, 8, 10), (2, 4, 6, 8, 10)],
            [(2, 2, 4, 6, 8, 10), (2, 2, 4, 6, 8, 10)],
        ],
    )
    def test_iter_array1d(self, dask_shape, iter_shape):
        dask_chunks = [2] * len(dask_shape)
        iter_chunks = [2] * len(iter_shape)
        dask_input = da.zeros(dask_shape, chunks=dask_chunks, dtype=np.uint16)
        iter_input = da.random.randint(0, 256, iter_shape, chunks=iter_chunks)

        def test_function(image, value):
            temp_image = image.copy()
            temp_image[:] = value
            return temp_image

        dask_output = dt._process_dask_array(dask_input, test_function, iter_input)
        data_output = dask_output.compute()
        iter_array = iter_input.compute()
        for i in np.ndindex(data_output.shape[:-2]):
            data = data_output[i]
            value = iter_array[i]
            assert (data == value).all()

    def test_too_large_iter_array(self):
        dask_input = da.zeros((4, 6, 8, 10), chunks=(2, 2, 2, 2), dtype=np.uint16)
        iter_input = da.zeros((4, 6, 2, 2, 4), chunks=(2, 2, 2, 2, 4))
        test_function = lambda a: 1
        with pytest.raises(ValueError):
            dt._process_dask_array(dask_input, test_function, iter_input)

    def test_wrong_nav_shape(self):
        dask_input = da.zeros((4, 6, 8, 10), chunks=(2, 2, 2, 2), dtype=np.uint16)
        iter_input = da.zeros((4, 3), chunks=(2, 2))
        test_function = lambda a: 1
        with pytest.raises(ValueError):
            dt._process_dask_array(dask_input, test_function, iter_input)

    def test_non_dask_iter_array(self):
        dask_input = da.zeros((4, 6, 8, 10), chunks=(2, 2, 2, 2), dtype=np.uint16)
        iter_input = np.zeros((4, 6))
        test_function = lambda a: 1
        with pytest.raises(ValueError):
            dt._process_dask_array(dask_input, test_function, iter_input)

    def test_chunks_not_aligned(self):
        dask_input = da.zeros((4, 6, 8, 10), chunks=(2, 2, 2, 2), dtype=np.uint16)
        iter_input = da.zeros((4, 6, 2), chunks=(2, 3, 2))
        test_function = lambda a: 1
        with pytest.raises(ValueError):
            dt._process_dask_array(dask_input, test_function, iter_input)


class TestGetIterArray:
    def test_too_large_iter_array(self):
        dask_array = da.zeros((4, 6, 8, 10), chunks=(2, 2, 2, 2), dtype=np.uint16)
        iter_array = da.zeros(
            (4, 6, 8, 10, 12), chunks=(2, 2, 2, 2, 2), dtype=np.uint16
        )
        with pytest.raises(ValueError):
            dt._get_iter_array(iter_array, dask_array)

    def test_wrong_nav_shape(self):
        dask_array = da.zeros((4, 6, 8, 10), chunks=(2, 2, 2, 2), dtype=np.uint16)
        iter_array0 = da.zeros((4, 6, 2), chunks=(2, 2, 2), dtype=np.uint16)
        iter_array1 = da.zeros((4, 5, 2), chunks=(2, 2, 2), dtype=np.uint16)

        dt._get_iter_array(iter_array0, dask_array)
        with pytest.raises(ValueError):
            dt._get_iter_array(iter_array1, dask_array)

    def test_non_dask_iter_array(self):
        dask_array = da.zeros((4, 6, 8, 10), chunks=(2, 2, 2, 2), dtype=np.uint16)
        iter_array = np.zeros((4, 6, 2), dtype=np.uint16)

        with pytest.raises(ValueError):
            dt._get_iter_array(iter_array, dask_array)

    def test_chunks_not_aligned(self):
        dask_array = da.zeros((4, 6, 8, 10), chunks=(2, 2, 2, 2), dtype=np.uint16)
        iter_array = da.zeros((4, 6, 2), chunks=(2, 2, 2), dtype=np.uint16)

        dask_array0 = dask_array[:, 1:]
        iter_array0 = iter_array[:, 1:]
        iter_array1 = iter_array[:, :-1]

        dt._get_iter_array(iter_array0, dask_array0)
        with pytest.raises(ValueError):
            dt._get_iter_array(iter_array1, dask_array0)

    @pytest.mark.parametrize(
        "iter_array_shape,chunk_shape",
        [[(4, 6, 8, 12), (2, 2, 4, 6)], [(4, 6, 2), (2, 2, 1)], [(4, 6), (2, 2)]],
    )
    def test_iter_array_shape_and_chunks(self, iter_array_shape, chunk_shape):
        dask_array = da.zeros((4, 6, 8, 10), chunks=(2, 2, 2, 2), dtype=np.uint16)
        iter_array = da.zeros(iter_array_shape, chunks=chunk_shape, dtype=np.uint16)

        iter_array_output = dt._get_iter_array(iter_array, dask_array)
        assert len(iter_array_output.shape) == len(dask_array.shape)
        nav_shape = len(dask_array.shape) - 2
        assert iter_array_output.chunks[:nav_shape] == dask_array.chunks[:nav_shape]
        chunk_sig_iter = iter_array_output.chunks[nav_shape:]
        chunk_sig_iter = tuple(np.array(chunk_sig_iter).squeeze())
        assert chunk_sig_iter == iter_array_output.shape[nav_shape:]


class TestGetDaskArray:
    def test_simple(self):
        s = Diffraction2D(np.zeros((2, 3, 10, 10)))
        array_out = dt._get_dask_array(s)
        assert hasattr(array_out, "compute")

    def test_chunk_shape(self):
        s = Diffraction2D(np.zeros((10, 10, 8, 8)))
        array_out = dt._get_dask_array(s, chunk_shape=5)
        assert array_out.chunksize[:2] == (5, 5)

    def test_chunk_bytes(self):
        s = Diffraction2D(np.zeros((10, 10, 8, 8)))
        array_out0 = dt._get_dask_array(s)
        array_out1 = dt._get_dask_array(s, chunk_bytes="25KiB")
        assert array_out0.chunksize[:2] != array_out1.chunksize[:2]

    def test_lazy_input(self):
        s = LazyDiffraction2D(da.zeros((20, 20, 30, 30), chunks=(10, 10, 10, 10)))
        array_out = dt._get_dask_array(s)
        assert s.data.chunks == array_out.chunks
        assert s.data.shape == array_out.shape


class TestGetChunking:
    def test_simple(self):
        s = LazyDiffraction2D(da.zeros((32, 32, 256, 256), dtype=np.uint16))
        chunks = dt._get_chunking(s)
        assert len(chunks) == 4

    def test_chunk_shape(self):
        s = LazyDiffraction2D(da.zeros((32, 32, 256, 256), dtype=np.uint16))
        chunks = dt._get_chunking(s, chunk_shape=16)
        assert chunks == ((16, 16), (16, 16), (256,), (256,))

    def test_chunk_bytes(self):
        s = LazyDiffraction2D(da.zeros((32, 32, 256, 256), dtype=np.uint16))
        chunks = dt._get_chunking(s, chunk_bytes="15MiB")
        assert chunks == ((8, 8, 8, 8), (8, 8, 8, 8), (256,), (256,))


class TestAlignSingleFrame:
    @pytest.mark.parametrize(
        "shifts", [[0, 0], [1, 0], [-1, 0], [0, -1], [-1, -1], [-2, -3], [-2, -5]]
    )
    def test_simple(self, shifts):
        x_size, y_size = 5, 9
        image = np.zeros((y_size, x_size), dtype=np.uint16)
        x, y = 3, 7
        image[y, x] = 7
        image_shifted = dt.align_single_frame(image, shifts)
        pos = np.s_[y + shifts[1], x + shifts[0]]
        assert image_shifted[pos] == 7
        image_shifted[pos] = 0
        assert not image_shifted.any()

    @pytest.mark.parametrize(
        "shifts,pos",
        [
            [[0.5, 0.5], np.s_[7:9, 3:5]],
            [[0.0, 0.5], np.s_[7:9, 3]],
            [[0.0, -0.5], np.s_[6:8, 3]],
            [[1.0, -1.5], np.s_[5:7, 4]],
        ],
    )
    def test_subpixel_integer_image(self, shifts, pos):
        x_size, y_size = 5, 9
        image = np.zeros((y_size, x_size), dtype=np.uint16)
        x, y = 3, 7
        image[y, x] = 8
        image_shifted = dt.align_single_frame(image, shifts, order=1)
        assert (image_shifted[pos] >= 2).all()
        image_shifted[pos] = 0
        assert not image_shifted.any()

    @pytest.mark.parametrize(
        "shifts,pos",
        [
            [[-1.0, -2.0], np.s_[5, 2]],
            [[-0.5, -2.0], np.s_[5, 2:4]],
            [[-0.5, -2.5], np.s_[4:6, 2:4]],
            [[-0.25, 0.0], np.s_[7, 2:4]],
        ],
    )
    def test_subpixel_float_image(self, shifts, pos):
        x_size, y_size = 5, 9
        image = np.zeros((y_size, x_size), dtype=np.float32)
        x, y = 3, 7
        image[y, x] = 9
        image_shifted = dt.align_single_frame(image, shifts, order=1)
        assert image_shifted[pos].sum() == 9
        image_shifted[pos] = 0
        assert not image_shifted.any()

    @pytest.mark.parametrize("shifts", [[-0.7, -2.7], [1.1, -1.1]])
    def test_not_subpixel_float_image(self, shifts):
        x_size, y_size = 5, 9
        image = np.zeros((y_size, x_size), dtype=np.float32)
        x, y = 3, 7
        image[y, x] = 9
        image_shifted = dt.align_single_frame(image, shifts, order=0)
        pos = np.s_[y + round(shifts[1]), x + round(shifts[0])]
        assert image_shifted[pos] == 9.0
        image_shifted[pos] = 0
        assert not image_shifted.any()


@pytest.mark.slow
class TestCenterOfMassArray:
    def test_simple(self):
        numpy_array = np.zeros((10, 10, 50, 50))
        numpy_array[:, :, 25, 25] = 1
        dask_array = da.from_array(numpy_array, chunks=(5, 5, 5, 5))

        data = dt._center_of_mass_array(dask_array)
        data = data.compute()
        assert data.shape == (2, 10, 10)
        assert (data == np.ones((2, 10, 10)) * 25).all()

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
        assert (data1 == np.ones((2, 10, 10)) * 25).all()
        mask_array_wrong_size = np.ones((25, 40))
        with pytest.raises(ValueError):
            dt._center_of_mass_array(dask_array, mask_array=mask_array_wrong_size)

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


@pytest.mark.slow
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


@pytest.mark.slow
class TestThresholdArray:
    @pytest.mark.parametrize("shape", [(20, 15), (12, 20, 15), (10, 12, 20, 15)])
    def test_shape(self, shape):
        numpy_array = np.ones(shape)
        slice_array = [slice(None)] * len(shape)
        slice_array[-1] = slice(5, 6)
        slice_array[-2] = slice(10, 11)
        numpy_array[slice_array] = 10000000
        dask_array = da.from_array(numpy_array)
        data = dt._threshold_array(dask_array, threshold_value=1)
        data = data.compute()
        assert data.shape == shape
        assert data.dtype == bool
        assert (data[slice_array] == np.ones((10, 12), dtype=bool)).all()
        data[slice_array] = False
        assert (data == np.zeros(shape)).all()

    def test_wrong_input_dimension(self):
        dask_array = da.zeros((3, 2, 5, 20, 20))
        with pytest.raises(ValueError):
            dt._threshold_array(dask_array)


@pytest.mark.slow
class TestRemoveBadPixels:
    def test_simple(self):
        data = np.ones((20, 30)) * 12
        data[5, 9] = 0
        data[2, 1] = 0
        dask_array = da.from_array(data, chunks=(5, 5))
        dead_pixels = dt._find_dead_pixels(dask_array)
        output = dt._remove_bad_pixels(dask_array, dead_pixels)
        output = output.compute()
        assert (output == 12).all()

    def test_3d(self):
        data = np.ones((5, 20, 30)) * 12
        data[:, 5, 9] = 0
        data[:, 2, 1] = 0
        dask_array = da.from_array(data, chunks=(5, 5, 5))
        dead_pixels = dt._find_dead_pixels(dask_array)
        output = dt._remove_bad_pixels(dask_array, dead_pixels)
        output = output.compute()
        assert (output == 12).all()

    def test_4d(self):
        data = np.ones((5, 10, 20, 30)) * 12
        data[:, :, 5, 9] = 0
        data[:, :, 2, 1] = 0
        dask_array = da.from_array(data, chunks=(5, 5, 5, 5))
        dead_pixels = dt._find_dead_pixels(dask_array)
        output = dt._remove_bad_pixels(dask_array, dead_pixels)
        output = output.compute()
        assert (output == 12).all()

    def test_3d_same_bad_pixel_array_shape(self):
        data = np.ones((5, 20, 30)) * 12
        data[2, 5, 9] = 0
        data[3, 2, 1] = 0
        dask_array = da.from_array(data, chunks=(5, 5, 5))
        bad_pixel_array = np.zeros(dask_array.shape)
        bad_pixel_array[2, 5, 9] = True
        bad_pixel_array[3, 2, 1] = True
        bad_pixel_array = da.from_array(bad_pixel_array, chunks=(5, 5, 5))
        output = dt._remove_bad_pixels(dask_array, bad_pixel_array)
        output = output.compute()
        assert (output == 12).all()

    def test_wrong_dask_array_shape_1d(self):
        data = np.ones((30))
        dask_array = da.from_array(data, chunks=(5))
        with pytest.raises(ValueError):
            dt._remove_bad_pixels(dask_array, dask_array)

    def test_wrong_shape_bad_pixel_array(self):
        data = np.ones((5, 10, 20, 30))
        dask_array = da.from_array(data, chunks=(5, 5, 5, 5))
        bad_pixel_array = da.zeros_like(dask_array)
        with pytest.raises(ValueError):
            dt._remove_bad_pixels(dask_array, bad_pixel_array[1, :, :, :])
        with pytest.raises(ValueError):
            dt._remove_bad_pixels(dask_array, bad_pixel_array[1, 1, :-2, :])

    def test_find_dead_pixels_wrong_input(self):
        dask_array = da.zeros((20,))
        with pytest.raises(ValueError):
            dt._find_dead_pixels(dask_array)

    def test_find_hot_pixels_wrong_input(self):
        dask_array = da.zeros((20,))
        with pytest.raises(ValueError):
            dt._find_hot_pixels(dask_array)


@pytest.mark.slow
class TestTemplateMatchBinaryImage:
    @pytest.mark.parametrize("x, y", [(13, 32), (76, 32), (87, 21), (43, 85)])
    def test_single_frame(self, x, y):
        disk_r = 5
        disk = sm.disk(disk_r)
        data = np.zeros(shape=(100, 100))

        data[y - disk_r : y + disk_r + 1, x - disk_r : x + disk_r + 1] = disk
        match = dt._template_match_binary_image_single_frame(data, disk)
        index = np.unravel_index(np.argmax(match), match.shape)
        assert (y, x) == index

    def test_chunk(self):
        x, y, disk_r = 76, 23, 5
        disk = sm.disk(disk_r)
        data = np.zeros(shape=(5, 10, 100, 90))
        data[:, :, y - disk_r : y + disk_r + 1, x - disk_r : x + disk_r + 1] = disk
        match_array = dt._template_match_binary_image_chunk(data, disk)
        assert data.shape == match_array.shape
        for ix, iy in np.ndindex(data.shape[:2]):
            match = match_array[ix, iy]
            index = np.unravel_index(np.argmax(match), match.shape)
            assert (y, x) == index

    def test_simple(self):
        data = np.ones((5, 3, 50, 40))
        disk = sm.disk(5)
        dask_array = da.from_array(data, chunks=(1, 1, 5, 5))
        match_array_dask = dt._template_match_with_binary_image(
            dask_array, binary_image=disk
        )
        match_array = match_array_dask.compute()
        assert match_array.shape == data.shape
        assert match_array.min() >= 0

    def test_position(self):
        disk_r = 5
        data = np.zeros((2, 3, 90, 100))
        # Nav top left, sig x=5, y=5
        data[0, 0, :11, :11] = sm.disk(disk_r)
        # Nav top centre, sig x=94, y=84
        data[0, 1, -11:, -11:] = sm.disk(disk_r)
        # Nav top right, sig x=94, y=5
        data[0, 2, :11, -11:] = sm.disk(disk_r)
        # Nav bottom left, sig x=5, y=84
        data[1, 0, -11:, :11] = sm.disk(disk_r)
        # Nav bottom centre, sig x=75, y=25
        data[1, 1, 20:31, 70:81] = sm.disk(disk_r)
        # Nav bottom right, sig x=55, y=75
        data[1, 2, 70:81, 50:61] = sm.disk(disk_r)
        binary_image = sm.disk(disk_r)
        dask_array = da.from_array(data, chunks=(1, 1, 5, 5))
        out_dask = dt._template_match_with_binary_image(
            dask_array, binary_image=binary_image
        )
        out = out_dask.compute()
        match00 = np.unravel_index(np.argmax(out[0, 0]), out[0, 0].shape)
        assert (5, 5) == match00
        match01 = np.unravel_index(np.argmax(out[0, 1]), out[0, 1].shape)
        assert (84, 94) == match01
        match02 = np.unravel_index(np.argmax(out[0, 2]), out[0, 2].shape)
        assert (5, 94) == match02
        match10 = np.unravel_index(np.argmax(out[1, 0]), out[1, 0].shape)
        assert (84, 5) == match10
        match11 = np.unravel_index(np.argmax(out[1, 1]), out[1, 1].shape)
        assert (25, 75) == match11
        match12 = np.unravel_index(np.argmax(out[1, 2]), out[1, 2].shape)
        assert (75, 55) == match12

    @pytest.mark.parametrize("nav_dims", [0, 1, 2, 3, 4])
    def test_array_different_dimensions(self, nav_dims):
        shape = list(np.random.randint(2, 6, size=nav_dims))
        shape.extend([50, 50])
        chunks = [1] * nav_dims
        chunks.extend([25, 25])
        dask_array = da.random.random(size=shape, chunks=chunks)
        binary_image = sm.disk(5)
        match_array_dask = dt._template_match_with_binary_image(
            dask_array, binary_image=binary_image
        )
        assert len(dask_array.shape) == nav_dims + 2
        assert dask_array.shape == match_array_dask.shape
        match_array = match_array_dask.compute()
        assert dask_array.shape == match_array.shape

    def test_1d_dask_array_error(self):
        binary_image = sm.disk(5)
        dask_array = da.random.random(size=50, chunks=10)
        with pytest.raises(ValueError):
            dt._template_match_with_binary_image(dask_array, binary_image=binary_image)


@pytest.mark.slow
class TestPeakFindDog:
    @pytest.mark.parametrize("x, y", [(112, 32), (170, 92), (54, 76), (10, 15)])
    def test_single_frame_one_peak(self, x, y):
        image = np.zeros(shape=(200, 100), dtype=np.float64)
        image[x, y] = 654
        min_sigma, max_sigma, sigma_ratio = 2, 5, 5
        threshold, overlap = 0.01, 1
        peaks = dt._peak_find_dog_single_frame(
            image,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            sigma_ratio=sigma_ratio,
            threshold=threshold,
            overlap=overlap,
        )
        assert (x, y) == (peaks[0, 0], peaks[0, 1])

    def test_single_frame_multiple_peak(self):
        image = np.zeros(shape=(200, 100), dtype=np.float64)
        peak_list = [[120, 76], [23, 54], [32, 78], [10, 15]]
        for x, y in peak_list:
            image[x, y] = 654
        min_sigma, max_sigma, sigma_ratio = 2, 5, 5
        threshold, overlap = 0.01, 1
        peaks = dt._peak_find_dog_single_frame(
            image,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            sigma_ratio=sigma_ratio,
            threshold=threshold,
            overlap=overlap,
        )
        assert len(peaks) == len(peak_list)
        for peak in peaks.tolist():
            assert peak in peak_list

    def test_single_frame_threshold(self):
        image = np.zeros(shape=(200, 100), dtype=np.float64)
        image[54, 29] = 100
        image[123, 54] = 20
        min_sigma, max_sigma, sigma_ratio, overlap = 2, 5, 5, 1
        peaks0 = dt._peak_find_dog_single_frame(
            image,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            sigma_ratio=sigma_ratio,
            threshold=0.01,
            overlap=overlap,
        )
        assert len(peaks0) == 2
        peaks1 = dt._peak_find_dog_single_frame(
            image,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            sigma_ratio=sigma_ratio,
            threshold=0.05,
            overlap=overlap,
        )
        assert len(peaks1) == 1

    def test_single_frame_normalize_value(self):
        image = np.zeros((100, 100), dtype=np.uint16)
        image[49:52, 49:52] = 100
        image[19:22, 9:12] = 10
        peaks0 = dt._peak_find_dog_single_frame(image, normalize_value=100)
        peaks1 = dt._peak_find_dog_single_frame(image, normalize_value=10)
        assert (peaks0 == [[50, 50]]).all()
        assert (peaks1 == [[50, 50], [20, 10]]).all()

    def test_chunk(self):
        data = np.zeros(shape=(2, 3, 200, 100), dtype=np.float64)
        data[0, 0, 50, 20] = 100
        data[0, 1, 51, 21] = 100
        data[0, 2, 52, 22] = 100
        data[1, 0, 53, 23] = 100
        data[1, 1, 54, 24] = 100
        data[1, 2, 55, 25] = 100
        min_sigma, max_sigma, sigma_ratio = 0.08, 1, 1.76
        threshold, overlap = 0.06, 0.01
        peaks = dt._peak_find_dog_chunk(
            data,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            sigma_ratio=sigma_ratio,
            threshold=threshold,
            overlap=overlap,
        )
        assert peaks[0, 0][0].tolist() == [50, 20]
        assert peaks[0, 1][0].tolist() == [51, 21]
        assert peaks[0, 2][0].tolist() == [52, 22]
        assert peaks[1, 0][0].tolist() == [53, 23]
        assert peaks[1, 1][0].tolist() == [54, 24]
        assert peaks[1, 2][0].tolist() == [55, 25]

    def test_chunk_normalize_value(self):
        data = np.zeros((2, 3, 100, 100), dtype=np.uint16)
        data[:, :, 49:52, 49:52] = 100
        data[:, :, 19:22, 9:12] = 10

        peak_array0 = dt._peak_find_dog_chunk(data, normalize_value=100)
        peak_array1 = dt._peak_find_dog_chunk(data, normalize_value=10)
        for ix, iy in np.ndindex(peak_array0.shape):
            assert (peak_array0[ix, iy] == [[50, 50]]).all()
            assert (peak_array1[ix, iy] == [[50, 50], [20, 10]]).all()

    def test_dask_array(self):
        data = np.zeros(shape=(2, 3, 200, 100), dtype=np.float64)
        data[0, 0, 50, 20] = 100
        data[0, 1, 51, 21] = 100
        data[0, 2, 52, 22] = 100
        data[1, 0, 53, 23] = 100
        data[1, 1, 54, 24] = 100
        data[1, 2, 55, 25] = 100
        dask_array = da.from_array(data, chunks=(1, 1, 200, 100))
        min_sigma, max_sigma, sigma_ratio = 0.08, 1, 1.76
        threshold, overlap = 0.06, 0.01
        peaks = dt._peak_find_dog(
            dask_array,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            sigma_ratio=sigma_ratio,
            threshold=threshold,
            overlap=overlap,
        )
        peaks = peaks.compute()
        assert peaks[0, 0][0].tolist() == [50, 20]
        assert peaks[0, 1][0].tolist() == [51, 21]
        assert peaks[0, 2][0].tolist() == [52, 22]
        assert peaks[1, 0][0].tolist() == [53, 23]
        assert peaks[1, 1][0].tolist() == [54, 24]
        assert peaks[1, 2][0].tolist() == [55, 25]

    def test_dask_array_normalize_value(self):
        data = np.zeros((2, 3, 100, 100), dtype=np.uint16)
        data[:, :, 49:52, 49:52] = 100
        data[:, :, 19:22, 9:12] = 10
        dask_array = da.from_array(data, chunks=(1, 1, 100, 100))
        peak_array0 = dt._peak_find_dog(dask_array, normalize_value=100)
        peak_array1 = dt._peak_find_dog(dask_array, normalize_value=10)
        for ix, iy in np.ndindex(peak_array0.shape):
            assert (peak_array0[ix, iy] == [[50, 50]]).all()
            assert (peak_array1[ix, iy] == [[50, 50], [20, 10]]).all()

    @pytest.mark.parametrize("nav_dims", [0, 1, 2, 3, 4])
    def test_array_different_dimensions(self, nav_dims):
        shape = list(np.random.randint(2, 6, size=nav_dims))
        shape.extend([50, 50])
        chunks = [1] * nav_dims
        chunks.extend([25, 25])
        dask_array = da.random.random(size=shape, chunks=chunks)
        peak_array_dask = dt._peak_find_dog(dask_array)
        assert len(peak_array_dask.shape) == nav_dims
        assert dask_array.shape[:-2] == peak_array_dask.shape
        peak_array = peak_array_dask.compute()
        assert dask_array.shape[:-2] == peak_array.shape

    def test_1d_dask_array_error(self):
        dask_array = da.random.random(size=50, chunks=10)
        with pytest.raises(ValueError):
            dt._peak_find_dog(dask_array)


@pytest.mark.slow
class TestPeakPositionRefinementCOM:
    def test_single_frame_peak(self):
        numpy_array = np.zeros((50, 50))
        numpy_array[25, 28] = 1
        numpy_array[10, 14] = 1
        peak = np.array([[27, 29], [11, 15]], np.int32)
        square_size = 6

        data = dt._peak_refinement_centre_of_mass_frame(numpy_array, peak, square_size)
        assert data[0][0] == 25.0
        assert data[0][1] == 28.0
        assert data[1][0] == 10.0
        assert data[1][1] == 14.0

        peak_outside = np.array(
            [
                [65.0, 10.0],
            ]
        )
        data_outside = dt._peak_refinement_centre_of_mass_frame(
            numpy_array, peak_outside, square_size
        )
        assert (peak_outside == data_outside).all()

    def test_chunk_peak(self):
        shape = (2, 2, 50, 50)
        numpy_array = np.zeros(shape)
        numpy_array[:, :, 25, 25] = 1

        peak_array = np.zeros((shape[0], shape[1], 1, 1), dtype=object)
        real_array = np.zeros((shape[:-2]), dtype=object)
        for index in np.ndindex(shape[:-2]):
            islice = np.s_[index]
            peak_array[islice][0, 0] = np.asarray([(27, 27)])
            real_array[islice] = np.asarray([(25, 25)])

        square_size = 12

        data = dt._peak_refinement_centre_of_mass_chunk(
            numpy_array, peak_array, square_size
        )
        assert data.shape == (2, 2)
        assert np.sum(data - real_array).sum() == 0

        peak_array_smaller = peak_array.squeeze()
        data = dt._peak_refinement_centre_of_mass_chunk(
            numpy_array, peak_array_smaller, square_size
        )

    def test_dask_array(self):
        numpy_array = np.zeros((10, 10, 50, 50))
        numpy_array[:, :, 25, 25] = 1

        peak_array = np.zeros((numpy_array.shape[:-2]), dtype=object)
        real_array = np.zeros((numpy_array.shape[:-2]), dtype=object)
        for index in np.ndindex(numpy_array.shape[:-2]):
            islice = np.s_[index]
            peak_array[islice] = np.asarray([(27, 27)])
            real_array[islice] = np.asarray([(25, 25)])

        dask_array = da.from_array(numpy_array, chunks=(5, 5, 5, 5))
        dask_peak_array = da.from_array(peak_array, chunks=(5, 5))

        square_size = 12

        data = dt._peak_refinement_centre_of_mass(
            dask_array, dask_peak_array, square_size
        )
        data = data.compute()
        assert data.shape == (10, 10)
        assert np.sum(data - real_array).sum() == 0

    @pytest.mark.parametrize("nav_dims", [0, 1, 2, 3, 4])
    def test_array_different_dimensions(self, nav_dims):
        shape = list(np.random.randint(2, 6, size=nav_dims))
        shape.extend([50, 50])
        chunks = [1] * nav_dims
        chunks.extend([25, 25])
        dask_array = da.random.random(size=shape, chunks=chunks)
        peak_array = np.zeros((dask_array.shape[:-2]), dtype=object)
        for index in np.ndindex(dask_array.shape[:-2]):
            islice = np.s_[index]
            peak_array[islice] = np.asarray([(27, 27)])
        square_size = 12
        peak_array_dask = da.from_array(peak_array, chunks=chunks[:-2])
        match_array_dask = dt._peak_refinement_centre_of_mass(
            dask_array, peak_array_dask, square_size
        )
        assert len(dask_array.shape) == nav_dims + 2
        match_array = match_array_dask.compute()
        assert peak_array_dask.shape == match_array.shape

    def test_wrong_input_sizes(self):
        dask_array = da.zeros((10, 9, 20, 20))
        peak_array = da.zeros((12, 6))
        with pytest.raises(ValueError):
            dt._peak_refinement_centre_of_mass(dask_array, peak_array, square_size=10)

    def test_wrong_input_not_dask(self):
        dask_array = da.zeros((9, 8, 10, 10))
        peak_array = da.zeros((9, 8))
        with pytest.raises(ValueError):
            dt._peak_refinement_centre_of_mass(
                dask_array.compute(), peak_array, square_size=10
            )
        with pytest.raises(ValueError):
            dt._peak_refinement_centre_of_mass(
                dask_array, peak_array.compute(), square_size=10
            )


@pytest.mark.slow
class TestBackgroundRemovalDOG:
    def test_single_frame_min_sigma(self):
        min_sigma = 10
        numpy_array = np.ones((50, 50))
        numpy_array[20:30, 20:30] = 5
        data = dt._background_removal_single_frame_dog(numpy_array, min_sigma=min_sigma)
        assert data.sum() != numpy_array.sum()
        assert data.shape == numpy_array.shape
        assert data[0, :].all() == 0

    def test_single_frame_max_sigma(self):
        max_sigma = 10
        numpy_array = np.ones((50, 50))
        numpy_array[20:30, 20:30] = 5
        data = dt._background_removal_single_frame_dog(numpy_array, max_sigma=max_sigma)
        assert data.sum() != numpy_array.sum()
        assert data.shape == numpy_array.shape
        assert data[0, :].all() == 0

    def test_dask_min_sigma(self):
        min_sigma = 10
        numpy_array = np.ones((10, 10, 50, 50))
        numpy_array[:, :20:30, 20:30] = 5
        dask_array = da.from_array(numpy_array, chunks=(2, 2, 50, 50))

        data = dt._background_removal_dog(dask_array, min_sigma=min_sigma)
        data.compute()
        assert data.sum() != numpy_array.sum()
        assert data.shape == numpy_array.shape
        assert data[:, :, 0, :].all() == 0

    def test_dask_max_sigma(self):
        max_sigma = 10
        numpy_array = np.ones((10, 10, 50, 50))
        numpy_array[:, :20:30, 20:30] = 5
        dask_array = da.from_array(numpy_array, chunks=(2, 2, 50, 50))

        data = dt._background_removal_dog(dask_array, max_sigma=max_sigma)
        data.compute()
        assert data.sum() != numpy_array.sum()
        assert data.shape == numpy_array.shape
        assert data[:, :, 0, :].all() == 0

    @pytest.mark.parametrize("nav_dims", [0, 1, 2, 3, 4])
    def test_array_different_dimensions(self, nav_dims):
        shape = list(np.random.randint(2, 6, size=nav_dims))
        shape.extend([50, 50])
        chunks = [1] * nav_dims
        chunks.extend([25, 25])
        dask_array = da.random.random(size=shape, chunks=chunks)
        match_array_dask = dt._background_removal_dog(dask_array)
        assert len(dask_array.shape) == nav_dims + 2
        assert dask_array.shape == match_array_dask.shape
        match_array = match_array_dask.compute()
        assert dask_array.shape == match_array.shape


@pytest.mark.slow
class TestBackgroundRemovalMedianFilter:
    def test_single_frame_footprint(self):
        footprint = 10
        numpy_array = np.ones((50, 50))
        numpy_array[20:30, 20:30] = 5
        data = dt._background_removal_single_frame_median(
            numpy_array, footprint=footprint
        )
        assert data.sum() != numpy_array.sum()
        assert data.shape == numpy_array.shape
        assert data[0, :].all() == 0

    def test_chunk_footprint(self):
        footprint = 10
        numpy_array = np.ones((10, 10, 50, 50))
        numpy_array[:, :20:30, 20:30] = 5
        data = dt._background_removal_chunk_median(numpy_array, footprint=footprint)
        assert data.sum() != numpy_array.sum()
        assert data.shape == numpy_array.shape
        assert data[:, :, 0, :].all() == 0

    def test_dask_footprint(self):
        footprint = 10
        numpy_array = np.ones((10, 10, 50, 50))
        numpy_array[:, :20:30, 20:30] = 5
        dask_array = da.from_array(numpy_array, chunks=(2, 2, 50, 50))

        data = dt._background_removal_median(dask_array, footprint=footprint)
        data = data.compute()
        assert data.sum() != numpy_array.sum()
        assert data.shape == numpy_array.shape
        assert data[:, :, 0, :].all() == 0

    @pytest.mark.parametrize("nav_dims", [0, 1, 2, 3, 4])
    def test_array_different_dimensions(self, nav_dims):
        shape = list(np.random.randint(2, 6, size=nav_dims))
        shape.extend([50, 50])
        chunks = [1] * nav_dims
        chunks.extend([25, 25])
        dask_array = da.random.random(size=shape, chunks=chunks)
        match_array_dask = dt._background_removal_median(dask_array)
        assert len(dask_array.shape) == nav_dims + 2
        assert dask_array.shape == match_array_dask.shape
        match_array = match_array_dask.compute()
        assert dask_array.shape == match_array.shape


@pytest.mark.slow
class TestBackgroundRemovalRadialMedian:
    def test_single_frame_centre(self):
        centre_x = 25
        centre_y = 25
        numpy_array = np.ones((50, 50))
        numpy_array[20:30, 20:30] = 5
        data = dt._background_removal_single_frame_radial_median(
            numpy_array, centre_x=centre_x, centre_y=centre_y
        )
        assert data.sum() != numpy_array.sum()
        assert data.shape == numpy_array.shape
        assert data[0, :].all() == 0

    def test_chunk_centre(self):
        centre_x = 25
        centre_y = 25
        numpy_array = np.ones((10, 10, 50, 50))
        numpy_array[:, :20:30, 20:30] = 5
        data = dt._background_removal_chunk_radial_median(
            numpy_array, centre_x=centre_x, centre_y=centre_y
        )
        assert data.sum() != numpy_array.sum()
        assert data.shape == numpy_array.shape
        assert data[:, :, 0, :].all() == 0

    def test_dask_centre(self):
        centre_x = 25
        centre_y = 25
        numpy_array = np.ones((10, 10, 50, 50))
        numpy_array[:, :20:30, 20:30] = 5
        dask_array = da.from_array(numpy_array, chunks=(2, 2, 50, 50))

        data = dt._background_removal_radial_median(
            dask_array, centre_x=centre_x, centre_y=centre_y
        )
        data = data.compute()
        assert data.sum() != numpy_array.sum()
        assert data.shape == numpy_array.shape
        assert (data[:, :, 0, :]).all() == 0

    @pytest.mark.parametrize("nav_dims", [0, 1, 2, 3, 4])
    def test_array_different_dimensions(self, nav_dims):
        centre_x = 25
        centre_y = 25
        shape = list(np.random.randint(2, 6, size=nav_dims))
        shape.extend([50, 50])
        chunks = [1] * nav_dims
        chunks.extend([25, 25])
        dask_array = da.random.random(size=shape, chunks=chunks)
        match_array_dask = dt._background_removal_radial_median(
            dask_array, centre_x=centre_x, centre_y=centre_y
        )
        assert len(dask_array.shape) == nav_dims + 2
        assert dask_array.shape == match_array_dask.shape
        match_array = match_array_dask.compute()
        assert dask_array.shape == match_array.shape


@pytest.mark.slow
class TestIntensityArray:
    def test_intensity_peaks_image_disk_r(self):
        numpy_array = np.zeros((50, 50))
        numpy_array[27, 29] = 2
        numpy_array[11, 15] = 1
        image = da.from_array(numpy_array, chunks=(50, 50))
        peak = np.array([[27, 29], [11, 15]], np.int32)
        peak_dask = da.from_array(peak, chunks=(1, 1))
        disk_r0 = 1
        disk_r1 = 2
        intensity0 = dt._intensity_peaks_image_single_frame(image, peak_dask, disk_r0)
        intensity1 = dt._intensity_peaks_image_single_frame(image, peak_dask, disk_r1)

        assert intensity0[0].all() == np.array([27.0, 29.0, 2 / 9]).all()
        assert intensity0[1].all() == np.array([11.0, 15.0, 1 / 9]).all()
        assert intensity1[0].all() == np.array([27.0, 29.0, 2 / 25]).all()
        assert intensity1[1].all() == np.array([11.0, 15.0, 1 / 25]).all()
        assert intensity0.shape == intensity1.shape == (2, 3)

    def test_intensity_peaks_image_region_outside_image(self):
        # If any part of the region around the peak (defined via disk_r) is outside
        # the image, the intensity should be 0
        image = np.ones((50, 50))
        peak = np.array([[10, 1], [1, 10], [10, 49], [49, 10]])
        intensity_list = dt._intensity_peaks_image_single_frame(image, peak, 2)
        for intensity in intensity_list:
            assert intensity[2] == 0

    def test_intensity_peaks_chunk(self):
        numpy_array = np.zeros((2, 2, 50, 50))
        numpy_array[:, :, 27, 27] = 1

        peak_array = np.zeros(
            (numpy_array.shape[0], numpy_array.shape[1]), dtype=object
        )
        for index in np.ndindex(numpy_array.shape[:-2]):
            islice = np.s_[index]
            peak_array[islice] = np.asarray([(27, 27)])

        dask_array = da.from_array(numpy_array, chunks=(1, 1, 25, 25))
        peak_array_dask = da.from_array(peak_array, chunks=(1, 1))
        disk_r = 2
        intensity_array = dt._intensity_peaks_image_chunk(
            dask_array, peak_array_dask, disk_r
        )

        assert intensity_array.shape == peak_array_dask.shape

    def test_intensity_peaks_dask(self):
        numpy_array = np.zeros((10, 10, 50, 50))
        numpy_array[:, :, 27, 27] = 1

        peak_array = np.zeros(
            (numpy_array.shape[0], numpy_array.shape[1]), dtype=object
        )
        for index in np.ndindex(numpy_array.shape[:-2]):
            islice = np.s_[index]
            peak_array[islice] = np.asarray([(27, 27)])

        dask_array = da.from_array(numpy_array, chunks=(5, 5, 5, 5))
        dask_peak_array = da.from_array(peak_array, chunks=(5, 5))

        disk_r = 2
        intensity_array = dt._intensity_peaks_image(dask_array, dask_peak_array, disk_r)
        intensity_array_computed = intensity_array.compute()
        assert intensity_array_computed.shape == peak_array.shape

    @pytest.mark.parametrize("nav_dims", [0, 1, 2, 3, 4])
    def test_array_different_dimensions(self, nav_dims):
        shape = list(np.random.randint(2, 6, size=nav_dims))
        shape.extend([50, 50])
        chunks = [1] * nav_dims
        chunks.extend([25, 25])
        dask_array = da.random.random(size=shape, chunks=chunks)
        peak_array = np.zeros((dask_array.shape[:-2]), dtype=object)
        for index in np.ndindex(dask_array.shape[:-2]):
            islice = np.s_[index]
            peak_array[islice] = np.asarray([(27, 27)])
        peak_array_dask = da.from_array(peak_array, chunks=chunks[:-2])
        match_array_dask = dt._intensity_peaks_image(dask_array, peak_array_dask, 5)
        assert len(dask_array.shape) == nav_dims + 2
        match_array = match_array_dask.compute()
        assert peak_array_dask.shape == match_array.shape

    def test_wrong_input_sizes(self):
        dask_array = da.zeros((10, 9, 20, 20))
        peak_array = da.zeros((12, 6))
        with pytest.raises(ValueError):
            dt._intensity_peaks_image(dask_array, peak_array, 5)

    def test_non_dask_array(self):
        data_array = np.ones((10, 10, 50, 50))
        data_array_dask = da.ones((10, 10, 50, 50), chunks=(2, 2, 25, 25))
        peak_array = np.empty((10, 10), dtype=object)
        peak_array_dask = da.from_array(peak_array, chunks=(2, 2))
        with pytest.raises(ValueError):
            dt._intensity_peaks_image(data_array, peak_array_dask, 5)
        with pytest.raises(ValueError):
            dt._intensity_peaks_image(data_array, peak_array, 5)
        with pytest.raises(ValueError):
            dt._intensity_peaks_image(data_array_dask, peak_array, 5)
        dt._intensity_peaks_image(data_array_dask, peak_array_dask, 5)

    def test_non_square_datasets(self):
        data_array_dask = da.ones((6, 16, 100, 50), chunks=(2, 2, 25, 25))
        peak_array_dask = da.empty((6, 16), chunks=(2, 2), dtype=object)
        dt._intensity_peaks_image(data_array_dask, peak_array_dask, 5)

    def test_different_chunks(self):
        data_array_dask = da.ones((6, 16, 100, 50), chunks=(6, 4, 50, 25))
        peak_array_dask = da.empty((6, 16), chunks=(3, 2), dtype=object)
        dt._intensity_peaks_image(data_array_dask, peak_array_dask, 5)


@pytest.mark.slow
class TestPeakFindLog:
    @pytest.mark.parametrize("x, y", [(112, 32), (170, 92), (54, 76), (10, 15)])
    def test_single_frame_one_peak(self, x, y):
        image = np.zeros(shape=(200, 100), dtype=np.float64)
        image[x, y] = 654
        min_sigma, max_sigma, num_sigma = 2, 5, 10
        threshold, overlap = 0.01, 1
        peaks = dt._peak_find_log_single_frame(
            image,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=threshold,
            overlap=overlap,
        )
        assert (x, y) == (peaks[0, 0], peaks[0, 1])

    def test_single_frame_multiple_peak(self):
        image = np.zeros(shape=(200, 100), dtype=np.float64)
        peak_list = [[120, 76], [23, 54], [32, 78], [10, 15]]
        for x, y in peak_list:
            image[x, y] = 654
        min_sigma, max_sigma, num_sigma = 2, 5, 10
        threshold, overlap = 0.01, 1
        peaks = dt._peak_find_log_single_frame(
            image,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=threshold,
            overlap=overlap,
        )
        assert len(peaks) == len(peak_list)
        for peak in peaks.tolist():
            assert peak in peak_list

    def test_single_frame_threshold(self):
        image = np.zeros(shape=(200, 100), dtype=np.float64)
        image[54, 29] = 100
        image[123, 54] = 20
        min_sigma, max_sigma, num_sigma, overlap = 2, 5, 10, 1
        peaks0 = dt._peak_find_log_single_frame(
            image,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=0.01,
            overlap=overlap,
        )
        assert len(peaks0) == 2
        peaks1 = dt._peak_find_log_single_frame(
            image,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=0.05,
            overlap=overlap,
        )
        assert len(peaks1) == 1

    def test_single_frame_normalize_value(self):
        image = np.zeros((100, 100), dtype=np.uint16)
        image[49:52, 49:52] = 100
        image[19:22, 9:12] = 10
        peaks0 = dt._peak_find_log_single_frame(image, normalize_value=100)
        peaks1 = dt._peak_find_log_single_frame(image, normalize_value=10)
        assert (peaks0 == [[50, 50]]).all()
        assert (peaks1 == [[50, 50], [20, 10]]).all()

    def test_chunk(self):
        data = np.zeros(shape=(2, 3, 200, 100), dtype=np.float64)
        data[0, 0, 50, 20] = 100
        data[0, 1, 51, 21] = 100
        data[0, 2, 52, 22] = 100
        data[1, 0, 53, 23] = 100
        data[1, 1, 54, 24] = 100
        data[1, 2, 55, 25] = 100
        min_sigma, max_sigma, num_sigma = 0.08, 1, 10
        threshold, overlap = 0.06, 0.01
        peaks = dt._peak_find_log_chunk(
            data,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=threshold,
            overlap=overlap,
        )
        assert peaks[0, 0][0].tolist() == [50, 20]
        assert peaks[0, 1][0].tolist() == [51, 21]
        assert peaks[0, 2][0].tolist() == [52, 22]
        assert peaks[1, 0][0].tolist() == [53, 23]
        assert peaks[1, 1][0].tolist() == [54, 24]
        assert peaks[1, 2][0].tolist() == [55, 25]

    def test_chunk_normalize_value(self):
        data = np.zeros((2, 3, 100, 100), dtype=np.uint16)
        data[:, :, 49:52, 49:52] = 100
        data[:, :, 19:22, 9:12] = 10

        peak_array0 = dt._peak_find_log_chunk(data, normalize_value=100)
        peak_array1 = dt._peak_find_log_chunk(data, normalize_value=10)
        for ix, iy in np.ndindex(peak_array0.shape):
            assert (peak_array0[ix, iy] == [[50, 50]]).all()
            assert (peak_array1[ix, iy] == [[50, 50], [20, 10]]).all()

    def test_dask_array(self):
        data = np.zeros(shape=(2, 3, 200, 100), dtype=np.float64)
        data[0, 0, 50, 20] = 100
        data[0, 1, 51, 21] = 100
        data[0, 2, 52, 22] = 100
        data[1, 0, 53, 23] = 100
        data[1, 1, 54, 24] = 100
        data[1, 2, 55, 25] = 100
        dask_array = da.from_array(data, chunks=(1, 1, 200, 100))
        min_sigma, max_sigma, num_sigma = 0.08, 1, 10
        threshold, overlap = 0.06, 0.01
        peaks = dt._peak_find_log(
            dask_array,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=threshold,
            overlap=overlap,
        )
        peaks = peaks.compute()
        assert peaks[0, 0][0].tolist() == [50, 20]
        assert peaks[0, 1][0].tolist() == [51, 21]
        assert peaks[0, 2][0].tolist() == [52, 22]
        assert peaks[1, 0][0].tolist() == [53, 23]
        assert peaks[1, 1][0].tolist() == [54, 24]
        assert peaks[1, 2][0].tolist() == [55, 25]

    def test_dask_array_normalize_value(self):
        data = np.zeros((2, 3, 100, 100), dtype=np.uint16)
        data[:, :, 49:52, 49:52] = 100
        data[:, :, 19:22, 9:12] = 10
        dask_array = da.from_array(data, chunks=(1, 1, 100, 100))
        peak_array0 = dt._peak_find_log(dask_array, normalize_value=100)
        peak_array1 = dt._peak_find_log(dask_array, normalize_value=10)
        for ix, iy in np.ndindex(peak_array0.shape):
            assert (peak_array0[ix, iy] == [[50, 50]]).all()
            assert (peak_array1[ix, iy] == [[50, 50], [20, 10]]).all()

    @pytest.mark.parametrize("nav_dims", [0, 1, 2, 3, 4])
    def test_array_different_dimensions(self, nav_dims):
        shape = list(np.random.randint(2, 6, size=nav_dims))
        shape.extend([50, 50])
        chunks = [1] * nav_dims
        chunks.extend([25, 25])
        dask_array = da.random.random(size=shape, chunks=chunks)
        peak_array_dask = dt._peak_find_log(dask_array)
        assert len(peak_array_dask.shape) == nav_dims
        assert dask_array.shape[:-2] == peak_array_dask.shape
        peak_array = peak_array_dask.compute()
        assert dask_array.shape[:-2] == peak_array.shape

    def test_1d_dask_array_error(self):
        dask_array = da.random.random(size=50, chunks=10)
        with pytest.raises(ValueError):
            dt._peak_find_log(dask_array)


@pytest.mark.slow
class TestCenterOfMass:
    def test_centerofmass(self):
        numpy_array = np.zeros((20, 20))
        numpy_array[10:15, 5:10] = 1
        cy, cx = dt._center_of_mass_hs(numpy_array)
        np.testing.assert_almost_equal(cx, 7)
        np.testing.assert_almost_equal(cy, 12)

    def test_get_experimental_square(self):
        numpy_array = np.zeros((20, 20))
        numpy_array[10:16, 5:11] = 1
        square_size = 6
        subf = dt._get_experimental_square(numpy_array, [13, 8], square_size)
        assert subf.shape[0] == 6
        assert subf.shape[1] == 6
        assert subf.all() == 1

    def test_get_experimental_square_wrong_input_odd_square_size(self):
        with pytest.raises(ValueError):
            dt._get_experimental_square(np.zeros((99, 99)), [13, 8], 9)

    def test_com_experimental_square(self):
        numpy_array = np.zeros((20, 20))
        numpy_array[10:16, 5:11] = 1
        square_size = 6
        subf = dt._center_of_mass_experimental_square(numpy_array, [13, 8], square_size)
        assert subf.shape[0] == 6
        assert subf.shape[1] == 6
        assert subf.sum() == (square_size - 1) ** 2

        subf1 = dt._center_of_mass_experimental_square(
            numpy_array, [40, 8], square_size
        )
        assert subf1 is None
