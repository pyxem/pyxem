# -*- coding: utf-8 -*-
# Copyright 2016-2024 The pyXem developers
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
import pyxem.utils._dask as dt
import pyxem.utils._background_subtraction as bt
import pyxem.utils.diffraction as et
import pyxem.utils._pixelated_stem_tools as pst


class TestSignalDimensionGetChunkSliceList:
    @pytest.mark.parametrize(
        "sig_chunks",
        [(10, 10), (5, 10), (5, 10), (5, 5), (20, 10), (20, 20)],
    )
    def test_chunksizes(self, sig_chunks):
        xchunk, ychunk = sig_chunks
        data = da.zeros((20, 20, 20, 20), chunks=(10, 10, ychunk, xchunk))
        chunk_slice_list = dt._get_signal_dimension_chunk_slice_list(data.chunks)
        assert len(data.chunks[-1]) * len(data.chunks[-2]) == len(chunk_slice_list)
        for chunk_slice in chunk_slice_list:
            xsize = chunk_slice[0].stop - chunk_slice[0].start
            ysize = chunk_slice[1].stop - chunk_slice[1].start
            assert xsize == xchunk
            assert ysize == ychunk

    def test_non_square_chunks(self):
        data = da.zeros((2, 2, 20, 20), chunks=(2, 2, 15, 15))
        chunk_slice_list = dt._get_signal_dimension_chunk_slice_list(data.chunks)
        assert chunk_slice_list[0] == (slice(0, 15, None), slice(0, 15, None))
        assert chunk_slice_list[1] == (slice(15, 20, None), slice(0, 15, None))
        assert chunk_slice_list[2] == (slice(0, 15, None), slice(15, 20, None))
        assert chunk_slice_list[3] == (slice(15, 20, None), slice(15, 20, None))

    def test_one_signal_chunk(self):
        data = da.zeros((2, 2, 20, 20), chunks=(1, 1, 20, 20))
        chunk_slice_list = dt._get_signal_dimension_chunk_slice_list(data.chunks)
        assert len(chunk_slice_list) == 1
        assert chunk_slice_list[0] == np.s_[0:20, 0:20]

    def test_rechunk(self):
        data = da.zeros((2, 2, 20, 20), chunks=(1, 1, 20, 20))
        data1 = data.rechunk((2, 2, 10, 10))
        chunk_slice_list = dt._get_signal_dimension_chunk_slice_list(data1.chunks)
        assert len(chunk_slice_list) == 4

    def test_slice_navigation(self):
        data = da.zeros((2, 2, 20, 20), chunks=(1, 1, 20, 20))
        data1 = data[0, 1]
        chunk_slice_list = dt._get_signal_dimension_chunk_slice_list(data1.chunks)
        assert len(chunk_slice_list) == 1
        assert chunk_slice_list[0] == np.s_[0:20, 0:20]


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
        chunk_slice = dt._get_signal_dimension_host_chunk_slice(x, y, data.chunks)
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
        chunk_slice = dt._get_signal_dimension_host_chunk_slice(x, y, data.chunks)
        assert chunk_slice == sig_slice


class TestAlignSingleFrame:
    @pytest.mark.parametrize(
        "shifts", [[0, 0], [1, 0], [-1, 0], [0, -1], [-1, -1], [-2, -3], [-2, -5]]
    )
    def test_simple(self, shifts):
        x_size, y_size = 5, 9
        image = np.zeros((y_size, x_size), dtype=np.uint16)
        x, y = 3, 7
        image[y, x] = 7
        image_shifted = dt._align_single_frame(image, shifts)
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
        image_shifted = dt._align_single_frame(image, shifts, order=1)
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
        image_shifted = dt._align_single_frame(image, shifts, order=1)
        assert image_shifted[pos].sum() == 9
        image_shifted[pos] = 0
        assert not image_shifted.any()

    @pytest.mark.parametrize("shifts", [[-0.7, -2.7], [1.1, -1.1]])
    def test_not_subpixel_float_image(self, shifts):
        x_size, y_size = 5, 9
        image = np.zeros((y_size, x_size), dtype=np.float32)
        x, y = 3, 7
        image[y, x] = 9
        image_shifted = dt._align_single_frame(image, shifts, order=0)
        pos = np.s_[y + round(shifts[1]), x + round(shifts[0])]
        assert image_shifted[pos] == 9.0
        image_shifted[pos] = 0
        assert not image_shifted.any()


class TestRemoveBadPixels:
    def test_simple(self):
        data = np.ones((20, 30)) * 12
        data[5, 9] = 0
        data[2, 1] = 0
        output = et.remove_bad_pixels(data, data == 0)
        assert (output == 12).all()


class TestTemplateMatchBinaryImage:
    @pytest.mark.parametrize("x, y", [(13, 32), (76, 32), (87, 21), (43, 85)])
    def test_single_frame(self, x, y):
        disk_r = 5
        disk = sm.disk(disk_r)
        data = np.zeros(shape=(100, 100))
        data[y - disk_r : y + disk_r + 1, x - disk_r : x + disk_r + 1] = disk
        match = et.normalize_template_match(data, disk)
        index = np.unravel_index(np.argmax(match), match.shape)
        assert (y, x) == index


class TestBackgroundRemoval:
    def test_median_sub(self):
        footprint = 10
        numpy_array = np.ones((50, 50))
        numpy_array[20:30, 20:30] = 5

        data = bt._subtract_median(numpy_array, footprint=footprint)
        assert data.sum() != numpy_array.sum()
        assert data.shape == numpy_array.shape
        assert data[0, :].all() == 0

    def test_radial_median_sub(self):
        center_x = 25
        center_y = 25
        numpy_array = np.ones((50, 50))
        numpy_array[20:30, 20:30] = 5
        data = bt._subtract_radial_median(
            numpy_array, center_x=center_x, center_y=center_y
        )
        assert data.sum() != numpy_array.sum()
        assert data.shape == numpy_array.shape
        assert data[0, :].all() == 0

    def test_dog_sub(self):
        min_sigma = 10
        numpy_array = np.ones((50, 50))
        numpy_array[20:30, 20:30] = 5
        data = bt._subtract_dog(numpy_array, min_sigma=min_sigma)
        assert data.sum() != numpy_array.sum()
        assert data.shape == numpy_array.shape
