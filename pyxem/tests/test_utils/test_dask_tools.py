# -*- coding: utf-8 -*-
# Copyright 2016-2020 The pyXem developers
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
import pyxem.utils.dask_tools as dt
import pyxem.utils.pixelated_stem_tools as pst
import pyxem.dummy_data.dask_test_data as dtd

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

@pytest.mark.slow
class TestFindDeadPixels:
    def test_2d(self):
        data = dtd._get_dead_pixel_test_data_2d()
        dead_pixels = dt._find_dead_pixels(data)
        assert data.shape == dead_pixels.shape
        dead_pixels = dead_pixels.compute()
        assert dead_pixels[14, 42]
        assert dead_pixels[2, 12]
        dead_pixels[14, 42] = False
        dead_pixels[2, 12] = False
        assert not dead_pixels.any()

    def test_3d(self):
        data = dtd._get_dead_pixel_test_data_3d()
        dead_pixels = dt._find_dead_pixels(data)
        assert data.shape[-2:] == dead_pixels.shape
        dead_pixels = dead_pixels.compute()
        assert dead_pixels[14, 42]
        assert dead_pixels[2, 12]
        dead_pixels[14, 42] = False
        dead_pixels[2, 12] = False
        assert not dead_pixels.any()

    def test_4d(self):
        data = dtd._get_dead_pixel_test_data_4d()
        dead_pixels = dt._find_dead_pixels(data)
        assert data.shape[-2:] == dead_pixels.shape
        dead_pixels = dead_pixels.compute()
        assert dead_pixels[14, 42]
        assert dead_pixels[2, 12]
        dead_pixels[14, 42] = False
        dead_pixels[2, 12] = False
        assert not dead_pixels.any()

    def test_dead_pixel_value(self):
        data = np.random.randint(10, 100, size=(20, 30))
        data[5, 9] = 3
        data[2, 1] = 3
        data = da.from_array(data, chunks=(5, 5))
        dead_pixels = dt._find_dead_pixels(data, dead_pixel_value=3)
        assert data.shape == dead_pixels.shape
        dead_pixels = dead_pixels.compute()
        assert dead_pixels[5, 9]
        assert dead_pixels[2, 1]
        dead_pixels[5, 9] = False
        dead_pixels[2, 1] = False
        assert not dead_pixels.any()

    def test_mask_array(self):
        data = dtd._get_dead_pixel_test_data_2d()
        mask_array = np.zeros((40, 50), dtype=np.bool)
        mask_array[:, 30:] = True
        dead_pixels = dt._find_dead_pixels(data, mask_array=mask_array)
        assert data.shape == dead_pixels.shape
        dead_pixels = dead_pixels.compute()
        assert dead_pixels[2, 12]
        dead_pixels[2, 12] = False
        assert not dead_pixels.any()

    def test_1d_wrong_shape(self):
        data = np.random.randint(10, 100, size=40)
        data = da.from_array(data, chunks=(5))
        with pytest.raises(ValueError):
            dt._find_dead_pixels(data)

@pytest.mark.slow
class TestFindHotPixels:
    def test_2d(self):
        data = dtd._get_hot_pixel_test_data_2d()
        hot_pixels = dt._find_hot_pixels(data)
        assert data.shape == hot_pixels.shape
        hot_pixels = hot_pixels.compute()
        assert hot_pixels[21, 11]
        assert hot_pixels[5, 38]
        hot_pixels[21, 11] = False
        hot_pixels[5, 38] = False
        assert not hot_pixels.any()

    def test_3d(self):
        data = dtd._get_hot_pixel_test_data_3d()
        hot_pixels = dt._find_hot_pixels(data)
        assert data.shape == hot_pixels.shape
        hot_pixels = hot_pixels.compute()
        assert hot_pixels[2, 21, 11]
        assert hot_pixels[1, 5, 38]
        hot_pixels[2, 21, 11] = False
        hot_pixels[1, 5, 38] = False
        assert not hot_pixels.any()

    def test_4d(self):
        data = dtd._get_hot_pixel_test_data_4d()
        hot_pixels = dt._find_hot_pixels(data)
        assert data.shape == hot_pixels.shape
        hot_pixels = hot_pixels.compute()
        assert hot_pixels[4, 2, 21, 11]
        assert hot_pixels[6, 1, 5, 38]
        hot_pixels[4, 2, 21, 11] = False
        hot_pixels[6, 1, 5, 38] = False
        assert not hot_pixels.any()

    def test_2d_mask(self):
        mask_array = np.zeros((40, 50), dtype=np.bool)
        mask_array[:, 30:] = True
        data = dtd._get_hot_pixel_test_data_2d()
        hot_pixels = dt._find_hot_pixels(data, mask_array=mask_array)
        assert data.shape == hot_pixels.shape
        hot_pixels = hot_pixels.compute()
        assert hot_pixels[21, 11]
        hot_pixels[21, 11] = False
        assert not hot_pixels.any()

    def test_3d_mask(self):
        mask_array = np.zeros((40, 50), dtype=np.bool)
        mask_array[:, 30:] = True
        data = dtd._get_hot_pixel_test_data_3d()
        hot_pixels = dt._find_hot_pixels(data, mask_array=mask_array)
        assert data.shape == hot_pixels.shape
        hot_pixels = hot_pixels.compute()
        assert hot_pixels[2, 21, 11]
        hot_pixels[2, 21, 11] = False
        assert not hot_pixels.any()

    def test_4d_mask(self):
        mask_array = np.zeros((40, 50), dtype=np.bool)
        mask_array[:, 30:] = True
        data = dtd._get_hot_pixel_test_data_4d()
        hot_pixels = dt._find_hot_pixels(data, mask_array=mask_array)
        assert data.shape == hot_pixels.shape
        hot_pixels = hot_pixels.compute()
        assert hot_pixels[4, 2, 21, 11]
        hot_pixels[4, 2, 21, 11] = False
        assert not hot_pixels.any()

    def test_threshold_multiplier(self):
        data = dtd._get_hot_pixel_test_data_2d()
        hot_pixels = dt._find_hot_pixels(data, threshold_multiplier=1000000)
        hot_pixels = hot_pixels.compute()
        assert not hot_pixels.any()

        hot_pixels = dt._find_hot_pixels(data, threshold_multiplier=-1000000)
        hot_pixels = hot_pixels.compute()
        assert hot_pixels.all()

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

    def test_single_frame_min_sigma(self):
        image = np.zeros(shape=(200, 100), dtype=np.float64)
        image[54, 29] = 100
        image[54, 32] = 100
        max_sigma, sigma_ratio = 5, 5
        threshold, overlap = 0.01, 0.1
        peaks0 = dt._peak_find_dog_single_frame(
            image,
            min_sigma=1,
            max_sigma=max_sigma,
            sigma_ratio=sigma_ratio,
            threshold=threshold,
            overlap=overlap,
        )
        assert len(peaks0) == 2
        peaks1 = dt._peak_find_dog_single_frame(
            image,
            min_sigma=2,
            max_sigma=max_sigma,
            sigma_ratio=sigma_ratio,
            threshold=threshold,
            overlap=overlap,
        )
        assert len(peaks1) == 1

    def test_single_frame_max_sigma(self):
        image = np.zeros(shape=(200, 100), dtype=np.float64)
        image[52:58, 22:28] = 100
        min_sigma, sigma_ratio = 0.1, 5
        threshold, overlap = 0.1, 0.01
        peaks = dt._peak_find_dog_single_frame(
            image,
            min_sigma=min_sigma,
            max_sigma=1.0,
            sigma_ratio=sigma_ratio,
            threshold=threshold,
            overlap=overlap,
        )
        assert len(peaks) > 1
        peaks = dt._peak_find_dog_single_frame(
            image,
            min_sigma=min_sigma,
            max_sigma=5.0,
            sigma_ratio=sigma_ratio,
            threshold=threshold,
            overlap=overlap,
        )
        assert len(peaks) == 1

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

    def test_chunk_peak(self):
        numpy_array = np.zeros((2, 2, 50, 50))
        numpy_array[:, :, 25, 25] = 1

        peak_array = np.zeros(
            (numpy_array.shape[0], numpy_array.shape[1], 1, 1), dtype=np.object
        )
        real_array = np.zeros((numpy_array.shape[:-2]), dtype=np.object)
        for index in np.ndindex(numpy_array.shape[:-2]):
            islice = np.s_[index]
            peak_array[islice][0, 0] = np.asarray([(27, 27)])
            real_array[islice] = np.asarray([(25, 25)])

        square_size = 12

        data = dt._peak_refinement_centre_of_mass_chunk(
            numpy_array, peak_array, square_size
        )
        assert data.shape == (2, 2)
        assert np.sum(data - real_array).sum() == 0

    def test_dask_array(self):
        numpy_array = np.zeros((10, 10, 50, 50))
        numpy_array[:, :, 25, 25] = 1

        peak_array = np.zeros((numpy_array.shape[:-2]), dtype=np.object)
        real_array = np.zeros((numpy_array.shape[:-2]), dtype=np.object)
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
        peak_array = np.zeros((dask_array.shape[:-2]), dtype=np.object)
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

    def test_chunk_min_sigma(self):
        min_sigma = 10
        numpy_array = np.ones((10, 10, 50, 50))
        numpy_array[:, :20:30, 20:30] = 5
        data = dt._background_removal_chunk_dog(numpy_array, min_sigma=min_sigma)
        assert data.sum() != numpy_array.sum()
        assert data.shape == numpy_array.shape
        assert data[:, :, 0, :].all() == 0

    def test_chunk_max_sigma(self):
        max_sigma = 10
        numpy_array = np.ones((10, 10, 50, 50))
        numpy_array[:, :20:30, 20:30] = 5
        data = dt._background_removal_chunk_dog(numpy_array, max_sigma=max_sigma)
        assert data.sum() != numpy_array.sum()
        assert data.shape == numpy_array.shape
        assert data[:, :, 0, :].all() == 0

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

    def test_intensity_peaks_chunk(self):
        numpy_array = np.zeros((2, 2, 50, 50))
        numpy_array[:, :, 27, 27] = 1

        peak_array = np.zeros(
            (numpy_array.shape[0], numpy_array.shape[1]), dtype=np.object
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
            (numpy_array.shape[0], numpy_array.shape[1]), dtype=np.object
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
        peak_array = np.zeros((dask_array.shape[:-2]), dtype=np.object)
        for index in np.ndindex(dask_array.shape[:-2]):
            islice = np.s_[index]
            peak_array[islice] = np.asarray([(27, 27)])
        peak_array_dask = da.from_array(peak_array, chunks=chunks[:-2])
        match_array_dask = dt._intensity_peaks_image(dask_array, peak_array_dask, 5)
        assert len(dask_array.shape) == nav_dims + 2
        match_array = match_array_dask.compute()
        assert peak_array_dask.shape == match_array.shape

    def test_non_dask_array(self):
        data_array = np.ones((10, 10, 50, 50))
        data_array_dask = da.ones((10, 10, 50, 50), chunks=(2, 2, 25, 25))
        peak_array = np.empty((10, 10), dtype=np.object)
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
        peak_array_dask = da.empty((6, 16), chunks=(2, 2), dtype=np.object)
        dt._intensity_peaks_image(data_array_dask, peak_array_dask, 5)

    def test_different_chunks(self):
        data_array_dask = da.ones((6, 16, 100, 50), chunks=(6, 4, 50, 25))
        peak_array_dask = da.empty((6, 16), chunks=(3, 2), dtype=np.object)
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

    def test_single_frame_min_sigma(self):
        image = np.zeros(shape=(200, 100), dtype=np.float64)
        image[54, 29] = 100
        image[54, 32] = 100
        max_sigma, num_sigma = 5, 10
        threshold, overlap = 0.01, 0.1
        peaks0 = dt._peak_find_log_single_frame(
            image,
            min_sigma=1,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=threshold,
            overlap=overlap,
        )
        assert len(peaks0) == 2
        peaks1 = dt._peak_find_log_single_frame(
            image,
            min_sigma=2,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=threshold,
            overlap=overlap,
        )
        assert len(peaks1) == 1

    def test_single_frame_max_sigma(self):
        image = np.zeros(shape=(200, 100), dtype=np.float64)
        image[52:58, 22:28] = 100
        min_sigma, num_sigma = 0.1, 10
        threshold, overlap = 0.1, 0.01
        peaks = dt._peak_find_log_single_frame(
            image,
            min_sigma=min_sigma,
            max_sigma=1.0,
            num_sigma=num_sigma,
            threshold=threshold,
            overlap=overlap,
        )
        assert len(peaks) > 1
        peaks = dt._peak_find_log_single_frame(
            image,
            min_sigma=min_sigma,
            max_sigma=5.0,
            num_sigma=num_sigma,
            threshold=threshold,
            overlap=overlap,
        )
        assert len(peaks) == 2

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

    def test_com_experimental_square(self):
        numpy_array = np.zeros((20, 20))
        numpy_array[10:16, 5:11] = 1
        square_size = 6
        subf = dt._center_of_mass_experimental_square(numpy_array, [13, 8], square_size)
        assert subf.shape[0] == 6
        assert subf.shape[1] == 6
        assert subf.sum() == (square_size - 1) ** 2
