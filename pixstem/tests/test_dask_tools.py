import pytest
import numpy as np
import dask.array as da
import skimage.morphology as sm
import pixstem.dask_tools as dt
import pixstem.pixelated_stem_tools as pst
import pixstem.dask_test_data as dtd


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


class TestRemoveBadPixels:

    def test_simple(self):
        data = np.ones((20, 30))*12
        data[5, 9] = 0
        data[2, 1] = 0
        dask_array = da.from_array(data, chunks=(5, 5))
        dead_pixels = dt._find_dead_pixels(dask_array)
        output = dt._remove_bad_pixels(dask_array, dead_pixels)
        output = output.compute()
        assert (output == 12).all()

    def test_3d(self):
        data = np.ones((5, 20, 30))*12
        data[:, 5, 9] = 0
        data[:, 2, 1] = 0
        dask_array = da.from_array(data, chunks=(5, 5, 5))
        dead_pixels = dt._find_dead_pixels(dask_array)
        output = dt._remove_bad_pixels(dask_array, dead_pixels)
        output = output.compute()
        assert (output == 12).all()

    def test_4d(self):
        data = np.ones((5, 10, 20, 30))*12
        data[:, :, 5, 9] = 0
        data[:, :, 2, 1] = 0
        dask_array = da.from_array(data, chunks=(5, 5, 5, 5))
        dead_pixels = dt._find_dead_pixels(dask_array)
        output = dt._remove_bad_pixels(dask_array, dead_pixels)
        output = output.compute()
        assert (output == 12).all()

    def test_3d_same_bad_pixel_array_shape(self):
        data = np.ones((5, 20, 30))*12
        data[2, 5, 9] = 0
        data[3, 2, 1] = 0
        dask_array = da.from_array(data, chunks=(5, 5, 5))
        bad_pixel_array = np.zeros_like(dask_array)
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


class TestTemplateMatchDisk:

    @pytest.mark.parametrize(
            "x, y", [(13, 32), (76, 32), (87, 21), (43, 85)])
    def test_single_frame(self, x, y):
        disk_r = 5
        disk = sm.disk(disk_r)
        data = np.zeros(shape=(100, 100))

        data[y-disk_r:y+disk_r+1, x-disk_r:x+disk_r+1] = disk
        match = dt._template_match_disk_single_frame(data, disk)
        index = np.unravel_index(np.argmax(match), match.shape)
        assert (y, x) == index

    def test_chunk(self):
        x, y, disk_r = 76, 23, 5
        disk = sm.disk(disk_r)
        data = np.zeros(shape=(5, 10, 100, 90))
        data[:, :, y-disk_r:y+disk_r+1, x-disk_r:x+disk_r+1] = disk
        match_array = dt._template_match_disk_chunk(data, disk)
        assert data.shape == match_array.shape
        for ix, iy in np.ndindex(data.shape[:2]):
            match = match_array[ix, iy]
            index = np.unravel_index(np.argmax(match), match.shape)
            assert (y, x) == index

    def test_simple(self):
        data = np.ones((5, 3, 50, 40))
        dask_array = da.from_array(data, chunks=(1, 1, 5, 5))
        match_array_dask = dt._template_match_disk(dask_array, disk_r=5)
        match_array = match_array_dask.compute()
        assert match_array.shape == data.shape

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
        dask_array = da.from_array(data, chunks=(1, 1, 5, 5))
        out_dask = dt._template_match_disk(dask_array, disk_r=disk_r)
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
        match_array_dask = dt._template_match_disk(dask_array, disk_r=5)
        assert len(dask_array.shape) == nav_dims + 2
        assert dask_array.shape == match_array_dask.shape
        match_array = match_array_dask.compute()
        assert dask_array.shape == match_array.shape

    def test_1d_dask_array_error(self):
        dask_array = da.random.random(size=50, chunks=10)
        with pytest.raises(ValueError):
            dt._template_match_disk(dask_array, disk_r=5)


class TestPeakFindDog:

    @pytest.mark.parametrize(
            "x, y", [(112, 32), (170, 92), (54, 76), (10, 15)])
    def test_single_frame_one_peak(self, x, y):
        image = np.zeros(shape=(200, 100), dtype=np.float64)
        image[x, y] = 654
        min_sigma, max_sigma, sigma_ratio = 2, 5, 5
        threshold, overlap = 0.01, 1
        peaks = dt._peak_find_dog_single_frame(
                image, min_sigma=min_sigma, max_sigma=max_sigma,
                sigma_ratio=sigma_ratio, threshold=threshold, overlap=overlap)
        assert (x, y) == (peaks[0, 0], peaks[0, 1])

    def test_single_frame_multiple_peak(self):
        image = np.zeros(shape=(200, 100), dtype=np.float64)
        peak_list = [[120, 76], [23, 54], [32, 78], [10, 15]]
        for x, y in peak_list:
            image[x, y] = 654
        min_sigma, max_sigma, sigma_ratio = 2, 5, 5
        threshold, overlap = 0.01, 1
        peaks = dt._peak_find_dog_single_frame(
                image, min_sigma=min_sigma, max_sigma=max_sigma,
                sigma_ratio=sigma_ratio, threshold=threshold, overlap=overlap)
        assert len(peaks) == len(peak_list)
        for peak in peaks.tolist():
            assert peak in peak_list

    def test_single_frame_threshold(self):
        image = np.zeros(shape=(200, 100), dtype=np.float64)
        image[54, 29] = 100
        image[123, 54] = 20
        min_sigma, max_sigma, sigma_ratio, overlap = 2, 5, 5, 1
        peaks0 = dt._peak_find_dog_single_frame(
                image, min_sigma=min_sigma, max_sigma=max_sigma,
                sigma_ratio=sigma_ratio, threshold=0.01, overlap=overlap)
        assert len(peaks0) == 2
        peaks1 = dt._peak_find_dog_single_frame(
                image, min_sigma=min_sigma, max_sigma=max_sigma,
                sigma_ratio=sigma_ratio, threshold=0.05, overlap=overlap)
        assert len(peaks1) == 1

    def test_single_frame_min_sigma(self):
        image = np.zeros(shape=(200, 100), dtype=np.float64)
        image[54, 29] = 100
        image[54, 32] = 100
        max_sigma, sigma_ratio = 5, 5
        threshold, overlap = 0.1, 0.1
        peaks0 = dt._peak_find_dog_single_frame(
                image, min_sigma=1, max_sigma=max_sigma,
                sigma_ratio=sigma_ratio, threshold=threshold, overlap=overlap)
        assert len(peaks0) == 2
        peaks1 = dt._peak_find_dog_single_frame(
                image, min_sigma=2, max_sigma=max_sigma,
                sigma_ratio=sigma_ratio, threshold=threshold, overlap=overlap)
        assert len(peaks1) == 1

    def test_single_frame_max_sigma(self):
        image = np.zeros(shape=(200, 100), dtype=np.float64)
        image[52:58, 22:28] = 100
        min_sigma, sigma_ratio = 0.1, 5
        threshold, overlap = 0.1, 0.01
        peaks = dt._peak_find_dog_single_frame(
                image, min_sigma=min_sigma, max_sigma=1.0,
                sigma_ratio=sigma_ratio, threshold=threshold, overlap=overlap)
        assert len(peaks) > 1
        peaks = dt._peak_find_dog_single_frame(
                image, min_sigma=min_sigma, max_sigma=5.0,
                sigma_ratio=sigma_ratio, threshold=threshold, overlap=overlap)
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
                data, min_sigma=min_sigma, max_sigma=max_sigma,
                sigma_ratio=sigma_ratio, threshold=threshold, overlap=overlap)
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
            dask_array, min_sigma=min_sigma, max_sigma=max_sigma,
            sigma_ratio=sigma_ratio, threshold=threshold, overlap=overlap)
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
