import pytest
import numpy as np
import dask.array as da
import pyxem.utils.dask_tools as dt
from pyxem.utils.peakfinders2D import find_peaks_dog, find_peaks_log


class TestPeakFindDog:
    @pytest.mark.parametrize("x, y", [(112, 32), (170, 92), (54, 76), (10, 15)])
    def test_single_frame_one_peak(self, x, y):
        image = np.zeros(shape=(200, 100), dtype=np.float64)
        image[x, y] = 654
        min_sigma, max_sigma, sigma_ratio = 2, 5, 5
        threshold, overlap = 0.01, 1
        peaks = find_peaks_dog(
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
        peaks = find_peaks_dog(
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
        peaks0 = find_peaks_dog(
            image,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            sigma_ratio=sigma_ratio,
            threshold=0.01,
            overlap=overlap,
        )
        assert len(peaks0) == 2
        peaks1 = find_peaks_dog(
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
        peaks0 = find_peaks_dog(
            image,
            min_sigma=1,
            max_sigma=max_sigma,
            sigma_ratio=sigma_ratio,
            threshold=threshold,
            overlap=overlap,
        )
        assert len(peaks0) == 2
        peaks1 = find_peaks_dog(
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
        peaks = find_peaks_dog(
            image,
            min_sigma=min_sigma,
            max_sigma=1.0,
            sigma_ratio=sigma_ratio,
            threshold=threshold,
            overlap=overlap,
        )
        assert len(peaks) > 1
        peaks = find_peaks_dog(
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
        peaks0 = find_peaks_dog(image, normalize=True)
        peaks1 = find_peaks_dog(image, normalize=False)
        assert (peaks0 == [[50, 50]]).all()
        assert (peaks1 == [[50, 50], [20, 10]]).all()

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
        peaks = dt._find_peak_dask_array(
            dask_array,
            find_peaks_dog,
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
        peak_array0 = dt._find_peak_dask_array(dask_array, find_peaks_dog,
                                               normalize=True)
        peak_array1 = dt._find_peak_dask_array(dask_array, find_peaks_dog,
                                               normalize=False)
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
        peak_array_dask = dt._find_peak_dask_array(dask_array, find_peaks_dog)
        assert len(peak_array_dask.shape) == nav_dims
        assert dask_array.shape[:-2] == peak_array_dask.shape
        peak_array = peak_array_dask.compute()
        assert dask_array.shape[:-2] == peak_array.shape

    def test_1d_dask_array_error(self):
        dask_array = da.random.random(size=50, chunks=10)
        with pytest.raises(ValueError):
            dt._find_peak_dask_array(dask_array, find_peaks_dog)


class TestPeakFindLog:
    @pytest.mark.parametrize("x, y", [(112, 32), (170, 92), (54, 76), (10, 15)])
    def test_single_frame_one_peak(self, x, y):
        image = np.zeros(shape=(200, 100), dtype=np.float64)
        image[x, y] = 654
        min_sigma, max_sigma, num_sigma = 2, 5, 10
        threshold, overlap = 0.01, 1
        peaks = find_peaks_log(
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
        peaks = find_peaks_log(
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
        peaks0 = find_peaks_log(
            image,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=0.01,
            overlap=overlap,
        )
        assert len(peaks0) == 2
        peaks1 = find_peaks_log(
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
        peaks0 = find_peaks_log(
            image,
            min_sigma=1,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=threshold,
            overlap=overlap,
        )
        assert len(peaks0) == 2
        peaks1 = find_peaks_log(
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
        peaks = find_peaks_log(
            image,
            min_sigma=min_sigma,
            max_sigma=1.0,
            num_sigma=num_sigma,
            threshold=threshold,
            overlap=overlap,
        )
        assert len(peaks) > 1
        peaks = find_peaks_log(
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
        peaks0 = find_peaks_log(image, normalize=True)
        peaks1 = find_peaks_log(image, normalize=False)
        assert (peaks0 == [[50, 50]]).all()
        assert (peaks1 == [[50, 50], [20, 10]]).all()

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
        peaks_dask = dt._find_peak_dask_array(
            dask_array,
            find_peaks_log,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=threshold,
            overlap=overlap,
        )
        peaks = peaks_dask.compute()
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
        peaks0_dask = dt._find_peak_dask_array(dask_array, find_peaks_log,
                                               normalize=True)
        peaks1_dask = dt._find_peak_dask_array(dask_array, find_peaks_log,
                                               normalize=False)
        peaks0 = peaks0_dask.compute()
        peaks1 = peaks1_dask.compute()
        for ix, iy in np.ndindex(peaks0.shape):
            assert (peaks0[ix, iy] == [[50, 50]]).all()
            assert (peaks1[ix, iy] == [[50, 50], [20, 10]]).all()

    @pytest.mark.parametrize("nav_dims", [0, 1, 2, 3, 4])
    def test_array_different_dimensions(self, nav_dims):
        shape = list(np.random.randint(2, 6, size=nav_dims))
        shape.extend([50, 50])
        chunks = [1] * nav_dims
        chunks.extend([25, 25])
        dask_array = da.random.random(size=shape, chunks=chunks)
        peak_array_dask = dt._find_peak_dask_array(dask_array,
                                                   find_peaks_log)
        assert len(peak_array_dask.shape) == nav_dims
        assert dask_array.shape[:-2] == peak_array_dask.shape
        peak_array = peak_array_dask.compute()
        assert dask_array.shape[:-2] == peak_array.shape

    def test_1d_dask_array_error(self):
        dask_array = da.random.random(size=50, chunks=10)
        with pytest.raises(ValueError):
            dt._find_peak_dask_array(dask_array, find_peaks_log)
