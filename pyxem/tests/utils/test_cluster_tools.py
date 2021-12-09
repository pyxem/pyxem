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
from numpy.random import randint
from pyxem.signals import Diffraction2D
import pyxem.utils.cluster_tools as ct


class TestFilterPeakList:
    def test_simple(self):
        peak_list = [
            [128, 129],
            [10, 0],
            [0, 120],
            [255, 123],
            [123, 255],
            [255, 255],
            [0, 0],
        ]
        peak_list_filtered = ct._filter_peak_list(peak_list)
        assert [[128, 129]] == peak_list_filtered

    def test_max_x_index(self):
        peak_list = [
            [128, 129],
            [10, 0],
            [0, 120],
            [256, 123],
            [123, 256],
            [256, 256],
            [0, 0],
        ]
        peak_list_filtered = ct._filter_peak_list(peak_list, max_x_index=256)
        assert [[128, 129], [123, 256]] == peak_list_filtered

    def test_max_y_index(self):
        peak_list = [
            [128, 129],
            [10, 0],
            [0, 120],
            [256, 123],
            [123, 256],
            [256, 256],
            [0, 0],
        ]
        peak_list_filtered = ct._filter_peak_list(peak_list, max_y_index=256)
        assert [[128, 129], [256, 123]] == peak_list_filtered


class TestFilterPeakListRadius:
    def test_simple(self):
        peak_list = np.random.randint(100, size=(1000, 2))
        peak_filtered_list0 = ct._filter_peak_list_radius(
            peak_list, xc=50, yc=50, r_min=30
        )
        assert len(peak_list) != len(peak_filtered_list0)
        peak_filtered_list1 = ct._filter_peak_list_radius(
            peak_list, xc=50, yc=50, r_min=1000
        )
        assert len(peak_filtered_list1) == 0

    def test_r_lim(self):
        peak_list = np.array([[50, 50], [50, 30]])
        peak_filtered_list0 = ct._filter_peak_list_radius(
            peak_list, xc=50, yc=50, r_min=19
        )
        assert (peak_filtered_list0 == np.array([[50, 30]])).all()
        peak_filtered_list1 = ct._filter_peak_list_radius(
            peak_list, xc=50, yc=50, r_min=21
        )
        assert len(peak_filtered_list1) == 0

    def test_r_max(self):
        peak_list = np.array([[50, 0], [0, 30]])
        peak_filtered_list0 = ct._filter_peak_list_radius(
            peak_list, xc=0, yc=0, r_max=40
        )
        assert (peak_filtered_list0 == np.array([[0, 30]])).all()
        peak_filtered_list1 = ct._filter_peak_list_radius(
            peak_list, xc=0, yc=0, r_max=52
        )
        assert (peak_filtered_list1 == np.array([[50, 0], [0, 30]])).all()
        peak_filtered_list2 = ct._filter_peak_list_radius(
            peak_list, xc=0, yc=0, r_max=81
        )
        assert len(peak_filtered_list2) == 2
        peak_filtered_list3 = ct._filter_peak_list_radius(
            peak_list, xc=0, yc=0, r_max=11
        )
        assert len(peak_filtered_list3) == 0

    def test_r_min_and_r_max(self):
        peak_list = np.array([[50, 0], [0, 30]])
        peak_filtered_list0 = ct._filter_peak_list_radius(
            peak_list, xc=0, yc=0, r_min=20, r_max=40
        )
        assert (peak_filtered_list0 == np.array([[0, 30]])).all()
        peak_filtered_list1 = ct._filter_peak_list_radius(
            peak_list, xc=0, yc=0, r_min=20, r_max=60
        )
        assert (peak_filtered_list1 == np.array([[50, 0], [0, 30]])).all()
        peak_filtered_list2 = ct._filter_peak_list_radius(
            peak_list, xc=0, yc=0, r_min=40, r_max=60
        )
        assert (peak_filtered_list2 == np.array([[50, 0]])).all()
        peak_filtered_list3 = ct._filter_peak_list_radius(
            peak_list, xc=0, yc=0, r_min=10, r_max=20
        )
        assert len(peak_filtered_list3) == 0
        peak_filtered_list4 = ct._filter_peak_list_radius(
            peak_list, xc=0, yc=0, r_min=90, r_max=99
        )
        assert len(peak_filtered_list4) == 0

    def test_xc_yc(self):
        peak_list = np.array([[50, 10], [50, 50]])
        peak_filtered_list = ct._filter_peak_list_radius(
            peak_list, xc=10, yc=50, r_min=10
        )
        assert (peak_filtered_list == np.array([[50, 50]])).all()

    def test_wrong_input_no_r_min_or_r_max(self):
        peak_list = np.array([[50, 10], [50, 50]])
        with pytest.raises(ValueError):
            ct._filter_peak_list_radius(peak_list, xc=10, yc=50)

    def test_r_min_larger_than_r_max(self):
        peak_list = np.array([[50, 10], [50, 50]])
        with pytest.raises(ValueError):
            ct._filter_peak_list_radius(peak_list, xc=10, yc=50, r_min=50, r_max=30)


class TestFilterPeakArrayRadius:
    def test_simple(self):
        peak_array = np.empty(shape=(2, 3), dtype=object)
        for ix, iy in np.ndindex(peak_array.shape):
            peak_array[ix, iy] = np.random.randint(30, 70, size=(1000, 2))
        peak_array_filtered = ct._filter_peak_array_radius(peak_array, 50, 50, r_min=10)
        assert peak_array_filtered.shape == (2, 3)
        for ix, iy in np.ndindex(peak_array_filtered.shape):
            assert len(peak_array_filtered[ix, iy]) != 1000

    def test_r_lim(self):
        peak_array = np.empty(shape=(2, 3), dtype=object)
        for ix, iy in np.ndindex(peak_array.shape):
            peak_array[ix, iy] = np.random.randint(30, 70, size=(1000, 2))
        peak_array_filtered = ct._filter_peak_array_radius(peak_array, 50, 50, r_min=50)
        for ix, iy in np.ndindex(peak_array_filtered.shape):
            assert len(peak_array_filtered[ix, iy]) == 0

    def test_r_max(self):
        peak_array = np.empty(shape=(2, 3), dtype=object)
        for ix, iy in np.ndindex(peak_array.shape):
            peak_array[ix, iy] = np.random.randint(30, 70, size=(1000, 2))
        peak_array_filtered0 = ct._filter_peak_array_radius(peak_array, 0, 0, r_max=20)
        for ix, iy in np.ndindex(peak_array_filtered0.shape):
            assert len(peak_array_filtered0[ix, iy]) == 0
        peak_array_filtered1 = ct._filter_peak_array_radius(peak_array, 0, 0, r_max=100)
        for ix, iy in np.ndindex(peak_array_filtered1.shape):
            assert len(peak_array_filtered1[ix, iy]) == 1000

    def test_r_min_and_r_max(self):
        peak_array = np.empty(shape=(2, 3), dtype=object)
        for ix, iy in np.ndindex(peak_array.shape):
            peak_array[ix, iy] = np.random.randint(30, 70, size=(1000, 2))
        peak_array_filtered0 = ct._filter_peak_array_radius(
            peak_array, 0, 0, r_min=20, r_max=25
        )
        for ix, iy in np.ndindex(peak_array_filtered0.shape):
            assert len(peak_array_filtered0[ix, iy]) == 0
        peak_array_filtered1 = ct._filter_peak_array_radius(
            peak_array, 0, 0, r_min=20, r_max=100
        )
        for ix, iy in np.ndindex(peak_array_filtered1.shape):
            assert len(peak_array_filtered1[ix, iy]) == 1000
        peak_array_filtered2 = ct._filter_peak_array_radius(
            peak_array, 0, 0, r_min=110, r_max=130
        )
        for ix, iy in np.ndindex(peak_array_filtered2.shape):
            assert len(peak_array_filtered2[ix, iy]) == 0

    def test_xc(self):
        peak_array = np.empty(shape=(2, 3), dtype=object)
        for ix, iy in np.ndindex(peak_array.shape):
            peak_array[ix, iy] = np.random.randint(30, 70, size=(1000, 2))
        peak_array_filtered = ct._filter_peak_array_radius(peak_array, 10, 50, r_min=10)
        for ix, iy in np.ndindex(peak_array_filtered.shape):
            assert len(peak_array_filtered[ix, iy]) == 1000

    def test_yc(self):
        peak_array = np.empty(shape=(2, 3), dtype=object)
        for ix, iy in np.ndindex(peak_array.shape):
            peak_list = np.random.randint(30, 70, size=(999, 2)).tolist()
            peak_list.append([10, 50])
            peak_array[ix, iy] = np.array(peak_list)
        peak_array_filtered = ct._filter_peak_array_radius(peak_array, 50, 10, r_min=10)
        for ix, iy in np.ndindex(peak_array_filtered.shape):
            assert len(peak_array_filtered[ix, iy]) == 999


class TestFindMaxIndices4DPeakArray:
    def test_simple(self):
        max_x, max_y = 255, 127
        peak_array0 = np.random.randint(10, max_x, size=(3, 4, 5000, 1))
        peak_array1 = np.random.randint(5, max_y, size=(3, 4, 5000, 1))
        peak_array = np.concatenate((peak_array0, peak_array1), axis=3)
        max_x_index, max_y_index = ct._find_max_indices_4D_peak_array(peak_array)
        assert max_x_index == max_x - 1
        assert max_y_index == max_y - 1


class TestFilter4DPeakArray:
    def test_simple(self):
        peak_array0 = randint(124, 132, size=(3, 4, 10, 2))
        peak_array1 = np.ones(shape=(3, 4, 3, 2)) * 255
        peak_array2 = np.zeros(shape=(3, 4, 3, 2))
        peak_array = np.concatenate((peak_array0, peak_array1, peak_array2), axis=2)
        peak_array_filtered = ct._filter_4D_peak_array(peak_array)
        for ix, iy in np.ndindex(peak_array_filtered.shape[:2]):
            peak_list = peak_array_filtered[ix, iy]
            for x, y in peak_list:
                assert x != 0
                assert x != 255
                assert y != 0
                assert y != 255

    def test_max_x_index_max_y_index(self):
        peak_array0 = randint(124, 132, size=(3, 4, 10, 2))
        peak_array1 = np.ones(shape=(3, 4, 3, 2)) * 256
        peak_array2 = np.zeros(shape=(3, 4, 3, 2))
        peak_array = np.concatenate((peak_array0, peak_array1, peak_array2), axis=2)
        peak_array_filtered = ct._filter_4D_peak_array(
            peak_array, max_x_index=256, max_y_index=256
        )
        for ix, iy in np.ndindex(peak_array_filtered.shape[:2]):
            peak_list = peak_array_filtered[ix, iy]
            for x, y in peak_list:
                assert x != 0
                assert x != 256
                assert y != 0
                assert y != 256

    def test_signal_axes(self):
        s = Diffraction2D(np.zeros(shape=(3, 4, 128, 128)))
        peak_array0 = randint(62, 67, size=(3, 4, 10, 2))
        peak_array1 = np.ones(shape=(3, 4, 3, 2)) * 127
        peak_array2 = np.zeros(shape=(3, 4, 3, 2))
        peak_array = np.concatenate((peak_array0, peak_array1, peak_array2), axis=2)
        peak_array_filtered = ct._filter_4D_peak_array(
            peak_array, signal_axes=s.axes_manager.signal_axes
        )
        for ix, iy in np.ndindex(peak_array_filtered.shape[:2]):
            peak_list = peak_array_filtered[ix, iy]
            for x, y in peak_list:
                assert x != 0
                assert x != 127
                assert y != 0
                assert y != 127

    def test_1d_nav(self):
        peak_array = randint(124, 132, size=(4, 10, 2))
        peak_array_filtered = ct._filter_4D_peak_array(peak_array)
        assert peak_array_filtered.shape == (4,)

    def test_3d_nav(self):
        peak_array = randint(124, 132, size=(2, 3, 4, 10, 2))
        peak_array_filtered = ct._filter_4D_peak_array(peak_array)
        assert peak_array_filtered.shape == (2, 3, 4)


class TestGetClusterDict:
    def test_simple(self):
        peak_array = randint(100, size=(100, 2))
        ct._get_cluster_dict(peak_array)

    def test_eps(self):
        peak_array0 = randint(6, size=(100, 2)) + 80
        peak_array1 = randint(6, size=(100, 2))
        peak_array = np.vstack((peak_array0, peak_array1))
        cluster_dict0 = ct._get_cluster_dict(peak_array)
        assert len(cluster_dict0) == 2
        assert len(cluster_dict0[0]) == 100
        assert len(cluster_dict0[1]) == 100
        cluster_dict1 = ct._get_cluster_dict(peak_array, eps=200)
        assert len(cluster_dict1) == 1
        assert len(cluster_dict1[0]) == 200

    def test_min_samples(self):
        peak_array0 = randint(6, size=(100, 2))
        peak_array = np.vstack((peak_array0, [[54, 21], [53, 20], [55, 22]]))
        cluster_dict0 = ct._get_cluster_dict(peak_array, min_samples=2)
        labels0 = sorted(list(cluster_dict0.keys()))
        assert labels0 == [0, 1]
        cluster_dict1 = ct._get_cluster_dict(peak_array, min_samples=4)
        labels1 = sorted(list(cluster_dict1.keys()))
        assert labels1 == [-1, 0]

    def test_three_clusters(self):
        peak_array0 = randint(6, size=(100, 2)) + 80
        peak_array1 = randint(6, size=(100, 2))
        peak_array = np.vstack(
            (
                peak_array0,
                peak_array1,
                [
                    [54, 21],
                ],
            )
        )
        cluster_dict = ct._get_cluster_dict(peak_array, min_samples=2)
        labels = sorted(list(cluster_dict.keys()))
        assert labels == [-1, 0, 1]


class TestSortClusterDict:
    def test_simple(self):
        n_centre, n_rest = 10, 20
        cluster_dict = {}
        cluster_dict[-1] = [
            [5, 100],
        ]
        cluster_dict[0] = randint(5, size=(n_centre, 2)).tolist()
        cluster_dict[1] = randint(100, 105, size=(n_rest, 2)).tolist()
        sorted_cluster_dict0 = ct._sort_cluster_dict(
            cluster_dict, centre_x=2, centre_y=2
        )
        assert len(sorted_cluster_dict0["centre"]) == n_centre
        assert len(sorted_cluster_dict0["rest"]) == n_rest
        assert len(sorted_cluster_dict0["none"]) == 1

        sorted_cluster_dict1 = ct._sort_cluster_dict(
            cluster_dict, centre_x=102, centre_y=102
        )
        assert len(sorted_cluster_dict1["centre"]) == n_rest
        assert len(sorted_cluster_dict1["rest"]) == n_centre
        assert len(sorted_cluster_dict1["none"]) == 1


class TestClusterAndSortPeakArray:
    def test_simple(self):
        peak_array0 = randint(124, 132, size=(3, 4, 10, 2))
        peak_array1 = randint(24, 32, size=(3, 4, 5, 2))
        peak_array2 = randint(201, 203, size=(3, 4, 1, 2))
        peak_array = np.concatenate((peak_array0, peak_array1, peak_array2), axis=2)
        peak_dict = ct._cluster_and_sort_peak_array(peak_array)
        assert len(peak_dict["centre"][0, 0]) == 10
        assert len(peak_dict["rest"][0, 0]) == 5
        assert len(peak_dict["none"][0, 0]) == 1

    def test_only_centre(self):
        peak_array = randint(124, 132, size=(3, 4, 10, 2))
        peak_dict = ct._cluster_and_sort_peak_array(peak_array)
        assert len(peak_dict["centre"][0, 0]) == 10
        assert len(peak_dict["rest"][0, 0]) == 0
        assert len(peak_dict["none"][0, 0]) == 0

    def test_only_none(self):
        peak_array = randint(201, 203, size=(3, 4, 1, 2))
        peak_dict = ct._cluster_and_sort_peak_array(peak_array)
        assert len(peak_dict["centre"][0, 0]) == 0
        assert len(peak_dict["rest"][0, 0]) == 0
        assert len(peak_dict["none"][0, 0]) == 1

    def test_eps(self):
        peak_array0 = randint(124, 132, size=(3, 4, 10, 2))
        peak_array1 = randint(24, 32, size=(3, 4, 5, 2))
        peak_array = np.concatenate((peak_array0, peak_array1), axis=2)
        peak_dict0 = ct._cluster_and_sort_peak_array(peak_array, eps=30)
        peak_dict1 = ct._cluster_and_sort_peak_array(peak_array, eps=300)
        assert len(peak_dict0["centre"][0, 0]) == 10
        assert len(peak_dict1["centre"][0, 0]) == 15

    def test_min_samples(self):
        peak_array0 = randint(124, 132, size=(3, 4, 10, 2))
        peak_array1 = randint(204, 208, size=(3, 4, 3, 2))
        peak_array = np.concatenate((peak_array0, peak_array1), axis=2)
        peak_dict0 = ct._cluster_and_sort_peak_array(peak_array, min_samples=4)
        peak_dict1 = ct._cluster_and_sort_peak_array(peak_array, min_samples=2)
        assert len(peak_dict0["none"][0, 0]) == 3
        assert len(peak_dict1["none"][0, 0]) == 0
        assert len(peak_dict1["rest"][0, 0]) == 3

    def test_different_centre(self):
        peak_array0 = randint(124, 132, size=(3, 4, 10, 2))
        peak_array1 = randint(24, 32, size=(3, 4, 5, 2))
        peak_array = np.concatenate((peak_array0, peak_array1), axis=2)
        peak_dict0 = ct._cluster_and_sort_peak_array(
            peak_array, centre_x=128, centre_y=128
        )
        peak_dict1 = ct._cluster_and_sort_peak_array(
            peak_array, centre_x=28, centre_y=28
        )
        assert len(peak_dict0["centre"][0, 0]) == 10
        assert len(peak_dict0["rest"][0, 0]) == 5
        assert len(peak_dict1["centre"][0, 0]) == 5
        assert len(peak_dict1["rest"][0, 0]) == 10

    @pytest.mark.parametrize("ndim", [0, 1, 2, 3, 4, 5, 6])
    def test_ndim(self, ndim):
        nav_shape = tuple(np.random.randint(2, 5, size=ndim))
        shape = nav_shape + (23, 2)
        peak_array = np.random.randint(99, size=shape)
        peak_dict = ct._cluster_and_sort_peak_array(peak_array)
        assert peak_dict["centre"].shape == nav_shape
        assert peak_dict["rest"].shape == nav_shape
        assert peak_dict["none"].shape == nav_shape


class TestAddPeakDictsToSignal:
    def test_simple(self):
        peak_dicts = {}
        peak_dicts["centre"] = randint(124, 132, size=(3, 4, 10, 2))
        peak_dicts["rest"] = randint(204, 212, size=(3, 4, 5, 2))
        peak_dicts["none"] = randint(10, 13, size=(3, 4, 2, 2))
        s = Diffraction2D(np.zeros((3, 4, 256, 256)))
        ct._add_peak_dicts_to_signal(s, peak_dicts)


class TestGetPeakArrayShape:
    @pytest.mark.parametrize("ndim", [0, 1, 2, 3, 4, 5, 6])
    def test_dtype_object(self, ndim):
        shape = tuple(np.random.randint(2, 5, size=ndim))
        peak_array = np.empty(shape, dtype=object)
        shape_output = ct._get_peak_array_shape(peak_array)
        assert shape == shape_output

    @pytest.mark.parametrize("ndim", [0, 1, 2, 3, 4, 5, 6])
    def test_not_dtype_object(self, ndim):
        nav_shape = tuple(np.random.randint(2, 5, size=ndim))
        shape = nav_shape + (23, 2)
        peak_array = np.random.randint(99, size=shape)
        shape_output = ct._get_peak_array_shape(peak_array)
        assert nav_shape == shape_output


class TestSortedClusterDictToMarkerList:
    def test_simple(self):
        sorted_cluster_dict = {}
        sorted_cluster_dict["centre"] = randint(10, size=(3, 4, 2, 2))
        sorted_cluster_dict["rest"] = randint(50, 60, size=(3, 4, 3, 2))
        sorted_cluster_dict["none"] = randint(90, size=(3, 4, 1, 2))
        marker_list = ct._sorted_cluster_dict_to_marker_list(sorted_cluster_dict)
        assert len(marker_list) == 2 + 3 + 1

    def test_size(self):
        marker_size = 30
        sorted_cluster_dict = {}
        sorted_cluster_dict["centre"] = randint(10, size=(3, 4, 2, 2))
        sorted_cluster_dict["rest"] = randint(50, 60, size=(3, 4, 3, 2))
        sorted_cluster_dict["none"] = randint(90, size=(3, 4, 1, 2))
        marker_list = ct._sorted_cluster_dict_to_marker_list(
            sorted_cluster_dict, size=marker_size
        )
        for marker in marker_list:
            assert marker.get_data_position("size") == marker_size

    def test_color(self):
        marker_color = "orange"
        sorted_cluster_dict = {}
        sorted_cluster_dict["centre"] = randint(10, size=(3, 4, 2, 2))
        sorted_cluster_dict["rest"] = randint(50, 60, size=(3, 4, 3, 2))
        sorted_cluster_dict["none"] = randint(90, size=(3, 4, 1, 2))
        sorted_cluster_dict["magic"] = randint(40, size=(3, 4, 1, 2))
        marker_list = ct._sorted_cluster_dict_to_marker_list(
            sorted_cluster_dict,
            color_rest=marker_color,
            color_centre=marker_color,
            color_none=marker_color,
        )
        for marker in marker_list[:-1]:
            assert marker.marker_properties["color"] == marker_color
        assert marker_list[-1].marker_properties["color"] == "cyan"
