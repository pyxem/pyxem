import numpy as np
from numpy.random import randint
from pixstem.pixelated_stem_class import PixelatedSTEM
import pixstem.cluster_tools as ct


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
        peak_array = np.vstack((peak_array0, peak_array1, [[54, 21], ]))
        cluster_dict = ct._get_cluster_dict(peak_array, min_samples=2)
        labels = sorted(list(cluster_dict.keys()))
        assert labels == [-1, 0, 1]


class TestSortClusterDict:

    def test_simple(self):
        n_centre, n_rest = 10, 20
        cluster_dict = {}
        cluster_dict[-1] = [[5, 100], ]
        cluster_dict[0] = randint(5, size=(n_centre, 2)).tolist()
        cluster_dict[1] = randint(100, 105, size=(n_rest, 2)).tolist()
        sorted_cluster_dict0 = ct._sort_cluster_dict(
                cluster_dict, centre_x=2, centre_y=2)
        assert len(sorted_cluster_dict0['centre']) == n_centre
        assert len(sorted_cluster_dict0['rest']) == n_rest
        assert len(sorted_cluster_dict0['none']) == 1

        sorted_cluster_dict1 = ct._sort_cluster_dict(
                cluster_dict, centre_x=102, centre_y=102)
        assert len(sorted_cluster_dict1['centre']) == n_rest
        assert len(sorted_cluster_dict1['rest']) == n_centre
        assert len(sorted_cluster_dict1['none']) == 1


class TestClusterAndSortPeakArray:

    def test_simple(self):
        peak_array0 = randint(124, 132, size=(3, 4, 10, 2))
        peak_array1 = randint(24, 32, size=(3, 4, 5, 2))
        peak_array2 = randint(201, 203, size=(3, 4, 1, 2))
        peak_array = np.concatenate(
                (peak_array0, peak_array1, peak_array2), axis=2)
        peak_dict = ct._cluster_and_sort_peak_array(peak_array)
        assert len(peak_dict['centre'][0, 0]) == 10
        assert len(peak_dict['rest'][0, 0]) == 5
        assert len(peak_dict['none'][0, 0]) == 1

    def test_eps(self):
        peak_array0 = randint(124, 132, size=(3, 4, 10, 2))
        peak_array1 = randint(24, 32, size=(3, 4, 5, 2))
        peak_array = np.concatenate((peak_array0, peak_array1), axis=2)
        peak_dict0 = ct._cluster_and_sort_peak_array(peak_array, eps=30)
        peak_dict1 = ct._cluster_and_sort_peak_array(peak_array, eps=300)
        assert len(peak_dict0['centre'][0, 0]) == 10
        assert len(peak_dict1['centre'][0, 0]) == 15

    def test_min_samples(self):
        peak_array0 = randint(124, 132, size=(3, 4, 10, 2))
        peak_array1 = randint(204, 208, size=(3, 4, 3, 2))
        peak_array = np.concatenate((peak_array0, peak_array1), axis=2)
        peak_dict0 = ct._cluster_and_sort_peak_array(peak_array, min_samples=4)
        peak_dict1 = ct._cluster_and_sort_peak_array(peak_array, min_samples=2)
        assert len(peak_dict0['none'][0, 0]) == 3
        assert len(peak_dict1['none'][0, 0]) == 0
        assert len(peak_dict1['rest'][0, 0]) == 3

    def test_different_centre(self):
        peak_array0 = randint(124, 132, size=(3, 4, 10, 2))
        peak_array1 = randint(24, 32, size=(3, 4, 5, 2))
        peak_array = np.concatenate((peak_array0, peak_array1), axis=2)
        peak_dict0 = ct._cluster_and_sort_peak_array(
                peak_array, centre_x=128, centre_y=128)
        peak_dict1 = ct._cluster_and_sort_peak_array(
                peak_array, centre_x=28, centre_y=28)
        assert len(peak_dict0['centre'][0, 0]) == 10
        assert len(peak_dict0['rest'][0, 0]) == 5
        assert len(peak_dict1['centre'][0, 0]) == 5
        assert len(peak_dict1['rest'][0, 0]) == 10


class TestAddPeakDictsToSignal:

    def test_simple(self):
        peak_dicts = {}
        peak_dicts['centre'] = randint(124, 132, size=(3, 4, 10, 2))
        peak_dicts['rest'] = randint(204, 212, size=(3, 4, 5, 2))
        peak_dicts['none'] = randint(10, 13, size=(3, 4, 2, 2))
        s = PixelatedSTEM(np.zeros((3, 4, 256, 256)))
        ct._add_peak_dicts_to_signal(s, peak_dicts)


class TestSortedClusterDictToMarkerList:

    def test_simple(self):
        sorted_cluster_dict = {}
        sorted_cluster_dict['centre'] = randint(10, size=(3, 4, 2, 2))
        sorted_cluster_dict['rest'] = randint(50, 60, size=(3, 4, 3, 2))
        sorted_cluster_dict['none'] = randint(90, size=(3, 4, 1, 2))
        marker_list = ct._sorted_cluster_dict_to_marker_list(
            sorted_cluster_dict)
        assert len(marker_list) == 2 + 3 + 1

    def test_size(self):
        marker_size = 30
        sorted_cluster_dict = {}
        sorted_cluster_dict['centre'] = randint(10, size=(3, 4, 2, 2))
        sorted_cluster_dict['rest'] = randint(50, 60, size=(3, 4, 3, 2))
        sorted_cluster_dict['none'] = randint(90, size=(3, 4, 1, 2))
        marker_list = ct._sorted_cluster_dict_to_marker_list(
            sorted_cluster_dict, size=marker_size)
        for marker in marker_list:
            assert marker.get_data_position('size') == marker_size

    def test_color(self):
        marker_color = 'orange'
        sorted_cluster_dict = {}
        sorted_cluster_dict['centre'] = randint(10, size=(3, 4, 2, 2))
        sorted_cluster_dict['rest'] = randint(50, 60, size=(3, 4, 3, 2))
        sorted_cluster_dict['none'] = randint(90, size=(3, 4, 1, 2))
        marker_list = ct._sorted_cluster_dict_to_marker_list(
            sorted_cluster_dict, color_rest=marker_color,
            color_centre=marker_color, color_none=marker_color)
        for marker in marker_list:
            assert marker.marker_properties['color'] == marker_color
