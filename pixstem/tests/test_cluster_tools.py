import numpy as np
from numpy.random import randint
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
