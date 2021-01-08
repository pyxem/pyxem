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
from sklearn.cluster import DBSCAN

from hyperspy.signals import Signal2D

from pyxem.signals import DiffractionVectors

# DiffractionVectors correspond to a single list of vectors, a map of vectors
# all of equal length, and the ragged case. A fixture is defined for each of
# these cases and all methods tested for it.


@pytest.fixture(
    params=[
        np.array(
            [
                [0.063776, 0.011958],
                [-0.035874, 0.131538],
                [0.035874, -0.131538],
                [0.035874, 0.143496],
                [-0.035874, -0.13951],
                [-0.115594, 0.123566],
                [0.103636, -0.11958],
                [0.123566, 0.151468],
            ]
        )
    ]
)
def diffraction_vectors_single(request):
    dvs = DiffractionVectors(request.param)
    dvs.axes_manager.set_signal_dimension(1)
    return dvs


@pytest.fixture(
    params=[
        np.array(
            [
                [
                    np.array(
                        [
                            [0.089685, 0.292971],
                            [0.017937, 0.277027],
                            [-0.069755, 0.257097],
                            [-0.165419, 0.241153],
                            [0.049825, 0.149475],
                            [-0.037867, 0.129545],
                            [-0.117587, 0.113601],
                        ]
                    ),
                    np.array(
                        [
                            [0.089685, 0.292971],
                            [0.017937, 0.277027],
                            [-0.069755, 0.257097],
                            [-0.165419, 0.241153],
                            [0.049825, 0.149475],
                            [-0.037867, 0.129545],
                            [-0.117587, 0.113601],
                            [0.149475, 0.065769],
                            [0.229195, 0.045839],
                            [0.141503, 0.025909],
                            [0.073741, 0.013951],
                            [0.001993, 0.001993],
                            [-0.069755, -0.009965],
                        ]
                    ),
                ],
                [
                    np.array(
                        [
                            [0.089685, 0.292971],
                            [0.017937, 0.277027],
                            [-0.069755, 0.257097],
                            [-0.165419, 0.241153],
                            [0.049825, 0.149475],
                            [-0.037867, 0.129545],
                            [-0.117587, 0.113601],
                            [0.149475, 0.065769],
                            [0.229195, 0.045839],
                            [0.141503, 0.025909],
                            [0.073741, 0.013951],
                        ]
                    ),
                    np.array([[0.001993, 0.001993]]),
                ],
            ],
            dtype=object,
        )
    ]
)
def diffraction_vectors_map(request):
    dvm = DiffractionVectors(request.param)
    dvm.axes_manager.set_signal_dimension(0)
    dvm.axes_manager[0].name = "x"
    dvm.axes_manager[1].name = "y"
    return dvm


def test_plot_diffraction_vectors(diffraction_vectors_map):
    with pytest.warns(UserWarning, match="distance_threshold=0 was given"):
        diffraction_vectors_map.plot_diffraction_vectors(
            xlim=1.0, ylim=1.0, distance_threshold=0
        )


def test_plot_diffraction_vectors_on_signal(
    diffraction_vectors_map, diffraction_pattern
):
    diffraction_vectors_map.plot_diffraction_vectors_on_signal(diffraction_pattern)


def test_get_cartesian_coordinates(diffraction_vectors_map):
    accelerating_voltage = 200
    camera_length = 0.2
    diffraction_vectors_map.calculate_cartesian_coordinates(
        accelerating_voltage, camera_length
    )
    # Coordinate conversion is tested in vector_utils. Just test that the
    # result is stored correctly
    assert diffraction_vectors_map.cartesian is not None
    assert (
        diffraction_vectors_map.axes_manager[0].name
        == diffraction_vectors_map.cartesian.axes_manager[0].name
    )


class TestMagnitudes:
    def test_get_magnitudes_single(self, diffraction_vectors_single):
        diffraction_vectors_single.get_magnitudes()

    @pytest.mark.filterwarnings("ignore::FutureWarning")  # deemed "safe enough"
    def test_get_magnitude_histogram_single(self, diffraction_vectors_single):
        diffraction_vectors_single.get_magnitude_histogram(bins=np.arange(0, 0.5, 0.1))

    def test_get_magnitudes_map(self, diffraction_vectors_map):
        diffraction_vectors_map.get_magnitudes()

    @pytest.mark.filterwarnings("ignore::FutureWarning")  # deemed "safe enough"
    def test_get_magnitude_histogram_map(self, diffraction_vectors_map):
        diffraction_vectors_map.get_magnitude_histogram(bins=np.arange(0, 0.5, 0.1))


class TestUniqueVectors:
    def test_get_unique_vectors_map_type(self, diffraction_vectors_map):
        unique_vectors = diffraction_vectors_map.get_unique_vectors()
        assert isinstance(unique_vectors, DiffractionVectors)

    def test_get_unique_vectors_single(self, diffraction_vectors_single):
        diffraction_vectors_single.get_unique_vectors()

    @pytest.mark.parametrize(
        "distance_threshold, answer",
        [
            (
                0.01,
                np.array(
                    [
                        [-0.165419, 0.241153],
                        [-0.117587, 0.113601],
                        [-0.069755, -0.009965],
                        [-0.069755, 0.257097],
                        [-0.037867, 0.129545],
                        [0.001993, 0.001993],
                        [0.017937, 0.277027],
                        [0.049825, 0.149475],
                        [0.073741, 0.013951],
                        [0.089685, 0.292971],
                        [0.141503, 0.025909],
                        [0.149475, 0.065769],
                        [0.229195, 0.045839],
                    ]
                ),
            ),
            (
                0.1,
                np.array(
                    [
                        [-0.117587, 0.249125],
                        [-0.077727, 0.121573],
                        [-0.021923, -0.001993],
                        [0.053811, 0.284999],
                        [0.049825, 0.149475],
                        [0.121573, 0.03520967],
                        [0.229195, 0.045839],
                    ]
                ),
            ),
        ],
    )
    def test_get_unique_vectors_map_values(
        self, diffraction_vectors_map, distance_threshold, answer
    ):
        unique_vectors = diffraction_vectors_map.get_unique_vectors(
            distance_threshold=distance_threshold
        )
        np.testing.assert_almost_equal(unique_vectors.data, answer)

    def test_get_unique_vectors_map_dbscan(self, diffraction_vectors_map):
        unique_dbscan = diffraction_vectors_map.get_unique_vectors(
            method="DBSCAN", return_clusters=True
        )
        assert isinstance(unique_dbscan[0], DiffractionVectors)
        assert isinstance(unique_dbscan[1], DBSCAN)

    @pytest.mark.parametrize(
        "distance_threshold, answer",
        [
            (
                0.01,
                np.array(
                    [
                        [-0.165419, 0.241153],
                        [-0.117587, 0.113601],
                        [-0.069755, -0.009965],
                        [-0.069755, 0.257097],
                        [-0.037867, 0.129545],
                        [0.001993, 0.001993],
                        [0.017937, 0.277027],
                        [0.049825, 0.149475],
                        [0.073741, 0.013951],
                        [0.089685, 0.292971],
                        [0.141503, 0.025909],
                        [0.149475, 0.065769],
                        [0.229195, 0.045839],
                    ]
                ),
            ),
            (
                0.1,
                np.array(
                    [
                        [-0.031888, 0.267062],
                        [-0.03520967, 0.13087367],
                        [0.10200536, 0.02699609],
                    ]
                ),
            ),
        ],
    )
    def test_get_unique_vectors_map_values_dbscan(
        self, diffraction_vectors_map, distance_threshold, answer
    ):
        unique_vectors = diffraction_vectors_map.get_unique_vectors(
            distance_threshold=distance_threshold, method="DBSCAN"
        )
        np.testing.assert_almost_equal(unique_vectors.data, answer)


class TestFilterVectors:
    def test_filter_magnitude_map_type(self, diffraction_vectors_map):
        filtered_vectors = diffraction_vectors_map.filter_magnitude(0.1, 1.0)
        assert isinstance(filtered_vectors, DiffractionVectors)

    def test_filter_magnitude_single_type(self, diffraction_vectors_single):
        filtered_vectors = diffraction_vectors_single.filter_magnitude(0.1, 1.0)
        assert isinstance(filtered_vectors, DiffractionVectors)

    def test_filter_magnitude_map(self, diffraction_vectors_map):
        filtered_vectors = diffraction_vectors_map.filter_magnitude(0.1, 1.0)
        ans = np.array(
            [
                [0.089685, 0.292971],
                [0.017937, 0.277027],
                [-0.069755, 0.257097],
                [-0.165419, 0.241153],
                [0.049825, 0.149475],
                [-0.037867, 0.129545],
                [-0.117587, 0.113601],
                [0.149475, 0.065769],
                [0.229195, 0.045839],
                [0.141503, 0.025909],
            ]
        )
        np.testing.assert_almost_equal(filtered_vectors.data[0][1], ans)

    def test_filter_magnitude_single(self, diffraction_vectors_single):
        filtered_vectors = diffraction_vectors_single.filter_magnitude(0.15, 1.0)
        ans = np.array(
            [[-0.115594, 0.123566], [0.103636, -0.11958], [0.123566, 0.151468]]
        )
        np.testing.assert_almost_equal(filtered_vectors.data, ans)

    def test_filter_detector_edge_map_type(self, diffraction_vectors_map):
        diffraction_vectors_map.detector_shape = (260, 240)
        diffraction_vectors_map.pixel_calibration = 0.001
        filtered_vectors = diffraction_vectors_map.filter_detector_edge(exclude_width=2)
        assert isinstance(filtered_vectors, DiffractionVectors)

    def test_filter_detector_edge_single_type(self, diffraction_vectors_single):
        diffraction_vectors_single.detector_shape = (260, 240)
        diffraction_vectors_single.pixel_calibration = 0.001
        filtered_vectors = diffraction_vectors_single.filter_detector_edge(
            exclude_width=10
        )
        assert isinstance(filtered_vectors, DiffractionVectors)

    def test_filter_detector_edge_map(self, diffraction_vectors_map):
        diffraction_vectors_map.detector_shape = (260, 240)
        diffraction_vectors_map.pixel_calibration = 0.001
        filtered_vectors = diffraction_vectors_map.filter_detector_edge(exclude_width=2)
        ans = np.array([[-0.117587, 0.113601]])
        np.testing.assert_almost_equal(filtered_vectors.data[0, 0], ans)

    def test_filter_detector_edge_single(self, diffraction_vectors_single):
        diffraction_vectors_single.detector_shape = (260, 240)
        diffraction_vectors_single.pixel_calibration = 0.001
        filtered_vectors = diffraction_vectors_single.filter_detector_edge(
            exclude_width=10
        )
        ans = np.array([[0.063776, 0.011958]])
        np.testing.assert_almost_equal(filtered_vectors.data, ans)


class TestDiffractingPixelsMap:
    def test_get_dpm_values(self, diffraction_vectors_map):
        answer = np.array([[7.0, 13.0], [11.0, 1.0]])
        xim = diffraction_vectors_map.get_diffracting_pixels_map()
        assert np.allclose(xim, answer)

    def test_get_dpm_type(self, diffraction_vectors_map):
        xim = diffraction_vectors_map.get_diffracting_pixels_map()
        assert isinstance(xim, Signal2D)

    def test_get_dpm_title(self, diffraction_vectors_map):
        xim = diffraction_vectors_map.get_diffracting_pixels_map()
        assert xim.metadata.General.title == "Diffracting Pixels Map"

    def test_get_dpm_in_range(self, diffraction_vectors_map):
        answer = np.array([[0.0, 3.0], [1.0, 1.0]])
        xim = diffraction_vectors_map.get_diffracting_pixels_map(in_range=(0, 0.1))
        assert np.allclose(xim, answer)

    def test_get_dpm_binary(self, diffraction_vectors_map):
        answer = np.array([[1.0, 1.0], [1.0, 1.0]])
        xim = diffraction_vectors_map.get_diffracting_pixels_map(binary=True)
        assert np.allclose(xim, answer)
