# -*- coding: utf-8 -*-
# Copyright 2017-2019 The pyXem developers
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
from pyxem.signals.diffraction_vectors import DiffractionVectors

# DiffractionVectors correspond to a single list of vectors, a map of vectors
# all of equal length, and the ragged case. A fixture is defined for each of
# these cases and all methods tested for it.


@pytest.fixture(params=[
    np.array([[0.063776, 0.011958],
              [-0.035874, 0.131538],
              [0.035874, -0.131538],
              [0.035874, 0.143496],
              [-0.035874, -0.13951],
              [-0.115594, 0.123566],
              [0.103636, -0.11958],
              [0.123566, 0.151468]])
])
def diffraction_vectors_single(request):
    dvs = DiffractionVectors(request.param)
    dvs.axes_manager.set_signal_dimension(1)
    return dvs


@pytest.fixture(params=[
    np.array([[np.array([[0.089685, 0.292971],
                         [0.017937, 0.277027],
                         [-0.069755, 0.257097],
                         [-0.165419, 0.241153],
                         [0.049825, 0.149475],
                         [-0.037867, 0.129545],
                         [-0.117587, 0.113601]]),
               np.array([[0.089685, 0.292971],
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
                         [-0.069755, -0.009965]])],
              [np.array([[0.089685, 0.292971],
                         [0.017937, 0.277027],
                         [-0.069755, 0.257097],
                         [-0.165419, 0.241153],
                         [0.049825, 0.149475],
                         [-0.037867, 0.129545],
                         [-0.117587, 0.113601],
                         [0.149475, 0.065769],
                         [0.229195, 0.045839],
                         [0.141503, 0.025909],
                         [0.073741, 0.013951]]),
               np.array([[0.001993, 0.001993]])]], dtype=object)
])
def diffraction_vectors_map(request):
    dvm = DiffractionVectors(request.param)
    dvm.axes_manager.set_signal_dimension(0)
    return dvm


def test_plot_diffraction_vectors(diffraction_vectors_map):
    diffraction_vectors_map.plot_diffraction_vectors(xlim=1., ylim=1.,
                                                     distance_threshold=0)


def test_plot_diffraction_vectors_on_signal(diffraction_vectors_map,
                                            diffraction_pattern):
    diffraction_vectors_map.plot_diffraction_vectors_on_signal(diffraction_pattern)


def test_get_cartesian_coordinates(diffraction_vectors_map):
    accelerating_volage = 200
    camera_length = 0.2
    diffraction_vectors_map.calculate_cartesian_coordinates(accelerating_volage,
                                                            camera_length)
    # Coordinate conversion is tested in vector_utils. Just test that the
    # result is stored correctly
    assert diffraction_vectors_map.cartesian is not None


class TestMagnitudes:

    def test_get_magnitudes_single(self, diffraction_vectors_single):
        diffraction_vectors_single.get_magnitudes()

    @pytest.mark.filterwarnings('ignore::FutureWarning')  # deemed "safe enough"
    def test_get_magnitude_histogram_single(self, diffraction_vectors_single):
        diffraction_vectors_single.get_magnitude_histogram(bins=np.arange(0, 0.5, 0.1))

    def test_get_magnitudes_map(self, diffraction_vectors_map):
        diffraction_vectors_map.get_magnitudes()

    @pytest.mark.filterwarnings('ignore::FutureWarning')  # deemed "safe enough"
    def test_get_magnitude_histogram_map(self, diffraction_vectors_map):
        diffraction_vectors_map.get_magnitude_histogram(bins=np.arange(0, 0.5, 0.1))


class TestUniqueVectors:

    def test_get_unique_vectors_map_type(self, diffraction_vectors_map):
        unique_vectors = diffraction_vectors_map.get_unique_vectors()
        assert isinstance(unique_vectors, DiffractionVectors)

    @pytest.mark.xfail(raises=ValueError)
    def test_get_unique_vectors_single(self, diffraction_vectors_single):
        diffraction_vectors_single.get_unique_vectors()

    @pytest.mark.parametrize('distance_threshold, answer', [
        (0.01, np.array([[0.089685, 0.292971],
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
                         [-0.069755, -0.009965]])),
        (0.1, np.array([[0.089685, 0.292971]])),
    ])
    def test_get_unique_vectors_map_values(self, diffraction_vectors_map,
                                           distance_threshold, answer):
        unique_vectors = diffraction_vectors_map.get_unique_vectors(
            distance_threshold=distance_threshold)
        np.testing.assert_almost_equal(unique_vectors.data, answer)


class TestDiffractingPixelMaps:

    def test_get_dpm_map(self, diffraction_vectors_map):
        diffraction_vectors_map.get_diffracting_pixels_map()

    def test_get_dpm_map_binary(self, diffraction_vectors_map):
        diffraction_vectors_map.get_diffracting_pixels_map(binary=True)
