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

from pyxem.signals.detector_coordinates2d import DetectorCoordinates2D
from pyxem.detectors.generic_flat_detector import GenericFlatDetector

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
def detector_coordinates_single(request):
    dcs = DetectorCoordinates2D(request.param)
    dcs.axes_manager.set_signal_dimension(1)
    return dcs


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
def detector_coordinates_map(request):
    dcm = DetectorCoordinates2D(request.param)
    return dcm


def test_plot_on_signal(detector_coordinates_map,
                        diffraction_pattern):
    detector_coordinates_map.plot_on_signal(diffraction_pattern)


class TestDiffractingPixelMaps:

    def test_get_dpm_map(self, detector_coordinates_map):
        detector_coordinates_map.get_npeaks_map()

    def test_get_dpm_map_binary(self, detector_coordinates_map):
        detector_coordinates_map.get_npeaks_map(binary=True)


class TestAsDiffractionVectors2D:

    def test_dcs_to_2d(self, detector_coordinates_single):
        detector_coordinates_single.as_diffraction_vectors2d(center=(10, 10),
                                                             calibration=0.01)

    def test_dcm_to_2d(self, detector_coordinates_map):
        detector_coordinates_map.as_diffraction_vectors2d(center=(10, 10),
                                                          calibration=0.01)


class TestAsDiffractionVectors3D:

    def test_dcs_to_3d(self, detector_coordinates_single):
        origin = [3.5, 3.5]
        detector = GenericFlatDetector(8, 8)
        detector_coordinates_single.as_diffraction_vectors3d(origin,
                                                             detector=detector,
                                                             detector_distance=1,
                                                             wavelength=1)

    def test_dcm_to_3d(self, detector_coordinates_map):
        origin = [3.5, 3.5]
        detector = GenericFlatDetector(8, 8)
        detector_coordinates_map.as_diffraction_vectors3d(origin,
                                                          detector=detector,
                                                          detector_distance=1,
                                                          wavelength=1)
