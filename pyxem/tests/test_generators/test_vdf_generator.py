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

from hyperspy.signals import Signal2D

from pyxem.generators.vdf_generator import (VDFGenerator,
                                            VDFSegmentGenerator)

from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D
from pyxem.signals.diffraction_vectors import DiffractionVectors
from pyxem.signals.vdf_image import VDFImage, VDFSegment


@pytest.fixture(params=[
    np.array([[1, 1],
              [2, 2]])
])
def diffraction_vectors(request):
    dvec = DiffractionVectors(request.param)
    dvec.axes_manager.set_signal_dimension(1)
    return dvec


@pytest.fixture
def vdf_generator(diffraction_pattern, diffraction_vectors):
    return VDFGenerator(diffraction_pattern, diffraction_vectors)


class TestVDFGenerator:

    def test_vdf_generator_init_with_vectors(self, diffraction_pattern):
        dvm = DiffractionVectors(np.array([[np.array([[1, 1],
                                                      [2, 2]]),
                                            np.array([[1, 1],
                                                      [2, 2],
                                                      [1, 2]])],
                                           [np.array([[1, 1],
                                                      [2, 2]]),
                                            np.array([[1, 1],
                                                      [2, 2]])]], dtype=object))
        dvm.axes_manager.set_signal_dimension(0)

        vdfgen = VDFGenerator(diffraction_pattern, dvm)
        assert isinstance(vdfgen.signal, ElectronDiffraction2D)
        assert isinstance(vdfgen.vectors, DiffractionVectors)

    def test_vdf_generator_init_without_vectors(self, diffraction_pattern):

        vdfgen = VDFGenerator(diffraction_pattern)
        assert isinstance(vdfgen.signal, ElectronDiffraction2D)
        assert isinstance(vdfgen.vectors, type(None))

    @pytest.mark.xfail(raises=ValueError)
    def test_vector_vdfs_without_vectors(self, diffraction_pattern):
        vdfgen = VDFGenerator(diffraction_pattern)
        vdfgen.get_vector_vdf_images(radius=2.)

    @pytest.mark.parametrize('radius, normalize', [
        (4., False),
        (4., True)
    ])
    def test_get_vector_vdf_images(
            self,
            vdf_generator: VDFGenerator,
            radius, normalize
    ):
        vdfs = vdf_generator.get_vector_vdf_images(radius, normalize)
        assert isinstance(vdfs, VDFImage)

    @pytest.mark.parametrize('k_min, k_max, k_steps, normalize', [
        (0., 4., 2, False),
        (0., 4., 2, True)
    ])
    def test_get_concentric_vdf_images(
            self,
            vdf_generator: VDFGenerator,
            k_min, k_max, k_steps, normalize
    ):
        vdfs = vdf_generator.get_concentric_vdf_images(k_min, k_max, k_steps,
                                                       normalize)
        assert isinstance(vdfs, VDFImage)


def test_vdf_generator_from_map(diffraction_pattern):
    dvm = DiffractionVectors(np.array([[np.array([[1, 1],
                                                  [2, 2]]),
                                        np.array([[1, 1],
                                                  [2, 2],
                                                  [1, 2]])],
                                       [np.array([[1, 1],
                                                  [2, 2]]),
                                        np.array([[1, 1],
                                                  [2, 2]])]], dtype=object))
    dvm.axes_manager.set_signal_dimension(0)

    vdfgen = VDFGenerator(diffraction_pattern, dvm)
    assert isinstance(vdfgen, VDFGenerator)


@pytest.fixture(params=[
    np.array([np.array([0, 0]), np.array([3, 3]), np.array([5, 5])])
])
def unique_vectors(request):
    uv = DiffractionVectors(request.param)
    uv.axes_manager.set_signal_dimension(0)
    return uv


@pytest.fixture
def signal_data():
    diff_sim_data = np.zeros((4, 5, 6, 6))
    s = ElectronDiffraction2D(diff_sim_data)
    s.inav[:2, :2].data[..., 0, 0] = 1
    s.inav[2, :2].data[..., 0, 0] = 1
    s.inav[2:, :3].data[..., 3, 3] = 1
    s.inav[1:, 2:].data[..., 5, 5] = 1
    return s


@pytest.fixture
def vdf_generator_seg(signal_data, unique_vectors):
    return VDFGenerator(signal_data, unique_vectors)


@pytest.fixture
def vdf_vector_images_seg(vdf_generator_seg):
    return vdf_generator_seg.get_vector_vdf_images(radius=1)


class TestVDFSegmentGenerator:

    def test_vdf_segment_generator(self, vdf_vector_images_seg: VDFImage):
        vdfsegmentgen = VDFSegmentGenerator(vdf_vector_images_seg)
        assert isinstance(vdfsegmentgen.vdf_images, VDFImage)
        assert isinstance(vdfsegmentgen.vectors, DiffractionVectors)

    @pytest.mark.parametrize('min_distance, min_size, max_size,'
                             'max_number_of_grains, marker_radius,'
                             'threshold, exclude_border',
                             [(1, 1, 20, 5, 1, False, 0),
                              (2, 3, 200, 10, 2, True, 1)])
    def test_get_vdf_segments(
            self, vdf_vector_images_seg,
            min_distance, min_size, max_size, max_number_of_grains,
            marker_radius, threshold, exclude_border):
        vdf_segment_generator = VDFSegmentGenerator(vdf_vector_images_seg)
        segs = vdf_segment_generator.get_vdf_segments(
            min_distance, min_size, max_size, max_number_of_grains,
            marker_radius, threshold, exclude_border)
        assert isinstance(segs, VDFSegment)
        assert isinstance(segs.segments, Signal2D)
        assert isinstance(segs.vectors_of_segments, DiffractionVectors)
