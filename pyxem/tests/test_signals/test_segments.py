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

from pyxem.generators.vdf_generator import VDFGenerator
from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D
from pyxem.signals.diffraction_vectors import DiffractionVectors
from pyxem.signals.segments import LearningSegment, VDFSegment


@pytest.fixture
def signal_data():
    s = ElectronDiffraction2D(np.zeros((4, 5, 6, 6)))
    s.inav[:2, :2].data[..., 0, 0] = 2
    s.inav[:2, :2].data[..., 0, 3] = 2
    s.inav[:2, :2].data[..., 3, 5] = 2

    s.inav[2:, :3].data[..., 3, 3] = 2
    s.inav[2:, :3].data[..., 3, 0] = 2

    s.inav[2, :2].data[..., 0, 0] = 1
    s.inav[2, :2].data[..., 0, 3] = 1
    s.inav[2, :2].data[..., 3, 5] = 1
    s.inav[2, :2].data[..., 3, 3] = 1
    s.inav[2, :2].data[..., 3, 0] = 1

    s.inav[:2, 2:].data[..., 5, 5] = 3
    s.inav[:2, 2:].data[..., 5, 0] = 3
    s.inav[:2, 2:].data[..., 5, 3] = 3

    s.inav[3:, 2:].data[..., 5, 5] = 3
    s.inav[3:, 2:].data[..., 5, 0] = 3
    return s


@pytest.fixture
def signal_decomposition(signal_data):
    signal_data.decomposition()
    signal_data.decomposition(algorithm='nmf', output_dimension=5)
    s_nmf = signal_data.get_decomposition_model(components=5)
    factors = s_nmf.get_decomposition_factors()
    loadings = s_nmf.get_decomposition_loadings()
    return factors, loadings


@pytest.fixture
def signal_learning_segment(signal_decomposition):
    return LearningSegment(signal_decomposition[0], signal_decomposition[1])


class TestLearningSegment:
    @pytest.mark.parametrize('corr_th_factors, corr_th_loadings',
                             [(-0.1, 0.6),
                              (0.5, 0.5)])
    def test_correlate_learning_segments(
            self, signal_learning_segment: LearningSegment, corr_th_factors,
            corr_th_loadings):
        learn_corr = signal_learning_segment.correlate_learning_segments(
            corr_th_factors, corr_th_loadings)
        assert isinstance(learn_corr.loadings, Signal2D)
        assert isinstance(learn_corr.factors, Signal2D)

    @pytest.mark.parametrize('min_intensity_threshold, min_distance,'
                             'min_size, max_size, max_number_of_grains,'
                             'marker_radius, threshold, exclude_border',
                             [(0.5, 1, 1, 50, 10, 1, False, False),
                              (1, 2, 4, 30, 10, 1, False, True),
                              (0.1, 1, 1, 100, 100, 1, True, True)])
    def test_separate_learning_segments(
            self, signal_learning_segment: LearningSegment,
            min_intensity_threshold, min_distance, min_size, max_size,
            max_number_of_grains, marker_radius, threshold, exclude_border):
        learn_seg = signal_learning_segment.separate_learning_segments(
            min_intensity_threshold, min_distance, min_size, max_size,
            max_number_of_grains, marker_radius, threshold, exclude_border)
        assert isinstance(learn_seg.loadings, Signal2D)
        assert isinstance(learn_seg.factors, Signal2D)


@pytest.fixture(params=[
    np.array([np.array([0, 0]), np.array([3, 0]), np.array([3, 5]),
              np.array([0, 5]), np.array([0, 3]), np.array([3, 3]),
              np.array([5, 3]), np.array([5, 5])])
])
def unique_vectors(request):
    uv = DiffractionVectors(request.param)
    uv.axes_manager.set_signal_dimension(0)
    return uv


@pytest.fixture
def vdf_segments(signal_data, unique_vectors):
    vdfgen = VDFGenerator(signal_data, unique_vectors)
    vdfs = vdfgen.get_vector_vdf_images(radius=1)
    return vdfs.get_vdf_segments()


class TestVDFSegment:

    @pytest.mark.parametrize('corr_threshold, vector_threshold,'
                             'segment_threshold',
                             [(0.1, 1, 1),
                              (0.9, 3, 2)])
    def test_correlate_segments(self, vdf_segments: VDFSegment,
                                corr_threshold, vector_threshold,
                                segment_threshold):
        corrsegs = vdf_segments.correlate_vdf_segments(
            corr_threshold, vector_threshold, segment_threshold)
        assert isinstance(corrsegs.segments, Signal2D)
        assert isinstance(corrsegs.vectors_of_segments, DiffractionVectors)
        assert isinstance(corrsegs.intensities, np.ndarray)

    @pytest.mark.xfail
    def test_corelate_segments_bad_thresholds(self, vdf_segments: VDFSegment):
        corrsegs = vdf_segments.correlate_vdf_segments(vector_threshold=4,segment_threshold=5)

    def test_get_virtual_electron_diffraction(self, vdf_segments: VDFSegment,
                                              signal_data):
        corrsegs = vdf_segments.correlate_vdf_segments(0.1, 1, 1)
        vs = corrsegs.get_virtual_electron_diffraction(
            calibration=1, sigma=1,
            shape=signal_data.axes_manager.signal_shape)
        assert isinstance(vs, ElectronDiffraction2D)
