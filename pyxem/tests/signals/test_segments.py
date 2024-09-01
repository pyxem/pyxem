# -*- coding: utf-8 -*-
# Copyright 2016-2024 The pyXem developers
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

import sys

from hyperspy.signals import Signal2D
import numpy as np
import pytest

from pyxem.signals import (
    ElectronDiffraction2D,
    DiffractionVectors,
    LearningSegment,
    VDFSegment,
    DiffractionVectors2D,
    Diffraction2D,
)


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
    signal_data.decomposition(algorithm="NMF", output_dimension=5, init="nndsvd")
    s_nmf = signal_data.get_decomposition_model(components=5)
    factors = s_nmf.get_decomposition_factors()
    loadings = s_nmf.get_decomposition_loadings()
    return factors, loadings


@pytest.fixture
def learning_segment(signal_decomposition):
    return LearningSegment(signal_decomposition[0], signal_decomposition[1])


class TestLearningSegment:
    def test_learning_ncc_matrix(self, learning_segment):
        ncc = learning_segment.get_ncc_matrix()
        # fmt: off
        ans = np.array(
            [
                [
                    [ 1.0,   -0.264, -0.506,  0.612, 0.0],
                    [-0.264,  1.0,   -0.401, -0.431, 0.0],
                    [-0.506, -0.401,  1.0,   -0.310, 0.0],
                    [ 0.612, -0.431, -0.310,  1.0,   0.0],
                    [ 0.0,    0.0,    0.0,    0.0,   1.0],
                ],
                [
                    [ 1.0,   -0.059, -0.073, -0.041, 0.0],
                    [-0.059,  1.0,   -0.073, -0.041, 0.0],
                    [-0.073, -0.073,  1.0,   -0.051, 0.0],
                    [-0.040, -0.041, -0.051,  1.0,   0.0],
                    [ 0.0,    0.0,    0.0,    0.0,   1.0],
                ],
            ]
        )
        # fmt: on
        assert np.allclose(ncc.data, ans, atol=1e-3)

    @pytest.mark.parametrize(
        "corr_th_factors, corr_th_loadings", [(-0.1, 0.6), (0.5, 0.5)]
    )
    def test_correlate_learning_segments(
        self, learning_segment: LearningSegment, corr_th_factors, corr_th_loadings
    ):
        learn_corr = learning_segment.correlate_learning_segments(
            corr_th_factors, corr_th_loadings
        )
        assert isinstance(learn_corr.loadings, Signal2D)
        assert isinstance(learn_corr.factors, Signal2D)

    @pytest.mark.parametrize(
        "min_intensity_threshold, min_distance,"
        "min_size, max_size, max_number_of_grains,"
        "marker_radius, threshold, exclude_border",
        [
            (0.5, 1, 1, 50, 10, 1, False, False),
            (1, 2, 4, 30, 10, 1, False, True),
            (0.1, 1, 1, 100, 100, 1, True, True),
        ],
    )
    def test_separate_learning_segments(
        self,
        learning_segment: LearningSegment,
        min_intensity_threshold,
        min_distance,
        min_size,
        max_size,
        max_number_of_grains,
        marker_radius,
        threshold,
        exclude_border,
    ):
        learn_seg = learning_segment.separate_learning_segments(
            min_intensity_threshold,
            min_distance,
            min_size,
            max_size,
            max_number_of_grains,
            marker_radius,
            threshold,
            exclude_border,
        )
        assert isinstance(learn_seg.loadings, Signal2D)
        assert isinstance(learn_seg.factors, Signal2D)


@pytest.fixture(
    params=[
        np.array(
            [
                np.array([0, 0]),
                np.array([3, 0]),
                np.array([3, 5]),
                np.array([0, 5]),
                np.array([0, 3]),
                np.array([3, 3]),
                np.array([5, 3]),
                np.array([5, 5]),
            ]
        )
    ]
)
def unique_vectors(request):
    uv = DiffractionVectors2D(request.param)
    return uv


@pytest.fixture
def vdf_segments(signal_data, unique_vectors):
    vdfgen = VirtualDarkFieldGenerator(signal_data, unique_vectors)
    vdfs = vdfgen.get_virtual_dark_field_images(radius=1)
    return vdfs.get_vdf_segments()


@pytest.fixture
def vdf_segments_cropped(vdf_segments):
    return VDFSegment(
        vdf_segments.segments.inav[:4], vdf_segments.vectors_of_segments.inav[:4]
    )
