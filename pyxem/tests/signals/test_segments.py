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

from hyperspy.signals import Signal2D

from pyxem.generators import VirtualDarkFieldGenerator
from pyxem.signals import (
    ElectronDiffraction2D,
    DiffractionVectors,
    LearningSegment,
    VDFSegment,
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
    signal_data.decomposition(algorithm="nmf", output_dimension=5)
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
        ans = np.array(
            [
                [
                    [1.0, -0.26413543, -0.50636968, 0.61237256, 0.0],
                    [-0.26413543, 1.0, -0.40125028, -0.43133109, 0.0],
                    [-0.50636968, -0.40125028, 1.0, -0.31008684, 0.0],
                    [0.61237256, -0.43133109, -0.31008684, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, -0.0588285, -0.07313363, -0.04087783, 0.0],
                    [-0.0588285, 1.0, -0.07312725, -0.04099601, 0.0],
                    [-0.07313363, -0.07312725, 1.0, -0.05096472, 0.0],
                    [-0.04087783, -0.04099601, -0.05096472, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                ],
            ]
        )
        np.testing.assert_almost_equal(ncc.data, ans)

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
    uv = DiffractionVectors(request.param)
    uv.axes_manager.set_signal_dimension(0)
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


class TestVDFSegment:
    def test_get_vdf_ncc_matrix(self, vdf_segments):
        ncc = vdf_segments.get_ncc_matrix()
        ans = np.array(
            [
                [
                    1.0,
                    -0.40125028,
                    -0.31008684,
                    -0.31008684,
                    1.0,
                    -0.40125028,
                    -0.31008684,
                    1.0,
                    -0.31008684,
                    -0.31008684,
                ],
                [
                    -0.40125028,
                    1.0,
                    -0.43133109,
                    0.10783277,
                    -0.40125028,
                    1.0,
                    -0.43133109,
                    -0.40125028,
                    -0.43133109,
                    0.10783277,
                ],
                [
                    -0.31008684,
                    -0.43133109,
                    1.0,
                    -0.25,
                    -0.31008684,
                    -0.43133109,
                    1.0,
                    -0.31008684,
                    1.0,
                    -0.25,
                ],
                [
                    -0.31008684,
                    0.10783277,
                    -0.25,
                    1.0,
                    -0.31008684,
                    0.10783277,
                    -0.25,
                    -0.31008684,
                    -0.25,
                    1.0,
                ],
                [
                    1.0,
                    -0.40125028,
                    -0.31008684,
                    -0.31008684,
                    1.0,
                    -0.40125028,
                    -0.31008684,
                    1.0,
                    -0.31008684,
                    -0.31008684,
                ],
                [
                    -0.40125028,
                    1.0,
                    -0.43133109,
                    0.10783277,
                    -0.40125028,
                    1.0,
                    -0.43133109,
                    -0.40125028,
                    -0.43133109,
                    0.10783277,
                ],
                [
                    -0.31008684,
                    -0.43133109,
                    1.0,
                    -0.25,
                    -0.31008684,
                    -0.43133109,
                    1.0,
                    -0.31008684,
                    1.0,
                    -0.25,
                ],
                [
                    1.0,
                    -0.40125028,
                    -0.31008684,
                    -0.31008684,
                    1.0,
                    -0.40125028,
                    -0.31008684,
                    1.0,
                    -0.31008684,
                    -0.31008684,
                ],
                [
                    -0.31008684,
                    -0.43133109,
                    1.0,
                    -0.25,
                    -0.31008684,
                    -0.43133109,
                    1.0,
                    -0.31008684,
                    1.0,
                    -0.25,
                ],
                [
                    -0.31008684,
                    0.10783277,
                    -0.25,
                    1.0,
                    -0.31008684,
                    0.10783277,
                    -0.25,
                    -0.31008684,
                    -0.25,
                    1.0,
                ],
            ]
        )
        np.testing.assert_almost_equal(ncc.data, ans)

    @pytest.mark.parametrize(
        "corr_threshold, vector_threshold," "segment_threshold",
        [(0.1, 1, 1), (0.9, 3, 2)],
    )
    def test_correlate_segments(
        self,
        vdf_segments: VDFSegment,
        corr_threshold,
        vector_threshold,
        segment_threshold,
    ):
        corrsegs = vdf_segments.correlate_vdf_segments(
            corr_threshold, vector_threshold, segment_threshold
        )
        assert isinstance(corrsegs.segments, Signal2D)
        assert isinstance(corrsegs.vectors_of_segments, DiffractionVectors)
        assert isinstance(corrsegs.intensities, np.ndarray)

    def test_correlate_segments_cropped(self, vdf_segments_cropped: VDFSegment):
        corrsegs = vdf_segments_cropped.correlate_vdf_segments(0.9, 1, 0)
        assert isinstance(corrsegs.segments, Signal2D)
        assert isinstance(corrsegs.vectors_of_segments, DiffractionVectors)
        assert isinstance(corrsegs.intensities, np.ndarray)

    def test_correlate_segments_small_vector_threshold(self, vdf_segments: VDFSegment):
        _ = vdf_segments.correlate_vdf_segments(
            corr_threshold=0.7, vector_threshold=0, segment_threshold=-1
        )

    def test_correlate_segments_bad_thresholds(self, vdf_segments: VDFSegment):
        with pytest.raises(
            ValueError,
            match="segment_threshold must be smaller than or equal to vector_threshold",
        ):
            _ = vdf_segments.correlate_vdf_segments(
                vector_threshold=4, segment_threshold=5
            )

    def test_get_virtual_electron_diffraction(
        self, vdf_segments: VDFSegment, signal_data
    ):
        corrsegs = vdf_segments.correlate_vdf_segments(0.1, 1, 1)
        vs = corrsegs.get_virtual_electron_diffraction(
            calibration=1, sigma=1, shape=signal_data.axes_manager.signal_shape
        )
        assert isinstance(vs, ElectronDiffraction2D)

    def test_get_virtual_electron_diffraction_no_intensities(
        self, vdf_segments: VDFSegment, signal_data
    ):
        vdf_segments.intensities = None
        with pytest.raises(
            ValueError,
            match="VDFSegment does not have the attribute intensities, required for this method",
        ):
            vdf_segments.get_virtual_electron_diffraction(
                calibration=1, sigma=1, shape=signal_data.axes_manager.signal_shape
            )

    def test_get_virtual_electron_diffraction_for_single_vectors(
        self, vdf_segments: VDFSegment, signal_data
    ):
        vs = vdf_segments.get_virtual_electron_diffraction(
            calibration=1, sigma=1, shape=signal_data.axes_manager.signal_shape
        )
        assert isinstance(vs, ElectronDiffraction2D)
