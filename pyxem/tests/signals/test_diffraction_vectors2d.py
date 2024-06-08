# -*- coding: utf-8 -*-
# Copyright 2016-2022 The pyXem developers
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

from hyperspy.signals import Signal2D, BaseSignal, Signal1D
import hyperspy.api as hs

from pyxem.signals import DiffractionVectors2D, DiffractionVectors1D


class TestDiffractionVectors2D:
    def setup_method(self):
        vectors = np.reshape(np.repeat(np.arange(0, 100), 2), (50, 4))
        self.vector = DiffractionVectors2D(vectors)

    def test_setup(self):
        assert isinstance(self.vector, DiffractionVectors2D)
        assert self.vector.ragged == False

    def test_magnitudes(self):
        magnitudes = self.vector.get_magnitudes()
        assert magnitudes.ragged == False
        mags = np.linalg.norm(self.vector.data[:, [0, 1]], axis=1)
        np.testing.assert_array_almost_equal(mags, magnitudes.data)
        assert len(magnitudes.axes_manager.signal_axes) == 1

    def test_filter_magnitudes(self):
        magnitudes = self.vector.filter_magnitude(min_magnitude=10, max_magnitude=40)
        mags = np.linalg.norm(self.vector.data[:, [0, 1]], axis=1)
        num_in_range = np.sum((mags > 10) * (mags < 40))
        assert num_in_range == magnitudes.data.shape[0]


class TestSingleDiffractionVectors2D:
    def setup_method(self):
        self.data = np.array(
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
        self.vector = DiffractionVectors2D(self.data)

    def test_filter_magnitude_single(self):
        filtered_vectors = self.vector.filter_magnitude(0.15, 1.0)
        ans = np.array(
            [[-0.115594, 0.123566], [0.103636, -0.11958], [0.123566, 0.151468]]
        )
        np.testing.assert_almost_equal(filtered_vectors.data, ans)

    def test_filter_detector_edge_single(self):
        self.vector.detector_shape = (260, 240)
        self.vector.column_scale = [0.001, 0.001]
        self.vector.column_offsets = [130 * 0.001, 120 * 0.001]
        filtered_vectors = self.vector.filter_detector_edge(exclude_width=10)
        ans = np.array([[0.063776, 0.011958]])
        np.testing.assert_almost_equal(filtered_vectors.data, ans)

    def test_cluster(self):
        clusterer = DBSCAN(eps=0.1, min_samples=2)
        clustered = self.vector.cluster(clusterer, min_vectors=3)
        assert clustered.data.shape[1] == 3
        assert clustered.data.shape[0] == 8
        assert clustered.axes_manager.signal_axes[0].size == 3
        assert clustered.ivec["cluster"].data.shape[0] == 8
        assert isinstance(clustered, DiffractionVectors2D)

    def test_slice(self):
        slic = self.vector.ivec[:, self.vector.ivec[1] > 0]
        assert slic.data.shape[0] == 5
        assert slic.data.shape[1] == 2


class TestVector2DSubclass:
    @pytest.fixture()
    def vectors(self):
        vectors = np.random.random((2, 2, 20, 2))
        v = DiffractionVectors2D(vectors)
        v.offsets = np.array([0, 0])
        v.scales = np.array([1, 1])
        v.column_names = ["x", "y"]
        return v

    def test_setup(self, vectors):
        assert isinstance(vectors, DiffractionVectors2D)
        assert isinstance(vectors, Signal2D)
        assert vectors.axes_manager.signal_shape == (2, 20)
        assert vectors.column_names == ["x", "y"]
        np.testing.assert_array_equal(vectors.scales, [1, 1])
        np.testing.assert_array_equal(vectors.offsets, [0, 0])

    @pytest.mark.parametrize("item", [0, "x"])
    def test_slice(self, vectors, item):
        sliced = vectors.ivec[item]
        assert isinstance(sliced, DiffractionVectors1D)
        assert isinstance(sliced, Signal1D)
        assert sliced.axes_manager.signal_shape == (20,)

    def test_flatten(self, vectors):
        flatten_diffraction_vectors = vectors.flatten_diffraction_vectors()
        assert isinstance(flatten_diffraction_vectors, DiffractionVectors2D)

    def test_flatten_twice(self, vectors):
        flatten_diffraction_vectors = vectors.flatten_diffraction_vectors()
        flatten_diffraction_vectors = (
            flatten_diffraction_vectors.flatten_diffraction_vectors()
        )
        assert isinstance(flatten_diffraction_vectors, DiffractionVectors2D)

    def test_to_markers(self, vectors):
        markers = vectors.to_markers()
        assert isinstance(markers, hs.plot.markers.Points)
        s = Signal2D(np.ones((2, 2, 10, 10)))
        s.add_marker(markers)

    def test_gt(self, vectors):
        gt_vectors = vectors > 1.1
        np.testing.assert_array_equal(False, gt_vectors.data)

    def test_gte(self, vectors):
        gt_vectors = vectors >= 1.1
        np.testing.assert_array_equal(False, gt_vectors.data)

    def test_lt(self, vectors):
        lt_vectors = vectors < 1.1
        np.testing.assert_array_equal(True, lt_vectors.data)

    def test_lte(self, vectors):
        lt_vectors = vectors <= 1.1
        np.testing.assert_array_equal(True, lt_vectors.data)

    def test_num_rows(self, vectors):
        assert vectors.num_rows == 20

    def test_from_peaks(self):
        with pytest.raises(NotImplementedError):
            DiffractionVectors2D.from_peaks(BaseSignal(np.ones((2, 2, 10, 10))))
