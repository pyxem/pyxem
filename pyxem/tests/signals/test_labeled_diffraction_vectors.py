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

import pytest

import numpy as np
import hyperspy.api as hs
from sklearn.cluster import DBSCAN

from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D
from pyxem.signals.labeled_diffraction_vectors2d import LabeledDiffractionVectors2D


class TestLabelledDiffractionVectors:
    @pytest.fixture
    def labeled_vectors(self):
        vectors = np.reshape(np.arange(1000), (100, 10))
        labels = np.repeat(np.arange(10), 10)
        vectors = np.hstack((vectors, labels[:, np.newaxis]))
        columns = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "label"]
        return LabeledDiffractionVectors2D(data=vectors, column_names=columns)

    def test_init(self, labeled_vectors):
        assert labeled_vectors.data.shape == (100, 11)

    def test_map_vectors(self, labeled_vectors):
        def func(x):
            return np.shape(x)[0]

        result = labeled_vectors.map_vectors(func, dtype=np.float64)
        assert result.shape == (10,)
        assert np.all(result == 10)  # all labels have 10 vectors

    def test_map_vectors_with_shape(self, labeled_vectors):
        def func(x):
            return np.shape(x)

        result = labeled_vectors.map_vectors(func, dtype=np.float64, shape=(2,))
        assert result.shape == (10, 2)
        assert np.all(result == [10, 11])  # all labels have 10 vectors

    @pytest.mark.parametrize("get_polygon", [True, False])
    def test_to_markers(self, labeled_vectors, get_polygon):
        s = ElectronDiffraction2D(np.zeros((10, 10, 10, 10)))
        if get_polygon:
            points, polygons = labeled_vectors.to_markers(
                signal=s, get_polygons=get_polygon
            )
            assert isinstance(polygons, hs.plot.markers.Polygons)
        else:
            points = labeled_vectors.to_markers(signal=s, get_polygon=get_polygon)
        assert isinstance(points, hs.plot.markers.Points)

    def test_cluster_labeled_vectors(self, labeled_vectors):
        scan = DBSCAN()
        clust = labeled_vectors.cluster_labeled_vectors(method=scan)
        assert clust.data.shape == (100, 12)
        assert clust.axes_manager.signal_axes[0].size == 12
        assert len(clust.column_names) == 12
        assert len(clust.units) == 12

    def test_cluster_labeled_vectors_custom(self, labeled_vectors):
        scan = DBSCAN()

        def max(arr, columns=[0, 1]):
            return np.max(arr[:, columns], axis=0)

        kwargs = {"label_index": -1, "dtype": float, "shape": (2,)}

        clust = labeled_vectors.cluster_labeled_vectors(
            method=scan, preprocessing=max, **kwargs
        )
        assert clust.data.shape == (100, 12)
        assert clust.axes_manager.signal_axes[0].size == 12
        assert len(clust.column_names) == 12
        assert len(clust.units) == 12

    def test_cluster_labeled_vectors_fail(self, labeled_vectors):
        scan = DBSCAN()
        with pytest.raises(ValueError):
            labeled_vectors.cluster_labeled_vectors(method=scan, preprocessing="fail")

    def test_plot_cluster_labeled_vectors_error(self, labeled_vectors):
        scan = DBSCAN()
        clust = labeled_vectors.cluster_labeled_vectors(method=scan)
        with pytest.raises(ValueError):
            clust.plot_clustered()

        with pytest.raises(ValueError):
            labeled_vectors.plot_clustered()

    def test_plot_cluster_labeled_vectors(self):
        vectors = np.reshape(np.arange(1000), (100, 10))
        labels = np.repeat(np.arange(10), 10)
        vectors = np.hstack((vectors, labels[:, np.newaxis]))
        columns = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "label"]
        vectors[:50, 0] = 10
        vectors[:50, 1] = 10
        vectors[50:, 0] = 20
        vectors[50:, 1] = 20
        labeled_vectors = LabeledDiffractionVectors2D(
            data=vectors, column_names=columns
        )
        scan = DBSCAN()
        clust = labeled_vectors.cluster_labeled_vectors(method=scan)
        clust.plot_clustered()

        clust.plot_clustered(labels=[0, 1])
        s = ElectronDiffraction2D(np.zeros((10, 10)))
        clust.plot_clustered(signal=s)

    def test_only_signal_axes_error(self):
        vectors = np.reshape(np.arange(1000), (100, 10))
        labels = np.repeat(np.arange(10), 10)
        vectors = np.hstack((vectors, labels[:, np.newaxis]))
        vects = np.stack([vectors, vectors])
        columns = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "label"]
        labeled_vectors = LabeledDiffractionVectors2D(data=vects, column_names=columns)
        with pytest.raises(ValueError):
            labeled_vectors.cluster_labeled_vectors(method=DBSCAN())
