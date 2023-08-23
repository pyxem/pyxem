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

from hyperspy.signals import Signal2D

from pyxem.signals import DiffractionVectors2D


class TestDiffractionVectors2D:
    def setup(self):
        vectors = np.reshape(np.repeat(np.arange(0, 100), 2), (50, 4))
        self.vector = DiffractionVectors2D(vectors)

    def test_setup(self):
        assert isinstance(self.vector, DiffractionVectors2D)

    def test_magnitudes(self):
        magnitudes = self.vector.get_magnitudes()
        mags = np.linalg.norm(self.vector, axis=1)
        np.testing.assert_array_almost_equal(mags, magnitudes.data)
        assert len(magnitudes.axes_manager.signal_axes) == 1

    def test_filter_magnitudes(self):
        magnitudes = self.vector.filter_magnitude(min_magnitude=10, max_magnitude=40)
        mags = np.linalg.norm(self.vector, axis=1)
        num_in_range = np.sum((mags > 10) * (mags < 40))
        assert num_in_range == magnitudes.data.shape[0]


class TestSingleDiffractionVectors2D:
    def setup(self):
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
