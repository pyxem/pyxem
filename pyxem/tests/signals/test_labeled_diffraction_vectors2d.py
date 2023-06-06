# -*- coding: utf-8 -*-
# Copyright 2016-2023 The pyXem developers
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
from pyxem.signals.labeled_diffraction_vectors2d import LabeledDiffractionVectors2D
import numpy as np
from numpy.testing import assert_array_equal


class TestLabeledDiffractionVectors2D:
    @pytest.fixture
    def labeled_array(self):
        v1 = np.array([[1, 2, 0],
                       [1, 3, 0],
                       [10, 2, 2],
                       [10, 3, 2],
                       [1, 2, 1],
                       [1, 3, 1],
                       [10, 2, 3],
                       [10, 3.5, 3],
                       ])
        return LabeledDiffractionVectors2D(v1)

    def test_vector_dist(self,
                         labeled_array):
        dis_matrix = labeled_array.get_dist_matrix()
        assert np.shape(dis_matrix) == (4, 4)
        assert dis_matrix[0, 1] == 0

    def test_map_function(self, labeled_array):
        m = labeled_array.map_vectors(np.mean,
                                      axis=0,
                                      dtype=float,
                                      shape=(3,))
        assert_array_equal(m[:, 0],
                           [1, 1, 10, 10])

    def test_cluster_labeled_vectors(self, labeled_array):
        labeled_v = labeled_array.cluster_labeled_vectors()
        assert_array_equal(labeled_v.data[:, -1],
                           [0, 0, 1, 1, 0, 0, 1, 1])
        assert labeled_v.axes_manager.signal_shape == (4, 8)

