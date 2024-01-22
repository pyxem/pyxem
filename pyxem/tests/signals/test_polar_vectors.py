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
from hyperspy._signals.lazy import LazySignal

from pyxem.signals.polar_vectors import PolarVectors, LazyPolarVectors


class TestDiffractionVectors:
    @pytest.fixture
    def polar_vectors(self):
        vectors = np.empty((5,), dtype=object)
        for i in np.ndindex(5):
            phis = np.linspace(0, 2 * np.pi, i[0] + 3)[:-1]  # 2 fold --> 6 fold
            ks = np.ones_like(phis)
            intens = np.ones_like(phis)
            vectors[i] = np.array([ks, phis, intens]).T
        return PolarVectors(
            data=vectors,
            column_names=["k", "phi", "intensity"],
            units=["1/nm", "rad", "a.u."],
        )

    def test_init(self, polar_vectors):
        assert polar_vectors.data.shape == (5,)
        assert polar_vectors.data[0].shape == (2, 3)
        assert polar_vectors.column_names == ["k", "phi", "intensity"]

    def test_get_inscribed_angles_value(self, polar_vectors):
        angles = polar_vectors.get_angles()
        assert angles.data.shape == (5,)
        assert angles.data[0].shape == (0, 5)
        assert angles.data[1].shape == (1, 5)
        for i in range(4):
            np.testing.assert_array_almost_equal(
                angles.data[i + 1][0], np.array([1, 2 * np.pi / (i + 3), 0, 1, 0])
            )

    @pytest.mark.parametrize("intensity_threshold", [None, 0.5])
    @pytest.mark.parametrize("min_k", [None, 0.3])
    def test_get_inscribed_angles(
        self,
        polar_vectors,
        intensity_threshold,
        min_k,
    ):
        polar_vectors.data[2][0, 1] = 2 * np.pi / 3
        angles = polar_vectors.get_angles(
            intensity_threshold=intensity_threshold,
            min_k=min_k,
            accept_threshold=0.1,
            min_angle=0.05,
        )
        assert angles.data.shape == (5,)
        assert angles.data[0].shape == (0, 5)
        assert angles.data[1].shape == (1, 5)
        assert angles.data[2].shape == (1, 5)  # repeated angle

    @pytest.mark.parametrize("cartesian", [True, False])
    def test_get_inscribed_angles_to_markers(self, polar_vectors, cartesian):
        angles = polar_vectors.get_angles()
        angles.to_markers(cartesian=cartesian)

    def test_as_lazy(self, polar_vectors):
        lazy = polar_vectors.as_lazy()
        assert isinstance(lazy, LazySignal)
        assert isinstance(lazy, LazyPolarVectors)
        assert lazy.data.shape == (5,)
        assert lazy.data[0].compute().shape == (2, 3)
        assert lazy.column_names == ["k", "phi", "intensity"]
