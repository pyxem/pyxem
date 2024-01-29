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

from pyxem.generators import get_DisplacementGradientMap
from pyxem.signals.tensor_field import DisplacementGradientMap, StrainMap
import hyperspy.api as hs
import pytest
import numpy as np


def vector_operation(z, M):
    """
    Extracts our input format and applies operations vector by vector

    Parameters
    ----------
    z : (n,2) np.array
        input vectors
    M : (2,2) np.array
        Tranformation matrix,
    Returns
    -------
    z_transformed : (n,2) np.array
        Output vectors
    """
    v_transformed = []
    for i in np.arange(0, z.shape[0]):
        v_transformed.append(np.matmul(M, z[i, :]))
    return np.asarray(v_transformed)


def rotation(z):
    theta = np.deg2rad(3)
    c, s = np.cos(theta), np.sin(theta)
    R = np.asarray(([c, -s], [s, c]))
    return vector_operation(z, R)


def uniform_expansion(z):
    return 1.1 * z


def stretch_in_x(z):
    M = np.asarray([[1.1, 0], [0, 1]])
    return vector_operation(z, M)


def generate_test_vectors(v):
    """
    We imagine we measured 4 sets of vectors, from 4 regions of sample, normal,
    rotated, uniform expansion and uniaxial expansion.
    """
    return np.asarray([[v, rotation(v)], [uniform_expansion(v), stretch_in_x(v)]])


def generate_displacment_map(vectors, return_residuals=False):
    deformed = hs.signals.Signal2D(generate_test_vectors(vectors))
    return get_DisplacementGradientMap(
        deformed, vectors, return_residuals=return_residuals
    )


def generate_strain_map(vectors):
    return generate_displacment_map(vectors).get_strain_maps()


class TestDisplacementGradientMap:
    four_vectors = np.asarray([[1, 0], [0, 1], [1, -1], [1, 1]])
    left_handed = np.asarray(([1, 1], [1, -1.2]))
    xy_vectors = np.asarray([[1, 0], [0, 1]])

    def setup_method(self):
        # reinitalizing to use inside functions...
        self.left_handed = np.asarray(([1, 1], [1, -1.2]))
        self.xy_vectors = np.asarray([[1, 0], [0, 1]])
        self.four_vectors = np.asarray([[1, 0], [0, 1], [1, -1], [1, 1]])

    @pytest.mark.parametrize("four_vectors", (four_vectors,))
    def test_generate_test_vectors(self, four_vectors):
        test_vectors = generate_test_vectors(four_vectors)
        assert np.shape(test_vectors) == (2, 2, 4, 2)
        assert np.shape(test_vectors[0, 0]) == four_vectors.shape
        assert np.shape(test_vectors[0, 1]) == four_vectors.shape

    @pytest.mark.parametrize("residuals", (True, False))
    @pytest.mark.parametrize("vectors", (four_vectors, xy_vectors, left_handed))
    def test_generate_displacement_map(self, vectors, residuals):
        if residuals:
            dis_map, r = generate_displacment_map(vectors, return_residuals=residuals)
        else:
            dis_map = generate_displacment_map(vectors, return_residuals=residuals)
        assert isinstance(dis_map, DisplacementGradientMap)

    @pytest.mark.parametrize("vectors", (four_vectors, xy_vectors, left_handed))
    def test_generate_strain_map(self, vectors):
        dis_map = generate_strain_map(vectors)
        assert isinstance(dis_map, StrainMap)

    def test_results_returned_correctly_in_same_basis(self):
        """Basic test of the summary statement for this section"""
        xy = generate_strain_map(self.xy_vectors)
        lh = generate_strain_map(self.left_handed)
        fv = generate_strain_map(self.four_vectors)
        np.testing.assert_almost_equal(xy.data, lh.data, decimal=2)
        np.testing.assert_almost_equal(xy.data, fv.data, decimal=2)

    def test_trivial_weight_function_case(self):
        """If weights are [1,1,1,1] the result should be the same as weights=None"""
        weights = [1, 1, 1, 1]
        xy = generate_strain_map(self.xy_vectors)
        deformed = hs.signals.Signal2D(generate_test_vectors(self.four_vectors))
        weight_strain_map = get_DisplacementGradientMap(
            deformed, self.four_vectors, weights=weights
        ).get_strain_maps()
        np.testing.assert_almost_equal(xy.data, weight_strain_map.data, decimal=2)

    def test_weight_function_behaviour(self):
        """Confirms that  a weight function [1,1,2,2] on [a,a,b,b] gives (2a+4b)/6 as the strain"""
        multi_vector_array = self.four_vectors
        strained_by_1pc_in_x = vector_operation(
            multi_vector_array, np.asarray([[1.01, 0], [0, 1]])
        )  # first  2
        strained_by_2pc_in_x = vector_operation(
            multi_vector_array, np.asarray([[1.02, 0], [0, 1]])
        )  # second 2
        weights = [1, 1, 2, 2]  # ((0.1*2 + 0.2*4)/6) = 0.166666
        vectors = np.concatenate(
            (strained_by_1pc_in_x[:, :2], strained_by_2pc_in_x[:, 2:]), axis=1
        )
        deformed = hs.signals.Signal2D(
            np.asarray([[vectors, vectors], [vectors, vectors]])
        )
        strain_map = get_DisplacementGradientMap(
            deformed, multi_vector_array, weights=weights
        ).get_strain_maps()
        np.testing.assert_almost_equal(
            strain_map.inav[0].isig[0, 0].data[0], -1.0166666 + 1, decimal=2
        )


""" Each of these basis should return the same results, in an xy basis"""


class TestLazyNotLazy:
    def setup_method(self):
        data = np.empty((4, 2), dtype=object)
        for iy, ix in np.ndindex(data.shape):
            data[iy, ix] = np.array([[5, 10], [10, 5]])
        self.vs = hs.signals.BaseSignal(data, ragged=True)

    def test_not_lazy(self):
        vs = self.vs
        vs_ref = vs.data[0, 0]
        D = get_DisplacementGradientMap(vs, vs_ref)
        assert D.axes_manager.navigation_shape == (2, 4)
        assert D.axes_manager.signal_shape == (3, 3)
        strain_map = D.get_strain_maps()

    def test_lazy(self):
        vs = self.vs.as_lazy()
        vs_ref = vs.data[0, 0]
        D = get_DisplacementGradientMap(vs, vs_ref)
        assert D.axes_manager.navigation_shape == (2, 4)
        assert D.axes_manager.signal_shape == (3, 3)
        strain_map = D.get_strain_maps()
