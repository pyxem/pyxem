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

from pyxem.generators import get_DisplacementGradientMap
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
    for i in np.arange(0, z.shape[1]):
        v_transformed.append(np.matmul(M, z[:, i]))
    return np.asarray(v_transformed).T


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


def generate_strain_map(vectors):
    deformed = hs.signals.Signal2D(generate_test_vectors(vectors))
    return get_DisplacementGradientMap(deformed, vectors).get_strain_maps()


@pytest.fixture()
def xy_vectors():
    xy = np.asarray([[1, 0], [0, 1]])
    return generate_strain_map(xy)


@pytest.fixture()
def left_handed():
    not_xy = np.asarray(([1, 1], [1, -1.2]))
    return generate_strain_map(not_xy)


@pytest.fixture()
def multi_vector():
    four_vectors = np.asarray([[1, 0, 1, 1], [0, 1, -1, 1]])
    return generate_strain_map(four_vectors)


""" Each of these basis should return the same results, in an xy basis"""


def test_results_returned_correctly_in_same_basis(
    xy_vectors, left_handed, multi_vector
):
    """ Basic test of the summary statement for this section """
    np.testing.assert_almost_equal(xy_vectors.data, left_handed.data, decimal=2)
    np.testing.assert_almost_equal(xy_vectors.data, multi_vector.data, decimal=2)


def test_trivial_weight_function_case(xy_vectors):
    """ If weights are [1,1,1,1] the result should be the same as weights=None"""
    weights = [1, 1, 1, 1]
    four_vectors = np.asarray([[1, 0, 1, 1], [0, 1, -1, 1]])
    deformed = hs.signals.Signal2D(generate_test_vectors(four_vectors))
    weight_strain_map = get_DisplacementGradientMap(
        deformed, four_vectors, weights=weights
    ).get_strain_maps()
    np.testing.assert_almost_equal(xy_vectors.data, weight_strain_map.data, decimal=2)


def test_weight_function_behaviour():
    """ Confirms that  a weight function [1,1,2,2] on [a,a,b,b] gives (2a+4b)/6 as the strain"""
    multi_vector_array = np.asarray([[1, 0, 1, 1], [0, 1, -1, 1]])
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
    deformed = hs.signals.Signal2D(np.asarray([[vectors, vectors], [vectors, vectors]]))
    strain_map = get_DisplacementGradientMap(
        deformed, multi_vector_array, weights=weights
    ).get_strain_maps()
    np.testing.assert_almost_equal(
        strain_map.inav[0].isig[0, 0].data[0], -1.0166666 + 1, decimal=2
    )
