decimal=2# -*- coding: utf-8 -*-
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

import numpy as np
import pytest
import hyperspy.api as hs

from pyxem.generators.displacement_gradient_tensor_generator import \
    get_DisplacementGradientMap, get_single_DisplacementGradientTensor

def vector_operation(z,M):
    """
    Extracts our input format and applies operations vector by vector

    Parameters
    ----------
    z :
        n | 2 array
    M :
        tranformation matrix, (2,2)
    Returns
    -------
    z_transformed :
        n | 2 array after transformation
    """

    v_transformed = []
    for i in np.arange(0,z.shape[1]):
        v_transformed.append(np.matmul(M,z[:,i]))
    return np.asarray(v_transformed).T

def rotation(z):
    theta = np.deg2rad(3)
    c, s = np.cos(theta), np.sin(theta)
    R = np.asarray(([c, -s], [s, c]))
    return vector_operation(z,R)

def uniform_expansion(z):
    return (1.1 * z)

def stretch_in_x(z):
    M = np.asarray([[1.1, 0], [0, 1]])
    return vector_operation(z,M)

def generate_test_vectors(v):
    """
    We imagine we measured 4 sets of vectors, from 4 regions of sample, normal,
    rotated, uniform expansion and uniaxial expansion.
    """
    return np.asarray([[v, rotation(v)],
                       [uniform_expansion(v), stretch_in_x(v)]])


def generate_strain_map(vectors):
    deformed = hs.signals.Signal2D(generate_test_vectors(vectors))
    st = get_DisplacementGradientMap(deformed, vectors).get_strain_maps()
    return st

"""
Our "sample" allows us to use a range of basis vectors, we try 4:
> xy
> a right handed: orthogonal set
> a left handed: not orthogonal, not right handed set
> Four vectors: (for the least squares method)
"""

@pytest.fixture()
def xy_vectors():
    xy = np.asarray([[1, 0], [0, 1]])
    s_xy = generate_strain_map(xy)
    return s_xy

@pytest.fixture()
def right_handed():
    oo = np.asarray(([1,1],[-1,1]))
    s_oo = generate_strain_map(oo)
    return s_oo

@pytest.fixture()
def left_handed():
    # not orthogonal, not normalised (all fine) but LEFT HANDED
    danger = np.asarray(([1,1],[1,-1.2]))
    s_da = generate_strain_map(danger)
    return s_da

@pytest.fixture()
def multi_vector():
    four_vectors = np.asarray([[1,0,1,1],[0,1,-1,1]])
    s_th = generate_strain_map(four_vectors)
    return s_th


def test_rotation(xy_vectors,right_handed,multi_vector):
    """
    We should always measure the same rotations, regardless of basis (as long as it's right handed)
    """
    xy_rot = xy_vectors.inav[3].data
    rh_rot = right_handed.inav[3].data
    mv_rot = multi_vector.inav[3].data

    np.testing.assert_almost_equal(xy_rot, rh_rot, decimal=2)  # rotations
    np.testing.assert_almost_equal(xy_rot, mv_rot, decimal=2)  # rotations

def test_left_vs_right_rotation(xy_vectors,left_handed):
    """
    We should pick up a minus sign if we compare left vs right handed basis
    """
    xy_rot = xy_vectors.inav[3].data
    lh_rot = np.multiply(left_handed.inav[3].data,-1)

    np.testing.assert_almost_equal(xy_rot, lh_rot, decimal=2)  # rotations

def test_trace(xy_vectors,right_handed,multi_vector):
    """
    Basis does effect strain measurement, but we can simply calculate suitable invarients.
    See https://en.wikipedia.org/wiki/Infinitesimal_strain_theory for details.
    """
    np.testing.assert_almost_equal(
        np.add(
            xy_vectors.inav[0].data, xy_vectors.inav[1].data), np.add(
            right_handed.inav[0].data, right_handed.inav[1].data), decimal=2)
    np.testing.assert_almost_equal(
        np.add(
            xy_vectors.inav[0].data, xy_vectors.inav[1].data), np.add(
            multi_vector.inav[0].data, multi_vector.inav[1].data), decimal=2)

def test_multi_vector_method_has_xy_as_basis(xy_vectors,multi_vector):
    xy = xy_vectors.data
    mv = multi_vector.data
    np.testing.assert_almost_equal(xy,mv, decimal=2)

def test_trivial_weight_function_case(xy_vectors):
    weights = [1,1,1,1]
    four_vectors = np.asarray([[1,0,1,1],[0,1,-1,1]])
    deformed = hs.signals.Signal2D(generate_test_vectors(four_vectors))
    weight_strain_map = get_DisplacementGradientMap(deformed,four_vectors, weights=weights).get_strain_maps()
    np.testing.assert_almost_equal(xy_vectors.data,weight_strain_map.data,decimal=2)

def test_weight_function_behaviour():
    multi_vector_array = np.asarray([[1,0,1,1],[0,1,-1,1]])
    strained_by_1pc_in_x = vector_operation(multi_vector_array,np.asarray([[1.01, 0], [0, 1]])) #first  2
    strained_by_2pc_in_x = vector_operation(multi_vector_array,np.asarray([[1.02, 0], [0, 1]])) #second 2
    weights = [1,1,2,2] # ((0.1*2 + 0.2*4)/6) = 0.166666
    vectors = np.concatenate((strained_by_1pc_in_x[:,:2],strained_by_2pc_in_x[:,2:]),axis=1)
    deformed = hs.signals.Signal2D(np.asarray([[vectors,vectors],[vectors,vectors]]))
    strain_map = get_DisplacementGradientMap(deformed, multi_vector_array, weights = weights).get_strain_maps()
    np.testing.assert_almost_equal(strain_map.inav[0].isig[0,0].data[0], -1.0166666 + 1 ,decimal=2)
