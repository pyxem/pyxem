# -*- coding: utf-8 -*-
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


def rotation(z):
    z = z.T  # [[Vx Ux],[Vy,Uy]] rotates incorrectly, so we transpose
    theta = np.deg2rad(3)
    c, s = np.cos(theta), np.sin(theta)
    R = np.asarray(([c, -s], [s, c]))
    zbar = np.matmul(R, z)
    return zbar.T  # back to the incorrectly rotating format we expect


def uniform_expansion(z):
    return (1.1 * z)


def stretch_in_x(z):
    M = np.asarray([[1.1, 0], [0, 1]])
    return np.matmul(M, z)


def generate_test_vectors(v):
    """
    We imagine we measured 4 sets of vectors, from 4 regions of sample, normal,
    rotated, uniform expansion and uniaxial expansion.
    """
    return np.asarray([[v, rotation(v)],
                       [uniform_expansion(v), stretch_in_x(v)]])


def generate_strain_map(vectors):
    dp = hs.signals.Signal2D(generate_test_vectors(vectors))
    st = get_DisplacementGradientMap(dp, vectors).get_strain_maps()
    return st


def get_arrays():
    """
    Our "sample" allows us to use a range of basis vectors, we try 3 sets:
    xy
    A pair at 45 degrees to xy, orthonormal
    A pair that are neither orthogonal to each other, nor normalised
    """
    xy = np.asarray([[1, 0], [0, 1]])
    oo = np.asarray(([1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]))
    danger = np.asarray(([1 / np.sqrt(2), 1 / np.sqrt(2)], [3, -2]))  # not orthogonal, not normalised
    s_xy = generate_strain_map(xy)
    s_oo = generate_strain_map(oo)
    s_da = generate_strain_map(danger)
    return s_xy, s_oo, s_da


def test_rotation():
    """
    We should always measure the same rotations, regardless of basis (note the tighter constraint
    avaliable for orthonormalised vectors)
    """
    s_xy, s_oo, s_da = get_arrays()
    np.testing.assert_almost_equal(s_xy.inav[3].data, s_oo.inav[3].data, decimal=5)  # rotations
    np.testing.assert_almost_equal(s_xy.inav[3].data, s_da.inav[3].data, decimal=2)  # rotations


def test_trace():
    """
    Basis does effect strain measurement, but we can simply calculate suitable invarients.
    See https://en.wikipedia.org/wiki/Infinitesimal_strain_theory for details.
    """
    s_xy, s_oo, s_da = get_arrays()
    np.testing.assert_almost_equal(
        np.add(
            s_xy.inav[0].data, s_xy.inav[1].data), np.add(
            s_oo.inav[0].data, s_oo.inav[1].data), decimal=5)
    np.testing.assert_almost_equal(
        np.add(
            s_xy.inav[0].data, s_xy.inav[1].data), np.add(
            s_da.inav[0].data, s_da.inav[1].data), decimal=2)
