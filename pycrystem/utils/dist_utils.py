# -*- coding: utf-8 -*-
# Copyright 2016 The PyCrystEM developers
#
# This file is part of  PyCrystEM.
#
#  PyCrystEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  PyCrystEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  PyCrystEM.  If not, see <http://www.gnu.org/licenses/>.
from __future__ import division

from scipy.constants import h, m_e, e, c, pi
import math
import numpy as np


"""
This module contains utility functions for dealing with distortion fields.
"""

def construct_displacement_gradient(ref):
    """Construct TensorField object containing displacement gradient
    tensor obtained via fitting a two dimensionsal scalable reference
    pattern.

    Parameters
    ----------

    ref : component

        The component object describing the

    Returns
    -------

    D : TensorField

        The

    """
    D = hs.signals.Signal(np.ones((dp.axes_manager.navigation_shape[1],
                                   dp.axes_manager.navigation_shape[0],
                                   3,3)))
    D.axes_manager.set_signal_dimension(2)

    D.data[:,:,0,0] = ref.d11.map['values']
    D.data[:,:,1,0] = ref.d12.map['values']
    D.data[:,:,2,0] = 0.
    D.data[:,:,0,1] = ref.d21.map['values']
    D.data[:,:,1,1] = ref.d22.map['values']
    D.data[:,:,2,1] = 0.
    D.data[:,:,0,2] = 0.
    D.data[:,:,1,2] = 0.
    D.data[:,:,2,2] = 1.

    return D


def _transform_basis2D(D, angle):
    """Method to transform the 2D basis in which a 2nd-order tensor is described.

    Parameters
    ----------

    D : 3x3 matrix


    angle : float
        The anti-clockwise rotation angle between the diffraction x/y axes and
        the x/y axes in which the tensor is to be specified.

    Returns
    -------

    T : TensorField
        Operates in place, replacing the original tensor field object with a
        tensor field described in the new basis.

    """

    a=angle*np.pi/180.0
    r11 = math.cos(a)
    r12 = math.sin(a)
    r21 = -math.sin(a)
    r22 = math.cos(a)

    R = np.array([[r11, r12, 0.],
                  [r21, r22, 0.],
                  [0.,  0.,  1.]])

    T = np.dot(np.dot(R, D), R.T)
    return T


def polar_decomposition(D, side='right'):
    """Perform polar decomposition on a signal containing matrices in

    D = RU

    where R

    Parameters
    ----------

    side : 'right' or 'left'

        The side of the matrix product on which the Hermitian semi-definite
        part of the polar decomposition appears. i.e. if 'right' D = RU and
        if 'left' D = UR.

    Returns
    -------

    R : RotationMatrix

        The orthogonal matrix describing

    U : TensorField

        The

    """

    R = hs.signals.Signal(np.ones((dp.axes_manager.navigation_shape[1],
                                   dp.axes_manager.navigation_shape[0],
                                   3,3)))
    R.axes_manager.set_signal_dimension(2)
    U = hs.signals.Signal(np.ones((dp.axes_manager.navigation_shape[1],
                                   dp.axes_manager.navigation_shape[0],
                                   3,3)))
    U.axes_manager.set_signal_dimension(2)

    for z, indices in zip(D._iterate_signal(),
                          D.axes_manager._array_indices_generator()):
        R.data[indices] = linalg.polar(D.data[indices], side=side)[0]
        U.data[indices] = linalg.polar(D.data[indices], side=side)[1]

    return R, U


def get_rotation_angle(R):
    """Return the

    Parameters
    ----------

    R : RotationMatrix

        RotationMatrix two dimensional signal object of the form:

         cos x  sin x
        -sin x  cos x

    Returns
    -------

    angle : float

    """
    arr_shape = (R.axes_manager._navigation_shape_in_array
                 if R.axes_manager.navigation_size > 0
                 else [1, ])
    T = np.zeros(arr_shape, dtype=object)


    for z, indices in zip(R._iterate_signal(),
                          R.axes_manager._array_indices_generator()):
        # T[indices] = math.acos(R.data[indices][0,0])
        # Insert check here to make sure that R is valid rotation matrix.
        # theta12 = math.asin(R.data[indices][0,1])
        T[indices] = -math.asin(R.data[indices][1,0])
        # theta22 = math.acos(R.data[10,30][1,1])
    X = hs.signals.Image(T.astype(float))

    return X
