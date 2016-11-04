# -*- coding: utf-8 -*-
# Copyright 2016 The PyCrystEM developers
#
# This file is part of PyCrystEM.
#
# PyCrystEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyCrystEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyCrystEM.  If not, see <http://www.gnu.org/licenses/>.
from __future__ import division

import scipy.linalg as linalg
import math
import numpy as np
import scipy.ndimage as ndi
from skimage import transform as tf
from skimage import morphology, filters
from skimage.morphology import square
from hyperspy.signals import BaseSignal, Signal2D

"""
This module contains utility functions for manipulating the results of strain
mapping analysis.
"""

def get_strain_maps(dp, component):
    """Gets the strain maps from model fitting results.

    Parameters
    ----------

    Returns
    -------
    """
    #TODO:
    D = construct_displacement_gradient(component)
    R, U = polar_decomposition(D)

    e11 = U.isig[0,0].as_signal2d(image_axes=e11.axes_manager.navigation_axes)
    e12 = U.isig[0,1].as_signal2d(image_axes=e12.axes_manager.navigation_axes)
    e21 = U.isig[1,0].as_signal2d(image_axes=e21.axes_manager.navigation_axes)
    e22 = U.isig[1,1].as_signal2d(image_axes=e22.axes_manager.navigation_axes)
    theta = get_rotation_angle(R)

    strain_results = Signal2D(np.ones((4, dp.axes_manager.navigation_shape[1],
                                       dp.axes_manager.navigation_shape[0])))

    strain_results.data[0] = 1. - e11.data
    strain_results.data[1] = 1. - e22.data
    strain_results.data[2] = e12.data
    strain_results.data[3] = theta.data

    return strain_results

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
    X = Signal2D(T.astype(float))

    return X
