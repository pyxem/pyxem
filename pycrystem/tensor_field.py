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

from hyperspy.signals import Signal2D
from hyperspy import roi
import numpy as np
from scipy.ndimage import variance
from pycrystem.utils.expt_utils import *

"""
Signal class for Tensor Fields
"""


class BaseTensorField(Signal2D):
    _signal_type = "tensor_field"

    def __init__(self, *args, **kwargs):
        Signal2D.__init__(self, *args, **kwargs)
        # Check that the signal dimensions are (3,3) for it to be a valid
        # TensorField

    def _transform_basis(self, R):
        """Method to transform the 2D basis in which a 2nd-order tensor is described.

        Parameters
        ----------

        R : 3x3 matrix

            Rotation matrix.

        Returns
        -------

        T : TensorField
            Operates in place, replacing the original tensor field object with a tensor
            field described in the new basis.

        """

        a=angle*np.pi/180.0
        r11 = math.cos(a)
        r12 = math.sin(a)
        r21 = -math.sin(a)
        r22 = math.cos(a)

        R = np.array([[r11, r12, 0.],
                      [r21, r22, 0.],
                      [0.,  0.,  1.]])

        T = np.dot(np.dot(R, self), R.T)
        return T
