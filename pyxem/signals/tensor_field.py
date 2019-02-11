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

from hyperspy.signals import Signal2D
import numpy as np
from scipy.linalg import polar
from hyperspy.utils import stack
import math

"""
Signal class for Tensor Fields
"""


def _polar_decomposition(image, side):
    """Perform a polar decomposition of a second rank tensor.

    Parameters
    ----------
    image : np.array()
        Matrix on which to form polar decomposition.
    side : string
        'left' or 'right' the side on which to perform polar decomposition.

    Returns
    -------
    U, R : np.array()
        Stretch and rotation matrices obtained by polar decomposition.

    """
    return np.array(polar(image, side=side))


def _get_rotation_angle(matrix):
    """Find the rotation angle associated with a given rotation matrix.

    Parameters
    ----------
    matrix : np.array()
        A rotation matrix.

    Returns
    -------
    angle :  np.array()
        Rotation angle associated with matrix.

    """
    return np.array(-math.asin(matrix[1, 0]))


class DisplacementGradientMap(Signal2D):
    _signal_type = "tensor_field"

    def __init__(self, *args, **kwargs):
        Signal2D.__init__(self, *args, **kwargs)
        # Check that the signal dimensions are (3,3) for it to be a valid
        # TensorField

    def polar_decomposition(self):
        """Perform polar decomposition on the second rank tensors describing
        the TensorField. The polar decomposition is right handed and given by
        :math:`D = RU`

        Returns
        -------
        R : TensorField
            The orthogonal matrix describing the rotation field.

        U : TensorField
            The strain tensor field.

        """
        RU = self.map(_polar_decomposition,
                      side='right',
                      inplace=False)
        return RU.isig[:, :, 0], RU.isig[:, :, 1]

    def get_strain_maps(self):
        """Obtain strain maps from the displacement gradient tensor at each
        navigation position in the small strain approximation.

        Returns
        -------

        strain_results : BaseSignal
            Signal of shape < 4 | , > , navigation order is e11,e22,e12,theta
        """
        R, U = self.polar_decomposition()

        e11 = -U.isig[0, 0].T + 1
        e12 = U.isig[0, 1].T
        e21 = U.isig[1, 0].T
        e22 = -U.isig[1, 1].T + 1
        theta = R.map(_get_rotation_angle, inplace=False)
        theta.axes_manager.set_signal_dimension(2)

        strain_results = stack([e11, e22, e12, theta])

        return strain_results
