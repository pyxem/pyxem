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
from hyperspy.component import Component
from pyxem.signals.tensor_field import DisplacementGradientMap
from skimage import transform as tf


class ScalableReferencePattern(Component):
    """Fixed diffraction pattern component which is scaled by a 2D affine
    transformation of the form:

        X = d11*x + d12*y
        Y = d21*x + d21*y

    The fixed two-dimensional pattern is defined by a single image which must
    be passed to the ScalableReferencePattern2D constructor, e.g.:

    .. code-block:: ipython

        In [1]: im = load('my_image_data.hdf5')
        In [2] : ref = components.ScalableFixedPattern(im.inav[11,30]))

    Attributes
    ----------

    D : list
        List containing matrix components for affine matrix

    order : 1, 3
        Interpolation order used when applying image transformation

    """

    def __init__(self, signal2D,
                 d11=1., d12=0.,
                 d21=0., d22=1.,
                 t1=0., t2=0.,
                 order=3
                 ):

        Component.__init__(self, ['d11', 'd12',
                                  'd21', 'd22',
                                  't1', 't2'])

        self._whitelist['signal2D'] = ('init,sig', signal2D)
        self.signal = signal2D
        self.order = order
        self.d11.value = d11
        self.d12.value = d12
        self.d21.value = d21
        self.d22.value = d22
        self.t1.value = t1
        self.t2.value = t2

    def function(self, x, y):

        signal2D = self.signal.data
        order = self.order
        d11 = self.d11.value
        d12 = self.d12.value
        d21 = self.d21.value
        d22 = self.d22.value
        t1 = self.t1.value
        t2 = self.t2.value

        D = np.array([[d11, d12, t1],
                      [d21, d22, t2],
                      [0., 0., 1.]])

        shifty, shiftx = np.array(signal2D.shape[:2]) / 2

        shift = tf.SimilarityTransform(translation=[-shiftx, -shifty])
        tform = tf.AffineTransform(matrix=D)
        shift_inv = tf.SimilarityTransform(translation=[shiftx, shifty])

        transformed = tf.warp(signal2D, (shift + (tform + shift_inv)).inverse,
                              order=order)

        return transformed

    def construct_displacement_gradient(self):
        """Construct a map of the displacement gradient tensor at each
        navigation position as determined by fitting an affine transformed
        reference pattern.

        Returns
        -------

        D : DisplacementGradientMap
            Signal containing the displacement gradient tensor at each
            navigation postion.

        """
        D = DisplacementGradientMap(np.ones(np.append(self.d11.map.shape,
                                                      (3, 3))))

        D.data[:, :, 0, 0] = self.d11.map['values']
        D.data[:, :, 1, 0] = self.d12.map['values']
        D.data[:, :, 2, 0] = 0.
        D.data[:, :, 0, 1] = self.d21.map['values']
        D.data[:, :, 1, 1] = self.d22.map['values']
        D.data[:, :, 2, 1] = 0.
        D.data[:, :, 0, 2] = 0.
        D.data[:, :, 1, 2] = 0.
        D.data[:, :, 2, 2] = 1.

        return D
