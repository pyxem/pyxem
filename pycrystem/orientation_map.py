# -*- coding: utf-8 -*-
# Copyright 2017 The PyCrystEM developers
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

from hyperspy.signals import BaseSignal
import numpy as np

"""
Signal class for crystallographic orientation maps.
"""


class OrientationMap(BaseSignal):
    """
    Signal class for crystallographic orientation maps.
    """
    _signal_type = "tensor_field"

    def __init__(self, *args, **kwargs):
        BaseSignal.__init__(self, *args, **kwargs)
        # Check that the signal dimensions are (3,3) for it to be a valid
        # TensorField

    def get_angle_map(self):
        """Obtain an orientation map specifed by the magnitude of the rotation
        angle at each navigation position.

        Returns
        -------
        angle_map : Signal2D
            Orientation map specifying the magnitude of the rotation angle at
            each navigation position.

        """
        best_angle = []
        for i in np.arange(self.shape[0]):
            for j in np.arange(self.shape[1]):
                best_angle.append(max(self[i,j]['gr'], key=self[i,j]['gr'].get))
        angle = []
        for euler in best_angle:
            angle.append(euler2axangle(euler[0], euler[1], euler[2])[1])
        angles = np.asarray(angle)
        angle_map = BaseSignal(angles.reshape(self.shape).T)
        angle_map.axes_manager.set_signal_dimension(0)
        angle_map.data = angle_map.data/pi * 180
        return angle_map.as_signal2D((0,1))

    def get_correlation_map(self):
        """Obtain an quality map for a pattern matching based indexation by
        plotting the correlation score at each navigation position.

        Returns
        -------

        """
        pass

    def get_reliability_map(self):
        pass

    def get_grains(self):
        pass

    def save_txt(self):
        """Save the orientation map as a txt file compatible with other
        packages.
        """
        pass
