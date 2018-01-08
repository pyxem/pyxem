# -*- coding: utf-8 -*-
# Copyright 2017 The pyXem developers
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

from hyperspy.signals import BaseSignal
from transforms3d.euler import euler2axangle
import numpy as np

"""
Signal class for crystallographic phase and orientation maps.
"""

def euler2axangle_signal(euler):
    return np.array(euler2axangle(euler[0], euler[1], euler[2])[1])

class CrystallographicMap(BaseSignal):

    def __init__(self, *args, **kwargs):
        BaseSignal.__init__(self, *args, **kwargs)
        self.axes_manager.set_signal_dimension(1)

    def get_phase_map(self):
        """Obtain a map of the best matching phase at each navigation position.

        """
        return self.isig[0].as_signal2D((0,1))

    def get_orientation_image(self):
        """Obtain an orientation image of the rotational angle associated with
        the crystal orientation at each navigation position.

        """
        eulers = self.isig[1:4]
        return eulers.map(euler2axangle_signal, inplace=False)

    def get_correlation_map(self):
        """Obtain a correlation map showing the highest correlation score at
        each navigation position.

        """
        return self.isig[4].as_signal2D((0,1))

    def get_reliability_map(self):
        """Obtain a reliability map showing the difference between the highest
        correlation scor and the next best score at each navigation position.
        """
        return self.isig[5].as_signal2D((0,1))

    def savetxt(self):
        pass
