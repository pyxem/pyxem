# -*- coding: utf-8 -*-
# Copyright 2017-2018 The pyXem developers
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

### Need a reconsideration wrt .get_crystallographic_map() method of MatchResults

def euler2axangle_signal(euler):
    return np.array(euler2axangle(euler[0], euler[1], euler[2])[1])

class CrystallographicMap(BaseSignal):
    """ 
    Stores a map of a SED scan. At each navigtion position there
    will be a phase, three angles, a correlation index and 1/2 reliability 
    scores. See the .get_crystallographic_maps() method
    """
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

    def get_reliability_map_orientation(self):
        """Obtain a reliability map showing the difference between the highest
        correlation score of the most suitable phase
        and the next best score (for the phase) at each navigation position.
        """

        return self.isig[5].as_signal2D((0,1))

    def get_reliability_map_phase(self):
        """Obtain a reliability map showing the difference between the highest
        correlation score of the most suitable phase
        and the next best score from a different phase at each navigation position.
        """
        return self.isig[6].as_signal2D((0,1))

    def get_modal_angles(self):
        """ Obtain the modal angles (and their prevelance)

            Returns
            ------
            scipy.ModeResult object
        """
        raise NotImplementedError("Under construction")
        
        #from scipy import stats
        #size = self.axes_manager.navigation_shape[0] * \
                   #self.axes_manager.navigation_shape[1]
        #return(stats.mode(self.isig[1:4,0].data.reshape(size,3)))


    def save_map(self, filename):
        """
        Save map so that in a format such that it can be imported into MTEX
        http://mtex-toolbox.github.io/
        
        Columns: 
        1 = phase id,
        2-4 = Euler angles in the zxz convention (radians),
        5 = Correlation score (only the best match is saved),
        6 = x co-ord in navigation space, 
        7 = y co-ord in navigation space.
        """
        results_array = np.zeros([0,7])
        for i in range (0, self.data.shape[1]):
            for j in range (0, self.data.shape[0]):
                # XXX
                # This won't work for a multiphase sample, can't guarentee [0] is the best fit
                try:
                    newrow = self.inav[i,j].data[0,0:5]
                except IndexError: #only 1 row at any given data point
                    newrow = self.inav[i,j].data[0:5]
                newrow = np.append(newrow, [i,j])
                results_array = np.vstack([results_array, newrow])
        np.savetxt('{filename}', results_array, delimiter = "\t", newline="\r\n")
