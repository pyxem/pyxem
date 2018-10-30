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
from hyperspy.signals import Signal2D
from transforms3d.euler import euler2quat, quat2axangle, euler2axangle
from transforms3d.quaternions import qmult, qinverse
import numpy as np
from tqdm import tqdm
from pyxem.utils.sim_utils import carry_through_navigation_calibration

"""
Signal class for crystallographic phase and orientation maps.
"""


def load_mtex_map(filename):
    """
    Loads a crystallographic map saved by previously saved via .save_map()
    """
    load_array = np.loadtxt(filename, delimiter='\t')
    x_max = np.max(load_array[:, 5]).astype(int)
    y_max = np.max(load_array[:, 6]).astype(int)
    # add one for zero indexing
    array = load_array.reshape(x_max + 1, y_max + 1, 7)
    cmap = Signal2D(array).transpose(navigation_axes=2)
    return CrystallographicMap(cmap.isig[:5])  # don't keep x/y


def _euler2axangle_signal(euler):
    """ Find the magnitude of a rotation"""
    return np.array(euler2axangle(euler[0], euler[1], euler[2])[1])


def _distance_from_fixed_angle(angle, fixed_angle):
    """
    Designed to be mapped, this function finds the smallest rotation between
    two rotations. It assumes a no-symmettry system.

    The algorithm involves converting angles to quarternions, then finding the
    appropriate misorientation. It is tested against the slower more complete
    version finding the joining rotation.

    """
    q_data = euler2quat(*np.deg2rad(angle), axes='rzxz')
    q_fixed = euler2quat(*np.deg2rad(fixed_angle), axes='rzxz')
    if np.abs(2 * (np.dot(q_data, q_fixed))**2 - 1) < 1:  # arcos will work
        # https://math.stackexchange.com/questions/90081/quaternion-distance
        theta = np.arccos(2 * (np.dot(q_data, q_fixed))**2 - 1)
    else:  # slower, but also good
        q_from_mode = qmult(qinverse(q_fixed), q_data)
        axis, theta = quat2axangle(q_from_mode)
        theta = np.abs(theta)

    return np.rad2deg(theta)


class CrystallographicMap(BaseSignal):
    """
    Stores a map of a SED scan. At each navigtion position there will be a
    phase, three angles, a correlation index and 1/2 reliability scores. See
    the .get_crystallographic_maps() method
    """

    def __init__(self, *args, **kwargs):
        BaseSignal.__init__(self, *args, **kwargs)
        self.axes_manager.set_signal_dimension(1)

    def get_phase_map(self):
        """Obtain a map of the best matching phase at each navigation position.
        """

        phase_map = self.isig[0].as_signal2D((0, 1))
        phase_map = carry_through_navigation_calibration(phase_map, self)

        return phase_map

    def get_orientation_map(self):
        """Obtain an orientation image of the rotational angle associated with
        the crystal orientation at each navigation position.

        Returns
        -------
        orientation_map : Signal2D
            The rotation angle assocaiated with the orientation at each
            navigation position.

        """
        eulers = self.isig[1:4]
        eulers.map(_euler2axangle_signal, inplace=True)
        orientation_map = eulers.as_signal2D((0, 1))
        orientation_map = carry_through_navigation_calibration(orientation_map, self)

        return orientation_map

    def get_correlation_map(self):
        """Obtain a correlation map showing the highest correlation score at
        each navigation position.

        Returns
        -------
        correlation_map : Signal2D
            The highest correlation score at each navigation position.

        """
        correlation_map = self.isig[4].as_signal2D((0, 1))
        correlation_map = carry_through_navigation_calibration(correlation_map, self)

        return correlation_map

    def get_reliability_map_orientation(self):
        """Obtain an orientation reliability map showing the difference between
        the highest correlation score and the second highest correlation score
        for the same phase oriented differently, at each navigation position.

        Returns
        -------
        reliability_map : Signal2D
            The difference between the highest correlation score and the second
            highest correlation score for the same phase oriented differently at
            each navigation position.

        """
        reliability_map = self.isig[5].as_signal2D((0, 1))
        reliability_map = carry_through_navigation_calibration(reliability_map, self)

        return reliability_map

    def get_reliability_map_phase(self):
        """Obtain a reliability map showing the difference between the highest
        correlation score of the most suitable phase and the next best score
        from a different phase at each navigation position.

        Returns
        -------
        reliability_map : Signal2D
            The difference between the highest correlation score and the second
            highest correlation score for the top scoring phase and the next
            best scoring phase at each navigation position.

        """
        reliability_map = self.isig[6].as_signal2D((0, 1))
        reliability_map = carry_through_navigation_calibration(reliability_map, self)

        return reliability_map

    def get_modal_angles(self):
        """Obtain the modal angles (and their fractional occurances).

        Returns
        -------
        modal_angles : list
            [modal_angles, fractional_occurance]
        """
        element_count = self.data.shape[0] * self.data.shape[1]
        euler_array = self.isig[1:4].data.reshape(element_count, 3)
        pairs, counts = np.unique(euler_array, axis=0, return_counts=True)
        return [pairs[counts.argmax()], counts[counts.argmax()] / np.sum(counts)]

    def get_distance_from_modal_angle(self):
        """Obtain the misorinetation with respect to the modal angle for the
        scan region, at each navigation position.

        NB: This view of the data is typically only useful when the orientation
        spread across the navigation axes is small.

        Returns
        -------
        mode_distance_map : list
            Misorientation with respect to the modal angle at each navigtion
            position.

        See Also
        --------
            method: save_mtex_map
        """
        modal_angle = self.get_modal_angles()[0]
        return self.isig[1:4].map(_distance_from_fixed_angle,
                                  fixed_angle=modal_angle, inplace=False)

    def save_mtex_map(self, filename):
        """
        Save map in a format such that it can be imported into MTEX
        http://mtex-toolbox.github.io/

        Columns:
        1 = phase id,
        2-4 = Euler angles in the zxz convention (radians),
        5 = Correlation score (only the best match is saved),
        6 = x co-ord in navigation space,
        7 = y co-ord in navigation space.
        """
        x_size_nav = self.data.shape[1]
        y_size_nav = self.data.shape[0]
        results_array = np.zeros((x_size_nav * y_size_nav, 7))
        for i in tqdm(range(0, x_size_nav), ascii=True):
            for j in range(0, y_size_nav):
                results_array[(j) * x_size_nav + i] = np.append(self.inav[i, j].data[0:5], [i, j])
        np.savetxt(filename, results_array, delimiter="\t", newline="\r\n")
