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

import numpy as np
from tqdm import tqdm

from hyperspy.signals import BaseSignal
from hyperspy.signals import Signal2D

from transforms3d.euler import euler2quat, quat2axangle, euler2axangle
from transforms3d.quaternions import qmult, qinverse

from pyxem.utils.sim_utils import transfer_navigation_axes

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
    """Crystallographic mapping results containing the best matching crystal
    phase and orientation at each navigation position with associated metrics.

    The Signal at each navigation position is an array of,

                    [phase, np.array((z,x,z)), dict(metrics)]

    which defines the phase, orientation as Euler angles in the zxz convention
    and metrics associated with the indexation / matching.

    Metrics depend on the method used (template matching vs. vector matching) to
    obtain the crystallographic map.

        'correlation'
        'match_rate'
        'total_error'
        'orientation_reliability'
        'phase_reliability'

    Atrributes
    ----------
    method : string
        Method used to obtain crystallographic mapping results, may be
        'template_matching' or 'vector_matching'.
    """
    
    def __init__(self, *args, **kwargs):
        BaseSignal.__init__(self, *args, **kwargs)
        self.axes_manager.set_signal_dimension(1)
        self.method = None

    def get_phase_map(self):
        """Obtain a map of the best matching phase at each navigation position.
        """
        phase_map = self.isig[0].as_signal2D((0, 1))
        phase_map = transfer_navigation_axes(phase_map, self)

        return phase_map

    def get_orientation_map(self):
        """Obtain a map of the rotational angle associated with the best
        matching crystal orientation at each navigation position.

        Returns
        -------
        orientation_map : Signal2D
            The rotation angle assocaiated with the orientation at each
            navigation position.

        """
        eulers = self.isig[1]
        eulers.map(_euler2axangle_signal, inplace=True)
        orientation_map = eulers.as_signal2D((0, 1))
        orientation_map = transfer_navigation_axes(orientation_map, self)

        return orientation_map

    def get_metric_map(self, metric):
        """Obtain a map of an indexation / matching metric at each navigation
        position.

        Parameters
        ----------
        metric : string
            String identifier for the indexation / matching metric to be
            mapped, for template_matching valid metrics are
                'correlation'
                'orientation_reliability'
                'phase_reliability'
            For vector_matching, valid metrics are;
                'match_rate'
                'ehkls'
                'total_error'
                'orientation_reliability'
                'phase_reliability'

        Returns
        -------
        metric_map : Signal2D
            A map of the specified metric at each navigation position.

        """
        if self.method=='template_matching':
            template_metrics = {
                'correlation': correlation,
                'orientation_reliability': orientation_reliability,
                'phase_reliability' : phase_reliability
            }
            if metric in metric_dict:
                metric_map = self.isig[2][metric].as_signal2D((0, 1))

            else:
                raise ValueError("The metric `{}` is not valid for template "
                                 "matching results. ")

        elif self.method=='vector_matching':
            vector_metrics = {
                'match_rate': match_rate,
                'ehkls': ehkls,
                'total_error': total_error,
                'orientation_reliability': orientation_reliability,
                'phase_reliability' : phase_reliability
            }
            if metric in metric_dict:
                metric_map = self.isig[2][metric].as_signal2D((0, 1))

            else:
                raise ValueError("The metric `{}` is not valid for vector "
                                 "matching results. ")

        else:
            raise ValueError("The crystallographic mapping method must be "
                             "specified, as an attribute, as either "
                             "template_matching or vector_matching.")

        metric_map = transfer_navigation_axes(metric_map, self)

        return metric_map

    def get_modal_angles(self):
        """Obtain the modal angles (and their fractional occurances).

        Returns
        -------
        modal_angles : list
            [modal_angles, fractional_occurance]
        """
        element_count = self.data.shape[0] * self.data.shape[1]
        euler_array = self.isig[1]

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
        return self.isig[1].map(_distance_from_fixed_angle,
                                fixed_angle=modal_angle, inplace=False)

    def save_mtex_map(self, filename):
        """Save map in a format such that it can be imported into MTEX
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
