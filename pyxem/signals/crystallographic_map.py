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
from tqdm import tqdm

from hyperspy.signals import BaseSignal
from hyperspy.signals import Signal2D

from transforms3d.euler import euler2quat, quat2axangle, euler2axangle
from transforms3d.quaternions import qmult, qinverse

from pyxem.utils.sim_utils import transfer_navigation_axes
from pyxem.utils.sim_utils import transfer_navigation_axes_to_signal_axes

"""
Signal class for crystallographic phase and orientation maps.
"""


def load_mtex_map(filename):
    """
    Loads a crystallographic map saved by previously saved via .save_map()

    Columns:
    1 = phase id,
    2-4 = Euler angles in the zxz convention (radians),
    5 = Correlation score (only the best match is saved),
    6 = x co-ord in navigation space,
    7 = y co-ord in navigation space.

    Parameters
    ----------
    filename : string
        Path to the file to be loaded.

    Returns
    -------
    crystallographic_map : CrystallographicMap
        Crystallographic map loaded from the specified file.

    """
    load_array = np.loadtxt(filename, delimiter='\t')
    # Add one for zero indexing
    x_max = np.max(load_array[:, 5]).astype(int) + 1
    y_max = np.max(load_array[:, 6]).astype(int) + 1
    crystal_data = np.empty((y_max, x_max, 3), dtype='object')
    for y in range(y_max):
        for x in range(x_max):
            load_index = y * x_max + x
            crystal_data[y, x] = [
                load_array[load_index, 0],
                load_array[load_index, 1:4],
                {'correlation': load_array[load_index, 4]}]
    return CrystallographicMap(crystal_data)


def _euler2axangle_signal(euler):
    """Converts an Euler triple into the axis-angle representation.

    Parameters
    ----------
    euler : np.array()
        Euler angles for a rotation.

    Returns
    -------
    asangle : np.array()
        Axis-angle representation of the rotation.

    """
    euler = euler[0]  # TODO: euler is a 1-element ndarray(dtype=object) with a tuple
    return np.rad2deg(euler2axangle(euler[0], euler[1], euler[2])[1])


def _distance_from_fixed_angle(angle, fixed_angle):
    """Designed to be mapped, this function finds the smallest rotation between
    two rotations. It assumes a no-symmettry system.

    The algorithm involves converting angles to quarternions, then finding the
    appropriate misorientation. It is tested against the slower more complete
    version finding the joining rotation.

    Parameters
    ----------
    angle : np.array()
        Euler angles representing rotation of interest.
    fixed_angle : np.array()
        Euler angles representing fixed reference rotation.

    Returns
    -------
    theta : np.array()
        Rotation angle between angle and fixed_angle.

    """
    angle = angle[0]
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


def _metric_from_dict(metric_dict, metric):
    """ Utility function for retrieving an entry in a dictionary. Used to map
    over dicts in signal space.

    Parameters
    ----------
    metric_dict : dict
        Dictionary to retrieve entry from
    metrics : string
        Name of the entry

    Returns
    -------
    entry
        Dictionary entry specified by metric.

    """
    return metric_dict[0][metric]


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

    Attributes
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
        phase_map = transfer_navigation_axes_to_signal_axes(phase_map, self)
        phase_map.change_dtype(np.int)

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
        orientation_map = transfer_navigation_axes_to_signal_axes(orientation_map, self)
        # Since vector matching results returns in object form, eulers inherits
        # it.
        orientation_map.change_dtype('float')

        return orientation_map

    def get_metric_map(self, metric):
        """Obtain a map of an indexation / matching metric at each navigation
        position.

        Parameters
        ----------
        metric : string
            String identifier for the indexation / matching metric to be
            mapped, for template matching valid metrics are {'correlation',
            'orientation_reliability', 'phase_reliability'}. For vector matching
            valid metrics are {'match_rate', 'ehkls', 'total_error',
            'orientation_reliability', 'phase_reliability'}.

        Returns
        -------
        metric_map : Signal2D
            A map of the specified metric at each navigation position.

        Notes
        -----
        For template matching, orientation reliability is given by
            100 * (1 - second_best_correlation/best_correlation)
        and phase reliability is given by
            100 * (1 - second_best_correlation_of_other_phase/best_correlation)

        For vector matching, orientation reliability is given by
            100 * (1 - lowest_error/second_lowest_error)
        and phase reliability is given by
            100 * (1 - lowest_error/lowest_error_of_other_phase)

        """
        if self.method == 'template_matching':
            template_metrics = [
                'correlation',
                'orientation_reliability',
                'phase_reliability',
            ]
            if metric in template_metrics:
                metric_map = self.isig[2].map(
                    _metric_from_dict,
                    metric=metric,
                    inplace=False).as_signal2D((0, 1))

            else:
                raise ValueError("The metric `{}` is not valid for template "
                                 "matching results.".format(metric))

        elif self.method == 'vector_matching':
            vector_metrics = [
                'match_rate',
                'ehkls',
                'total_error',
                'orientation_reliability',
                'phase_reliability',
            ]
            if metric in vector_metrics:
                metric_map = self.isig[2].map(
                    _metric_from_dict,
                    metric=metric,
                    inplace=False).as_signal2D((0, 1))

            else:
                raise ValueError("The metric `{}` is not valid for vector "
                                 "matching results.".format(metric))

        else:
            raise ValueError("The crystallographic mapping method must be "
                             "specified, as an attribute, as either "
                             "template_matching or vector_matching.")

        metric_map = transfer_navigation_axes_to_signal_axes(metric_map, self)

        return metric_map

    def get_modal_angles(self):
        """Obtain the modal angles (and their fractional occurances).

        Returns
        -------
        modal_angles : list
            [modal_angles, fractional_occurance]
        """
        # Extract the euler arrays by flattening, creating a continuous list
        # and converting it to an array again
        euler_array = np.array(self.isig[1].data.ravel().tolist())

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
        results_array[:, 0] = self.isig[0].data.ravel()
        results_array[:, 1:4] = np.array(self.isig[1].data.tolist()).reshape(-1, 3)
        score_metric = 'correlation' if self.method == 'template_matching' else 'match_rate'
        results_array[:, 4] = self.get_metric_map(score_metric).data.ravel()
        x_indices = np.arange(x_size_nav)
        y_indices = np.arange(y_size_nav)
        results_array[:, 5:7] = np.array(np.meshgrid(x_indices, y_indices)).T.reshape(-1, 2)
        np.savetxt(filename, results_array, delimiter="\t", newline="\r\n")
