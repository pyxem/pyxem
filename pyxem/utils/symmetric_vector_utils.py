# -*- coding: utf-8 -*-
# Copyright 2016-2024 The pyXem developers
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

import itertools
import numpy as np

##################################################
# Symmetry Methods for Analyzing Glassy Materials
##################################################


def get_angles(angles):
    """This function takes a list of angles and returns the angles between each pair of angles.
    This is useful for finding the angles between three vectors.

    Parameters
    ----------
    angles: np.ndarray
        An array of angles in radians.  This is a 2D array with shape (n, 3) where n is the number
        of combinations and 3 is specific combination of angles to determine the difference between.

    """
    all_angles = np.abs(np.triu(np.subtract.outer(angles, angles)))
    all_angles = all_angles[all_angles != 0]
    all_angles[all_angles > np.pi] = np.pi - np.abs(
        all_angles[all_angles > np.pi] - np.pi
    )
    return all_angles


def get_filtered_combinations(
    pks,
    num,
    radial_index=0,
    angle_index=1,
    intensity_index=None,
    intensity_threshold=None,
    min_angle=None,
    min_k=None,
):
    """
    Creates combinations of `num` peaks but forces at least one of the combinations to have
    an intensity higher than the `intensity_threshold`.
    This filter is useful for finding high intensity features but not losing lower intensity
    paired features which contribute to symmetry etc.

    Parameters
    ----------
    pks : np.ndarray
        The diffraction vectors to be analyzed
    num : int
        The number of peaks to be combined
    radial_index : int, optional
        The index of the radial component of the diffraction vectors, by default 0
    angle_index : int, optional
        The index of the angular component of the diffraction vectors, by default 1
    intensity_index : int, optional
        The index of the intensity component of the diffraction vectors, by default None
    intensity_threshold : float, optional
        The minimum intensity of the diffraction vectors to be considered, by default None
    min_angle: float, optional
        The minimum angle between two diffraction vectors. This ignores small angles which are
        likely to be from the same feature or unphysical.
    min_k : float, optional
        The minimum difference between the radial component of the diffraction vectors to be
        considered from the same feature, by default 0.05
    """
    if intensity_threshold is not None and intensity_index is not None:
        intensity = pks[:, intensity_index]
        intensity_bool = intensity > intensity_threshold
        pks = pks[intensity_bool]
        intensity = intensity[intensity_bool]
    else:
        intensity = np.ones(len(pks))
    angles = pks[:, angle_index]
    k = pks[:, radial_index]

    angle_combos = np.array(list(itertools.combinations(angles, num)))
    k_combos = np.array(list(itertools.combinations(k, num)))
    intensity_combos = np.array(list(itertools.combinations(intensity, num)))

    if len(angle_combos) == 0:
        angle_combos = np.zeros(shape=(0, 3))
    # Filtering out combinations where there are two peaks close to each other
    if min_angle is not None:
        in_range = (
            np.abs(angle_combos[:, :, np.newaxis] - angle_combos[:, np.newaxis, :])
            < min_angle
        )

        above_angle = np.any(
            np.sum(
                in_range,
                axis=1,
            )
            < 2,
            axis=1,
        )  # See if there are two angles that are too close to each other

    else:
        above_angle = True
    # Filtering out combinations of diffraction vectors at different values for k.
    # This could be faster if we sort the values by k first and then only get the combinations
    # of the values within a certain range.
    if min_k is not None and len(k_combos) > 0:
        mean_k = np.mean(k_combos, axis=1)
        abs_k = np.abs(np.subtract(k_combos, mean_k[:, np.newaxis]))
        in_k_range = np.mean(abs_k, axis=1) < min_k
    else:
        in_k_range = True
    in_combos = above_angle * in_k_range
    if len(angle_combos) == 0:
        combos = angle_combos
        combos_k = []
        combo_inten = []
    elif np.all(in_combos):
        combos = angle_combos
        combos_k = np.mean(k_combos, axis=1)
        combo_inten = np.mean(intensity_combos, axis=1)
    else:
        combos = angle_combos[in_combos]
        combos_k = np.mean(k_combos[in_combos], axis=1)
        combo_inten = np.mean(intensity_combos[in_combos], axis=1)
    return combos, combos_k, combo_inten


def get_three_angles(
    pks,
    k_index=0,
    angle_index=1,
    intensity_index=2,
    intensity_threshold=None,
    accept_threshold=0.05,
    min_k=0.05,
    min_angle=None,
):
    """
    This function takes the angle between three points and determines the angle between them,
    returning the angle if it is repeated using the `accept_threshold` to measure the acceptable
    difference between angle a and angle b
           o
           |
           |_   angle a
           | |
           x--------o
           |_|
           |    angle b
           |
           o

    Parameters
    ----------
    pks : np.ndarray
        The diffraction vectors to be analyzed
    k_index : int, optional
        The index of the radial component of the diffraction vectors, by default 0
    angle_index : int, optional
        The index of the angular component of the diffraction vectors, by default 1
    intensity_index : int, optional
        The index of the intensity component of the diffraction vectors, by default 2
    intensity_threshold : float, optional
        The minimum intensity of the diffraction vectors to be considered, by default None
    accept_threshold : float, optional
        The maximum difference between angle a and angle b to be considered the same angle,
        by default 0.05
    min_k : float, optional
        The minimum difference between the radial component of the diffraction vectors to be
        considered from the same feature, by default 0.05
    min_angle: float, optional
        The minimum angle between two diffraction vectors. This ignores small angles which are
        likely to be from the same feature or unphysical.

    Returns
    -------
    three_angles : np.ndarray
        An array of angles between three diffraction vectors.  The columns are:
        [k, delta phi, min-angle, intensity, reduced-angle]
    """
    three_angles = []
    combos, combo_k, combo_inten = get_filtered_combinations(
        pks,
        3,
        radial_index=k_index,
        angle_index=angle_index,
        intensity_index=intensity_index,
        intensity_threshold=intensity_threshold,
        min_angle=min_angle,
        min_k=min_k,
    )
    for c, k, inten in zip(combos, combo_k, combo_inten):
        angular_seperations = get_angles(c)
        min_ind = np.argmin(angular_seperations)
        min_sep = angular_seperations[min_ind]
        angular_seperations = np.delete(angular_seperations, min_ind)
        in_range = np.abs((angular_seperations - min_sep)) < accept_threshold
        is_symetric = np.any(in_range)
        if is_symetric:
            # take the average of the two smaller angles
            avg_sep = np.mean((np.array(angular_seperations)[in_range][0], min_sep))
            min_angle = np.min(c)
            num_times = np.round(min_angle / min_sep)
            three_angles.append(
                [
                    k,
                    avg_sep,
                    min_angle,
                    inten,
                    np.abs(min_angle - (num_times * min_sep)),
                ]
            )
    if len(three_angles) == 0:
        three_angles = np.empty((0, 5))
    return np.array(three_angles)
