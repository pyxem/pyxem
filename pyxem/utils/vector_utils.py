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
import math

from transforms3d.axangles import axangle2mat


def detector_to_fourier(k_xy, wavelength, camera_length):
    """Maps two-dimensional Cartesian coordinates in the detector plane to
    three-dimensional coordinates in reciprocal space, with origo in [000].

    The detector uses a left-handed coordinate system, while the reciprocal
    space uses a right-handed coordinate system.

    Parameters
    ----------
    k_xy : np.array()
        Cartesian coordinates in detector plane, in reciprocal Ångström.
    wavelength : float
        Electron wavelength in Ångström.
    camera_length : float
        Camera length in metres.

    Returns
    -------
    k : np.array()
        Array of Cartesian coordinates in reciprocal space relative to [000].

    """

    if k_xy.shape == (1,) and k_xy.dtype == 'object':
        # From ragged array
        k_xy = k_xy[0]

    # The calibrated positions of the diffraction spots are already the x and y
    # coordinates of the k vector on the Ewald sphere. The radius is given by
    # the wavelength. k_z is calculated courtesy of Pythagoras, then offset by
    # the Ewald sphere radius.

    k_z = np.sqrt(1 / (wavelength**2) - np.sum(k_xy**2, axis=1)) - 1 / wavelength

    # Stack the xy-vector and the z vector to get the full k
    k = np.hstack((k_xy, k_z[:, np.newaxis]))
    return k


def calculate_norms(z):
    """Calculates the norm of an array of cartesian vectors. For use with map().

    Parameters
    ----------
    z : np.array()
        Array of cartesian vectors.

    Returns
    -------
    norms : np.array()
        Array of vector norms.
    """
    return np.linalg.norm(z, axis=1)


def calculate_norms_ragged(z):
    """Calculates the norm of an array of cartesian vectors. For use with map()
    when applied to a ragged array.

    Parameters
    ----------
    z : np.array()
        Array of cartesian vectors.

    Returns
    -------
    norms : np.array()
        Array of vector norms.
    """
    norms = []
    for i in z[0]:
        norms.append(np.linalg.norm(i))
    return np.asarray(norms)


def get_rotation_matrix_between_vectors(k1, k2, ref_k1, ref_k2):
    """Calculates the rotation matrix to two experimentally measured
    diffraction vectors from the corresponding vectors in a reference structure.

    Parameters
    ----------
    k1 : np.array()
        Experimentally measured scattering vector 1.
    k2 : np.array()
        Experimentally measured scattering vector 2.
    ref_k1 : np.array()
        Reference scattering vector 1.
    ref_k2 : np.array()
        Reference scattering vector 2.

    Returns
    -------
    R : np.array()
        Rotation matrix describing transformation from experimentally measured
        scattering vectors to equivalent reference vectors.
    """
    ref_plane_normal = np.cross(ref_k1, ref_k2)
    k_plane_normal = np.cross(k1, k2)
    axis = np.cross(ref_plane_normal, k_plane_normal)
    # Avoid 0 degree including angle
    if np.linalg.norm(axis) == 0:
        R = np.identity(3)
    else:
        # Rotate ref plane into k plane
        angle = get_angle_cartesian(ref_plane_normal, k_plane_normal)
        R1 = axangle2mat(axis, angle)
        rot_ref_k1, rot_ref_k2 = R1.dot(ref_k1), R1.dot(ref_k2)

        # Rotate ref vectors in plane
        angle1 = get_angle_cartesian(k1, rot_ref_k1)
        angle2 = get_angle_cartesian(k2, rot_ref_k2)
        angle = 0.5 * (angle1 + angle2)
        # k plane normal still the same
        R2 = axangle2mat(k_plane_normal, angle)

        # Total rotation is the combination of to plane R1 and in plane R2
        R = R2.dot(R1)

    return R


def get_indices_from_distance_matrix(distances, distance_threshold):
    """Checks if the distances from one vector in vlist to all other vectors in
    vlist are larger than distance_threshold.

    Parameters
    ----------
    distances : np.array()
        Array of distances between vectors.
    distance_threshold : float
        The distance threshold for a vector to be retained.

    Returns
    -------
    new_indices : np.array()
        Array of vectors with distances greater than the threshold.
    """
    new_indices = []
    l = np.shape(distances)[0]
    for i in range(np.shape(distances)[1]):
        if (np.sum(distances[:, i] > distance_threshold) == l):
            new_indices = np.append(new_indices, i)
    return np.array(new_indices, dtype=np.int)


def get_npeaks(found_peaks):
    """Returns the number of entries in a list. For use with map().

    Parameters
    ----------
    found_peaks : np.array()
        Array of found peaks.

    Returns
    -------
    len : int
        The number of peaks in the array.
    """
    return len(found_peaks[0])


def get_angle_cartesian(a, b):
    """Compute the angle between two vectors in a cartesian coordinate system.

    Parameters
    ----------
    a, b : array-like with 3 floats
        The two directions to compute the angle between.

    Returns
    -------
    angle : float
        Angle between `a` and `b` in radians.
    """
    determinant = np.linalg.norm(a) * np.linalg.norm(b)
    if determinant == 0:
        return 0.0
    return math.acos(max(-1.0, min(1.0, np.dot(a, b) / determinant)))
