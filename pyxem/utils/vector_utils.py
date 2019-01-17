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
import math

from transforms3d.axangles import axangle2mat


def detector_to_fourier(z, wavelength, camera_length):
    """Maps two-dimensional Cartesian coordinates in the detector plane to
    three-dimensional coordinates in reciprocal space.

    Parameters
    ----------
    z : np.array()
        Array of Cartesian coordinates in the detector plane.
    wavelength : float
        Electron wavelength in Angstroms.
    camera_length : float
        Camera length in metres.

    Returns
    -------
    k : np.array()
        Array of Cartesian coordinates in reciprocal space.

    """
    # specify experimental parameters
    # TODO: z is a 1-element ndarray(dtype='object') containing the coordinates
    z = z[0]
    camera_length = np.ones(len(z)) * camera_length
    camera_length = np.reshape(camera_length, (-1, 1))
    # reshape and scale 2D vectors
    k = np.hstack((z, camera_length))
    # Compute norm of each row in k1
    k_norm = np.sqrt(np.sum(k * k, 1))
    k /= k_norm[:, np.newaxis]
    # TODO: Why do we subtract 1 (or k_norm before normalizing) from the camera length component (correct?)
    k[:, 2] -= 1
    k *= 1 / wavelength
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
    norms = []
    for i in z:
        norms.append(np.linalg.norm(i))
    return np.asarray(norms)


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
    """Calculates the rotation matrix between two experimentally measured
    diffraction vectors and the corresponding vectors in a reference structure.

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
    ref_nv = np.cross(ref_k1, ref_k2)
    k_nv = np.cross(k1, k2)
    # avoid 0 degree including angle
    if min(norm(ref_nv), norm(k_nv)) == 0.:
        R = np.identity(3)
    else:
        axis = np.cross(ref_nv, k_nv)
        angle = np.rad2deg(acos(ref_nv.dot(k_nv) / (norm(ref_nv) * norm(k_nv))))
        R1 = axangle2mat(axis, angle)
        # rotate ref_q1,2 plane to q1,2 plane
        rot_ref_k1, rot_ref_k2 = R1.dot(ref_k1), R1.dot(ref_k2)
        # avoid math domain error
        cos1 = max(min(k1.dot(rot_ref_k1) / (np.linalg.norm(rot_ref_k1) * np.linalg.norm(k1)), 1.), -1.)
        cos2 = max(min(k2.dot(rot_ref_k2) / (np.linalg.norm(rot_ref_k2) * np.linalg.norm(k2)), 1.), -1.)
        angle1 = np.rad2deg(acos(cos1))
        angle2 = np.rad2deg(acos(cos2))
        angle = (angle1 + angle2) / 2.
        axis = np.cross(rot_ref_k1, k1)
        R2 = axangle2mat(axis, angle)
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
