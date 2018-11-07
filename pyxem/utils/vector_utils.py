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
    camera_length = np.ones(len(z)) * camera_length
    camera_length = np.reshape(camera_length, (-1, 1))
    # reshape and scale 2D vectors
    k1 = np.hstack((z, camera_length))
    k1_norm = np.sqrt(np.diag(k1.dot(k1.T)))
    k1_norm = k1_norm.reshape((-1, 1)).repeat(3, axis=1)
    k1 = k1 / k1_norm
    # Sort third component
    k0 = np.asarray([0., 0., 1.])
    k0 = k0.reshape((1,-1)).repeat(z, axis=0)
    k = 1. / wavelength * (k1 - k0)

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
    try:
        angle = math.acos(min(1.0, np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    except BaseException:
        angle = math.acos(max(-1.0, np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))

    return angle
