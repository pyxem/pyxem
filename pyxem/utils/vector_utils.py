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


def detector_px_to_3D_kspace(peak_coord, ai):
    """Converts the detector 2d coordinate, in pixel units, to the respective 3D
    coordinate in the kspace, using the pyFAI Geometry object.

    Parameters
    ----------
    peak_coord: np.array
        An array with the diffraction vectors of a single scanning coordinate,
        in pixel units of the detector.
    ai: pyFAI.azimuthalIntegrator.AzimuthalIntegrator object
        A pyFAI Geometry object, containing all the detector geometry parameters.

    Returns
    ----------
    g_xyz: np.array
        Array composed of [g_x, g_y, g_z] values for the peaks in the scanning
        coordinate, changed from pixel units to Angstrom^-1.
    """
    #Get the geometry parameters necessary for this transormation (all in metres)
    det2sample_dist = ai.dist
    wavelength = ai.wavelength

    #Transform each pixel unit to the actual disctance in metres, using the
    #pyFAI module function
    if peak_coord.shape == (1,) and peak_coord.dtype == 'object':
        # From ragged array
        peak_coord = peak_coord[0]
    #x= peak_coord[:,0]
    #y= peak_coord[:,1]
    zyx =  ai.calc_pos_zyx(d1=peak_coord[:,1], d2=peak_coord[:,0])

    #Get the polar coordinate angles, in a 3D Edwald circunference:
    #Note: zyx[2]==x, zyx[1]==y
    #Vector moduli 'r' from the beam centre to the coordinate at the detector
    #for each peak.
    r = np.sqrt(zyx[2]**2 + zyx[1]**2)
    #Phi angles (from z axis) for each peak:
    phi = np.arctan(r/det2sample_dist)
    #2 Theta angles (between x and y axis) for each peak. Use arctan2 to get the
    #right quadrant sign:
    two_theta = np.arctan2(np.sqrt(zyx[1]**2), zyx[2])
    #TODO: Account for slight difference in the z-axis when computing angles
    # (stored in the zyx[0])

    #Convert each x and y to the respective gx, gy and gz values, using 3D
    #geometry. Multiply by the pixel sign:
    sin_phi = np.sin(phi) #For memory saving
    gx = (1/wavelength)*sin_phi*np.cos(two_theta)
    gy = (1/wavelength)*sin_phi*np.sin(two_theta)
    gz = (1/wavelength)*(np.cos(phi) - 1)

    #Append the reciprocal vectors in one single array, while flipping the
    #vector form, resembling the input array. Convert from m-1 to A-1:
    g_xyz = np.hstack((gx[:,np.newaxis], gy[:,np.newaxis], gz[:,np.newaxis])) * 1e-10

    return g_xyz


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


def normalize_or_zero(v):
    """Normalize `v`, or return the vector directly if it has zero length.

    Parameters
    ----------
    v : np.array()
        Single vector or array of vectors to be normalized.
    """
    norms = np.linalg.norm(v, axis=-1)
    nonzero_mask = norms > 0
    if np.any(nonzero_mask):
        v[nonzero_mask] /= norms[nonzero_mask].reshape(-1, 1)


def get_rotation_matrix_between_vectors(from_v1, from_v2, to_v1, to_v2):
    """Calculates the rotation matrix from one pair of vectors to the other.
    Handles multiple to-vectors from a single from-vector.

    Find `R` such that `v_to = R @ v_from`.

    Parameters
    ----------
    from_v1, from_v2 : np.array()
        Vector to rotate _from_.
    to_v1, to_v2 : np.array()
        Nx3 array of vectors to rotate _to_.

    Returns
    -------
    R : np.array()
        Nx3x3 list of rotation matrices between the vector pairs.
    """
    # Find normals to rotate around
    plane_normal_from = np.cross(from_v1, from_v2, axis=-1)
    plane_normal_to = np.cross(to_v1, to_v2, axis=-1)
    plane_common_axes = np.cross(plane_normal_from, plane_normal_to, axis=-1)

    # Try to remove normals from degenerate to-planes by replacing them with
    # the rotation axes between from and to vectors.
    to_degenerate = np.isclose(np.sum(np.abs(plane_normal_to), axis=-1), 0.0)
    plane_normal_to[to_degenerate] = np.cross(from_v1, to_v1[to_degenerate], axis=-1)
    to_degenerate = np.isclose(np.sum(np.abs(plane_normal_to), axis=-1), 0.0)
    plane_normal_to[to_degenerate] = np.cross(from_v2, to_v2[to_degenerate], axis=-1)

    # Normalize the axes used for rotation
    normalize_or_zero(plane_normal_to)
    normalize_or_zero(plane_common_axes)

    # Create rotation from-plane -> to-plane
    common_valid = ~np.isclose(np.sum(np.abs(plane_common_axes), axis=-1), 0.0)
    angles = get_angle_cartesian_vec(np.broadcast_to(plane_normal_from, plane_normal_to.shape), plane_normal_to)
    R1 = np.empty((angles.shape[0], 3, 3))
    if np.any(common_valid):
        R1[common_valid] = np.array([axangle2mat(axis, angle, is_normalized=True)
                                     for axis, angle in zip(plane_common_axes[common_valid], angles[common_valid])])
    R1[~common_valid] = np.identity(3)

    # Rotate from-plane into to-plane
    rot_from_v1 = np.matmul(R1, from_v1)
    rot_from_v2 = np.matmul(R1, from_v2)

    # Create rotation in the now common plane

    # Find the average angle
    angle1 = get_angle_cartesian_vec(rot_from_v1, to_v1)
    angle2 = get_angle_cartesian_vec(rot_from_v2, to_v2)
    angles = 0.5 * (angle1 + angle2)
    # Negate angles where the rotation where the rotation axis points the
    # opposite way of the to-plane normal. Einsum gives list of dot
    # products.
    neg_angle_mask = np.einsum('ij,ij->i', np.cross(rot_from_v1, to_v1, axis=-1), plane_normal_to) < 0
    np.negative(angles, out=angles, where=neg_angle_mask)

    # To-plane normal still the same
    R2 = np.array([axangle2mat(axis, angle, is_normalized=True) for axis, angle in zip(plane_normal_to, angles)])

    # Total rotation is the combination of to plane R1 and in plane R2
    R = np.matmul(R2, R1)

    return R


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


def get_angle_cartesian_vec(a, b):
    """Compute the angles between two lists of vectors in a cartesian
    coordinate system.

    Parameters
    ----------
    a, b : np.array()
        The two lists of directions to compute the angle between in Nx3 float
        arrays.

    Returns
    -------
    angles : np.array()
        List of angles between `a` and `b` in radians.
    """
    if a.shape != b.shape:
        raise ValueError('The shape of a {} and b {} must be the same.'.format(a.shape, b.shape))

    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    denom_nonzero = denom != 0.0
    angles = np.zeros(a.shape[0])
    angles[denom_nonzero] = np.arccos(np.clip(
        np.sum(a[denom_nonzero] * b[denom_nonzero], axis=-1) / denom[denom_nonzero], -1.0, 1.0)).ravel()
    return angles


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
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return math.acos(max(-1.0, min(1.0, np.dot(a, b) / denom)))
