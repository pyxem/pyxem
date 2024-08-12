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

"""Utils for operating on 2D Diffraction Patterns."""

import itertools
from copy import deepcopy
import math

import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from transforms3d.axangles import axangle2mat

__all__ = [
    "detector_to_fourier",
    "calculate_norms",
    "calculate_norms_ragged",
    "filter_vectors_ragged",
    "filter_vectors_edge_ragged",
    "normalize_or_zero",
    "get_rotation_matrix_between_vectors",
    "get_npeaks",
    "get_angle_cartesian_vec",
    "get_angle_cartesian",
    "filter_vectors_near_basis",
    "vectors_to_polar",
    "to_cart_three_angles",
    "polar_to_cartesian",
    "get_vectors_mesh",
    "get_angles",
    "get_filtered_combinations",
    "convert_to_markers",
    "points_to_polygon",
    "points_to_poly_collection",
    "column_mean",
    "vectors2image",
    "get_three_angles",
]


def detector_to_fourier(k_xy, wavelength, camera_length):
    """Maps two-dimensional Cartesian coordinates in the detector plane to
    three-dimensional coordinates in reciprocal space, with origo in [000].

    The detector uses a left-handed coordinate system, while the reciprocal
    space uses a right-handed coordinate system.

    Parameters
    ----------
    k_xy : numpy.ndarray
        Cartesian coordinates in detector plane, in reciprocal Ångström.
    wavelength : float
        Electron wavelength in Ångström.
    camera_length : float
        Camera length in metres.

    Returns
    -------
    k : numpy.ndarray
        Array of Cartesian coordinates in reciprocal space relative to [000].

    """

    if k_xy.shape == (1,) and k_xy.dtype == "object":
        # From ragged array
        k_xy = k_xy

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
    z : numpy.ndarray
        Array of cartesian vectors.

    Returns
    -------
    norms : numpy.ndarray
        Array of vector norms.
    """
    return np.linalg.norm(z, axis=1)


def calculate_norms_ragged(z):
    """Calculates the norm of an array of cartesian vectors. For use with map()
    when applied to a ragged array.

    Parameters
    ----------
    z : numpy.ndarray
        Array of cartesian vectors.

    Returns
    -------
    norms : numpy.ndarray
        Array of vector norms.
    """
    norms = []
    for i in z:
        norms.append(np.linalg.norm(i))
    return np.asarray(norms)


def filter_vectors_ragged(z, min_magnitude, max_magnitude, columns=[0, 1]):
    """Filters the diffraction vectors to accept only those with magnitudes
    within a user specified range.

    Parameters
    ----------
    min_magnitude : float
        Minimum allowed vector magnitude.
    max_magnitude : float
        Maximum allowed vector magnitude.

    Returns
    -------
    filtered_vectors : numpy.ndarray
        Diffraction vectors within allowed magnitude tolerances.
    """
    # Calculate norms
    norms = np.linalg.norm(z[:, columns], axis=1)
    # Filter based on norms
    norms[norms < min_magnitude] = 0
    norms[norms > max_magnitude] = 0
    filtered_vectors = z[np.where(norms)]

    return filtered_vectors


def filter_vectors_edge_ragged(z, x_threshold, y_threshold):
    """Filters the diffraction vectors to accept only those not within a user
    specified proximity to detector edge.

    Parameters
    ----------
    x_threshold : float
        Maximum x-coordinate in calibrated units.
    y_threshold : float
        Maximum y-coordinate in calibrated units.

    Returns
    -------
    filtered_vectors : numpy.ndarray
        Diffraction vectors within allowed tolerances.
    """
    # Filter x / y coordinates
    z[np.absolute(z.T[0]) > x_threshold] = 0
    z[np.absolute(z.T[1]) > y_threshold] = 0
    filtered_vectors = z[np.where(z.T[0])]

    return filtered_vectors


def normalize_or_zero(v):
    """Normalize `v`, or return the vector directly if it has zero length.

    Parameters
    ----------
    v : numpy.ndarray
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
    from_v1, from_v2 : numpy.ndarray
        Vector to rotate _from_.
    to_v1, to_v2 : numpy.ndarray
        Nx3 array of vectors to rotate _to_.

    Returns
    -------
    R : numpy.ndarray
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
    angles = get_angle_cartesian_vec(
        np.broadcast_to(plane_normal_from, plane_normal_to.shape), plane_normal_to
    )
    R1 = np.empty((angles.shape[0], 3, 3))
    if np.any(common_valid):
        R1[common_valid] = np.array(
            [
                axangle2mat(axis, angle, is_normalized=True)
                for axis, angle in zip(
                    plane_common_axes[common_valid], angles[common_valid]
                )
            ]
        )
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
    neg_angle_mask = (
        np.einsum("ij,ij->i", np.cross(rot_from_v1, to_v1, axis=-1), plane_normal_to)
        < 0
    )
    np.negative(angles, out=angles, where=neg_angle_mask)

    # To-plane normal still the same
    R2 = np.array(
        [
            axangle2mat(axis, angle, is_normalized=True)
            for axis, angle in zip(plane_normal_to, angles)
        ]
    )

    # Total rotation is the combination of to plane R1 and in plane R2
    R = np.matmul(R2, R1)

    return R


def get_npeaks(found_peaks):
    """Returns the number of entries in a list. For use with map().

    Parameters
    ----------
    found_peaks : numpy.ndarray
        Array of found peaks.

    Returns
    -------
    len : int
        The number of peaks in the array.
    """
    return len(found_peaks)


def get_angle_cartesian_vec(a, b):
    """Compute the angles between two lists of vectors in a cartesian
    coordinate system.

    Parameters
    ----------
    a, b : numpy.ndarray
        The two lists of directions to compute the angle between in Nx3 float
        arrays.

    Returns
    -------
    angles : numpy.ndarray
        List of angles between `a` and `b` in radians.
    """
    if a.shape != b.shape:
        raise ValueError(
            "The shape of a {} and b {} must be the same.".format(a.shape, b.shape)
        )

    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    denom_nonzero = denom != 0.0
    angles = np.zeros(a.shape[0])
    angles[denom_nonzero] = np.arccos(
        np.clip(
            np.sum(a[denom_nonzero] * b[denom_nonzero], axis=-1) / denom[denom_nonzero],
            -1.0,
            1.0,
        )
    ).ravel()
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


def filter_vectors_near_basis(vectors, basis, columns=[0, 1], distance=None):
    """
    Filter an array of vectors to only the list of closest vectors
    to some set of basis vectors.  Only vectors within some `distance`
    are considered.  If no vector is within the `distance` np.nan is
    returned for that vector.

    Parameters
    ----------
    vectors: array-like
        A two dimensional array of vectors where each row identifies a new vector

    basis: array-like
        A two dimensional array of vectors where each row identifies a vector.

    columns: list
        A list of columns to consider when comparing vectors.  The default is [0,1]

    Returns
    -------
    closest_vectors: array-like
        An array of vectors which are the closest to the basis considered.
    """
    vectors = vectors[:, columns]
    if basis.ndim == 1:  # possible bug in hyperspy map with iterating basis
        basis = basis.reshape(1, -1)
    if len(vectors) == 0:
        vectors = np.empty(basis.shape)
        vectors[:, :] = np.nan
        return vectors
    distance_mat = cdist(vectors, basis)
    closest_index = np.argmin(distance_mat, axis=0)
    min_distance = distance_mat[closest_index, np.arange(len(basis), dtype=int)]
    closest_vectors = vectors[closest_index]
    if distance is not None:
        if closest_vectors.dtype == int:
            closest_vectors = closest_vectors.astype(float)

        closest_vectors[min_distance > distance, :] = np.nan
    return closest_vectors


def _reverse_pos(peaks, ind=2):
    """Reverses the position of the peaks in the signal.

    This is useful for plotting the peaks on top of a hyperspy signal as the returned peaks are in
    reverse order to the hyperspy markers. Only points up to the index are reversed.

    Parameters
    ----------
    peaks : numpy.ndarray
        Array of peaks to be reversed.
    ind : int
        The index of the position to be reversed.

    """
    new_data = np.empty(peaks.shape[:-1] + (2,))
    for i in range(ind):
        new_data[..., (-i - 1)] = peaks[..., i]
    return new_data


def cluster(
    data, method, columns, column_scale_factors, min_vectors=None, remove_nan=True
):
    vectors = data[:, columns]
    if remove_nan:
        isnan = ~np.isnan(vectors).any(axis=1)
        vectors = vectors[isnan]
    vectors = vectors / np.array(column_scale_factors)
    clusters = method.fit(vectors)
    labels = clusters.labels_
    if min_vectors is not None:
        label, counts = np.unique(labels, return_counts=True)
        below_min_v = label[counts < min_vectors]
        labels[np.isin(labels, below_min_v)] = -1
    one_neg = np.array((-1,) * len(data))[
        :,
        np.newaxis,
    ]
    vectors_and_labels = np.hstack([data, one_neg])
    if remove_nan:
        vectors_and_labels[isnan, -1] = labels
    return vectors_and_labels


def only_signal_axes(func):
    def wrapper(*args, **kwargs):
        self = args[0]
        if self.has_navigation_axis:
            raise ValueError(
                "This function is not supported for signals with a navigation axis"
            )

        return func(*args, **kwargs)

    return wrapper


def vectors_to_polar(vectors, columns=None):
    """Converts a list of vectors to polar coordinates.

    Parameters
    ----------
    vectors : numpy.ndarray
        Array of vectors.
    columns:
        The x and y columns to be used to calculate the
        polar vector.


    Returns
    -------
    polar_vectors : numpy.ndarray
        Array of vectors in polar coordinates.
    """
    if columns is None:
        columns = [0, 1]
    polar_vectors = np.empty(vectors.shape)
    polar_vectors[:, 0] = np.linalg.norm(vectors[:, columns], axis=1)
    polar_vectors[:, 1] = np.arctan2(vectors[:, columns[1]], vectors[:, columns[0]])
    polar_vectors[:, 2:] = vectors[:, 2:]
    return polar_vectors


def to_cart_three_angles(vectors):
    k = vectors[:, 0]
    delta_phi = vectors[:, 1]
    min_angle = vectors[:, 2]
    angles = np.repeat(min_angle, 3)
    angles[1::3] += delta_phi
    angles[2::3] += delta_phi * 2
    k = np.repeat(k, 3)

    return np.vstack([k * np.cos(angles), k * np.sin(angles)]).T


def polar_to_cartesian(vectors):
    k = vectors[:, 0]
    phi = vectors[:, 1]
    return np.vstack([k * np.cos(phi), k * np.sin(phi)]).T


def get_vectors_mesh(g1_norm, g2_norm, g_norm_max, angle=0.0, shear=0.0):
    """
    Calculate vectors coordinates of a mesh defined by a norm, a rotation and
    a shear component.

    Parameters
    ----------
    g1_norm, g2_norm : float
        The norm of the two vectors of the mesh.
    g_norm_max : float
        The maximum value for the norm of each vector.
    angle : float, optional
        The rotation of the mesh in degree.
    shear : float, optional
        The shear of the mesh. It must be in the interval [0, 1].
        The default is 0.0.

    Returns
    -------
    numpy.ndarray
        x and y coordinates of the vectors of the mesh

    """

    def rotation_matrix(angle):
        return np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )

    def shear_matrix(shear):
        return np.array([[1.0, shear], [0.0, 1.0]])

    if shear < 0 or shear > 1:
        raise ValueError("The `shear` value must be in the interval [0, 1].")

    order1 = int(np.ceil(g_norm_max / g1_norm))
    order2 = int(np.ceil(g_norm_max / g2_norm))
    order = max(order1, order2)

    x = np.arange(-g1_norm * order, g1_norm * (order + 1), g1_norm)
    y = np.arange(-g2_norm * order, g2_norm * (order + 1), g2_norm)

    xx, yy = np.meshgrid(x, y)
    vectors = np.stack(np.meshgrid(x, y)).reshape((2, (2 * order + 1) ** 2))

    transformation = rotation_matrix(np.radians(angle)) @ shear_matrix(shear)

    vectors = transformation @ vectors
    norm = np.linalg.norm(vectors, axis=0)

    return vectors[:, norm <= g_norm_max].T


##################################################
# Symmetry Methods for Analyzing Glassy Materials
##################################################


def get_angles(angles):
    """This function takes a list of angles and returns the angles between each pair of angles.
    This is useful for finding the angles between three vectors.

    Parameters
    ----------
    angles: numpy.ndarray
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
    pks : numpy.ndarray
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
    difference between angle a and angle b.

    Parameters
    ----------
    pks : numpy.ndarray
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
    three_angles : numpy.ndarray
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


# =============================================
# Functions for use with the map_vectors method
# =============================================


def column_mean(vectors, columns):
    """Calculate the mean of the columns of a set of vectors. Useful for calculating the mean
    using the map_vectors method.

    Parameters
    ----------
    vectors: numpy.ndarray
        The vectors to be used to calculate the mean
    columns:
        The columns to be used to calculate the mean.
    """
    return np.mean(vectors[:, columns], axis=0)


##############################
# Plotting Diffraction Vectors
##############################


def vectors2image(
    vectors,
    image_size,
    scales,
    offsets,
    indexes=None,
):
    """Convert a set of vectors to an image by binning the vectors into a 2D histogram.

    Parameters
    ----------
    vectors: numpy.ndarray
        The vectors to be binned
    image_size  : tuple
        The size of the image to be produced
    scales: tuple
        The scales of the image to be produced
    offsets: tuple
        The offsets of the image to be produced
    indexes: tuple
        The indexes of the vectors to be used to create the image.
        If None, the first two columns are used.

    Returns
    -------

    """
    if indexes is None:
        indexes = [0, 1]
    red_points = vectors[:, indexes]
    image_range = (
        (offsets[0], offsets[0] + image_size[0] * scales[0]),
        (offsets[1], offsets[1] + image_size[1] * scales[1]),
    )
    im, _, _ = np.histogram2d(
        red_points[:, 0], red_points[:, 1], bins=image_size, range=image_range
    )
    return im


def points_to_poly_collection(points, hull_index=(0, 1)):
    """Convert a set of points to a polygon collection by creating a polygon. The method takes the
    points given and finds the outermost points to create the polygon.

    Parameters
    ----------
    points: numpy.ndarray
        The points to be used to create the polygon (N x 2)
    hull_index:
        The index of the points to be used to create the polygon. The default is (0, 1) which
        means that the first two columns of the points are used to create the polygon.
    """
    try:
        hull = ConvexHull(points[:, hull_index][:, ::-1])
    except:
        return np.array([[0, 0], [0, 0], [0, 0]])
    return hull.points[hull.vertices]


def points_to_polygon(points, num_points=50):
    """Convert a set of points to a polygon by creating a polygon. The method takes the points
    given and finds the outermost points to create the polygon. The number of points in the
    polygon defines the resolution of the polygon.

    Parameters
    ----------
    points: numpy.ndarray
        The points to be used to create the polygon (N x 2)
    num_points
        The number of points on each side (top, bottom, left, right) of the polygon
        used to create the edges of the polygon.

    Returns
    -------
    numpy.ndarray
        The vertices of the polygon

    """
    if len(points) == 0:
        return np.array([[0, 0], [0, 0], [0, 0]])
    sorted_points = points[np.argsort(points[:, 1])]
    min_x = sorted_points[0, 1]
    max_x = sorted_points[-1, 1]
    sorted_index = np.linspace(min_x, max_x, num_points)

    lo = np.searchsorted(sorted_points[:, 1], sorted_index[:-1], side="left")
    hi = np.searchsorted(sorted_points[:, 1], sorted_index[1:], side="right")

    min_points = []
    for l, h, x in zip(lo, hi, sorted_index):
        if l == h:
            pass
        else:
            min_points.append(
                [
                    np.min(sorted_points[l:h][:, 0]),
                    x,
                ]
            )

    max_points = []
    for l, h, x in zip(lo, hi, sorted_index):
        if l == h:
            pass
        else:
            max_points.append([np.max(sorted_points[l:h][:, 0]), x])
    all_points = min_points + max_points[::-1]
    all_points = np.array(all_points)
    return all_points


def convert_to_markers(
    peaks,
    signal,
):
    """Convert a set of (flattened) peaks to a set of markers for plotting. Note that the
    function below only works for 4D signals.


    Parameters
    ----------
    peaks: DiffractionVectors2D
        The peaks to be converted to markers
    signal: pyxem.signals.Signal2D
        The 4-D signal to plot the peaks on

    Returns
    -------

    """
    new_peaks = deepcopy(peaks.data)
    x_axis, y_axis = signal.axes_manager.navigation_axes
    new_peaks[:, 0] = np.round((new_peaks[:, 0] - x_axis.offset) / x_axis.scale)
    new_peaks[:, 1] = np.round((new_peaks[:, 1] - y_axis.offset) / y_axis.scale)
    ind = np.lexsort((new_peaks[:, 1], new_peaks[:, 0]))
    sorted_peaks = new_peaks[ind]
    shape = signal.axes_manager._navigation_shape_in_array
    by_ind_peaks = np.empty(shape, dtype=object)
    by_ind_colors = np.empty(shape, dtype=object)
    num_labels = np.max(new_peaks[:, -1])
    colors_by_index = (
        np.random.random((int(num_labels + 1), 3)) * 0.9
    )  # (Stay away from white)
    colors_by_index = np.vstack((colors_by_index, [1, 1, 1]))
    low_x_ind = np.searchsorted(sorted_peaks[:, 0], range(0, shape[1]), side="left")
    high_x_ind = np.searchsorted(
        sorted_peaks[:, 0], range(1, shape[1] + 1), side="left"
    )
    for i, (lo_x, hi_x) in enumerate(zip(low_x_ind, high_x_ind)):
        x_inds = sorted_peaks[lo_x:hi_x]
        low_y_ind = np.searchsorted(x_inds[:, 1], range(0, shape[0]), side="left")
        high_y_ind = np.searchsorted(x_inds[:, 1], range(1, shape[0] + 1), side="left")
        for j, (lo_y, hi_y) in enumerate(zip(low_y_ind, high_y_ind)):
            x_values = x_inds[lo_y:hi_y, 2]
            y_values = x_inds[lo_y:hi_y, 3]
            labels = np.array(x_inds[lo_y:hi_y, -1], dtype=int)
            by_ind_peaks[j, i] = np.stack((y_values, x_values), axis=1)
            by_ind_colors[j, i] = colors_by_index[labels]
    return by_ind_peaks, by_ind_colors, colors_by_index
