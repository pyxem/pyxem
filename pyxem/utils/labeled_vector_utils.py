import itertools
from copy import deepcopy

import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull


# =============================================
# Functions for use with the map_vectors method
# =============================================


def upwrap_flattened(pks):
    r = np.linalg.norm(pks[:, (2, 3)], axis=1)  # ignore the intensity
    theta = np.arctan2(pks[:, 2], pks[:, 3])
    return np.stack((pks[:, :2], r, theta, pks[:, 4:]), axis=1)


def reduced_subtract(min_vector, v2):
    return np.abs(v2 - np.round(v2 / min_vector) * min_vector)


def filter_mag(z, min_mag, max_mag):
    norm = np.linalg.norm(z[:, :2], axis=1)
    in_range = norm < max_mag * (norm > min_mag)
    return z[in_range]


def get_vector_dist(v1, v2):
    """Return the average minimum distance between the list of vectors v1 and v2.
    This is a modified distance matrix which is a measure of the similarity between the two point clouds.
    """
    d = cdist(v1, v2)
    distance = np.mean([np.mean(np.min(d, axis=0)), np.mean(np.min(d, axis=1))])
    return distance


def column_mean(vectors, columns=None):
    if columns is None:
        columns = [0, 1]
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
    try:
        hull = ConvexHull(points[:, hull_index][:, ::-1])
    except:
        return np.array([[0, 0], [0, 0], [0, 0]])
    return hull.points[hull.vertices]


def points_to_polygon(points, num_points=50):
    if len(points) == 0:
        return np.array([[0, 0], [0, 0], [0, 0]])
    sorted_points = points[np.argsort(points[:, 0])]
    min_x = sorted_points[0, 0]
    max_x = sorted_points[-1, 0]
    sorted_index = np.linspace(min_x, max_x, num_points)

    lo = np.searchsorted(sorted_points[:, 0], sorted_index[:-1], side="left")
    hi = np.searchsorted(sorted_points[:, 0], sorted_index[1:], side="right")

    min_points = []
    for l, h, x in zip(lo, hi, sorted_index):
        if l == h:
            pass
        else:
            min_points.append(
                [
                    np.min(sorted_points[l:h][:, 1]),
                    x,
                ]
            )

    max_points = []
    for l, h, x in zip(lo, hi, sorted_index):
        if l == h:
            pass
        else:
            max_points.append([np.max(sorted_points[l:h][:, 1]), x])
    all_points = min_points + max_points[::-1]
    all_points = np.array(all_points)
    return all_points


def angles_to_markers(angles, signal, k=4.0, polar=True, return_marker=False, **kwargs):
    new_angles = deepcopy(angles.data)
    x_axis, y_axis = signal.axes_manager.navigation_axes
    new_angles[:, 0] = np.round((new_angles[:, 0] - x_axis.offset) / x_axis.scale)
    new_angles[:, 1] = np.round((new_angles[:, 1] - y_axis.offset) / y_axis.scale)
    ind = np.lexsort((new_angles[:, 1], new_angles[:, 0]))
    sorted_peaks = new_angles[ind]
    x, y = signal.axes_manager.signal_axes
    shape = signal.axes_manager.navigation_shape
    by_ind_peaks = np.empty(shape, dtype=object)
    by_ind_colors = np.empty(shape, dtype=object)
    num_labels = np.max(new_angles[:, -1])
    colors_by_index = (
        np.random.random((int(num_labels + 1), 3)) * 0.9
    )  # (Stay away from white)
    colors_by_index = np.vstack((colors_by_index, [1, 1, 1]))
    low_x_ind = np.searchsorted(sorted_peaks[:, 0], range(0, shape[0]), side="left")
    high_x_ind = np.searchsorted(
        sorted_peaks[:, 0], range(1, shape[0] + 1), side="left"
    )
    for i, (lo_x, hi_x) in enumerate(zip(low_x_ind, high_x_ind)):
        x_inds = sorted_peaks[lo_x:hi_x]
        low_y_ind = np.searchsorted(x_inds[:, 1], range(0, shape[1]), side="left")
        high_y_ind = np.searchsorted(x_inds[:, 1], range(1, shape[1] + 1), side="left")
        for j, (lo_y, hi_y) in enumerate(zip(low_y_ind, high_y_ind)):
            initial_theta = x_inds[lo_y:hi_y, 2]
            angle_seperation = x_inds[lo_y:hi_y, 3]
            labels = np.array(x_inds[lo_y:hi_y, -1], dtype=int)
            angles = [initial_theta + angle_seperation * i for i in [0, 1, 2]]
            y_values = np.hstack([np.cos(a) * k for a in angles])
            x_values = np.hstack([np.sin(a) * k for a in angles])
            labels = np.hstack((labels, labels, labels))
            by_ind_peaks[i, j] = np.stack((y_values, x_values), axis=1)
            by_ind_colors[i, j] = colors_by_index[labels]
    return by_ind_peaks, by_ind_colors, colors_by_index


def convert_to_markers(
    peaks,
    signal,
):
    new_peaks = deepcopy(peaks.data)
    x_axis, y_axis = signal.axes_manager.navigation_axes
    new_peaks[:, 0] = np.round((new_peaks[:, 0] - x_axis.offset) / x_axis.scale)
    new_peaks[:, 1] = np.round((new_peaks[:, 1] - y_axis.offset) / y_axis.scale)
    ind = np.lexsort((new_peaks[:, 1], new_peaks[:, 0]))
    sorted_peaks = new_peaks[ind]
    shape = signal.axes_manager.navigation_shape
    by_ind_peaks = np.empty(shape, dtype=object)
    by_ind_colors = np.empty(shape, dtype=object)
    num_labels = np.max(new_peaks[:, -1])
    colors_by_index = (
        np.random.random((int(num_labels + 1), 3)) * 0.9
    )  # (Stay away from white)
    colors_by_index = np.vstack((colors_by_index, [1, 1, 1]))
    low_x_ind = np.searchsorted(sorted_peaks[:, 0], range(0, shape[0]), side="left")
    high_x_ind = np.searchsorted(
        sorted_peaks[:, 0], range(1, shape[0] + 1), side="left"
    )
    for i, (lo_x, hi_x) in enumerate(zip(low_x_ind, high_x_ind)):
        x_inds = sorted_peaks[lo_x:hi_x]
        low_y_ind = np.searchsorted(x_inds[:, 1], range(0, shape[1]), side="left")
        high_y_ind = np.searchsorted(x_inds[:, 1], range(1, shape[1] + 1), side="left")
        for j, (lo_y, hi_y) in enumerate(zip(low_y_ind, high_y_ind)):
            x_values = x_inds[lo_y:hi_y, 2]
            y_values = x_inds[lo_y:hi_y, 3]
            labels = np.array(x_inds[lo_y:hi_y, -1], dtype=int)
            by_ind_peaks[i, j] = np.stack((y_values, x_values), axis=1)
            by_ind_colors[i, j] = colors_by_index[labels]
    return by_ind_peaks, by_ind_colors, colors_by_index
