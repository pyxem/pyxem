import numpy as np
import itertools
from scipy.spatial.distance import cdist


# =============================================
# Functions for use with the map_vectors method
# =============================================


def upwrap_flattened(pks):
    r = np.linalg.norm(pks[:, (2, 3)], axis=1)  # ignore the intensity
    theta = np.arctan2(pks[:, 2], pks[:, 3])
    return np.stack((pks[:, :2], r, theta, pks[:, 4:]), axis=1)


def reduced_subtract(min_vector, v2):
    return np.abs(v2 - np.round(v2 / min_vector) * min_vector)


def get_angles(angles):
    all_angles = np.abs(np.triu(np.subtract.outer(angles, angles)))
    all_angles = all_angles[all_angles != 0]
    all_angles[all_angles > np.pi] = np.pi - np.abs(
        all_angles[all_angles > np.pi] - np.pi
    )
    return all_angles


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
    red_points = np.round((red_points / scales) - offsets)
    red_points = red_points.astype(int)
    im = np.zeros(image_size)
    im[red_points[:, 0], red_points[:, 1]] = 1
    return im


def column_mean(vectors, columns=None):
    if columns is None:
        columns = [0, 1]
    return np.mean(vectors[:, columns], axis=0)


##################################################
# Symmetry Methods for Analyzing Glassy Materials
##################################################


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
    """
    angles = pks[:, angle_index]
    k = pks[:, radial_index]

    angle_combos = list(itertools.combinations(angles, num))
    k_combos = list(itertools.combinations(k, num))
    # Filtering out combinations with only diffraction from vectors below the intensity threshold
    if intensity_threshold is not None and intensity_index is not None:
        intensity = pks[:, intensity_index]
        intensity_combos = itertools.combinations(intensity, num)
        has_min_intensity = np.array(
            [any(np.array(i) > intensity_threshold) for i in intensity_combos]
        )
    else:
        has_min_intensity = True
    # Filtering out combinations where there are two peaks close to each other
    if min_angle is not None:
        above_angle = np.array(
            [
                all(
                    [
                        np.abs(np.subtract(*c)) > min_angle
                        for c in itertools.combinations(a, 2)
                    ]
                )
                for a in angle_combos
            ]
        )
    else:
        above_angle = True
    # Filtering out combinations of diffraction vectors at different values for k.
    if min_k is not None:
        in_k_range = np.array(
            [np.mean(np.abs(np.subtract(np.mean(k), k))) < min_k for k in k_combos]
        )
    else:
        in_k_range = True
    in_combos = above_angle * has_min_intensity * in_k_range
    if np.all(in_combos):
        combos = angle_combos
        combos_k = [np.mean(ks) for ks in k_combos]
    else:
        combos = [c for c, in_c in zip(angle_combos, in_combos) if in_c]
        combos_k = [
            np.mean(ks) for ks, in_range in zip(k_combos, in_combos) if in_range
        ]
    return combos, combos_k


def get_three_angles(
    pks,
    k_index=0,
    angle_index=1,
    intensity_index=2,
    intensity_threshold=None,
    accept_threshold=0.05,
    min_k=0.05,
    return_min=True,
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
    """
    three_angles = []
    min_angles = []
    combos, combo_k = get_filtered_combinations(
        pks,
        3,
        radial_index=k_index,
        angle_index=angle_index,
        intensity_index=intensity_index,
        intensity_threshold=intensity_threshold,
        min_angle=min_angle,
        min_k=min_k,
    )
    for c, k in zip(combos, combo_k):
        angular_seperations = get_angles(c)
        try:
            min_ind = np.argmin(angular_seperations)
            min_sep = angular_seperations[min_ind]
            angular_seperations = np.delete(angular_seperations, min_ind)
            is_symetric = np.any(
                np.abs((angular_seperations - min_sep)) < accept_threshold
            )
            if is_symetric:
                if not return_min:
                    for a in angular_seperations:
                        three_angles.append(a)
                    three_angles.append(min_sep)
                else:
                    min_angle = np.min(c)
                    num_times = np.round(min_angle / min_sep)
                    three_angles.append(
                        [
                            min_angle,
                            min_sep,
                            np.abs(min_angle - (num_times * min_sep)),
                            k,
                        ]
                    )
        except ValueError:
            print("error")
            print(c)
            print(combos)
            pass
    if len(three_angles) == 0:
        three_angles = np.empty((0, 4))
    return np.array(three_angles)




##############################
# Plotting Diffraction Vectors
##############################
from scipy.spatial import ConvexHull, convex_hull_plot_2d, QhullError
import numpy as np
from copy import deepcopy


def vectors2image(
    vectors,
    image_size,
    scales,
    offsets,
    indexes=[0, 1],
):
    red_points = vectors[:, indexes]
    im = np.zeros(image_size)
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


def convert_to_markers(peaks, signal,):
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
