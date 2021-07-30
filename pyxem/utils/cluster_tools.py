# -*- coding: utf-8 -*-
# Copyright 2016-2021 The pyXem developers
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
from sklearn import cluster
from hyperspy.misc.utils import isiterable
import pyxem.utils.marker_tools as mt


def _find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def _find_max_indices_4D_peak_array(peak_array):
    """Find maximum indices in a 4D peak array.

    Parameters
    ----------
    peak_array : 4D NumPy array

    Returns
    -------
    max_indices : tuple
        (max_x_index, max_y_index)

    Examples
    --------
    >>> import pyxem.utils.cluster_tools as ct
    >>> peak_array0 = np.random.randint(10, 255, size=(3, 4, 5000, 1))
    >>> peak_array1 = np.random.randint(5, 127, size=(3, 4, 5000, 1))
    >>> peak_array = np.concatenate((peak_array0, peak_array1), axis=3)
    >>> max_x_index, max_y_index = ct._find_max_indices_4D_peak_array(
    ...     peak_array)

    """
    max_x_index, max_y_index = 0, 0
    for ix, iy in np.ndindex(peak_array.shape[:2]):
        temp_peak_array = peak_array[ix, iy]
        x_max = int(temp_peak_array[:, 0].max())
        y_max = int(temp_peak_array[:, 1].max())
        if x_max > max_x_index:
            max_x_index = x_max
        if y_max > max_y_index:
            max_y_index = y_max
    return max_x_index, max_y_index


def _filter_4D_peak_array(
    peak_array, signal_axes=None, max_x_index=255, max_y_index=255
):
    """Remove false positives at the outer edges.

    Parameters
    ----------
    peak_array : NumPy array
    signal_axes : HyperSpy signal axes axes_manager, optional
    max_x_index, max_y_index : scalar, optional
        Default 255.

    Examples
    --------
    >>> import pyxem.utils.cluster_tools as ct
    >>> peak_array = np.random.randint(0, 255, size=(3, 4, 100, 2))
    >>> peak_array_filtered = ct._filter_4D_peak_array(peak_array)

    See Also
    --------
    _filter_peak_array_radius
    _filter_peak_list
    _filter_peak_list_radius

    """
    if signal_axes is not None:
        max_x_index = signal_axes[0].high_index
        max_y_index = signal_axes[1].high_index
    peak_array_shape = _get_peak_array_shape(peak_array)
    peak_array_filtered = np.empty(shape=peak_array_shape, dtype=object)
    for index in np.ndindex(peak_array_shape):
        islice = np.s_[index]
        peak_list_filtered = _filter_peak_list(
            peak_array[islice], max_x_index=max_x_index, max_y_index=max_y_index
        )
        peak_array_filtered[islice] = np.array(peak_list_filtered)
    return peak_array_filtered


def _filter_peak_list(peak_list, max_x_index=255, max_y_index=255):
    """Remove false positive peaks at the outer edges.

    Parameters
    ----------
    peak_list : 2D NumPy or 2D list
        [[x0, y0], [x1, y1], ...]
    max_x_index, max_y_index : int, optional
        Default 255.

    Returns
    -------
    peak_list_filtered : 2D NumPy array or 2D list

    Examples
    --------
    >>> import pyxem.utils.cluster_tools as ct
    >>> peak_list = [[128, 129], [10, 52], [0, 120], [255, 123], [123, 255]]
    >>> ct._filter_peak_list(peak_list)
    [[128, 129], [10, 52]]

    See Also
    --------
    _filter_peak_array_radius
    _filter_4D_peak_array
    _filter_peak_list_radius

    """
    peak_list_filtered = []
    for x, y in peak_list:
        if x == 0:
            pass
        elif y == 0:
            pass
        elif x == max_x_index:
            pass
        elif y == max_y_index:
            pass
        else:
            peak_list_filtered.append([x, y])
    return peak_list_filtered


def _filter_peak_array_radius(peak_array, xc, yc, r_min=None, r_max=None):
    """Remove peaks from a peak_array, based on distance from a point.

    Parameters
    ----------
    peak_array : NumPy array
        In the form [[[[y0, x0], [y1, x1]]]]
    xc, yc : scalars, NumPy array
        Centre position
    r_min, r_max : scalar
        Remove peaks which are within r_min and r_max distance from the centre.
        One of them must be specified.

    Returns
    -------
    peak_array_filtered : NumPy array
        Similar to peak_array input, but with the too-close peaks
        removed.

    See Also
    --------
    _filter_peak_list
    _filter_4D_peak_array
    _filter_peak_list_radius

    """
    if not isiterable(xc):
        xc = np.ones(peak_array.shape[:2]) * xc
    if not isiterable(yc):
        yc = np.ones(peak_array.shape[:2]) * yc
    peak_array_filtered = np.empty(shape=peak_array.shape[:2], dtype=object)
    for iy, ix in np.ndindex(peak_array.shape[:2]):
        temp_xc, temp_yc = xc[iy, ix], yc[iy, ix]
        peak_list_filtered = _filter_peak_list_radius(
            peak_array[iy, ix], xc=temp_xc, yc=temp_yc, r_min=r_min, r_max=r_max
        )
        peak_array_filtered[iy, ix] = np.array(peak_list_filtered)
    return peak_array_filtered


def _filter_peak_list_radius(peak_list, xc, yc, r_min=None, r_max=None):
    """Remove peaks based on distance to some point.

    Parameters
    ----------
    peak_list : NumPy array
        In the form [[y0, x0], [y1, x1], ...]
    xc, yc : scalars
        Centre position
    r_min, r_max : scalar
        Remove peaks which are within r_min and r_max distance from the centre.
        One of them must be specified.

    Returns
    -------
    peak_filtered_list : NumPy array
        Similar to peak_list input, but with the too-close peaks
        removed.

    Examples
    --------
    >>> import pyxem.utils.cluster_tools as ct
    >>> peak_list = np.array([[128, 32], [128, 127]])
    >>> ct._filter_peak_list_radius(peak_list, 128, 128, r_min=10)
    array([[128,  32]])

    See Also
    --------
    _filter_peak_array_radius
    _filter_peak_list
    _filter_4D_peak_array

    """
    dist = np.hypot(peak_list[:, 1] - xc, peak_list[:, 0] - yc)
    if (r_min is None) and (r_max is None):
        raise ValueError("Either r_min or r_max must be specified")
    if (r_min is not None) and (r_max is not None):
        if r_max < r_min:
            raise ValueError(
                "r_min ({0}) must be smaller than r_max ({1})".format(r_min, r_max)
            )
    filter_list = np.ones_like(dist, dtype=bool)
    if r_min is not None:
        temp_filter_list = dist > r_min
        filter_list[:] = np.logical_and(filter_list, temp_filter_list)
    if r_max is not None:
        temp_filter_list = dist < r_max
        filter_list[:] = np.logical_and(filter_list, temp_filter_list)
    peak_filtered_list = peak_list[filter_list]
    return peak_filtered_list


def _get_cluster_dict(peak_array, eps=30, min_samples=2):
    """Sort peaks into cluster using sklearn's DBSCAN.

    Each cluster is given its own label, with the unclustered
    having the label -1.

    Parameters
    ----------
    peak_array : 2D numpy array
        In the form [[x0, y0], [x1, y1], ...], i.e. shape = 2
    eps : scalar
        For the DBSCAN clustering algorithm
    min_samples : int
        Minimum number of peaks in each cluster

    Returns
    -------
    cluster_dict : dict
        The peaks are sorted into a dict with the cluster label as the key.

    Example
    -------
    >>> import numpy as np
    >>> peak_array = np.random.randint(1000, size=(100, 2))
    >>> import pyxem.utils.cluster_tools as ct
    >>> cluster_dict = ct._get_cluster_dict(peak_array)
    >>> cluster0 = cluster_dict[0]

    """
    dbscan = cluster.DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(peak_array)
    label_list = dbscan.labels_

    label_unique_list = sorted(list(set(label_list)))
    cluster_dict = {}
    for label_unique in label_unique_list:
        cluster_dict[label_unique] = []

    for peak, label in zip(peak_array, label_list):
        cluster_dict[label].append(peak.tolist())
    return cluster_dict


def _sort_cluster_dict(cluster_dict, centre_x=128, centre_y=128):
    """Sort clusters into centre, rest and unclustered.

    Parameters
    ----------
    cluster_dict : dict
    centre_x : scalar, optional
        Default 128
    centre_y : scalar, optional
        Default 128

    Returns
    -------
    sorted_cluster_dict : dict
        Centre cluster has key 'centre', the others 'rest',
        and lastly the unclustered 'none'.

    Examples
    --------
    >>> import numpy as np
    >>> peak_array0 = np.random.randint(6, size=(100, 2)) + 128
    >>> peak_array1 = np.random.randint(6, size=(100, 2)) + 200
    >>> peak_array = np.vstack((peak_array0, peak_array1, [[100, 0], ]))
    >>> import pyxem.utils.cluster_tools as ct
    >>> cluster_dict = ct._get_cluster_dict(peak_array)
    >>> sorted_cluster_dict = ct._sort_cluster_dict(cluster_dict)
    >>> cluster_centre = sorted_cluster_dict['centre']
    >>> cluster_rest = sorted_cluster_dict['rest']
    >>> cluster_none = sorted_cluster_dict['none']

    Different centre position

    >>> sorted_cluster_dict = ct._sort_cluster_dict(
    ...     cluster_dict, centre_x=200, centre_y=200)

    """
    label_list, closest_list = [], []
    for label, cluster_list in cluster_dict.items():
        label_list.append(label)
        cluster_array = np.array(cluster_list)
        r = np.hypot(cluster_array[:, 0] - centre_x, cluster_array[:, 1] - centre_y)
        closest_list.append(_find_nearest(r, 0))

    icentre_label = np.argmin(closest_list)
    centre_label = label_list[icentre_label]

    sorted_cluster_dict = {"none": [], "centre": [], "rest": []}
    for label, cluster_list in cluster_dict.items():
        if label == -1:
            sorted_cluster_dict["none"] = cluster_list
        elif label == centre_label:
            sorted_cluster_dict["centre"] = cluster_list
        else:
            sorted_cluster_dict["rest"].extend(cluster_list)
    return sorted_cluster_dict


def _get_peak_array_shape(peak_array):
    """Find the navigation shape of a peak array

    This is necessary due to the peak_array.shape will be different
    depending if the array is the more common object dtype, or
    something else.

    Parameters
    ----------
    peak_array : NumPy array

    Returns
    -------
    peak_array_shape : tuple

    """
    if peak_array.dtype == object:
        peak_array_shape = peak_array.shape
    else:
        peak_array_shape = peak_array.shape[:-2]
    return peak_array_shape


def _cluster_and_sort_peak_array(
    peak_array, eps=30, min_samples=2, centre_x=128, centre_y=128
):
    """Cluster and sort a 4D peak array into centre, rest and unclustered.

    Parameters
    ----------
    peak_array : NumPy array
    eps : scalar, optional
        Default 30, passed to sklearn's DBSCAN
    min_samples : scalar, optional
        Default 2, passed to sklearn's DBSCAN
    centre_x, centre_y : scalar, optional
        Default 128

    Returns
    -------
    peak_dicts : dict
        Different peaks sorted into either peak_dicts['centre'],
        peak_dicts['rest'], peak_dicts['none'] (for the unclustered points).

    Example
    -------
    >>> peak_array0 = np.random.randint(124, 132, size=(2, 4, 10, 2))
    >>> peak_array1 = np.random.randint(204, 208, size=(2, 4, 10, 2))
    >>> peak_array = np.concatenate((peak_array0, peak_array1), axis=2)
    >>> import pyxem.utils.cluster_tools as ct
    >>> peak_dicts = ct._cluster_and_sort_peak_array(peak_array)
    >>> peak_array_centre = peak_dicts['centre']
    >>> peak_array_rest = peak_dicts['rest']
    >>> peak_array_none = peak_dicts['none']

    """
    peak_array_shape = _get_peak_array_shape(peak_array)
    peak_centre_array = np.empty(shape=peak_array_shape, dtype=object)
    peak_rest_array = np.empty(shape=peak_array_shape, dtype=object)
    peak_none_array = np.empty(shape=peak_array_shape, dtype=object)
    for index in np.ndindex(peak_array_shape):
        islice = np.s_[index]
        cluster_dict = _get_cluster_dict(
            peak_array[islice], eps=eps, min_samples=min_samples
        )
        sorted_cluster_dict = _sort_cluster_dict(
            cluster_dict, centre_x=centre_x, centre_y=centre_y
        )
        peak_centre_array[islice] = sorted_cluster_dict["centre"]
        peak_rest_array[islice] = sorted_cluster_dict["rest"]
        peak_none_array[islice] = sorted_cluster_dict["none"]

    peak_dicts = {}
    peak_dicts["centre"] = peak_centre_array
    peak_dicts["rest"] = peak_rest_array
    peak_dicts["none"] = peak_none_array
    return peak_dicts


def _add_peak_dicts_to_signal(
    signal,
    peak_dicts,
    color_centre="red",
    color_rest="blue",
    color_none="cyan",
    size=20,
):
    """Visualize the results of peak_dicts through markers in a Signal.

    Parameters
    ----------
    signal : HyperSpy Signal2D, PixelatedSTEM
    peak_dicts : dicts
    color_centre, color_rest, color_none : string, optional
        Color of the markers. Default 'red', 'blue', 'cyan'.
    size : scalar, optional
        Size of the markers. Default 20

    Example
    -------
    >>> peak_dicts = {}
    >>> peak_dicts['centre'] = np.random.randint(99, size=(2, 3, 10, 2))
    >>> peak_dicts['rest'] = np.random.randint(99, size=(2, 3, 3, 2))
    >>> peak_dicts['none'] = np.random.randint(99, size=(2, 3, 2, 2))
    >>> s = pxm.signals.Diffraction2D(np.random.random((2, 3, 100, 100)))
    >>> import pyxem.utils.cluster_tools as ct
    >>> ct._add_peak_dicts_to_signal(s, peak_dicts)
    >>> s.plot()

    """
    mt.add_peak_array_to_signal_as_markers(
        signal, peak_dicts["centre"], color=color_centre, size=size
    )
    mt.add_peak_array_to_signal_as_markers(
        signal, peak_dicts["rest"], color=color_rest, size=size
    )
    mt.add_peak_array_to_signal_as_markers(
        signal, peak_dicts["none"], color=color_none, size=size
    )


def _sorted_cluster_dict_to_marker_list(
    sorted_cluster_dict,
    signal_axes=None,
    color_centre="blue",
    color_rest="red",
    color_none="green",
    size=20,
):
    """Make a list of markers with different colors from a sorted cluster dict

    Parameters
    ----------
    sorted_cluster_dict : dict
        dict with clusters sorted into 'centre', 'rest' and 'none'
        lists.
    signal_axes : HyperSpy axes_manager, optional
    color_centre, color_rest, color_none : string, optional
        Color of the markers. Default 'blue', 'red' and 'green'.
    size : scalar, optional
        Size of the markers.

    Returns
    -------
    marker_list : list of HyperSpy markers

    Examples
    --------
    >>> from numpy.random import randint
    >>> sorted_cluster_dict = {}
    >>> sorted_cluster_dict['centre'] = randint(10, size=(3, 4, 10, 2))
    >>> sorted_cluster_dict['rest'] = randint(50, 60, size=(3, 4, 10, 2))
    >>> sorted_cluster_dict['none'] = randint(90, size=(3, 4, 2, 2))
    >>> import pyxem.utils.cluster_tools as ct
    >>> marker_list = ct._sorted_cluster_dict_to_marker_list(
    ...     sorted_cluster_dict)
    >>> import pyxem.utils.marker_tools as mt
    >>> s = pxm.signals.Diffraction2D(np.random.random((3, 4, 100, 100)))
    >>> mt._add_permanent_markers_to_signal(s, marker_list)

    Different colors

    >>> marker_list = ct._sorted_cluster_dict_to_marker_list(
    ...     sorted_cluster_dict, color_centre='green', color_rest='cyan',
    ...     color_none='purple', size=15)

    """
    marker_list = []
    for label, cluster_list in sorted_cluster_dict.items():
        if label == "centre":
            color = color_centre
        elif label == "rest":
            color = color_rest
        elif label == "none":
            color = color_none
        else:
            color = "cyan"
        temp_markers = mt._get_4d_points_marker_list(
            cluster_list, signal_axes=signal_axes, color=color, size=size
        )
        marker_list.extend(temp_markers)
    return marker_list
