import numpy as np
from sklearn import cluster
import pixstem.marker_tools as mt


def _find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def _get_cluster_dict(peak_array, eps=30, min_samples=2):
    """Sort peaks into cluster using sklearn's DBSCAN.

    Each cluster is given its own label, with the unclustered
    having the label -1.

    Parameters
    ----------
    peak_array : 2D numpy array
        In the form [[x0, y0], [x1, y1], ...]
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
    >>> import pixstem.cluster_tools as ct
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
    >>> import pixstem.cluster_tools as ct
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
        r = np.hypot(cluster_array[:, 0] - centre_x,
                     cluster_array[:, 1] - centre_y)
        closest_list.append(_find_nearest(r, 0))

    icentre_label = np.argmin(closest_list)
    centre_label = label_list[icentre_label]

    sorted_cluster_dict = {'none': [], 'centre': [], 'rest': []}
    for label, cluster_list in cluster_dict.items():
        if label == -1:
            sorted_cluster_dict['none'] = cluster_list
        elif label == centre_label:
            sorted_cluster_dict['centre'] = cluster_list
        else:
            sorted_cluster_dict['rest'].extend(cluster_list)
    return sorted_cluster_dict


def _sorted_cluster_dict_to_marker_list(
        sorted_cluster_dict, signal_axes=None,
        color_centre='blue', color_rest='red', color_none='green', size=20):
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
    >>> import pixstem.cluster_tools as ct
    >>> marker_list = ct._sorted_cluster_dict_to_marker_list(
    ...     sorted_cluster_dict)
    >>> import pixstem.marker_tools as mt
    >>> s = ps.PixelatedSTEM(np.random.random((3, 4, 100, 100)))
    >>> mt._add_permanent_markers_to_signal(s, marker_list)

    Different colors

    >>> marker_list = ct._sorted_cluster_dict_to_marker_list(
    ...     sorted_cluster_dict, color_centre='green', color_rest='cyan',
    ...     color_none='purple', size=15)

    """
    marker_list = []
    for label, cluster_list in sorted_cluster_dict.items():
        if label == 'centre':
            color = color_centre
        elif label == 'rest':
            color = color_rest
        elif label == 'none':
            color = color_none
        else:
            color = 'cyan'
        temp_markers = mt._get_4d_marker_list(
                cluster_list, signal_axes=signal_axes, color=color, size=20)
        marker_list.extend(temp_markers)
    return marker_list
