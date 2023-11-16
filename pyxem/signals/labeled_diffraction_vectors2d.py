# - * - coding: utf - 8 -*-
# Copyright 2016-2023 The pyXem developers
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


from pyxem.signals.diffraction_vectors2d import DiffractionVectors2D
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt


from sklearn.cluster import OPTICS
import hyperspy.api as hs

from pyxem.utils.labeled_vector_utils import (
    get_vector_dist,
    column_mean,
    vectors2image,
    convert_to_markers,
    points_to_polygon,
)


class LabeledDiffractionVectors2D(DiffractionVectors2D):
    """The labeled diffraction vectors 2d signal"""

    _signal_dimension = 2
    _signal_type = "labeled_diffraction_vectors"

    @property
    def unique_counts(self):
        _, counts = np.unique(self.data, return_counts=True, axis=1)
        return

    @property
    def is_clustered(self):
        return self.metadata.get_item("Vectors.is_clustered", False)

    @is_clustered.setter
    def is_clustered(self, value):
        self.metadata.set_item("Vectors.is_clustered", value)

    def get_dist_matrix(self, max_search=None):
        """Returns a distance matrix from a list of vectors with labels in the last column of the dataset."""
        vectors = self.data
        max_label = int(np.max(vectors[:, -1])) + 1
        labels = vectors[:, -1]
        cross_corr_matrix = np.zeros((max_label, max_label))
        label_order = labels.argsort()
        labels = labels[label_order]  # Order the labels
        vectors = vectors[label_order]  # Order the vectors

        sorted_index = np.arange(max_label)
        lo = np.searchsorted(labels, sorted_index, side="left")
        hi = np.searchsorted(labels, sorted_index, side="right")

        if max_search is not None:
            mean_pos = self.map_vectors(
                column_mean, columns=[0, 1], label_index=-1, dtype=float, shape=(2,)
            )
            dist_mat = cdist(mean_pos, mean_pos)

            in_range = dist_mat < max_search
        else:
            in_range = np.ones((max_label, max_label))
        for i, (l, h) in enumerate(zip(lo, hi)):
            v1 = vectors[l:h][:, :2]
            for j, (l2, h2) in enumerate(zip(lo, hi)):
                if in_range[i, j]:
                    v2 = vectors[l2:h2][:, :2]
                    cross_corr_matrix[i, j] = get_vector_dist(v1, v2)
                else:
                    cross_corr_matrix[i, j] = 10000
        return cross_corr_matrix

    def map_vectors(self, func, dtype, label_index=-1, shape=None, **kwargs):
        """
        Parameters
        ----------
        func: func
            The function to be applied to each label
        label_index: int
            The index of the label to be analyzed. Usually this is the last column
        dtype: np.dtype
            The dtype used to initialize the output array
        shape: None or tuple
            The shape for the output of the `func`
        kwargs:
            Any additional arguments passed to `func`
        Returns
        -------
        signal:
            The signal from the map_vectors function. Each index on the navigation axes
            is related to the label for each vector.
        """
        vectors = self.data
        labels = vectors[:, label_index]
        label_order = labels.argsort()
        labels = labels[label_order]  # Order the labels
        vectors = vectors[label_order]  # Order the vectors

        sorted_index = np.arange(0, np.max(labels) + 1)
        lo = np.searchsorted(labels, sorted_index, side="left")
        hi = np.searchsorted(labels, sorted_index, side="right")

        if shape is not None:
            ans = np.empty((len(sorted_index),) + shape, dtype=dtype)
        else:
            ans = np.empty(len(sorted_index), dtype=dtype)
        for i, (l, h) in enumerate(zip(lo, hi)):
            ans[i] = func(vectors[l:h], **kwargs)
        return ans

    def plot_clustered(
        self,
        nav_columms=None,
        signal_columns=None,
        navigation_pixels=(105, 105),
        scales=None,
        offsets=None,
        labels=None,
        signal=None,
        figsize=None,
    ):
        if signal_columns is None:
            signal_columns = [2, 3]
        if nav_columms is None:
            nav_columms = [0, 1]
        if scales is None:
            nav = self.data[:, nav_columms]
            xscale = (np.max(nav[:, 0]) - np.min(nav[:, 0])) / (
                navigation_pixels[0] - 1
            )
            yscale = (np.max(nav[:, 1]) - np.min(nav[:, 1])) / (
                navigation_pixels[1] - 1
            )
            scales = (xscale, yscale)
            print(scales)
        if offsets is None:
            nav = self.data[:, nav_columms]
            offsets = (np.min(nav[:, 0]), np.min(nav[:, 1]))
        if not self.is_clustered:
            raise ValueError(
                "You must first cluster the dataset using the "
                "`cluster_labeled_vectors` function."
            )
        num_clusters = int(np.max(self.data[:, -1]) + 1)
        fig, axs = plt.subplots(2, num_clusters, figsize=figsize)
        if labels is None:
            labels = range(num_clusters)
        for i in labels:
            clustered_peaks = self.data[self.data[:, -1] == i]
            unique_peaks = np.unique(clustered_peaks[:, -2])
            for p in unique_peaks:
                is_p = clustered_peaks[:, -2] == p
                pks = clustered_peaks[is_p]
                img = vectors2image(
                    pks,
                    image_size=navigation_pixels,
                    scales=scales,
                    offsets=offsets,
                    indexes=nav_columms,
                )
                img[img == 0] = np.nan
                ext = tuple((np.array(scales) * navigation_pixels) + offsets)
                ext = (offsets[0], ext[0], offsets[1], ext[1])
                axs[0, i].imshow(img, alpha=0.2, extent=ext)
            if signal is not None:
                axs[1, i].imshow(signal.data, extent=signal.axes_manager.signal_extent)
            axs[1, i].scatter(
                clustered_peaks[:, signal_columns[1]],
                clustered_peaks[:, signal_columns[0]],
            )
        return fig, axs

    def cluster_labeled_vectors(
        self, method, columns=None, preprocessing="mean",replace_nan=-100, **kwargs
    ):
        """A function to cluster the labeled vectors in the dataset.

        Parameters
        ----------
        method: sklearn.base.ClusterMixin
            The clustering method to be used. This is a class that implements the ``fit`` method
        columns: None or list
            The columns to be used for clustering. If None, the first two columns are used
        preprocessing: str or callable
            The function to be applied to each label clustering. If 'mean', the mean of the
            vectors is used. If callable, the function is applied to each label and the result
            is used for clustering.
        """
        if columns is None:
            columns = [0, 1]
        if preprocessing == "mean":
            preprocessing = column_mean
            kwargs = {"label_index": -1, "dtype": float, "shape": (2,)}
        elif callable(preprocessing):
            preprocessing = preprocessing
        else:
            raise ValueError("The preprocessing must be either 'mean' or a function")
        to_cluster_vectors = self.map_vectors(
            preprocessing,
            columns=columns,
            **kwargs,
        )
        to_cluster_vectors[np.isnan(to_cluster_vectors)] = replace_nan
        clustering = method.fit(to_cluster_vectors)
        labels = clustering.labels_
        initial_labels = self.data[:, -1].astype(int)
        new_labels = labels[initial_labels]
        new_labels[initial_labels == -1] = -1
        print(f"{np.max(labels) + 1} : Clusters Found!")
        vectors_and_labels = np.hstack([self.data, new_labels[:, np.newaxis]])
        new_signal = self._deepcopy_with_new_data(data=vectors_and_labels)
        new_signal.axes_manager.signal_axes[0].size = (
            new_signal.axes_manager.signal_axes[0].size + 1
        )
        new_signal.is_clustered = True
        return new_signal

    def to_markers(self, signal, get_polygons=False,num_points=10, **kwargs):
        marker_list = []

        offsets, colors, colors_by_index = convert_to_markers(self, signal)
        points = hs.plot.markers.Points(offsets=offsets.T, color=colors.T, **kwargs)
        marker_list.append(points)
        if get_polygons:
            verts = self.map_vectors(points_to_polygon, num_points=num_points, dtype=object)
            verts = list(verts)
            polygons = hs.plot.markers.Polygons(
                verts=verts, alpha=0.5, color=colors_by_index, linewidth=2
            )
            marker_list.append(polygons)
        return marker_list
