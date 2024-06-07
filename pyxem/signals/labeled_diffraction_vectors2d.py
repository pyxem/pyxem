# - * - coding: utf - 8 -*-
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


from pyxem.signals.diffraction_vectors2d import DiffractionVectors2D
import numpy as np
import matplotlib.pyplot as plt


import hyperspy.api as hs

from pyxem.utils.vectors import (
    column_mean,
    vectors2image,
    convert_to_markers,
    points_to_polygon,
)
from pyxem.utils.vectors import only_signal_axes


class LabeledDiffractionVectors2D(DiffractionVectors2D):
    """The labeled diffraction vectors 2d signal"""

    _signal_dimension = 2
    _signal_type = "labeled_diffraction_vectors"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_clustered = False

    @property
    def is_clustered(self):
        return self.metadata.VectorMetadata["is_clustered"]

    @is_clustered.setter
    def is_clustered(self, value):
        self.metadata.VectorMetadata["is_clustered"] = value

    @only_signal_axes
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

    @only_signal_axes
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
        """Plot the clustered vectors and the characteristic diffraction patterns

        Parameters
        ----------
        nav_columms: None or list
            The columns to be used for the navigation axes
        signal_columns: None or list
            The columns to be used for the signal axes
        navigation_pixels: tuple
            The number of pixels in the navigation axes
        scales: None or tuple
            The scales for the navigation axes
        offsets: None or tuple
            The offsets for the navigation axes
        labels: None or list
            The labels to be plotted
        signal: None or pyxem.signals.ElectronDiffraction2D
            The signal to be plotted on the
        figsize: None or tuple
            The figure size to be used
        """
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
        if num_clusters == 0:
            raise ValueError("No clusters found!")
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

    @only_signal_axes
    def cluster_labeled_vectors(
        self, method, columns=None, preprocessing="mean", **kwargs
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
        # Remove the nan values but we need to keep track of the original labels
        not_nan = np.logical_not(np.any(np.isnan(to_cluster_vectors), axis=1))
        num_labels = len(to_cluster_vectors)
        to_cluster_vectors = to_cluster_vectors[not_nan]
        clustering = method.fit(to_cluster_vectors)
        labels = clustering.labels_
        initial_labels = self.data[:, -1].astype(int)
        # Replace the values with the original
        actual_labels = np.full(num_labels, -1)
        actual_labels[not_nan] = labels
        new_labels = actual_labels[initial_labels]
        new_labels[initial_labels == -1] = -1
        print(f"{np.max(labels) + 1} : Clusters Found!")
        vectors_and_labels = np.hstack([self.data, new_labels[:, np.newaxis]])
        new_signal = self._deepcopy_with_new_data(data=vectors_and_labels)
        new_signal.ragged = False  # Need to reset this (Remove once hyperspy checks for object --> ragged)
        new_signal.axes_manager.signal_axes[0].size = (
            new_signal.axes_manager.signal_axes[0].size + 1
        )
        new_signal.is_clustered = True
        new_signal.column_names = np.append(self.column_names, ["cluster_label"])
        new_signal.units = np.append(self.units, ["n.a."])
        return new_signal

    @only_signal_axes
    def to_markers(self, signal, get_polygons=False, num_points=10, **kwargs):
        """Convert the labeled vectors to markers

        Parameters
        ----------
        signal: pyxem.signals.ElectronDiffraction2D
            The signal which the markers will be plotted on
        get_polygons: bool
            If True, both the vectors and the polygons will be returned
        num_points: int
            The number of points to be used to create the polygon
        kwargs:
            Any additional arguments to be passed to the :class:`~hyperspy.api.plot.Points` Maker

        Returns
        -------
        points:
            The points to be plotted
        polygons:
            The polygons to be plotted. Only returned if `get_polygons` is True

        """
        offsets, colors, colors_by_index = convert_to_markers(self, signal)
        points = hs.plot.markers.Points(offsets=offsets.T, edgecolor=colors.T, **kwargs)
        if get_polygons:
            verts = self.map_vectors(
                points_to_polygon, num_points=num_points, dtype=object
            )
            verts = list(verts)
            polygons = hs.plot.markers.Polygons(
                verts=verts, alpha=0.5, color=colors_by_index, linewidth=2
            )
            return points, polygons
        return points
