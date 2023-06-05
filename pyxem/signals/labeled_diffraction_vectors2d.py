# -*- coding: utf-8 -*-
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

from pyxem.utils.labeled_vector_utils import get_vector_dist


class LabeledDiffractionVectors2D(DiffractionVectors2D):
    """The labeled diffraction vectors 2d signal
    """
    @property
    def unique_counts(self):
        _, counts = np.unique(self.data, return_counts=True, axis=1)
        return

    @property
    def is_clustered(self):
        return self.metadata.Vectors["is_clustered"]

    @is_clustered.setter
    def is_clustered(self, value):
        self.metadata["Vectors.is_clustered"] = value

    def get_dist_matrix(self,
                        max_search=None
                        ):
        """Returns a distance matrix from a list of vectors with labels in the last column of the dataset.
        """
        vectors=self.data
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
            mean_pos = self.map_vectors(np.mean,
                                        label_index=-1,
                                        axis=0,
                                        dtype=float,
                                        shape=(2,))
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

    def map_vectors(self,
                    func,
                    dtype,
                    label_index=-1,
                    shape=None,
                    **kwargs):
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
        vectors= self.data
        labels = vectors[:,label_index]
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

    def cluster_labeled_vectors(self,
                                method="DBSCAN",
                                metric="mean_nn",
                                eps=2,
                                max_search=None,
                                min_samples=2):
        vectors = self.data
        if metric == "mean_nn":
            if max_search is None:
                max_search = eps * 10
            dist_matrix = self.get_dist_matrix(max_search=max_search)
        elif isinstance(metric, np.ndarray):
            dist_matrix = metric
        else:
            raise (ValueError(f"The metric: {metric} must be one of ['mean_nn'] or"
                              f"a numpy distance matrix"))

        if method == "DBSCAN":
            d = DBSCAN(eps=eps,
                       min_samples=min_samples,
                       metric="precomputed").fit(dist_matrix)
            labels = d.labels_
            initial_labels = vectors[:, -1].astype(int)
            new_labels = labels[initial_labels]
            new_labels[initial_labels == -1] = -1
            print(f"{np.max(labels) + 1} : Clusters Found!")
            vectors_and_labels = np.hstack([vectors, new_labels[:, np.newaxis]])
            new_signal = self._deepcopy_with_new_data(data=vectors_and_labels)
            self.is_clustered = True
            return new_signal

    def plot_clustered(self):
        if not self.is_clustered:
            raise ValueError("You must first cluster the dataset using the "
                             "`cluster_labeled_vectors` function.")

