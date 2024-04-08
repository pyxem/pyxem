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


import numpy as np
from warnings import warn

from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN

from hyperspy.signals import Signal2D
from hyperspy.roi import CircleROI
from pyxem.signals import DiffractionVectors
import hyperspy.api as hs


class DiffractionVectors2D(DiffractionVectors, Signal2D):
    """DiffractionVectors signal with the same number of vectors at each navigation
    position.

    This class can also be used to represent a list of vectors with no navigation axis.
    """

    _signal_dimension = 2
    _signal_type = "diffraction_vectors"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_unique_vectors(
        self,
        distance_threshold=0.01,
        method="distance_comparison",
        min_samples=1,
        return_clusters=False,
        columns=(-2, -1),
    ):
        """Returns diffraction vectors considered unique by:
        strict comparison, distance comparison with a specified
        threshold, or by clustering using DBSCAN [1].

        Parameters
        ----------
        distance_threshold : float
            The minimum distance between diffraction vectors for them to
            be considered unique diffraction vectors. If
            distance_threshold==0, the unique vectors will be determined
            by strict comparison.
        method : str
            The method to use to determine unique vectors. Valid methods
            are 'strict', 'distance_comparison' and 'DBSCAN'.
            'strict' returns all vectors that are strictly unique and
            corresponds to distance_threshold=0.
            'distance_comparison' checks the distance between vectors to
            determine if some should belong to the same unique vector,
            and if so, the unique vector is iteratively updated to the
            average value.
            'DBSCAN' relies on the DBSCAN [1] clustering algorithm, and
            uses the Eucledian distance metric.
        min_samples : int, optional
            The minimum number of not strictly identical vectors within
            one cluster for the cluster to be considered a core sample,
            i.e. to not be considered noise. Only used for method='DBSCAN'.
        return_clusters : bool, optional
            If True (False is default), the DBSCAN clustering result is
            returned. Only used for method='DBSCAN'.

        References
        ----------
        [1] "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html"

        Returns
        -------
        unique_peaks : DiffractionVectors
            The unique diffraction vectors.
        clusters : DBSCAN
            The results from the clustering, given as class DBSCAN.
            Only returned if method='DBSCAN' and return_clusters=True.
        """

        # A distance_threshold of 0 implies a strict comparison. So in that
        # case, a warning is raised unless the specified method is 'strict'.
        data = self.data[:, columns]
        if distance_threshold == 0:
            if method != "strict":
                warn(
                    "distance_threshold=0 was given, and therefore "
                    "a strict comparison is used, even though the "
                    "specified method was {}".format(method)
                )
                method = "strict"

        if method == "strict":
            unique_peaks = np.unique(data, axis=0)

        elif method == "distance_comparison":
            unique_vectors, unique_counts = np.unique(data, axis=0, return_counts=True)

            unique_peaks = np.array([[0, 0]])
            unique_peaks_counts = np.array([0])

            while unique_vectors.shape[0] > 0:
                unique_vector = unique_vectors[0]
                distances = distance_matrix(np.array([unique_vector]), unique_vectors)
                indices = np.where(distances < distance_threshold)[1]

                new_count = indices.size
                new_unique_peak = np.array(
                    [
                        np.average(
                            unique_vectors[indices],
                            weights=unique_counts[indices],
                            axis=0,
                        )
                    ]
                )

                unique_peaks = np.append(unique_peaks, new_unique_peak, axis=0)

                unique_peaks_counts = np.append(unique_peaks_counts, new_count)
                unique_vectors = np.delete(unique_vectors, indices, axis=0)
                unique_counts = np.delete(unique_counts, indices, axis=0)
            unique_peaks = np.delete(unique_peaks, [0], axis=0)

        elif method == "DBSCAN":
            # All peaks are clustered by DBSCAN so that peaks within
            # one cluster are separated by distance_threshold or less.
            unique_vectors, unique_vectors_counts = np.unique(
                data, axis=0, return_counts=True
            )
            clusters = DBSCAN(
                eps=distance_threshold, min_samples=min_samples, metric="euclidean"
            ).fit(unique_vectors, sample_weight=unique_vectors_counts)
            unique_labels, unique_labels_count = np.unique(
                clusters.labels_, return_counts=True
            )
            unique_peaks = np.zeros((unique_labels.max() + 1, 2))

            # For each cluster, a center of mass is calculated based
            # on all the peaks within the cluster, and the center of
            # mass is taken as the final unique vector position.
            for n in np.arange(unique_labels.max() + 1):
                peaks_n_temp = unique_vectors[clusters.labels_ == n]
                peaks_n_counts_temp = unique_vectors_counts[clusters.labels_ == n]
                unique_peaks[n] = np.average(
                    peaks_n_temp, weights=peaks_n_counts_temp, axis=0
                )

        # Manipulate into DiffractionVectors class
        if unique_peaks.size > 0:
            unique_peaks = DiffractionVectors2D(unique_peaks)
        if return_clusters and method == "DBSCAN":
            return unique_peaks, clusters
        else:
            return unique_peaks

    def __lt__(self, other):
        return self._deepcopy_with_new_data(self.data < other)

    def __le__(self, other):
        return self._deepcopy_with_new_data(self.data <= other)

    def __gt__(self, other):
        return self._deepcopy_with_new_data(self.data > other)

    def __ge__(self, other):
        return self._deepcopy_with_new_data(self.data >= other)

    def from_peaks(cls, **kwargs):
        raise NotImplementedError("This method is not implemented for 2D vectors")

    def filter_detector_edge(self, exclude_width, columns=[-2, -1]):
        """Filter the diffraction vectors to accept only those not within a
        user specified proximity to the detector edge.

        Parameters
        ----------
        exclude_width : int
            The width of the region adjacent to the detector edge from which
            vectors will be excluded.
        Returns
        -------
        filtered_vectors : DiffractionVectors
            Diffraction vectors within allowed detector region.
        """
        values = self.data[:, columns]
        offsets = np.array(self.column_offsets)[columns]
        scales = np.array(self.column_scale)[columns]

        xlow = -offsets[0] + scales[0] * exclude_width
        xhigh = (
            -offsets[0]
            + (scales[0] * self.detector_shape[0])
            - (scales[0] * exclude_width)
        )

        ylow = -offsets[1] + scales[1] * exclude_width
        yhigh = (
            -offsets[1]
            + (scales[1] * self.detector_shape[1])
            - (scales[1] * exclude_width)
        )

        inbounds = (
            (values[:, 0] > xlow)
            * (values[:, 0] < xhigh)
            * (values[:, 1] > ylow)
            * (values[:, 1] < yhigh)
        )
        new_data = self.data[inbounds]

        return DiffractionVectors2D(new_data)

    # Functions that only work for 2D Vectors with no navigation axis

    @property
    def has_navigation_axis(self):
        return len(self.axes_manager.navigation_axes) > 0

    def to_roi(self, radius=0.1, columns=None, include_labels=False, **kwargs):
        """
        Convert the diffraction vectors to regions of interest (ROIs) for creating virtual images.

        Parameters
        ----------
        radius: float
            The radius of the ROIs in calibrated units
        columns: list
            The columns to use for the ROIs. If None, columns 0 and 1 are used to create the ROIs.
        include_labels: bool
            If True, the labels for each ROI are returned as well using hyperspy's
             :class:`hyperspy.api.plot.markers.Texts` class.
        kwargs: dict
            Keyword arguments to pass to the :class:`hyperspy.api.plot.markers.Texts` class.
        """
        if self.has_navigation_axis:
            raise NotImplementedError(
                "This method is not implemented for 2D vectors with a navigation axis"
            )
        if columns is None:
            columns = [0, 1]

        rois = [
            CircleROI(vector[columns[1]], vector[columns[0]], r=radius)
            for vector in self.data
        ]
        if include_labels:
            pos = [(vector[columns[1]], vector[columns[0]]) for vector in self.data]
            label = [f"{i}" for i in range(len(self.data))]
            texts = hs.plot.markers.Texts(offsets=pos, texts=label, **kwargs)
            return rois, texts
        else:
            return rois
