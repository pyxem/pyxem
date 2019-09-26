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

from hyperspy.signals import BaseSignal, Signal1D
from hyperspy.api import markers

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN

from warnings import warn

from pyxem.signals import push_metadata_through
from pyxem.signals import transfer_navigation_axes
from pyxem.utils.vector_utils import detector_to_fourier
from pyxem.utils.vector_utils import calculate_norms, calculate_norms_ragged
from pyxem.utils.vector_utils import get_npeaks
from pyxem.utils.expt_utils import peaks_as_gvectors
from pyxem.utils.plot import generate_marker_inputs_from_peaks

"""
Signal class for diffraction vectors.

There are two cases that are supported:

1. A map of diffraction vectors, which will in general be a ragged signal of
signals. It the navigation dimensions of the map and contains a signal for each
peak at every position.

2. A list of diffraction vectors with dimensions < n | 3 > where n is the
number of peaks.
"""


class DiffractionVectors3D(BaseSignal):
    """Three-dimensional diffraction vectors in reciprocal Angstrom units.

    Attributes
    ----------
    hkls : np.array()
        Array of Miller indices associated with each diffraction vector
        following indexation.
    """
    _signal_type = "diffraction_vectors"

    def __init__(self, *args, **kwargs):
        self, args, kwargs = push_metadata_through(self, *args, **kwargs)
        super().__init__(*args, **kwargs)
        self.cartesian = None
        self.hkls = None

    @classmethod
    def from_peaks(cls, peaks, center, calibration):
        """Takes a list of peak positions (pixel coordinates) and returns
        an instance of `Diffraction3D`

        Parameters
        ----------
        peaks : Signal
            Signal containing lists (np.array) of pixel coordinates specifying
            the reflection positions
        center : np.array
            Diffraction pattern center in array indices.
        calibration : np.array
            Calibration in reciprocal Angstroms per pixels for each of the dimensions.

        Returns
        -------
        vectors : :obj:`pyxem.signals.diffraction_vectors.DiffractionVectors2D`
            List of diffraction vectors
        """
        gvectors = peaks.map(peaks_as_gvectors,
                             center=center,
                             calibration=calibration,
                             inplace=False)

        vectors = cls(gvectors)
        vectors.axes_manager.set_signal_dimension(0)

        return vectors

    def get_magnitudes(self, *args, **kwargs):
        """Calculate the magnitude of diffraction vectors.

        Parameters
        ----------
        *args:
            Arguments to be passed to map().
        **kwargs:
            Keyword arguments to map().

        Returns
        -------
        magnitudes : BaseSignal
            A signal with navigation dimensions as the original diffraction
            vectors containging an array of gvector magnitudes at each
            navigation position.

        """
        # If ragged the signal axes will not be defined
        if len(self.axes_manager.signal_axes) == 0:
            magnitudes = self.map(calculate_norms_ragged,
                                  inplace=False,
                                  *args, **kwargs)
        # Otherwise easier to calculate.
        else:
            magnitudes = BaseSignal(calculate_norms(self))
            magnitudes.axes_manager.set_signal_dimension(0)

        return magnitudes

    def get_magnitude_histogram(self, bins, *args, **kwargs):
        """Obtain a histogram of gvector magnitudes.

        Parameters
        ----------
        bins : numpy array
            The bins to be used to generate the histogram.
        *args:
            Arguments to get_magnitudes().
        **kwargs:
            Keyword arguments to get_magnitudes().

        Returns
        -------
        ghis : Signal1D
            Histogram of gvector magnitudes.

        """
        gmags = self.get_magnitudes(*args, **kwargs)

        if len(self.axes_manager.signal_axes) == 0:
            glist = []
            for i in gmags._iterate_signal():
                for j in np.arange(len(i[0])):
                    glist.append(i[0][j])
            gs = np.asarray(glist)
            gsig = Signal1D(gs)
            ghis = gsig.get_histogram(bins=bins)

        else:
            ghis = gmags.get_histogram(bins=bins)

        ghis.axes_manager.signal_axes[0].name = 'k'
        ghis.axes_manager.signal_axes[0].units = '$A^{-1}$'

        return ghis

    def get_unique_vectors(self, distance_threshold=0.01,
                           method='distance_comparison', min_samples=1,
                           return_clusters=False):
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
        method : string
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
        [1] https://scikit-learn.org/stable/modules/generated/sklearn.
            cluster.DBSCAN.html

        Returns
        -------
        unique_peaks : DiffractionVectors2D
            The unique diffraction vectors.
        clusters : DBSCAN
            The results from the clustering, given as class DBSCAN.
            Only returned if method='DBSCAN' and return_clusters=True.
        """
        # Flatten the array of peaks to reach dimension (n, 2), where n
        # is the number of peaks.
        peaks_all = np.concatenate([
            peaks.ravel() for peaks in self.data.flat]).reshape(-1, 2)

        # A distance_threshold of 0 implies a strict comparison. So in that
        # case, a warning is raised unless the specified method is 'strict'.
        if distance_threshold == 0:
            if method is not 'strict':
                warn(message='distance_threshold=0 was given, and therefore ' +
                     'a strict comparison is used, even though the ' +
                     'specified method was ' + method + '.')
                method = 'strict'

        if method == 'strict':
            unique_peaks = np.unique(peaks_all, axis=0)

        elif method == 'DBSCAN':
            # All peaks are clustered by DBSCAN so that peaks within
            # one cluster are separated by distance_threshold or less.
            unique_vectors, unique_vectors_counts = np.unique(
                peaks_all, axis=0, return_counts=True)
            clusters = DBSCAN(
                eps=distance_threshold, min_samples=min_samples,
                metric='euclidean').fit(
                unique_vectors, sample_weight=unique_vectors_counts)
            unique_labels, unique_labels_count = np.unique(
                clusters.labels_, return_counts=True)
            unique_peaks = np.zeros((unique_labels.max() + 1, 2))
            # For each cluster, a center of mass is calculated based
            # on all the peaks within the cluster, and the center of
            # mass is taken as the final unique vector position.
            for n in np.arange(unique_labels.max() + 1):
                peaks_n_temp = unique_vectors[clusters.labels_ == n]
                peaks_n_counts_temp = unique_vectors_counts[
                    clusters.labels_ == n]
                unique_peaks[n] = np.average(
                    peaks_n_temp, weights=peaks_n_counts_temp,
                    axis=0)
        # Manipulate into DiffractionVectors2D class
        if unique_peaks.size > 0:
            unique_peaks = DiffractionVectors2D(unique_peaks)
            unique_peaks.axes_manager.set_signal_dimension(1)
        if return_clusters and method == 'DBSCAN':
            return unique_peaks, clusters
        else:
            return unique_peaks

    def get_diffracting_pixels_map(self, binary=False):
        """Map of the number of vectors at each navigation position.

        Parameters
        ----------
        binary : boolean
            If True a binary image with diffracting pixels taking value == 1 is
            returned.

        Returns
        -------
        crystim : Signal2D
            2D map of diffracting pixels.
        """
        crystim = self.map(get_npeaks, inplace=False).as_signal2D((0, 1))

        if binary == True:
            crystim = crystim == 1

        crystim.change_dtype('float')

        # Set calibration to same as signal
        x = crystim.axes_manager.signal_axes[0]
        y = crystim.axes_manager.signal_axes[1]

        x.name = 'x'
        x.scale = self.axes_manager.navigation_axes[0].scale
        x.units = 'nm'

        y.name = 'y'
        y.scale = self.axes_manager.navigation_axes[0].scale
        y.units = 'nm'

        return crystim

    def as_detector_coordinates2d(self, beam_energy,
                                  camera_length,
                                  pixel_size):
        pass
