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
Signal class for detector coordinates.

"""


class DetectorCoordinates2D(BaseSignal):
    """Coordinates defining positions on a two-dimensional detector in pixel
    units.

    """
    _signal_type = "detector_coordinates"

    def __init__(self, *args, **kwargs):
        self, args, kwargs = push_metadata_through(self, *args, **kwargs)
        super().__init__(*args, **kwargs)

    def plot_detector_coordinates(self, xlim=1.0, ylim=1.0,
                                  unique_vectors=None,
                                  distance_threshold=0.01,
                                  method='distance_comparison',
                                  min_samples=1,
                                  image_to_plot_on=None,
                                  image_cmap='gray',
                                  plot_label_colors=False,
                                  distance_threshold_all=0.005): # pragma: no cover
        """Plot the unique diffraction vectors.

        Parameters
        ----------
        xlim : float
            The maximum x coordinate in reciprocal Angstroms to be plotted.
        ylim : float
            The maximum y coordinate in reciprocal Angstroms to be plotted.
        unique_vectors : DetectorCoordinates2D, optional
            The unique vectors to be plotted (optional). If not given, the
            unique vectors will be found by get_unique_vectors.
        distance_threshold : float, optional
            The minimum distance in reciprocal Angstroms between diffraction
            vectors for them to be considered unique diffraction vectors.
            Will be passed to get_unique_vectors if no unique vectors are
            given.
        method : str
            The method to use to determine unique vectors, if not given.
            Valid methods are 'strict', 'distance_comparison' and 'DBSCAN'.
            'strict' returns all vectors that are strictly unique and
            corresponds to distance_threshold=0.
            'distance_comparison' checks the distance between vectors to
            determine if some should belong to the same unique vector,
            and if so, the unique vector is iteratively updated to the
            average value.
            'DBSCAN' relies on the DBSCAN [1] clustering algorithm, and
            uses the Eucledian distance metric.
        min_samples : int, optional
            The minimum number of not identical vectors within one cluster
            for it to be considered a core sample, i.e. to not be considered
            noise. Will be passed to get_unique_vectors if no unique vectors
            are given. Only used if method=='DBSCAN'.
        image_to_plot_on : BaseSignal, optional
            If provided, the vectors will be plotted on top of this image.
            The image must be calibrated in terms of offset and scale.
        image_cmap : str, optional
            The colormap to plot the image in.
        plot_label_colors : bool, optional
            If True (default is False), also the vectors contained within each
            cluster will be plotted, with colors according to their
            cluster membership. If True, the unique vectors will be
            calculated by get_unique_vectors. Requires on method=='DBSCAN'.
        distance_threshold_all : float, optional
            The minimum distance, in calibrated units, between diffraction
            vectors inside one cluster for them to be plotted. Only used if
            plot_label_colors is True and requires method=='DBSCAN'.

        Returns
        -------
        fig : matplotlib figure
            The plot as a matplotlib figure.

        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        offset, scale = 0., 1.
        if image_to_plot_on is not None:
            offset = image_to_plot_on.axes_manager[-1].offset
            scale = image_to_plot_on.axes_manager[-1].scale
            ax.imshow(image_to_plot_on, cmap=image_cmap)
        else:
            ax.set_xlim(-xlim, xlim)
            ax.set_ylim(ylim, -ylim)
            ax.set_aspect('equal')

        if plot_label_colors is True and method == 'DBSCAN':
            clusters = self.get_unique_vectors(
                distance_threshold, method='DBSCAN', min_samples=min_samples,
                return_clusters=True)[1]
            labs = clusters.labels_[clusters.core_sample_indices_]
            # Get all vectors from the clustering not considered noise
            cores = clusters.components_
            if cores.size == 0:
                warn('No clusters were found. Check parameters, or '
                     'use plot_label_colors=False.')
            else:
                peaks = DetectorCooreinates2D(cores)
                peaks.axes_manager.set_signal_dimension(1)
                # Since this original number of vectors can be huge, we
                # find a reduced number of vectors that should be plotted, by
                # running a new clustering on all the vectors not considered
                # noise, considering distance_threshold_all.
                peaks = peaks.get_unique_vectors(
                    distance_threshold_all, min_samples=1,
                    return_clusters=False)
                peaks_all_len = peaks.data.shape[0]
                labels_to_plot = np.zeros(peaks_all_len)
                peaks_to_plot = np.zeros((peaks_all_len, 2))
                # Find the labels of each of the peaks to plot by referring back
                # to the list of labels for the original vectors.
                for n, peak in zip(np.arange(peaks_all_len), peaks):
                    index = distance_matrix([peak.data], cores).argmin()
                    peaks_to_plot[n] = cores[index]
                    labels_to_plot[n] = labs[index]
                # Assign a color value to each label, and shuffle these so that
                # adjacent clusters hopefully get distinct colors.
                cmap_lab = get_cmap('gist_rainbow')
                lab_values_shuffled = np.arange(np.max(labels_to_plot) + 1)
                np.random.shuffle(lab_values_shuffled)
                labels_steps = np.array(list(map(
                    lambda n: lab_values_shuffled[int(n)], labels_to_plot)))
                labels_steps = labels_steps / (np.max(labels_to_plot) + 1)
                # Plot all peaks
                for lab, peak in zip(labels_steps, peaks_to_plot):
                    ax.plot((peak[0] - offset) / scale,
                            (peak[1] - offset) / scale, '.',
                            color=cmap_lab(lab))
        if unique_vectors is None:
            unique_vectors = self.get_unique_vectors(
                distance_threshold, method=method, min_samples=min_samples)
        # Plot the unique vectors
        ax.plot((unique_vectors.data.T[0] - offset) / scale,
                (unique_vectors.data.T[1] - offset) / scale, 'kx')
        plt.tight_layout()
        plt.axis('off')
        return fig

    def plot_on_signal(self, signal, *args, **kwargs):
        """Plot the diffraction vectors on a signal.

        Parameters
        ----------
        signal : Diffraction2D
            The Diffraction2D signal object on which to plot the peaks.
            This signal must have the same navigation dimensions as the peaks.
        *args :
            Arguments passed to signal.plot()
        **kwargs :
            Keyword arguments passed to signal.plot()
        """
        mmx, mmy = generate_marker_inputs_from_peaks(self)
        signal.plot(*args, **kwargs)
        for mx, my in zip(mmx, mmy):
            m = markers.point(x=mx, y=my, color='red', marker='x')
            signal.add_marker(m, plot_marker=True, permanent=False)

    def as_diffraction_vectors2d(self, center, calibration,
                                 *args, **kwargs):
        """Transform detector coordinates to two-dimensional diffraction vectors
        with coordinates in calibrated units of reciprocal Angstroms.

        Note that this transformation corresponds to making a flat Ewald sphere
        approximation.

        Parameters
        ----------
        center :
        calibration :
        *args : arguments
            Arguments to be passed to the map method.
        **kwargs : keyword arguments
            Keyword arguments to be passed to the map method.

        Returns
        -------
        vectors : DiffractionVectors2D
            Object containing two-dimensional reciprocal space vectors with
            coordinates [k_x, k_y] for each detector coorinate. The
            navigation dimensions are unchanged.
        """
        vectors = self.map(peaks_as_gvectors,
                           center=np.array(self.axes_manager.signal_shape) / 2 - 0.5,
                           calibration=self.axes_manager.signal_axes[0].scale,
                           inplace=False,
                           parallel=False,
                           *args, **kwargs)
        transfer_navigation_axes(vectors, self)

        return vectors

    def as_diffraction_vectors3d(self,
                                 azimuthal_integrator,
                                 *args, **kwargs):
        """Transform detector coordinates to three-dimensional diffraction vectors
        with coordinates in calibrated units of reciprocal Angstroms.

        Parameters
        ----------
        azimuthal_integrator : pyFAI.azimuthalIntegrator.AzimuthalIntegrator
            A pyFAI Geometry object, containing all the detector geometry
            parameters.
        *args : arguments
            Arguments to be passed to the plot method.
        **kwargs : keyword arguments
            Keyword arguments to be passed to the plot method.

        Returns
        -------
        vectors : DiffractionVectors3D
            Object containing three-dimensional reciprocal space vectors with
            coordinates [k_x, k_y, k_z] for each detector coorinate. The
            navigation dimensions are unchanged.
        """
        vectors = self.map(detector_px_to_3D_kspace,
                             ai=azimuthal_integrator,
                             inplace=False,
                             parallel=False
                             *args, **kwargs)
        transfer_navigation_axes(vectors, self)

        return vectors
