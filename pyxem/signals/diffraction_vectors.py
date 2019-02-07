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
from scipy.spatial import distance_matrix

from pyxem.utils.sim_utils import transfer_navigation_axes
from pyxem.utils.vector_utils import detector_to_fourier
from pyxem.utils.vector_utils import calculate_norms, calculate_norms_ragged
from pyxem.utils.vector_utils import get_indices_from_distance_matrix
from pyxem.utils.vector_utils import get_npeaks

from pyxem.utils.plot import generate_marker_inputs_from_peaks

"""
Signal class for diffraction vectors.

There are two cases that are supported:

1. A map of diffraction vectors, which will in general be a ragged signal of
signals. It the navigation dimensions of the map and contains a signal for each
peak at every position.

2. A list of diffraction vectors with dimensions < n | 2 > where n is the
number of peaks.
"""


class DiffractionVectors(BaseSignal):
    """Crystallographic mapping results containing the best matching crystal
    phase and orientation at each navigation position with associated metrics.

    Attributes
    ----------
    cartesian : np.array()
        Array of 3-vectors describing Cartesian coordinates associated with
        each diffraction vector.
    hkls : np.array()
        Array of Miller indices associated with each diffraction vector
        following indexation.
    """
    _signal_type = "diffraction_vectors"

    def __init__(self, *args, **kwargs):
        BaseSignal.__init__(self, *args, **kwargs)
        self.cartesian = None
        self.hkls = None

    def plot_diffraction_vectors(self, xlim, ylim, distance_threshold):
        """Plot the unique diffraction vectors.

        Parameters
        ----------
        xlim : float
            The maximum x coordinate to be plotted.
        ylim : float
            The maximum y coordinate to be plotted.
        distance_threshold : float
            The minimum distance between diffraction vectors to be passed to
            get_unique_vectors.

        Returns
        -------
        fig : matplotlib figure
            The plot as a matplot lib figure.

        """
        # Find the unique gvectors to plot.
        unique_vectors = self.get_unique_vectors(distance_threshold)
        # Plot the gvector positions
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(unique_vectors.data.T[0], -unique_vectors.data.T[1], 'ro')
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)
        ax.set_aspect('equal')
        return fig

    def plot_diffraction_vectors_on_signal(self, signal, *args, **kwargs):
        """Plot the diffraction vectors on a signal.

        Parameters
        ----------
        signal : ElectronDiffraction
            The ElectronDiffraction signal object on which to plot the peaks.
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

    def get_unique_vectors(self,
                           distance_threshold=0):
        """Obtain the unique diffraction vectors.

        Parameters
        ----------
        distance_threshold : float
            The minimum distance between diffraction vectors for them to be
            considered unique diffraction vectors.

        Returns
        -------
        unique_vectors : DiffractionVectors
            A DiffractionVectors object containing only the unique diffraction
            vectors in the original object.
        """
        if (self.axes_manager.navigation_dimension == 2):
            gvlist = np.array([self.data[0, 0][0]])
        else:
            raise ValueError("This method only works for ragged vector maps!")

        for i in self._iterate_signal():
            vlist = i[0]
            distances = distance_matrix(gvlist, vlist)
            new_indices = get_indices_from_distance_matrix(distances,
                                                           distance_threshold)
            gvlist_new = vlist[new_indices]
            if gvlist_new.any():
                gvlist = np.concatenate((gvlist, gvlist_new), axis=0)

        # An internal check, just to be sure.
        delete_indices = []
        l = np.shape(gvlist)[0]
        distances = distance_matrix(gvlist, gvlist)
        for i in range(np.shape(distances)[1]):
            if (np.sum(distances[:, i] <= distance_threshold) > 1):
                delete_indices = np.append(delete_indices, i)
        gvecs = np.delete(gvlist, delete_indices, axis=0)

        # Manipulate into DiffractionVectors class
        unique_vectors = DiffractionVectors(gvecs)
        unique_vectors.axes_manager.set_signal_dimension(1)

        return unique_vectors

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

    def calculate_cartesian_coordinates(self, accelerating_voltage, camera_length,
                                        *args, **kwargs):
        """Get cartesian coordinates of the diffraction vectors.

        Parameters
        ----------
        accelerating_voltage : float
            The acceleration voltage with which the data was acquired.
        camera_length : float
            The camera length in meters.
        """
        # Imported here to avoid circular dependency
        from pyxem.utils.sim_utils import get_electron_wavelength
        wavelength = get_electron_wavelength(accelerating_voltage)
        self.cartesian = self.map(detector_to_fourier,
                                  wavelength=wavelength,
                                  camera_length=camera_length * 1e10,
                                  inplace=False,
                                  parallel=False,  # TODO: For testing
                                  *args, **kwargs)
        transfer_navigation_axes(self.cartesian, self)
