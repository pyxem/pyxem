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
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

from warnings import warn

from diffsims.utils.sim_utils import get_electron_wavelength

from pyxem.signals import push_metadata_through
from pyxem.signals import transfer_navigation_axes
from pyxem.utils.vector_utils import detector_to_fourier
from pyxem.utils.vector_utils import calculate_norms, calculate_norms_ragged
from pyxem.utils.vector_utils import get_npeaks
from pyxem.utils.expt_utils import peaks_as_gvectors
from pyxem.utils.plot import generate_marker_inputs_from_peaks

from pyxem.signals.diffraction_vectors2d import DiffractionVectors2D
from pyxem.signals.diffraction_vectors3d import DiffractionVectors3D

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

    def get_npeaks_map(self, binary=False):
        """Map of the number of peaks at each navigation position.

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
        npeaks_map = self.map(get_npeaks, inplace=False).as_signal2D((0, 1))
        # If binary set values greater than 0 to 1
        if binary == True:
            npeaks_map = npeaks_map == 1
        # Ensure floating point output
        npeaks_map.change_dtype('float')
        # Set calibration to same as signal
        for i in [0,1]:
            axis = npeaks_map.axes_manager.signal_axes[i]
            axis.scale = self.axes_manager.navigation_axes[i].scale
            axis.units = 'nm'
            axis.name = 'x' if i == 0 else 'y'

        return npeaks_map

    def as_diffraction_vectors2d(self, center, calibration,
                                 *args, **kwargs):
        """Transform detector coordinates to two-dimensional diffraction vectors
        with coordinates in calibrated units of reciprocal Angstroms.

        Note that this transformation corresponds to making a flat Ewald sphere
        approximation.

        Parameters
        ----------
        center : array like, optional
            Coordinates of the diffraction pattern center in pixel units.
        calibration : float
            Calibrated pixel size in units of reciprocal Angstroms per pixel.
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
                           calibration=calibration,
                           center=center,
                           inplace=False,
                           ragged=True,
                           *args, **kwargs)
        vectors = DiffractionVectors2D(vectors)
        vectors.axes_manager.set_signal_dimension(0)
        # Set coordinates in vectors attributes
        vectors.detector_coordinates = self
        # Transfer navigation axes from the detector coordinates object to the
        # new DiffractionVectors2D object.
        transfer_navigation_axes(vectors, self)

        return vectors

    def as_diffraction_vectors3d(self,
                                 detector,
                                 origin,
                                 detector_distance,
                                 *args, **kwargs):
        """Transform detector coordinates to three-dimensional diffraction vectors
        with coordinates in calibrated units of reciprocal Angstroms.

        Parameters
        ----------
        detector : pyFAI.detectors.Detector object
            A pyFAI detector used for the AzimuthalIntegrator.
        origin : np.array_like
            This parameter should either be a list or numpy.array with two
            coordinates ([x_origin,y_origin]), or an array of the same shape as
            the navigation axes, with an origin (with the shape
            [x_origin,y_origin]) at each navigation location.
        detector_distance : float
            Detector distance in meters passed to pyFAI AzimuthalIntegrator.
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
        # Define pyFAI azimuthal integrator
        p1, p2 = origin[0] * detector.pixel1, origin[1] * detector.pixel2
        ai = AzimuthalIntegrator(dist=detector_distance, poni1=p1, poni2=p2,
                                 detector=detector, wavelength=wavelength,
                                 **kwargs_for_integrator)
        # Map detector coordinates to 3D reciprocal space
        vectors = self.map(detector_px_to_3D_kspace,
                           ai=ai,
                           inplace=False,
                           parallel=False,
                           *args, **kwargs)
        vectors = DiffractionVectors3D(vectors)
        transfer_navigation_axes(vectors, self)

        return vectors
