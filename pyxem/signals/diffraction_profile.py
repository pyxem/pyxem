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
"""Signal class for Electron Diffraction radial profiles

"""

import numpy as np
from hyperspy.signals import Signal1D
from pyxem.signals.diffraction_peaks import DiffractionPeaks


class ElectronDiffractionProfile(Signal1D):
    _signal_type = "diffraction_profile"

    def __init__(self, *args, **kwargs):
        Signal1D.__init__(self, *args, **kwargs)

    def detector_axes_to_1D_kspace(self, beam_wavelen, det2sample_len, pixel_len):
        """Converts the detector 1D coordinates in px, to the respective 1D coordinates in the kspace, in Angstom^-1, using purely geometrical arguments. 
        Only use for DiffractionProfile class. 2D version of the detector_to_3D_kspace from the class DiffractionVectors.
        Args:
        -------
            beam_wavelen (float):
                Wavelength of the scanning beam, in Amstrong.
            det2sample_len (float): 
                Distance from detector to sample, in Amstrong. IMPORTANT: Exact distance obtained from the calibration file.
            pixel_len (float):
                Length of the pixel in the detector, in micrometres
        Returns:
        -------
            self:
                DiffractionProfile has been modified from px cordinates to angstrom^-1
        """
        #Extract the pixel-coordinates as an array
        px = np.array(self.axes_manager['k'].axis)
        #Convert each pixel to the actual disctance in Angstrom
        x = px*10000*pixel_len
        # Get the theta angle (in 1D Edwald circunference it is arctan(px/d)) for each x:
        theta = np.arctan(x/(det2sample_len))
        #Convert each x to the respective gx value:
        gx = (1/beam_wavelen)*np.sin(theta)
        #Convert each x to the respective gy value:
        gy = (1/beam_wavelen)*(1-np.cos(theta))
        #Get the diffraction vector g magnitude:
        g = np.sqrt(gx**2 + gy**2)

        # Replace pixel coordinates in the ElectronDiffractionProfile to the kx values:
        self.axes_manager['k'].axis = g
        self.axes_manager['k'].name = '$\mid g \mid$'

    def find_peaks_1D(self, *args, **kwargs):
        """Find peaks in a DiffractionProfile using the O-Haver function (Hyperspy). It returns the DiffractionPeaks class.
        Parameters
        ----------
            slope threshold : float 
                Higher values will neglect broader features
            amp/intensity threshold: float
                Intensity below which peaks are ignored
            *args
                Inherited parameters from Hyperspy find_peaks1D_ohaver
        Returns
        ----------
            peaks: DiffractionPeaks
                A DiffractionPeaks object with navigation dimensions identical to the original ElectronDiffraction object.
                Each signal is a BaseSignal object contiaining the diffraction peak "position" found at each navigation position, the "intensity" and the "height" of each peak.
                
                TO DO: The intensity of each peak is stored in the "intensity" attribute, with navigation dimensions identical to the original object.
        """
        peaks = self.find_peaks1D_ohaver(*args, **kwargs)
        #Create a DiffractionPeaks object
        peaks = DiffractionPeaks(peaks)
        peaks.axes_manager.set_signal_dimension(0)

        # Set calibration to same as signal
        x = peaks.axes_manager.navigation_axes[0]
        y = peaks.axes_manager.navigation_axes[1]

        x.name = 'x'
        x.scale = self.axes_manager.navigation_axes[0].scale
        x.units = 'nm'

        y.name = 'y'
        y.scale = self.axes_manager.navigation_axes[1].scale
        y.units = 'nm'
        """
        TO DO:
            1. Extract the intensity value from each peak in each navigation axis, and store it in peak.intensity attribute.
            2. Only store peak "position" in the BaseSignal data, instead of the three returns from the find_peaks1D_ohaver function.
        """
        return peaks