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
from pyxem.utils.peak_utils import mapping_indeces_dictionary


class ElectronDiffractionProfile(Signal1D):
    _signal_type = "diffraction_profile"

    def __init__(self, *args, **kwargs):
        Signal1D.__init__(self, *args, **kwargs)

    def detector_px_to_1D_kspace(self, wavelength, det2sample_len, px_size):
        """Converts the detector 1D coordinates in px, to the respective 1D coordinates in the kspace, in Angstom^-1, using purely geometrical arguments. 
        Only use for DiffractionProfile class. 2D version of the detector_to_3D_kspace from the class DiffractionVectors.
        Args:
        -------
            wavelength (float):
                Wavelength of the scanning beam, in Amstrong.
            det2sample_len (float): 
                Distance from detector to sample, in Amstrong. IMPORTANT: Distance obtained from the calibration file.
            px_size (float):
                Length of the pixel in the detector, in micrometres
        Returns:
        -------
            self:
                DiffractionProfile has been modified from px cordinates to angstrom^-1
        """
        #Extract the pixel-coordinates as an array
        px = np.array(self.axes_manager['k'].axis)
        #Convert each pixel to the actual disctance in Angstrom
        x = px*1e4*px_size
        # Get the two_theta angle (in 1D Edwald circunference it is arctan(px/d)) for each x:
        two_theta = np.arctan2(x,det2sample_len)
        # #Convert each x to the respective gx value:
        # gx = (1/wavelength)*np.sin(two_theta)
        # #Convert each x to the respective gy value:
        # gy = (1/wavelength)*(1-np.cos(two_theta))
        # #Get the diffraction vector g magnitude:
        # g = np.sqrt(gx**2 + gy**2)

        g = 2 * (1/wavelength) * np.sin(two_theta/2)

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
                Each datapoint is an array of the position/magnitude of the diffraction peaks. The intensity of each peak is stored in the "intensity" attribute, with navigation dimensions identical to the original object.
        """
        #Find peaks using the o'Haver function from Hyperspy. It returns a BaseSignal object, in which each data point is a dictionary containing "position" and "height" of each peak.
        peaks = self.find_peaks1D_ohaver(*args, **kwargs)
        #Create a DiffractionPeaks object.
        peaks = DiffractionPeaks(peaks)
        peaks.axes_manager.set_signal_dimension(0)

        #Extract the intensity and store it as an attribute.
        peaks.intensity = peaks.map(mapping_indeces_dictionary, key='height', inplace=False)
        #Extract the peak position. Replace each data point (a dictionary) for a peak position/magnitude array.
        peaks.map(mapping_indeces_dictionary, key='position', inplace=True)

        # For diffraction profiles with navigation axes, transfer them to the DiffractionPeaks object:
        if self.axes_manager.navigation_axes != ():
            x = peaks.axes_manager.navigation_axes[0]
            y = peaks.axes_manager.navigation_axes[1]

            x.name = 'x'
            x.scale = self.axes_manager.navigation_axes[0].scale
            x.units = 'nm'

            y.name = 'y'
            y.scale = self.axes_manager.navigation_axes[1].scale
            y.units = 'nm'
        
        return peaks