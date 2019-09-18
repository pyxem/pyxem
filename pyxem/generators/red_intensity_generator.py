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

"""Reduced intensity generator and associated tools.


"""
import numpy as np
import matplotlib.pyplot as plt

from hyperspy.signals import Signal1D

from pyxem.signals.electron_diffraction1d import ElectronDiffraction1D
from pyxem.signals.reduced_intensity1d import ReducedIntensity1D

from pyxem.components.scattering_fit_component import ScatteringFitComponent
from pyxem.utils.ri_utils import scattering_to_signal
from pyxem.signals import transfer_navigation_axes
from pyxem.signals import transfer_signal_axes


class ReducedIntensityGenerator():
    """Generates a reduced intensity 1D profile for a specified diffraction radial
    profile.


    Parameters
    ----------
    signal : ElectronDiffraction1D
        An electron diffraction radial profile.
    """

    def __init__(self, signal, *args, **kwargs):
        self.signal = signal
        self.cutoff = [0, signal.axes_manager.signal_axes[0].size - 1]
        self.nav_size = [signal.axes_manager.navigation_axes[0].size,
                         signal.axes_manager.navigation_axes[1].size]
        self.sig_size = [signal.axes_manager.signal_axes[0].size]
        self.background_fit = None  # added in one of the fits below.
        self.normalisation = None

    def set_diffraction_calibration(self, calibration):
        """
        Defines calibration for the signal axis variable s in terms of
        A^-1 per pixel. Note that s is defined here as
        s = 2 sin(theta)/lambda = 1/d.

        Parameters
        ----------
        calibration: float
                    Calibration in terms of A^-1 per pixel.
        """
        self.signal.axes_manager.signal_axes[0].scale = calibration
        return

    def set_cutoff_vector(self, s_min, s_max):
        """
        Scattering vector cutoff for the purposes of fitting an atomic scattering
        factor to the 1D profile. Specified in terms of s (in inverse angstroms).
        s is defined as s = 2 sin(theta)/lambda = 1/d.

        Parameters
        ----------
        s_min: float
                    Minimum scattering vector amplitude for cutoff.
        s_max: float
                    Maximum scattering vector amplitude for cutoff.
        """
        #s_scale = self.signal.axes_manager.signal_axes[0].scale
        self.cutoff = [s_min, s_max]
        return

    def fit_atomic_scattering(self, elements, fracs,
                              N=1., C=0., scattering_factor='lobato',
                              plot_fit=True):
        """Fits a diffraction intensity profile to the background using
        FIT = N * sum(ci * (fi^2) + C)

        The cutoff for the scattering factor fit to s is defined via the function
        set_cutoff_vector above.

        Parameters
        ----------
        elements: list of str
                    A list of elements present (by symbol). No order is necessary.
                    Example: ['Ca', 'C', 'O'] (for CaCO3)
        fracs: list of float
                    A list of fraction of the respective elements. Should sum to 1.
                    Example: [0.2, 0.2, 0.6] (for CaCO3)
        N : float
                    The "slope" of the fit.
        C : float
                    An additive constant to the fit.
        scattering_factor : str
                    Type of scattering parameters fitted. Default is lobato.
                    See scattering_fit_component for more details.
        plot_fit: bool
                    A bool to decide if the fit from scattering is plotted
                    after fitting.
        """

        fit_model = self.signal.create_model()
        background = ScatteringFitComponent(elements, fracs, N, C, scattering_factor)

        fit_model.append(background)
        fit_model.set_signal_range(self.cutoff)
        fit_model.multifit()
        fit_model.reset_signal_range()
        if plot_fit == True:
            fit_model.plot()
        C_values = background.C.as_signal()
        N_values = background.N.as_signal()
        s_size = self.sig_size[0]
        s_scale = self.signal.axes_manager.signal_axes[0].scale
        fit, normalisation = scattering_to_signal(elements, fracs, N_values,
                                                  C_values, s_size, s_scale, scattering_factor)
        # self.fit = np.array(background.sum_squares).reshape(
        #            self.nav_size[0],self.nav_size[1],self.sig_size[0])

        self.normalisation = normalisation  # change this
        self.background_fit = fit
        return

    def subtract_bkgd_pattern(self, bkgd_pattern):
        """Subtracts a background pattern from the signal. This method will edit
        self.signal.

        Parameters
        ----------
        Bkgd_pattern : np.array
                    A numpy array of a single line profile of the same resolution
                    (same number of pixels) as the radial profile.
        """
        self.signal = self.signal - bkgd_pattern

        return

    def mask_from_bkgd_pattern(self, mask_pattern, mask_threshold=1):
        """Uses a background pattern with a threshold, and sets that part of
        the signal to zero, effectively adding a mask. This can be used to mask
        the central beam. This method will edit self.signal.

        Parameters
        ----------
        mask_pattern : np.array
                    A numpy array line profile of the same resolution
                    as the radial profile.
        mask_threshold : int or float
                    An integer or float threshold. Any pixel in the
                    mask_pattern with lower intensity is kept, any with
                    higher or equal is set to zero.
        """

        mask_array = mask_pattern < mask_threshold

        self.signal = self.signal * mask_array.astype(float)

        return

    def get_reduced_intensity(self):
        """Obtains a reduced intensity profile from the radial profile.

        Parameters
        ----------
        s_cutoff : list of float
                    A list of the form [s_min, s_max] to change the s_cutoff
                    from the fit.
        """

        # define numerical cutoff to remove certain data parts
        s_scale = self.signal.axes_manager.signal_axes[0].scale

        s = np.arange(self.signal.axes_manager.signal_axes[0].size,
                      dtype='float64')
        s *= self.signal.axes_manager.signal_axes[0].scale
        # remember axes scale and size!
        reduced_intensity = (4 * np.pi * s *
                             np.divide((self.signal.data - self.background_fit),
                                       self.normalisation))

        #ri = ReducedIntensityProfile(reduced_intensity.data[:,:,num_min:num_max])
        ri = ReducedIntensity1D(reduced_intensity)
        transfer_navigation_axes(ri,self.signal)
        transfer_signal_axes(ri,self.signal)

        return ri
