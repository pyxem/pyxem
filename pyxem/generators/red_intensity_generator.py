# -*- coding: utf-8 -*-
# Copyright 2017-2018 The pyXem developers
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

from pyxem.signals.diffraction_profile import ElectronDiffractionProfile
from pyxem.signals.reduced_intensity_profile import ReducedIntensityProfile

from pyxem.components.scattering_fit_component import ScatteringFitComponent
from pyxem.utils.ri_utils import scattering_to_signal


class ReducedIntensityGenerator():
    """Generates a reduced intensity profile for a specified diffraction radial
    profile.


    Parameters
    ----------
    signal : ElectronDiffractionProfile
        An electron diffraction radial average profile.
    """

    def __init__(self, signal, *args, **kwargs):
        self.signal = signal
        self.cutoff = [0, signal.axes_manager.signal_axes[0].size - 1]
        self.nav_size = [signal.axes_manager.navigation_axes[0].size,
                         signal.axes_manager.navigation_axes[1].size]
        self.sig_size = [signal.axes_manager.signal_axes[0].size]
        self.background_fit = None  # added in one of the fits below.
        self.normalisation = None

    def specify_scattering_calibration(self, calibration):
        """
        Defines calibration for the signal axis variable s in terms of
        A^-1 per pixel.
        """
        self.signal.axes_manager.signal_axes[0].scale = calibration
        return

    def specify_cutoff_vector(self, s_min, s_max):
        """
        Specified in terms of s (in inverse angstroms).
        """
        #s_scale = self.signal.axes_manager.signal_axes[0].scale
        self.cutoff = [s_min, s_max]
        return

    def fit_atomic_scattering(self, elements, fracs,
                              N=1., C=0., type='lobato',
                              plot_fit=True):
        """Fits a diffraction intensity profile to the background using
        FIT = N * sum(ci * (fi^2) + C)

        NOTE: define s cutoff via the function specify_cutoff_vector
        s_cutoff is given as a function of scattering vector

        Parameters
        ----------
        elements: a list of elements present (by symbol)
        fracs: a list of fraction of the respective elements
        N = the "slope"
        C = an additive constant
        type = type of scattering parameters fitted. Default is lobato.
                See scattering_fit_component for more details.
        plot_fit: a bool to decide if the fit from scattering is plotted
        """

        fit_model = self.signal.create_model()
        background = ScatteringFitComponent(elements, fracs, N, C, type)

        fit_model.append(background)
        fit_model.set_signal_range(self.cutoff)
        fit_model.multifit()
        fit_model.reset_signal_range()
        if plot_fit == True:
            fit_model.plot()
        if self.nav_size[0] == 1 and self.nav_size[1] == 1:
            fit = fit_model.as_signal()
            normalisation = background.square_sum  # change this
        else:
            C_values = background.C.as_signal()
            N_values = background.N.as_signal()
            s_size = self.sig_size[0]
            s_scale = self.signal.axes_manager.signal_axes[0].scale
            fit, normalisation = scattering_to_signal(elements, fracs, N_values,
                                                      C_values, s_size, s_scale, type)
        # self.fit = np.array(background.sum_squares).reshape(
        #            self.nav_size[0],self.nav_size[1],self.sig_size[0])

        self.normalisation = normalisation  # change this
        self.background_fit = fit
        return

    def subtract_bkgd_pattern(self, bkgd_pattern):
        """Fits a diffraction intensity profile to the signal by using a
        diffraction pattern from an area with no sample in it. This is to
        reduce the effects of the central beam. This method will edit
        self.signal.

        Parameters
        ----------
        Bkgd_pattern : A numpy array line profile of the same resolution
        as the radial profile
        """
        self.signal = self.signal - bkgd_pattern

        return

    def get_reduced_intensity(self, cutoff=None):
        if cutoff:
            self.cutoff = cutoff
        else:
            cutoff = self.cutoff

        # define numerical cutoff to remove certain data parts
        s_scale = self.signal.axes_manager.signal_axes[0].scale
        num_min, num_max = int(cutoff[0] / s_scale), int(cutoff[1] / s_scale)

        s = np.arange(self.signal.axes_manager.signal_axes[0].size,
                      dtype='float64')
        s *= self.signal.axes_manager.signal_axes[0].scale
        # remember axes scale and size!
        reduced_intensity = (4 * np.pi * s *
                             np.divide((self.signal.data - self.background_fit),
                                       self.normalisation))

        #ri = ReducedIntensityProfile(reduced_intensity.data[:,:,num_min:num_max])
        ri = ReducedIntensityProfile(reduced_intensity)
        ax_old = self.signal.axes_manager.navigation_axes
        ri.axes_manager.navigation_axes[0].scale = ax_old[0].scale
        ri.axes_manager.navigation_axes[0].units = ax_old[0].units
        ri.axes_manager.navigation_axes[0].name = ax_old[0].name
        if len(ax_old) > 1:
            ri.axes_manager.navigation_axes[1].scale = ax_old[1].scale
            ri.axes_manager.navigation_axes[1].units = ax_old[1].units
            ri.axes_manager.navigation_axes[1].name = ax_old[1].name

        ri_axis = ri.axes_manager.signal_axes[0]
        ri_axis.name = 's'
        ri_axis.scale = self.signal.axes_manager.signal_axes[0].scale
        ri_axis.units = '$A^{-1}$'

        return ri
