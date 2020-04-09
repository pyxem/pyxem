# -*- coding: utf-8 -*-
# Copyright 2017-2020 The pyXem developers
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

"""
Signal class for two-dimensional diffraction data in polar coordinates.
"""

from hyperspy.signals import Signal2D, BaseSignal
from hyperspy._signals.lazy import LazySignal

from pyxem.utils.exp_utils_polar import angular_correlation, angular_power, variance, mean_mask
import numpy as np


class Correlation2D(Signal2D):
    _signal_type = "correlation"

    def __init__(self, *args, **kwargs):
        """
        Create a PolarDiffraction2D object from a numpy.ndarray.

        Parameters
        ----------
        *args :
            Passed to the __init__ of Signal2D. The first arg should be
            a numpy.ndarray
        **kwargs :
            Passed to the __init__ of Signal2D
        """
        super().__init__(*args, **kwargs)

        self.decomposition.__func__.__doc__ = BaseSignal.decomposition.__doc__

    def get_angular_power(self, ** kwargs):
        """
        Calculate a power spectrum from the correlation signal

        Parameters
        ----------
        method : str
            'FFT' gives fourier transformation of the angular power spectrum.  Currently the only method available
        """
        power_signal = self.map(power_spectrum,
                                method=method,
                                inplace=False,
                                show_progressbar=False)
        passed_meta_data = self.metadata.as_dictionary()
        if self.metadata.has_item('Masks'):
            del (passed_meta_data['Masks'])
        power = Power2D(power_signal)
        power.axes_manager.navigation_axes = self.axes_manager.navigation_axes

        power.set_axes(-2,
                       name="FourierCoefficient",
                       scale=1,
                       units="a.u.",
                       offset=.5)
        power.set_axes(-1,
                       name="k",
                       scale=self.axes_manager[-1].scale,
                       units=self.axes_manager[-1].units,
                       offset=self.axes_manager[-1].offset)
        return power

    def get_summed_angular_power(self, ** kwargs):
        """Returns the power spectrum of the angular auto-correlation function
         in the form of a Signal2D class.

         This gives the fourier decomposition of the radial correlation. Due to
         nyquist sampling the number of fourier coefficients will be equal to the
         angular range.

         Parameters
         ---------------
         mask: Numpy array or Signal2D
             A mask of values to ignore of shape equal to the signal shape
         normalize: bool
             Normalize the radial correlation by the average value at some radius.
        inplace: bool
            From hyperspy.signal.map(). inplace=True means the signal is
            overwritten.
         Returns
         --------------
         power: Signal2D
             The power spectrum of the Signal2D"""
        pow = self.map(angular_power, mask=mask,normalize=normalize, inplace=inplace, **kwargs)
        if inplace:
            self.set_signal_type("Signal2D")  # It should already be a Signal 2D object...
            self.axes_manager.signal_axes[0].scale = self.axes_manager.signal_axes[0].scale
            self.axes_manager.signal_axes[1].scale = 1
            self.axes_manager.signal_axes[1].name = "Fourier Coefficient"
            self.axes_manager.signal_axes[1].unit = "a.u"
            self.axes_manager.signal_axes[1].offset = 0.5
        else:
            pow.set_signal_type("Signal2D")  # It should already be a Signal 2D object...
            pow.axes_manager.signal_axes[0].scale = self.axes_manager.signal_axes[0].scale
            pow.axes_manager.signal_axes[1].scale = 1
            pow.axes_manager.signal_axes[1].name = "Fourier Coefficient"
            pow.axes_manager.signal_axes[1].unit = "a.u"
            pow.axes_manager.signal_axes[1].offset = 0.5
        return pow

class LazyPolarDiffraction2D(LazySignal, PolarDiffraction2D):

    pass