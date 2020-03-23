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
"""Signal class for Electron Diffraction radial profiles

"""

import numpy as np

from hyperspy.api import interactive
from hyperspy.signals import Signal2D
from hyperspy._signals.lazy import LazySignal

from pyxem.signals import push_metadata_through
from pyxem.utils.exp_utils_polar import angular_correlation, angular_power


class PolarDiffraction2D(Signal2D):
    _signal_type = "polar_diffraction2d"

    def __init__(self, *args, **kwargs):
        """
        Create an PolarDiffraction2D object from a hs.Signal2D or np.array.

        Parameters
        ----------
        *args :
            Passed to the __init__ of Signal2D. The first arg should be
            either a numpy.ndarray or a Signal2D
        **kwargs :
            Passed to the __init__ of Signal2D
        """
        self, args, kwargs = push_metadata_through(self, *args, **kwargs)
        super().__init__(*args, **kwargs)

        self.decomposition.__func__.__doc__ = BaseSignal.decomposition.__doc__

    def get_angular_correlation(self, mask=None, normalize=True,
                                inplace=False, **kwargs):
        """
        Returns the angular auto-correlation function in the form of a Signal2D class.

        The angular correlation measures the angular symmetry by computing the self or auto
        correlation. The equation being calculated is
        $ C(\phi,k,n)= \frac{ <I(\theta,k,n)*I(\theta+\phi,k,n)>_\theta-<I(\theta,k,n)>^2}{<I(\theta,k,n)>^2}$

        Parameters
        ---------------
        mask: Numpy array or Signal2D
            A bool mask of values to ignore of shape equal to the signal shape.  If it is equal
            to the size of the array a different mask is applied for every signal. Used for bad
             pixels or beam stop.
        normalize: bool
            Normalize the radial correlation by the average value at some radius.
        kwargs: dict
            Any additional options for the hyperspy.BaseSignal.map() function
        Returns
        --------------
        correlation: Signal2D
            The radial correlation for the signal2D
        """
        if self.axes_manager.signal_shape() == np.shape(mask): # for a static mask
            correlation = self.map(angular_correlation, mask=mask, normalize=normalize,inplace=inplace, **kwargs)
        elif self.axes_manager.shape() == np.shape(mask):  # for a changing mask
            mask = np.reshape(mask, newshape=(-1, *reversed(self.axes_manager.signal_shape)))
            correlation = self._map_iterate(angular_correlation,iterating_kwargs=(('mask', mask),),
                                            normalize=True, inplace=inplace, **kwargs)

        if inplace:
            self.set_signal_type("Signal2D")  # It should already be a Signal 2D object...
            self.axes_manager.signal_axes[1].name = "Angular Correlation, $/phi$"
        else:
            correlation.axes_manager.signal_axes[1].name = "Angular Correlation, $/phi$"

        return correlation

    def get_angular_power(self, mask=None, normalize=True, inplace=False,** kwargs):
        """ Returns the power spectrum of the angular auto-correlation function
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
         Returns
         --------------
         power: Signal2D
             The power spectrum of the Signal2D"""

        if self.axes_manager.signal_shape() == np.shape(mask): # for a static mask
            pow = self.map(angular_power, mask=mask,normalize=normalize,inplace=inplace)
        elif self.axes_manager.shape() == np.shape(mask):  # for a changing mask
            mask = np.reshape(mask, newshape=(-1, *reversed(self.axes_manager.signal_shape)))
            pow = self._map_iterate(angular_power,iterating_kwargs=(('mask', mask),),
                                    normalize=True, inplace=inplace)

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


