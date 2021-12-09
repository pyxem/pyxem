# -*- coding: utf-8 -*-
# Copyright 2016-2021 The pyXem developers
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

"""Signal class for two-dimensional diffraction data in polar coordinates."""

from hyperspy.signals import Signal2D
from hyperspy._signals.lazy import LazySignal

from pyxem.utils.correlation_utils import _correlation, _power, _pearson_correlation


class PolarDiffraction2D(Signal2D):
    _signal_type = "polar_diffraction"

    def __init__(self, *args, **kwargs):
        """Create a PolarDiffraction2D object from a numpy.ndarray.

        Parameters
        ----------
        *args :
            Passed to the __init__ of Signal2D. The first arg should be
            a numpy.ndarray
        **kwargs :
            Passed to the __init__ of Signal2D
        """
        super().__init__(*args, **kwargs)

    def get_angular_correlation(
        self, mask=None, normalize=True, inplace=False, **kwargs
    ):
        r"""Returns the angular auto-correlation function in the form of a Signal2D class.

        The angular correlation measures the angular symmetry by computing the self or auto
        correlation. The equation being calculated is
        $ C(\phi,k,n)= \frac{ <I(\theta,k,n)*I(\theta+\phi,k,n)>_\theta-<I(\theta,k,n)>^2}{<I(\theta,k,n)>^2}$

        Parameters
        ----------
        mask: Numpy array or Signal2D
            A bool mask of values to ignore of shape equal to the signal shape.  If the mask
            is a BaseSignal than it is iterated with the polar signal
        normalize: bool
            Normalize the radial correlation by the average value at some radius.
        kwargs: dict
            Any additional options for the hyperspy.BaseSignal.map() function
        inplace: bool
            From hyperspy.signal.map(). inplace=True means the signal is
            overwritten.

        Returns
        -------
        correlation: Signal2D
            The radial correlation for the signal2D

        Examples
        --------
        Basic example, no mask applied and normalization applied.
        >polar.get_angular_correlation()
        Angular correlation with a static matst for

        """
        correlation = self.map(
            _correlation,
            axis=1,
            mask=mask,
            normalize=normalize,
            inplace=inplace,
            **kwargs
        )
        if inplace:
            self.set_signal_type("correlation")
            correlation_axis = self.axes_manager.signal_axes[0]
        else:
            correlation.set_signal_type("correlation")
            correlation_axis = correlation.axes_manager.signal_axes[0]
        correlation_axis.name = "Angular Correlation, $/phi$"
        return correlation

    def get_angular_power(self, mask=None, normalize=True, inplace=False, **kwargs):
        """Returns the power spectrum of the angular auto-correlation function
        in the form of a Signal2D class.

        This gives the fourier decomposition of the radial correlation. Due to
        nyquist sampling the number of fourier coefficients will be equal to the
        angular range.

        Parameters
        ----------
        mask: Numpy array or Signal2D
            A bool mask of values to ignore of shape equal to the signal shape.  If the mask
            is a BaseSignal than it is iterated with the polar signal
         normalize: bool
             Normalize the radial correlation by the average value at some radius.
        inplace: bool
            From hyperspy.signal.map(). inplace=True means the signal is
            overwritten.

        Returns
        -------
        power: Signal2D
            The power spectrum of the Signal2D
        """
        power = self.map(
            _power, axis=1, mask=mask, normalize=normalize, inplace=inplace, **kwargs
        )
        if inplace:
            self.set_signal_type("power")
            fourier_axis = self.axes_manager.signal_axes[0]
        else:
            power.set_signal_type("power")
            fourier_axis = power.axes_manager.signal_axes[0]
        fourier_axis.name = "Fourier Coefficient"
        fourier_axis.units = "a.u"
        fourier_axis.offset = 0.5
        fourier_axis.scale = 1
        return power

    def get_pearson_correlation(self, mask=None, krange=None, inplace=False, **kwargs):
        """Returns the pearson rotational correlation in the form of a Signal2D class.

        Parameters
        ----------
        mask: Numpy array
            A bool mask of values to ignore of shape equal to the signal shape. True for
            elements masked, False for elements unmasked
        krange: tuple
            The range of k values in corresponding unit for segment correlation (None if use
            the entire pattern)
        inplace: bool
            From hyperspy.signal.map(). inplace=True means the signal is
            overwritten.

        Returns
        -------
        correlation: Signal2D
        """
        if krange is None:
            correlation = self._map_iterate(_pearson_correlation, mask=mask, inplace=inplace, **kwargs)
        else:
            self_slice = self.isig[:, krange[0]:krange[1]]
            if mask is not None:
                mask_signal = Signal2D(mask)
                mask_signal.axes_manager.signal_axes[1].scale = self.axes_manager[-1].scale
                mask_slice = mask_signal.isig[:, krange[0]:krange[1]]
                correlation = self_slice._map_iterate(_pearson_correlation, mask=mask_slice, inplace=inplace, **kwargs)
            else:
                correlation = self_slice._map_iterate(_pearson_correlation, inplace=inplace, **kwargs)

        if inplace:
            self.set_signal_type("symmetry")
            rho_axis = self.axes_manager.signal_axes[0]
        else:
            correlation.set_signal_type("symmetry")
            rho_axis = correlation.axes_manager.signal_axes[0]
            correlation.axes_manager.navigation_axes = self.axes_manager.navigation_axes
        rho_axis.name = "Radians"
        rho_axis.units = 'rad'
        rho_axis.scale = self.axes_manager[-2].scale
        return correlation


class LazyPolarDiffraction2D(LazySignal, PolarDiffraction2D):

    pass
