# -*- coding: utf-8 -*-
# Copyright 2016-2022 The pyXem developers
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


from hyperspy.signals import Signal2D
from hyperspy._signals.lazy import LazySignal

from pyxem.utils.correlation_utils import _correlation, _power, _pearson_correlation


class PolarDiffraction2D(Signal2D):
    """Signal class for two-dimensional diffraction data in polar coordinates."""

    _signal_type = "polar_diffraction"

    def get_angular_correlation(
        self, mask=None, normalize=True, inplace=False, **kwargs
    ):
        r"""Calculate the angular auto-correlation function in the form of a Signal2D class.

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
            The radial correlation for the signal2D, when inplace is False,
            otherwise None

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
        s = self if inplace else correlation
        theta_axis = s.axes_manager.signal_axes[0]

        theta_axis.name = "Angular Correlation, $ \Delta \Theta$"
        theta_axis.offset = 0

        s.set_signal_type("correlation")
        return correlation

    def get_angular_power(self, mask=None, normalize=True, inplace=False, **kwargs):
        """Calculate the power spectrum of the angular auto-correlation function
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
            The power spectrum of the Signal2D, when inplace is False, otherwise
            return None
        """
        power = self.map(
            _power, axis=1, mask=mask, normalize=normalize, inplace=inplace, **kwargs
        )
        
        s = self if inplace else power
        s.set_signal_type("power")
        fourier_axis = s.axes_manager.signal_axes[0]

        fourier_axis.name = "Fourier Coefficient"
        fourier_axis.units = "a.u"
        fourier_axis.scale = 1

        return power

    def get_pearson_correlation(self, mask=None, krange=None, inplace=False, **kwargs):
        """Calculate the pearson rotational correlation in the form of a Signal2D class.

        Parameters
        ----------
        mask: Numpy array
            A bool mask of values to ignore of shape equal to the signal shape.
            True for elements masked, False for elements unmasked
        krange: tuple of int or float
            The range of k values for segment correlation. If type is ``int``,
            the value is taken as the axis index. If type is ``float`` the
            value is in corresponding unit.
            If None (default), use the entire pattern .
        inplace: bool
            From hyperspy.signal.map(). inplace=True means the signal is
            overwritten.

        Returns
        -------
        correlation: Signal2D,
            The pearson rotational correlation when inplace is False, otherwise
            return None
        """
        # placeholder to handle inplace playing well with cropping and mapping
        s_ = self
        if krange is not None:
            if inplace:
                s_.crop(-1, start=krange[0], end=krange[1])
            else:
                s_ = self.isig[:, krange[0]:krange[1]]

            if mask is not None:
                mask = Signal2D(mask)
                # When float krange are used, axis calibration is required
                mask.axes_manager[-1].scale = self.axes_manager[-1].scale
                mask.axes_manager[-1].offset = self.axes_manager[-1].offset
                mask.crop(-1, start=krange[0], end=krange[1])

        correlation = s_.map(
            _pearson_correlation,
            mask=mask,
            inplace=inplace,
            **kwargs
            )

        s = s_ if inplace else correlation
        s.set_signal_type("correlation")
        
        rho_axis = s.axes_manager.signal_axes[0]
        rho_axis.name = "Pearson Correlation, $ \Delta \Theta$"
        rho_axis.offset = 0
        rho_axis.units = 'rad'
        rho_axis.scale = self.axes_manager[-2].scale
        
        return correlation


class LazyPolarDiffraction2D(LazySignal, PolarDiffraction2D):

    pass
