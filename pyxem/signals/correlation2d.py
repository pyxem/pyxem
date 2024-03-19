# -*- coding: utf-8 -*-
# Copyright 2016-2024 The pyXem developers
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


from fractions import Fraction as frac
import numpy as np

from hyperspy.signals import Signal2D
from hyperspy._signals.lazy import LazySignal

from pyxem.utils._correlations import (
    _get_interpolation_matrix,
    _symmetry_stem,
    _corr_to_power,
)
from pyxem.signals.common_diffraction import CommonDiffraction


class Correlation2D(Signal2D, CommonDiffraction):
    """Signal class for Correlation data after applying some correlation along some axis."""

    _signal_type = "correlation"

    def get_angular_power(self, inplace=False, **kwargs):
        """Returns the power spectrum of the angular auto-correlation function
        in the form of a Signal2D class.

        This gives the fourier decomposition of the radial correlation. Due to
        Nyquist sampling the number of fourier coefficients will be equal to the
        angular range.

        Parameters
        ----------
        mask: Numpy array or Signal2D
            A bool mask of values to ignore of shape equal to the signal shape.  If the mask
            is a BaseSignal than it is iterated with the polar signal
         normalize: bool
             Normalize the radial correlation by the average value at some radius.
        inplace: bool
            From :meth:`~hyperspy.signal.BaseSignal.map`. inplace=True means the signal is
            overwritten.

        Returns
        -------
        power: Signal2D
            The power spectrum of the Signal2D
        """
        power = self.map(_corr_to_power, inplace=inplace, **kwargs)

        s = self if inplace else power

        s.set_signal_type("power")
        fourier_axis = s.axes_manager.signal_axes[0]

        fourier_axis.name = "Fourier Coefficient"
        fourier_axis.units = "a.u"
        fourier_axis.scale = 1
        return power

    def get_summed_angular_power(self, inplace=False, **kwargs):
        """Returns the power spectrum of the summed angular auto-correlation function
        over all real space positions.  Averages the angular correlation.

        Parameters
        ----------
        inplace: bool
            From :meth:`~hyperspy.signal.BaseSignal.map`. inplace=True means the signal is
            overwritten.

        Returns
        -------
        power: Power2D
            The power spectrum of summed angular correlation
        """
        power = self.nansum().map(_corr_to_power, inplace=inplace, **kwargs)

        s = self if inplace else power

        s.set_signal_type("power")
        fourier_axis = s.axes_manager.signal_axes[0]

        fourier_axis.name = "Fourier Coefficient"
        fourier_axis.units = "a.u"
        fourier_axis.scale = 1
        return power

    def get_symmetry_coefficient(
        self,
        symmetries=[2, 3, 4, 5, 6, 7, 8, 9, 10],
        angular_range=0,
        method="average",
        include_duplicates=False,
        normalize=True,
        **kwargs
    ):
        """This function is for finding and extracting information about clusters
        based on the angular symmetries. This a pretty catch all method which has
        a couple of different operating principles.
        If k_range=None the function uses a 3 dimensional version of blob finding and attempts to
        find blobs in the 3 dimensional space, x,y,k for each symmetry.
        Parameters
        ------------
        symmetries: list
            The symmetries to calculate
        method: str
            One of max or average
        include_duplicates: bool
            Include duplicates like 2 and 4
        :return:
        """
        angles = [set(frac(j, i) for j in range(0, i)) for i in symmetries]
        if not include_duplicates:
            already_used = set()
            new_angles = []
            for a in angles:
                new_angles.append(a.difference(already_used))
                already_used = already_used.union(a)
            angles = new_angles
        num_angles = [len(a) for a in angles]
        interp = [
            _get_interpolation_matrix(
                a,
                angular_range,
                num_points=self.axes_manager.signal_axes[0].size,
                method=method,
            )
            for a in angles
        ]

        signals = self.map(
            _symmetry_stem, interpolation=interp, inplace=False, method=method, **kwargs
        )
        if method in ["max", "first"]:
            normalize = False
        if normalize:
            signals.data = np.divide(signals.data, num_angles)
        # 2-D signal (x,y,k,s) for each symmetry
        signals.axes_manager.signal_axes[0].scale = 1
        signals.axes_manager.signal_axes[0].name = "Symmetry"
        signals.axes_manager.signal_axes[0].offset = symmetries[0]
        signals.axes_manager.signal_axes[0].unit = None
        return signals


class LazyCorrelation2D(LazySignal, Correlation2D):
    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
