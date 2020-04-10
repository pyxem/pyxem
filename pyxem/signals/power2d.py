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
from hyperspy.drawing.utils import plot_images


class Power2D(Signal2D):
    _signal_type = "power"

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

    def get_i_vs_k(self, symmetry=None):
        """ Get the intensity versus k for the summed diffraction patterns

        Parameters
        ----------
        symmetry: int or array-like
            specific integers or list of symmetries to average over when creating the map of the correlations.
        Returns
        ----------
        i: Signal-2D
            The intensity as a function of k for some signal
        """
        if symmetry is None:
            i = self.isig[:, :].sum(axis=[0, 1, 2])

        elif isinstance(symmetry, int):
            i = self.isig[symmetry, :].sum()
            print(i)

        else:
            i = Signal1D(data=np.zeros(self.axes_manager.signal_shape[1]))
            for sym in symmetry:
               i = self.isig[sym, :].sum() + i
        return i

    def get_map(self, k_region=[3.0, 6.0], symmetry=None):
        """Creates a 2 dimensional map of from the power spectrum.

        Parameters
        ----------
        k_region: array-like
           upper and lower k values to integrate over, allows both ints and floats for indexing
        symmetry: int or array-like
            specific integers or list of symmetries to average over when creating the map of the correlations.
        Returns
        ----------
        symmetry_map: 2-d array
            2 dimensional map of from the power spectrum
        """
        if symmetry is None:
            sym_map = self.isig[:, k_region[0]:k_region[1]].sum(axis=[-1, -2]).transpose()

        elif isinstance(symmetry, int):
            sym_map = self.isig[symmetry, k_region[0]:k_region[1]].sum(axis=[-1]).transpose()

        else:
            sym_map = Signal2D(data=np.zeros(self.axes_manager.navigation_shape))
            for sym in symmetry:
                sym_map = self.isig[sym, k_region[0]:k_region[1]].sum(axis=[-1]).transpose() + sym_map
        return sym_map

    def plot_symmetries(self, k_region=[3.0, 6.0], symmetry=[2, 4, 6, 8, 10], *args, **kwargs):
        """Plots the symmetries in the list of symmetries. Plot symmetries takes all of the arguements that imshow does.

        Parameters
        -------------
         k_region: array-like
           upper and lower k values to integrate over, allows both ints and floats for indexing
        symmetry: list
            specific integers or list of symmetries to average over when creating the map of the correlations.
        """
        summed = [self.get_map(k_region=k_region)]
        maps = summed + [self.get_map(k_region=k_region, symmetry=i) for i in symmetry]
        l = ["summed"] + [str(i) +"-fold" for i in symmetry]
        plot_images(images=maps, label=l, *args, **kwargs)

class LazyPolarDiffraction2D(LazySignal, Power2D):

    pass