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

    def get_map(self, k_region=None, symmetry=None):
        """Creates a 2 dimensional map of from the power spectrum.

        Parameters
        ----------
        k_region: array-like
           upper and lower k values to integrate over, allows both ints and floats for indexing
        symmetry: int or array-like
            specific integers or list of symmetries to average over when creating the map of the correlations.

        Returns
        -------
        symmetry_map: 2-d array
            2 dimensional map of from the power spectrum
        """
        if k_region is None:
            k_region = [0, -1]
        if symmetry is None:
            sym_map = (
                self.isig[:, k_region[0] : k_region[1]].sum(axis=[-1, -2]).transpose()
            )

        elif isinstance(symmetry, int):
            sym_map = (
                self.isig[symmetry, k_region[0] : k_region[1]]
                .sum(axis=[-1])
                .transpose()
            )

        else:
            sym_map = Signal2D(data=np.zeros(self.axes_manager.navigation_shape))
            for sym in symmetry:
                sym_map = (
                    self.isig[sym, k_region[0] : k_region[1]].sum(axis=[-1]).transpose()
                    + sym_map
                )
        return sym_map

    def plot_symmetries(self, k_region=None, symmetry=None, *args, **kwargs):
        """Plots the symmetries in the list of symmetries. Plot symmetries takes all of the arguements that imshow does.

        Parameters
        ----------
         k_region: array-like
           upper and lower k values to integrate over, allows both ints and floats for indexing
        symmetry: list or None
            specific integers or list of symmetries to average over when creating the map of the correlations.
            If None, defaults to [2, 4, 6, 8, 10]
        """

        symmetry = [2, 4, 6, 8, 10] if symmetry is None else symmetry

        if k_region is None:
            k_region = [0, -1]
        summed = [self.get_map(k_region=k_region)]
        maps = summed + [self.get_map(k_region=k_region, symmetry=i) for i in symmetry]
        labels = ["summed"] + [str(i) + "-fold" for i in symmetry]
        plot_images(images=maps, label=labels, *args, **kwargs)


class LazyPower2D(LazySignal, Power2D):

    pass
