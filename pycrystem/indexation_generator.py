# -*- coding: utf-8 -*-
# Copyright 2016 The PyCrystEM developers
#
# This file is part of PyCrystEM.
#
# PyCrystEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyCrystEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyCrystEM.  If not, see <http://www.gnu.org/licenses/>.
"""Indexation generator and associated tools.

"""

from __future__ import division

import numpy as np
from hyperspy.signals import BaseSignal
from tqdm import tqdm

from .utils import correlate
from .utils.plot import plot_correlation_map

class IndexationGenerator():
    """Generates an indexer for data using a number of methods.
    """
    def __init__(self, signal, library):
        """Initialises the indexer with a diffraction signal and library to be
        correlated in template matching.

        Parameters
        ----------
        signal : :class:`ElectronDiffraction`
            The signal of electron diffraction patterns to be indexed.

        library : :class: `DiffractionLibrary`
            The library of simulated diffraction patterns for indexation

        """
        self.signal = signal
        self.library = library

    def correlate(self, show_progressbar=True):
        signal = self.signal
        library = self.library

        output_array = np.zeros(signal.axes_manager.navigation_shape,
                                dtype=object)
        for image, index in tqdm(zip(signal._iterate_signal(),
                signal.axes_manager._array_indices_generator()),
                disable=not show_progressbar,
                total=signal.axes_manager.navigation_size):
            correlations = Correlation()
            for euler_angle, diffraction_pattern in library.items():
                correlation = correlate(image, diffraction_pattern)
                correlations[euler_angle] = correlation
            output_array[index[::-1]] = correlations
        return output_array


class Correlation(dict):
    """Maps angles to correlation indices.

    Some useful properties and methods are defined.

    """

    @property
    def angles(self):
        """Returns the angles (keys) as a list."""
        return list(self.keys())

    @property
    def correlations(self):
        """Returns the correlations (values) as a list."""
        return list(self.values())

    @property
    def best_angle(self):
        """Returns the angle with the highest correlation index."""
        return max(self, key=self.get)

    @property
    def best_correlation(self):
        """Returns the highest correlation index."""
        return self[self.best_angle]

    @property
    def best(self):
        """Returns angle and value of the highest correlation index."""
        return self.best_angle, self.best_correlation

    def filter_best(self, axes=(-2, -1,)):
        """Reduces the dimensionality of the angles.

        Returns a `Correlation` with only those angles that are unique in
        `axes`. Where there are duplicates, only the angle with the highest
        correlation is retained.

        Parameters
        ----------
        axes : tuple, optional
            The indices of the angles along which to optimise. Default is the
            last *two* indices.

        Returns
        -------
        Correlation

        """
        best_correlations = {}
        for angle in self:
            correlation = self[angle]
            angle = tuple(np.array(angle)[axes,])
            if angle in best_correlations and correlation < best_correlations[angle]:
                continue
            best_correlations[angle] = correlation
        return Correlation(best_correlations)

    def as_signal(self, resolution=np.pi/180, interpolation_method='cubic', fill_value=0.):
        """Returns the correlation as a hyperspy signal.

        Interpolates between angles where necessary to produce a consistent
        grid.

        Parameters
        ----------
        resolution : float, optional
            Resolution of the interpolation, in radians.
        interpolation_method : 'nearest' | 'linear' | 'cubic'
            The method used for interpolation. See
            :func:`scipy.interpolate.griddata` for more details.

        Returns
        -------
        :class:`hyperspy.signals.BaseSignal`

        """
        indices = np.array(self.angles)
        if interpolation_method == 'nearest' and indices.shape[1] > 2:
            raise TypeError("`interpolation_method='nearest'` only works with data of two dimensions or less. Try using `filter_best`.")
        extremes = [slice(q.min(), q.max(), resolution) for q in indices.T]
        z = np.array(self.correlations)
        grid_n = tuple(e for e in np.mgrid[extremes])
        grid = griddata(indices, z, grid_n, method=interpolation_method, fill_value=fill_value)
        return BaseSignal(grid)

    def plot(self, **kwargs):
        angles = np.array(self.angles)
        if angles.shape[1] != 2:
            raise NotImplementedError("Plotting is only available for two-dimensional angles. Try using `filter_best`.")
        angles = angles[:, (1, 0)]
        domain = []
        domain.append((angles[:, 0].min(), angles[:, 0].max()))
        domain.append((angles[:, 1].min(), angles[:, 1].max()))
        correlations = np.array(self.correlations)
        ax = plot_correlation_map(angles, correlations, phi=domain[0], theta=domain[1], **kwargs)
        ax.scatter(self.best_angle[1], self.best_angle[0], c='r', zorder=2, edgecolor='none')
        return ax
