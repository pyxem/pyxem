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
"""Diffraction pattern library generator and associated tools.

"""

from __future__ import division

import numpy as np
from scipy.interpolate import griddata

from hyperspy.signals import BaseSignal
from pymatgen.transformations.standard_transformations \
    import RotationTransformation
from transforms3d.euler import euler2axangle
from tqdm import tqdm

from .utils import correlate
from .utils.plot import plot_correlation_map


class DiffractionLibraryGenerator(object):
    """
    Computes a series of electron diffraction patterns using a kinematical model
    """

    def __init__(self, electron_diffraction_calculator):
        """Initialises the library with a diffraction calculator.

        Parameters
        ----------
        electron_diffraction_calculator : :class:`ElectronDiffractionCalculator`
            The calculator used for the diffraction patterns.

        """
        self.electron_diffraction_calculator = electron_diffraction_calculator

    def get_diffraction_library(self, structure, orientations):
        """Calculates a list of diffraction data for a structure.

        The structure is rotated to each orientation in `orientations` and the
        diffraction pattern is calculated each time.

        .. todo:: convert `RotationTransformation` to general transformation and
            `orientations` to `deformations`

        Parameters
        ----------
        structure : :class:`Structure`
            The structure for which to derive the library.
        orientations : list of tuple
            tuple[0] is an array specifying the axis of rotation
            tuple[1] is the angle of rotation in radians

        Returns
        -------
        diffraction_library : dict of :class:`DiffractionSimulation`
            Mapping of Euler angles of rotation to diffraction data objects.

        """

        diffraction_library = DiffractionLibrary()
        diffractor = self.electron_diffraction_calculator

        for orientation in tqdm(orientations):
            axis, angle = euler2axangle(orientation[0], orientation[1], orientation[2], 'rzyz')
            rotation = RotationTransformation(axis, angle,
                                              angle_in_radians=True)
            rotated_structure = rotation.apply_transformation(structure)
            data = diffractor.calculate_ed_data(rotated_structure)
            diffraction_library[tuple(orientation)] = data

        return diffraction_library


class DiffractionLibrary(dict):

    def plot(self):
        """Plots the library interactively.

        .. todo::
            Implement this method.

        """
        pass

    def correlate(self, image: np.ndarray, show_progressbar=True):
        """Finds the correlation between an image and the entire library.

        Parameters
        ----------
        image : :class:`numpy.ndarray`
            A numpy array of the data to be correlated.

        Returns
        -------
        correlations : Correlation
            A mapping of Euler angles to correlation values.

        """
        correlations = Correlation()
        for euler_angle, diffraction_pattern in tqdm(self.items(), disable=not show_progressbar, leave=False):
            correlation = correlate(image,
                                    diffraction_pattern)
            correlations[euler_angle] = correlation
        return correlations

    def index(self, image):
        correlations = self.correlate(image)
        return correlations

    def set_calibration(self, calibration):
        """Sets the scale of every diffraction pattern simulation in the
        library.

        Parameters
        ----------
        calibration : {:obj:`float`, :obj:`tuple` of :obj:`float`}, optional
            The x- and y-scales of the patterns, with respect to the original
            reciprocal angstrom coordinates.

        """
        for diffraction_pattern in self.values():
            diffraction_pattern.calibration = calibration
        return self

    def set_offset(self, offset):
        """Sets the offset of every diffraction pattern simulation in the
        library.

        Parameters
        ----------
        offset : :obj:`tuple` of :obj:`float`, optional
            The x-y offset of the patterns in reciprocal angstroms. Defaults to
            zero in each direction.

        """
        assert len(offset) == 2
        for diffraction_pattern in self.values():
            diffraction_pattern.offset = offset
        return self


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
