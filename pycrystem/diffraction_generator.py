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
"""Kinematical diffraction pattern calculation.

"""


from __future__ import division

import numpy as np
from pycrystem.utils.sim_utils import get_electron_wavelength,\
    get_structure_factors
from pymatgen.util.plotting_utils import get_publication_quality_plot


class ElectronDiffractionCalculator(object):
    """Computes electron diffraction patterns for a crystal structure.

    1. Calculate reciprocal lattice of structure. Find all reciprocal points
       within the limiting sphere given by :math:`\\frac{2}{\\lambda}`.

    2. For each reciprocal point :math:`\\mathbf{g_{hkl}}` corresponding to
       lattice plane :math:`(hkl)`, compute the Bragg condition
       :math:`\\sin(\\theta) = \\frac{\\lambda}{2d_{hkl}}`

    3. The intensity of each reflection is then given in the kinematic
       approximation as the modulus square of the structure factor.
           .. math::
                I_{hkl} = F_{hkl}F_{hkl}^*

    .. todo::
        Include camera length, when implemented.
    .. todo::
        Refactor the excitation error to a structure property.

    Parameters
    ----------
    accelerating_voltage : float
        The accelerating voltage of the microscope in kV
    reciprocal_radius : float
        The maximum radius of the sphere of reciprocal space to sample, in
        reciprocal angstroms.
    excitation_error : float
        The maximum extent of the relrods in reciprocal angstroms. Typically
        equal to 1/{specimen thickness}.

    """

    def __init__(self, accelerating_voltage, reciprocal_radius, excitation_error):

        self.wavelength = get_electron_wavelength(accelerating_voltage)
        self.reciprocal_radius = reciprocal_radius
        self.excitation_error = excitation_error

    def calculate_ed_data(self, structure):
        """Calculates the Electron Diffraction data for a structure.

        Parameters
        ----------
        structure : Structure
            The structure for which to derive the diffraction pattern. Note that
            the structure must be rotated to the appropriate orientation.

        Returns
        -------
        DiffractionSimulation
            The data associated with this structure and diffraction setup.

        """
        wavelength = self.wavelength
        latt = structure.lattice

        # Obtain crystallographic reciprocal lattice points within `max_r`.
        reciprocal_lattice = latt.reciprocal_lattice_crystallographic
        fractional_coordinates = \
            reciprocal_lattice.get_points_in_sphere([[0, 0, 0]], [0, 0, 0],
                                                    self.reciprocal_radius,
                                                    zip_results=False)[0]
        cartesian_coordinates = reciprocal_lattice.get_cartesian_coords(
            fractional_coordinates)

        # Identify points intersecting the Ewald sphere within maximum
        # excitation error and the magnitude of their excitation error.
        radius = 1/wavelength
        r = np.sqrt(np.sum(np.square(cartesian_coordinates[:, :2]), axis=1))
        theta = np.arcsin(r/radius)
        z_sphere = radius * (1 - np.cos(theta))
        proximity = np.absolute(z_sphere - cartesian_coordinates[:, 2])
        intersection = proximity < self.excitation_error

        intersection_coordinates = cartesian_coordinates[intersection]
        intersection_indices = fractional_coordinates[intersection]
        proximity = proximity[intersection]
        intersection_intensities = \
            self.get_peak_intensities(
                structure,
                intersection_indices,
                proximity
            )

        return DiffractionSimulation(
            coordinates=intersection_coordinates,
            indices=intersection_indices,
            intensities=intersection_intensities
        )

    @staticmethod
    def get_peak_intensities(structure, indices, proximities):
        """Calculates peak intensities.

        The peak intensity is a combination of the structure factor for a given
        peak and the position the Ewald sphere intersects the relrod. In this
        implementation, the intensity scales linearly with proximity.

        Parameters
        ----------
        structure : Structure
            The structure for which to derive the structure factors.
        indices : array-like
            The fractional coordinates of the peaks for which to calculate the
            structure factor.
        proximities : array-like
            The distances between the Ewald sphere and the peak centres.

        Returns
        -------
        peak_intensities : array-like
            The intensities of the peaks.

        """
        structure_factors = get_structure_factors(indices, structure)
        peak_relative_proximity = 1 - proximities / np.max(proximities)
        peak_intensities = np.sqrt(structure_factors * peak_relative_proximity)
        return peak_intensities


class DiffractionSimulation:

    def __init__(self, coordinates=None, indices=None, intensities=None,
                 scale=1., offset=(0., 0.)):
        """Holds the result of a given diffraction pattern.

        coordinates : array-like
            The x-y coordinates of points in reciprocal space.
        indices : array-like
            The indices of the reciprocal lattice points that intersect the
            Ewald sphere.
        proximity : array-like
            The distance between the reciprocal lattice points that intersect
            the Ewald sphere and the Ewald sphere itself in reciprocal
            angstroms.
        scale : {:obj:`float`, :obj:`tuple` of :obj:`float`}, optional
            The x- and y-scales of the pattern, with respect to the original
            reciprocal angstrom coordinates.
        offset : :obj:`tuple` of :obj:`float`, optional
            The x-y offset of the pattern in reciprocal angstroms. Defaults to
            zero in each direction.
        """
        self.coordinates = coordinates
        self.indices = indices
        self.intensities = intensities
        self._scale = (1., 1.)
        self.scale = scale
        self.offset = offset

    @property
    def calibrated_coordinates(self):
        """Offset and scaled coordinates."""
        coordinates = np.copy(self.coordinates)
        coordinates[:, 0] += self.offset[0]
        coordinates[:, 1] += self.offset[1]
        coordinates[:, 0] *= self.scale[0]
        coordinates[:, 1] *= self.scale[1]
        return coordinates

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale):
        if isinstance(scale, float) or isinstance(scale, int):
            self._scale = (scale, scale)
        elif len(scale) == 2:
            self._scale = scale
        else:
            raise ValueError("`scale` must be a float, int, or length-2 tuple"
                             "of floats or ints.")

    def plot(self):
        """Returns the diffraction data as a plot.

        Notes
        -----
        Run `.show()` on the result of this method to display the plot.

        """
        plt = get_publication_quality_plot(10, 10)
        plt.scatter(
            self.coordinates[:, 0],
            self.coordinates[:, 1],
            s=self.intensities
        )
        plt.axis('equal')
        plt.xlabel("Reciprocal Dimension ($A^{-1}$)")
        plt.ylabel("Reciprocal Dimension ($A^{-1}$)")
        plt.tight_layout()
        return plt
