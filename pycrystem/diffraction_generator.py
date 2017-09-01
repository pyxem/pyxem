# -*- coding: utf-8 -*-
# Copyright 2017 The PyCrystEM developers
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

"""Electron diffraction pattern simulation.

"""


from __future__ import division

import numpy as np
from hyperspy.components2d import Expression

from pycrystem.diffraction_signal import ElectronDiffraction
from pycrystem.utils.sim_utils import get_electron_wavelength,\
    get_kinematical_intensities
from pymatgen.util.plotting import get_publication_quality_plot


_GAUSSIAN2D_EXPR = \
    "intensity * exp(" \
    "-((x-cx)**2 / (2 * sigma ** 2)" \
    " + (y-cy)**2 / (2 * sigma ** 2))" \
    ")"


class ElectronDiffractionCalculator(object):
    """Computes electron diffraction patterns for a crystal structure.

    1. Calculate reciprocal lattice of structure. Find all reciprocal points
       within the limiting sphere given by :math:`\\frac{2}{\\lambda}`.

    2. For each reciprocal point :math:`\\mathbf{g_{hkl}}` corresponding to
       lattice plane :math:`(hkl)`, compute the Bragg condition
       :math:`\\sin(\\theta) = \\frac{\\lambda}{2d_{hkl}}`

    3. The intensity of each reflection is then given in the kinematic
       approximation as the modulus square of the structure factor.
       :math:`I_{hkl} = F_{hkl}F_{hkl}^*`

    Parameters
    ----------
    accelerating_voltage : float
        The accelerating voltage of the microscope in kV.
    max_excitation_error : float
        The maximum extent of the relrods in reciprocal angstroms. Typically
        equal to 1/{specimen thickness}.
    debye_waller_factors : dict of str : float
        Maps element names to their temperature-dependent Debye-Waller factors.

    """
    # TODO: Include camera length, when implemented.
    # TODO: Refactor the excitation error to a structure property.

    def __init__(self,
                 accelerating_voltage,
                 max_excitation_error,
                 debye_waller_factors=None):
        self.wavelength = get_electron_wavelength(accelerating_voltage)
        self.max_excitation_error = max_excitation_error
        self.debye_waller_factors = debye_waller_factors or {}

    def calculate_ed_data(self, structure, reciprocal_radius):
        """Calculates the Electron Diffraction data for a structure.

        Parameters
        ----------
        structure : Structure
            The structure for which to derive the diffraction pattern. Note that
            the structure must be rotated to the appropriate orientation.
        reciprocal_radius : float
            The maximum radius of the sphere of reciprocal space to sample, in
            reciprocal angstroms.

        Returns
        -------
        DiffractionSimulation
            The data associated with this structure and diffraction setup.

        """
        # Specify variables used in calculation
        wavelength = self.wavelength
        max_excitation_error = self.max_excitation_error
        debye_waller_factors = self.debye_waller_factors
        latt = structure.lattice

        # Obtain crystallographic reciprocal lattice points within `max_r` and
        # g-vector magnitudes for intensity calculations.
        recip_latt = latt.reciprocal_lattice_crystallographic
        recip_pts, g_hkls = \
            recip_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0],
                                            reciprocal_radius,
                                            zip_results=False)[:2]
        cartesian_coordinates = recip_latt.get_cartesian_coords(recip_pts)

        # Identify points intersecting the Ewald sphere within maximum
        # excitation error and store the magnitude of their excitation error.
        radius = 1 / wavelength
        r = np.sqrt(np.sum(np.square(cartesian_coordinates[:, :2]), axis=1))
        theta = np.arcsin(r / radius)
        z_sphere = radius * (1 - np.cos(theta))
        proximity = np.absolute(z_sphere - cartesian_coordinates[:, 2])
        intersection = proximity < max_excitation_error
        # Mask parameters corresponding to excited reflections.
        intersection_coordinates = cartesian_coordinates[intersection]
        intersection_indices = recip_pts[intersection]
        proximity = proximity[intersection]
        g_hkls = g_hkls[intersection]

        # Calculate diffracted intensities based on a kinematical model.
        intensities = get_kinematical_intensities(structure,
                                                  intersection_indices,
                                                  g_hkls,
                                                  proximity,
                                                  max_excitation_error,
                                                  debye_waller_factors)

        # Threshold peaks included in simulation based on minimum intensity.
        peak_mask = intensities > 1e-20
        intensities = intensities[peak_mask]
        intersection_coordinates = intersection_coordinates[peak_mask]
        intersection_indices = intersection_indices[peak_mask]

        return DiffractionSimulation(coordinates=intersection_coordinates,
                                     indices=intersection_indices,
                                     intensities=intensities)


class DiffractionSimulation:
    """Holds the result of a given diffraction pattern.

    Parameters
    ----------

    coordinates : array-like, shape [n_points, 2]
        The x-y coordinates of points in reciprocal space.
    indices : array-like, shape [n_points, 3]
        The indices of the reciprocal lattice points that intersect the
        Ewald sphere.
    intensities : array-like, shape [n_points, ]
        The intensity of the reciprocal lattice points.
    calibration : float or tuple of float, optional
        The x- and y-scales of the pattern, with respect to the original
        reciprocal angstrom coordinates.
    offset : tuple of float, optional
        The x-y offset of the pattern in reciprocal angstroms. Defaults to
        zero in each direction.
    """

    def __init__(self, coordinates=None, indices=None, intensities=None,
                 calibration=1., offset=(0., 0.), with_direct_beam=False):
        """Initializes the DiffractionSimulation object with data values for
        the coordinates, indices, intensities, calibration and offset.
        """
        self._coordinates = None
        self.coordinates = coordinates
        self.indices = indices
        self._intensities = None
        self.intensities = intensities
        self._calibration = (1., 1.)
        self.calibration = calibration
        self.offset = offset
        self.with_direct_beam = with_direct_beam

    @property
    def calibrated_coordinates(self):
        """ndarray : Coordinates converted into pixel space."""
        coordinates = np.copy(self.coordinates)
        coordinates[:, 0] += self.offset[0]
        coordinates[:, 1] += self.offset[1]
        coordinates[:, 0] /= self.calibration[0]
        coordinates[:, 1] /= self.calibration[1]
        return coordinates

    @property
    def calibration(self):
        """tuple of float : The x- and y-scales of the pattern, with respect to
        the original reciprocal angstrom coordinates."""
        return self._calibration

    @calibration.setter
    def calibration(self, calibration):
        if isinstance(calibration, float) or isinstance(calibration, int):
            self._calibration = (calibration, calibration)
        elif len(calibration) == 2:
            self._calibration = calibration
        else:
            raise ValueError("`calibration` must be a float or length-2"
                             "tuple of floats.")

    @property
    def direct_beam_mask(self):
        """ndarray : If `with_direct_beam` is True, returns a True array for all
        points. If `with_direct_beam is False, returns a True array with False
        in the position of the direct beam."""
        if self.with_direct_beam:
            return np.ones_like(self._intensities, dtype=bool)
        else:
            return np.sum(self._coordinates == 0., axis=1) != 3

    @property
    def coordinates(self):
        """ndarray : The coordinates of all unmasked points."""
        return self._coordinates[self.direct_beam_mask]

    @coordinates.setter
    def coordinates(self, coordinates):
        self._coordinates = coordinates

    @property
    def intensities(self):
        """ndarray : The intensities of all unmasked points."""
        return self._intensities[self.direct_beam_mask]

    @intensities.setter
    def intensities(self, intensities):
        self._intensities = intensities

    def plot(self, ax=None):
        """Plots the simulated electron diffraction pattern with a logarithmic
        intensity scale.

        Run `.show()` on the result of this method to display the plot.

        Parameters
        ----------
        ax : :obj:`matplotlib.axes.Axes`, optional
            A `matplotlib` axes instance. Used if the plot needs to be in a
            figure that has been created elsewhere.

        """
        if ax is None:
            plt = get_publication_quality_plot(10, 10)
            ax = plt.gca()
        ax.scatter(
            self.coordinates[:, 0],
            self.coordinates[:, 1],
            s=np.log10(self.intensities)
        )
        ax.set_xlabel("Reciprocal Dimension ($A^{-1}$)")
        ax.set_ylabel("Reciprocal Dimension ($A^{-1}$)")
        return ax

    def as_signal(self, size, sigma, max_r):
        """Returns the diffraction data as an ElectronDiffraction signal with
        two-dimensional Gaussians representing each diffracted peak.

        Parameters
        ----------
        size : int
            Side length (in pixels) for the signal to be simulated.
        sigma : float
            Standard deviation of the Gaussian function to be plotted.
        max_r : float
            Half the side length in reciprocal Angstroms. Defines the signal's
            calibration.

        """
        # Plot a 2D Gaussian at each peak position.
        # TODO: Update this method so plots intensity at each position and then
        # convolves with a Gaussian to make faster - needs interpolation care.
        dp_dat = 0
        l = np.linspace(-max_r, max_r, size)
        x, y = np.meshgrid(l, l)
        coords = self.coordinates[:, :2]
        g = Expression(_GAUSSIAN2D_EXPR, 'Gaussian2D', module='numexpr')
        for (cx, cy), intensity in zip(coords, self.intensities):
            g.intensity.value = intensity
            g.sigma.value = sigma
            g.cx.value = cx
            g.cy.value = cy
            dp_dat += g.function(x, y)

        dp = ElectronDiffraction(dp_dat)
        dp.set_calibration(2*max_r/size)

        return dp
