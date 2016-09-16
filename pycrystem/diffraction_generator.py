# -*- coding: utf-8 -*-
# Copyright 2016 The PyCrystEM developers
#
# This file is part of  PyCrystEM.
#
#  PyCrystEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  PyCrystEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  PyCrystEM.  If not, see <http://www.gnu.org/licenses/>.
from __future__ import division

from scipy.constants import h, m_e, e, c, pi
import math
import numpy as np
from .sim_utils import *

"""
This module implements an Electron Diffraction pattern calculator.
"""

__author_ = "Duncan Johnstone"
__copyright__ = "Copyright 2016, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Duncan Johnstone"
__email__ = "duncanjohnstone@live.co.uk"
__date__ = 9/15/16


class ElectronDiffractionCalculator(object):
    """
    Computes Electron Diffraction patterns of a crystal structure in any orientation.


    1. Calculate reciprocal lattice of structure. Find all reciprocal points
       within the limiting sphere given by :math:`\\frac{2}{\\lambda}`.

    2. For each reciprocal point :math:`\\mathbf{g_{hkl}}` corresponding to
       lattice plane :math:`(hkl)`, compute the Bragg condition
       :math:`\\sin(\\theta) = \\frac{\\lambda}{2d_{hkl}}`

    3. The intensity of each reflection is then given in the kinematic
       approximation as the modulus square of the structure factor.
       .. math::
           I_{hkl} = F_{hkl}F_{hkl}^*
    """

    def __init__(self, accelerating_voltage):
        """
        Initialises the Electron Diffraction calculator with a given accelerating voltage
        and camera length.


        Args:
            accelerating_voltage (float): The accelerating voltage of the microscope in kV

            camera_length (float): The camera length at which the diffraction pattern
                is to be calculated in cm.
        """
        self.wavelength = get_electron_wavelength(accelerating_voltage)

    def get_ed_data(self, structure, max_r, excitation_error):
        """
        Calculates the Electron Diffraction data for a structure.

        Args:
            structure (Structure): Input structure

            max_r (float): The maximum angle

        Returns:
            (Electron Diffraction pattern) in the form of
        """
        wavelength = self.wavelength
        latt = structure.lattice

        # Obtain crystallographic reciprocal lattice points within range.
        recip_latt = latt.reciprocal_lattice_crystallographic
        recip_zip = recip_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], max_r)
        recip_pts = recip_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0],
                                                    max_r, zip_results=False)[0]
        recip_cart = recip_latt.get_cartesian_coords(recip_pts)

        # Identify points intersecting the Ewald sphere within maximum excitation error
        # and the magnitude of their excitation error.
        radius = 1/wavelength
        r = np.sqrt(np.sum(np.square(recip_cart[:, :2]), axis=1))
        theta = np.arcsin(r/radius)
        z_sphere = radius * (1 - np.cos(theta))
        proximity = np.absolute(z_sphere - recip_cart[:, 2])
        intersection = proximity < excitation_error

        icoords = recip_cart[intersection]
        sfcoords = structure.lattice.reciprocal_lattice_crystallographic.get_fractional_coords(icoords)

        data= [icoords, sfcoords, proximity, intersection]

        return data

    def get_ed_plot(self, structure, data):
        """
        Returns the Electron Diffraction plot as a matplotlib.pyplot.
        Args:
            structure (Structure): Input structure
            max_theta (float): Tuple for range of
                two_thetas to calculate in degrees. Defaults to (0, 90). Set to
                None if you want all diffracted beams within the limiting
                sphere of radius 2 / wavelength.
            annotate_peaks (bool): Whether to annotate the peaks with plane
                information.
        Returns:
            (matplotlib.pyplot)
        """
        from pymatgen.util.plotting_utils import get_publication_quality_plot
        plt = get_publication_quality_plot(10, 10)
        plt.scatter(data[0][:, 0], data[0][:, 1],
                    s=np.sqrt(get_structure_factors(data[1],
                                                    structure))*(1-data[2][data[3]]))
        plt.axis('equal')
        plt.xlabel("Reciprocal Dimension ($A^{-1}$)")
        plt.ylabel("Reciprocal Dimension ($A^{-1}$)")
        plt.tight_layout()
        return plt

    def show_ed_plot(self, structure, data):
        """
        Shows the Electron Diffraction plot.
        Args:
            structure (Structure): Input structure
            max_theta (float): Float for maximum angle.
            annotate_peaks (bool): Whether to annotate the peaks with plane
                information.
        """
        self.get_ed_plot(structure, data).show()
