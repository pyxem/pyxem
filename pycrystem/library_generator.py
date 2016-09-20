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

from math import radians, sin

from transforms3d.euler import euler2axangle

import numpy as np

from pymatgen.transformations.standard_transformations import RotationTransformation, DeformStructureTransformation

"""
This module implements a Diffraction Library Generator.
"""

__author_ = "Duncan Johnstone"
__copyright__ = "Copyright 2016, Python Crystallographic Electron Microscopy"
__version__ = "0.1"
__maintainer__ = "Duncan Johnstone"
__email__ = "duncanjohnstone@live.co.uk"
__date__ = 9/15/16

LAUE = ["-1", "2/m", "mmm", "4/m", "4/mmm",
        "-3", "-3m", "6/m", "6/mmm", "m-3", "m-3m"]


class DiffractionLibraryGenerator(object):
    """
    Computes a series of electron diffraction patterns using a kinematical model
    """

    def __init__(self, accelerating_voltage, max_r, excitation_error):
        """
        Initialises the Diffraction Library Generator with parameters for
        calculation of the diffraction patterns.

        Args:
            accelerating_voltage (float): The accelerating voltage of the microscope in kV

            camera_length (float): The camera length at which the diffraction pattern
                is to be calculated in cm.
        """
        self.accelerating_voltage = accelerating_voltage
        self.max_r = max_r
        self.excitation_error = excitation_error

    def get_diffraction_library(self, structure, orientations):
        """
        Calculates a library of electron diffraction paterns for a given structure
        and list of orientations.

        Args:
            structure (Structure): Input structure

            orientations (axis-angle): Whether to return scaled intensities. The maximum
                peak is set to 1. Defaults to True.

            max_theta (float): The maximum angle

        """
        diffraction_library=[]
        ediff = ElectronDiffractionCalculator(self.accelerating_voltage)
        for ori in orientations:
            rot = DeformStructureTransformation(ori)
            rotated_structure = rot.apply_transformation(structure)
            data = ediff.get_ed_data(rotated_structure, self.max_r, self.excitation_error)
            diffraction_library.append(data)

        return diffraction_library

    def get_diffraction_library_plot(self, diffraction_library):
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

    def show_diffraction_library_plot(self, diffraction_library):
        """
        Shows the Electron Diffraction plot.
        Args:
            structure (Structure): Input structure
            max_theta (float): Float for maximum angle.
            annotate_peaks (bool): Whether to annotate the peaks with plane
                information.
        """
        self.get_ed_plot(data).show()


def equispaced_s2_grid(theta_range, phi_range, resolution=2.5):
    """Creates rotations approximately equispaced on a sphere.

    Parameters
    ----------
    theta_range : tuple of float
        (theta_min, theta_max)
        The range of allowable polar angles.
    phi_range : tuple of float
        (phi_min, phi_max)
        The range of allowable azimuthal angles.
    resolution : float
        The angular resolution of the grid in degrees.

    Returns
    -------
    s2_grid : list of tuple
        tuple[0] is an array specifying the axis of rotation
        tuple[1] is the angle of rotation in radians

    """
    theta_min, theta_max = [radians(t) for t in theta_range]
    phi_min, phi_max = [radians(r) for r in phi_range]
    resolution = radians(resolution)
    n_theta = int(theta_max/resolution)
    theta_grid = np.linspace(theta_min, theta_max, n_theta+1)
    phi_grid = []
    for j, theta in enumerate(theta_grid):
        steps = max(round(sin(theta) * phi_max / theta_max * n_theta), 1)
        phi = phi_min\
            + np.arange(steps) * (phi_max - phi_min) / steps \
            + (j % 2) * (phi_max - phi_min) / steps / 2
        phi_grid.append(phi)
    s2_grid = np.array(
        [(theta, phi) for phis, theta in zip(phi_grid, theta_grid) for phi in
         phis])
    x_rotations = np.zeros((len(s2_grid),))
    s2_grid = [euler2axangle(ai, aj, ak, 'sxyz') for ai, aj, ak in
               zip(x_rotations, s2_grid[:, 0], s2_grid[:, 1])]
    return s2_grid
