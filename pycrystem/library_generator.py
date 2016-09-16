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
