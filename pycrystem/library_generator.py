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

from pymatgen.transformations.standard_transformations \
    import RotationTransformation
from transforms3d.euler import axangle2euler

LAUE = ["-1", "2/m", "mmm", "4/m", "4/mmm",
        "-3", "-3m", "6/m", "6/mmm", "m-3", "m-3m"]


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
        diffraction_library : dict of :class:`DiffractionData`
            Mapping of Euler angles of rotation to diffraction data objects.

        """

        diffraction_library = DiffractionLibrary()
        diffractor = self.electron_diffraction_calculator

        for orientation in orientations:
            axis = orientation[0]
            angle = orientation[1]
            euler = axangle2euler(axis, angle)
            rotation = RotationTransformation(axis, angle)
            rotated_structure = rotation.apply_transformation(structure)
            data = diffractor.calculate_ed_data(rotated_structure)
            diffraction_library[euler] = data

        return diffraction_library


class DiffractionLibrary(dict):

    def plot(self):
        """Plots the library interactively.

        ..todo:: Implement this method.

        """
        pass
