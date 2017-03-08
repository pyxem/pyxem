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
from hyperspy.signals import BaseSignal
from pymatgen.transformations.standard_transformations \
    import RotationTransformation
from scipy.interpolate import griddata
from tqdm import tqdm
from transforms3d.euler import euler2axangle
from scipy.constants import pi


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

    def get_diffraction_library(self,
                                structure_library,
                                calibration,
                                representation='euler'):
        """Calculates a list of diffraction data for a structure.

        The structure is rotated to each orientation in `orientations` and the
        diffraction pattern is calculated each time.

        .. todo:: convert `RotationTransformation` to general transformation and
            `orientations` to `deformations`

        Parameters
        ----------
        structure : :class:`Structure`
            List of structures for which to derive the library.
        orientations : list of tuple
            tuple[0] is an array specifying the axis of rotation
            tuple[1] is the angle of rotation in radians
        format :

        Returns
        -------
        diffraction_library : dict of :class:`DiffractionSimulation`
            Mapping of Euler angles of rotation to diffraction data objects.

        """
        #TODO: update this method to include multiple phases and to incorporate
        #crystal symmetry properly
        diffraction_library = DiffractionLibrary()
        diffractor = self.electron_diffraction_calculator

        for key in structure_library.keys():
            phase_diffraction_library = dict()
            structure = structure_library[key][0]
            orientations = structure_library[key][1]
            for orientation in orientations:
                if representation=='axis-angle':
                    axis = [orientation[0], orientation[1], orientation[2]]
                    angle = orientation[3] / 180 * pi
                if representation=='euler':
                    axis, angle = euler2axangle(orientation[0], orientation[1],
                                                orientation[2], 'rzyz')
                rotation = RotationTransformation(axis, angle,
                                                  angle_in_radians=True)
                rotated_structure = rotation.apply_transformation(structure)
                data = diffractor.calculate_ed_data(rotated_structure)
                data.calibration = calibration
                phase_diffraction_library[tuple(orientation)] = data
            diffraction_library[key] = phase_diffraction_library
        return diffraction_library


class DiffractionLibrary(dict):
    """Maps structure and orientation (Euler angles) to simulated diffraction
    data.
    """

    def set_calibration(self, calibration):
        """Sets the scale of every diffraction pattern simulation in the
        library.

        Parameters
        ----------
        calibration : {:obj:`float`, :obj:`tuple` of :obj:`float`}, optional
            The x- and y-scales of the patterns, with respect to the original
            reciprocal angstrom coordinates.

        """
        for key in self.keys():
            diff_lib = self[key]
            for diffraction_pattern in diff_lib.values():
                diffraction_pattern.calibration = calibration

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
        for key in self.keys():
            diff_lib = self[key]
            for diffraction_pattern in diff_lib.values():
                diffraction_pattern.offset = offset

    def plot(self):
        """Plots the library interactively.

        """
        #TODO: Update this so not so stupid and therefore faster
        from pycrystem.diffraction_signal import ElectronDiffraction
        sim_diff_dat = []
        for key in self.keys():
            for ori in self[key].keys():
                dpi = self[key][ori].as_signal(128, 0.03, 1)
                sim_diff_dat.append(dpi.data)
        ppt_test = ElectronDiffraction(sim_diff_dat)
        ppt_test.plot()
