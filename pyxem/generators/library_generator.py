# -*- coding: utf-8 -*-
# Copyright 2017-2018 The pyXem developers
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

"""Diffraction pattern library generator and associated tools.

"""

import numpy as np
from pyxem.libraries.diffraction_library import DiffractionLibrary
from tqdm import tqdm
from transforms3d.euler import euler2mat
import diffpy.structure


class DiffractionLibraryGenerator(object):
    """
    Computes a library of electron diffraction patterns for specified atomic
    structures and orientations.
    """

    def __init__(self, electron_diffraction_calculator):
        """Initialises the library with a diffraction calculator.

        Parameters
        ----------
        electron_diffraction_calculator : :class:`DiffractionGenerator`
            The calculator used for the diffraction patterns.

        """
        self.electron_diffraction_calculator = electron_diffraction_calculator

    def get_diffraction_library(self,
                                structure_library,
                                calibration,
                                reciprocal_radius,
                                half_shape,
                                with_direct_beam=True
                                ):
        """Calculates a dictionary of diffraction data for a library of crystal
        structures and orientations.

        Each structure in the structure library is rotated to each associated
        orientation and the diffraction pattern is calculated each time.

        Angles must be in the Euler representation (Z,X,Z) and in degrees

        Parameters
        ----------
        structure_library : pyxem:StructureLibrary Object
            Dictionary of structures and associated orientations for which
            electron diffraction is to be simulated.

        calibration : float
            The calibration of experimental data to be correlated with the
            library, in reciprocal Angstroms per pixel.

        reciprocal_radius : float
            The maximum g-vector magnitude to be included in the simulations.

        half_shape: tuple
            The half shape of the target patterns, for 144x144 use (72,72) etc

        Returns
        -------
        diffraction_library : dict of :class:`DiffractionSimulation`
            Mapping of crystal structure and orientation to diffraction data
            objects.

        """
        # Define DiffractionLibrary object to contain results
        diffraction_library = DiffractionLibrary()
        # The electron diffraction calculator to do simulations
        diffractor = self.electron_diffraction_calculator
        structure_library = structure_library.struct_lib
        # Iterate through phases in library.
        for key in structure_library.keys():
            phase_diffraction_library = dict()
            structure = structure_library[key][0]
            a, b, c = structure.lattice.a, structure.lattice.b, structure.lattice.c
            alpha = structure.lattice.alpha
            beta = structure.lattice.beta
            gamma = structure.lattice.gamma
            orientations = structure_library[key][1]
            # Iterate through orientations of each phase.
            for orientation in tqdm(orientations, leave=False):
                _orientation = np.deg2rad(orientation)
                matrix = euler2mat(_orientation[0],
                                   _orientation[1],
                                   _orientation[2], 'rzxz')

                latt_rot = diffpy.structure.lattice.Lattice(a, b, c,
                                                            alpha, beta, gamma,
                                                            baserot=matrix)
                structure.placeInLattice(latt_rot)

                # Calculate electron diffraction for rotated structure
                data = diffractor.calculate_ed_data(structure,
                                                    reciprocal_radius,
                                                    with_direct_beam)
                # Calibrate simulation
                data.calibration = calibration
                pattern_intensities = data.intensities
                pixel_coordinates = np.rint(
                    data.calibrated_coordinates[:, :2] + half_shape).astype(int)
                # Construct diffraction simulation library, removing those that
                # contain no peaks
                if len(pattern_intensities) > 0:
                    phase_diffraction_library[tuple(orientation)] = \
                        {'Sim': data, 'intensities': pattern_intensities,
                         'pixel_coords': pixel_coordinates,
                         'pattern_norm': np.sqrt(np.dot(pattern_intensities,
                                                        pattern_intensities))}
                    diffraction_library[key] = phase_diffraction_library
        return diffraction_library
