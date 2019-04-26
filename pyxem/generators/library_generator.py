# -*- coding: utf-8 -*-
# Copyright 2017-2019 The pyXem developers
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
import itertools
from tqdm import tqdm
from transforms3d.euler import euler2mat
import diffpy.structure

from pyxem.libraries.diffraction_library import DiffractionLibrary
from pyxem.libraries.vector_library import DiffractionVectorLibrary

from pyxem.utils.sim_utils import get_points_in_sphere
from pyxem.utils.sim_utils import simulate_rotated_structure
from pyxem.utils.vector_utils import get_angle_cartesian


class DiffractionLibraryGenerator(object):
    """Computes a library of electron diffraction patterns for specified atomic
    structures and orientations.
    """

    def __init__(self, electron_diffraction_calculator):
        """Initialises the generator with a diffraction calculator.

        Parameters
        ----------
        electron_diffraction_calculator : :class:`DiffractionGenerator`
            The calculator used to simulate diffraction patterns.
        """
        self.electron_diffraction_calculator = electron_diffraction_calculator

    def get_diffraction_library(self,
                                structure_library,
                                calibration,
                                reciprocal_radius,
                                half_shape,
                                with_direct_beam=True):
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
        half_shape : tuple
            The half shape of the target patterns, for 144x144 use (72,72) etc
        with_direct_beam : bool
            Include the direct beam in the library.

        Returns
        -------
        diffraction_library : :class:`DiffractionLibrary`
            Mapping of crystal structure and orientation to diffraction data
            objects.

        """
        # Define DiffractionLibrary object to contain results
        diffraction_library = DiffractionLibrary()
        # The electron diffraction calculator to do simulations
        diffractor = self.electron_diffraction_calculator
        # Iterate through phases in library.
        for phase_name in structure_library.struct_lib.keys():
            phase_diffraction_library = dict()
            structure = structure_library.struct_lib[phase_name][0]
            orientations = structure_library.struct_lib[phase_name][1]

            num_orientations = len(orientations)
            simulations = np.empty(num_orientations, dtype='object')
            pixel_coords = np.empty(num_orientations, dtype='object')
            intensities = np.empty(num_orientations, dtype='object')
            # Iterate through orientations of each phase.
            for i, orientation in enumerate(tqdm(orientations, leave=False)):
                matrix = euler2mat(*np.deg2rad(orientation), 'rzxz')
                simulation = simulate_rotated_structure(diffractor, structure, matrix, reciprocal_radius, with_direct_beam)

                # Calibrate simulation
                simulation.calibration = calibration
                pixel_coordinates = np.rint(
                    simulation.calibrated_coordinates[:, :2] + half_shape).astype(int)

                # Construct diffraction simulation library
                simulations[i] = simulation
                pixel_coords[i] = pixel_coordinates
                intensities[i] = simulation.intensities

            diffraction_library[phase_name] = {
                'simulations': simulations,
                'orientations': orientations,
                'pixel_coords': pixel_coords,
                'intensities': intensities,
            }

        # Pass attributes to diffraction library from structure library.
        diffraction_library.identifiers = structure_library.identifiers
        diffraction_library.structures = structure_library.structures
        diffraction_library.diffraction_generator = diffractor
        diffraction_library.reciprocal_radius = reciprocal_radius
        diffraction_library.with_direct_beam = with_direct_beam

        return diffraction_library


class VectorLibraryGenerator(object):
    """Computes a library of diffraction vectors and pairwise inter-vector
    angles for a specified StructureLibrary.
    """

    def __init__(self, structure_library):
        """Initialises the library with a diffraction calculator.

        Parameters
        ----------
        structure_library : :class:`StructureLibrary`
            The StructureLibrary defining structures to be
        """
        self.structures = structure_library

    def get_vector_library(self,
                           reciprocal_radius):
        """Calculates a library of diffraction vectors and pairwise inter-vector
        angles for a library of crystal structures.

        Parameters
        ----------
        reciprocal_radius : float
            The maximum g-vector magnitude to be included in the library.

        Returns
        -------
        vector_library : :class:`DiffractionVectorLibrary`
            Mapping of phase identifier to a numpy array with entries in the
            form: [hkl1, hkl2, len1, len2, angle] ; lengths are in reciprocal
            Angstroms and angles are in radians.

        """
        # Define DiffractionVectorLibrary object to contain results
        vector_library = DiffractionVectorLibrary()
        # Get structures from structure library
        structure_library = self.structures.struct_lib
        # Iterate through phases in library.
        for phase_name in structure_library.keys():
            # Get diffpy.structure object associated with phase
            structure = structure_library[phase_name][0]
            # Get reciprocal lattice points within reciprocal_radius
            recip_latt = structure.lattice.reciprocal()
            indices, coordinates, distances = get_points_in_sphere(
                recip_latt,
                reciprocal_radius)

            # Iterate through all pairs calculating interplanar angle
            phase_vector_pairs = []
            for comb in itertools.combinations(np.arange(len(indices)), 2):
                i, j = comb[0], comb[1]
                # Specify hkls and lengths associated with the crystal structure.
                # TODO: This should be updated to reflect systematic absences
                if np.count_nonzero(coordinates[i]) == 0 or np.count_nonzero(coordinates[j]) == 0:
                    continue  # Ignore combinations including [000]
                hkl1 = indices[i]
                hkl2 = indices[j]
                len1 = distances[i]
                len2 = distances[j]
                if len1 < len2:  # Keep the longest first
                    hkl1, hkl2 = hkl2, hkl1
                    len1, len2 = len1, len2
                angle = get_angle_cartesian(coordinates[i], coordinates[j])
                phase_vector_pairs.append(np.array([hkl1, hkl2, len1, len2, angle]))
            vector_library[phase_name] = np.array(phase_vector_pairs)

        # Pass attributes to diffraction library from structure library.
        vector_library.identifiers = self.structures.identifiers
        vector_library.structures = self.structures.structures

        return vector_library
