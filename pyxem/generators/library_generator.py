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

import itertools
import math

import numpy as np
from tqdm import tqdm
from transforms3d.euler import euler2mat

from pyxem.libraries.diffraction_library import DiffractionLibrary
from pyxem.libraries.vector_library import DiffractionVectorLibrary

from pyxem.utils.sim_utils import get_points_in_sphere
from pyxem.utils.sim_utils import simulate_rotated_structure
from pyxem.utils.vector_utils import get_angle_cartesian_vec


class DiffractionLibraryGenerator:
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


class VectorLibraryGenerator:
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
            Mapping of phase identifier to phase information in dictionary
            format.
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
            miller_indices, coordinates, distances = get_points_in_sphere(
                recip_latt,
                reciprocal_radius)

            # Create pair_indices for selecting all point pair combinations
            num_indices = len(miller_indices)
            pair_a_indices, pair_b_indices = np.mgrid[:num_indices, :num_indices]

            # Only select one of the permutations and don't pair an index with
            # itself (select above diagonal)
            upper_indices = np.triu_indices(num_indices, 1)
            pair_a_indices = pair_a_indices[upper_indices].ravel()
            pair_b_indices = pair_b_indices[upper_indices].ravel()

            # Mask off origin (0, 0, 0)
            origin_index = num_indices // 2
            pair_a_indices = pair_a_indices[pair_a_indices != origin_index]
            pair_b_indices = pair_b_indices[pair_b_indices != origin_index]

            pair_indices = np.vstack([pair_a_indices, pair_b_indices])

            # Create library entries
            angles = get_angle_cartesian_vec(coordinates[pair_a_indices], coordinates[pair_b_indices])
            pair_distances = distances[pair_indices.T]
            # Ensure longest vector is first
            len_sort = np.fliplr(pair_distances.argsort(axis=1))
            # phase_index_pairs is a list of [hkl1, hkl2]
            phase_index_pairs = np.take_along_axis(miller_indices[pair_indices.T], len_sort[:, :, np.newaxis], axis=1)
            # phase_measurements is a list of [len1, len2, angle]
            phase_measurements = np.column_stack((np.take_along_axis(pair_distances, len_sort, axis=1), angles))

            # Only keep unique triplets
            unique_measurements, unique_measurement_indices = np.unique(phase_measurements, axis=0, return_index=True)
            vector_library[phase_name] = {
                'indices': phase_index_pairs[unique_measurement_indices],
                'measurements': unique_measurements
            }

        # Pass attributes to diffraction library from structure library.
        vector_library.identifiers = self.structures.identifiers
        vector_library.structures = self.structures.structures
        vector_library.reciprocal_radius = reciprocal_radius

        return vector_library
