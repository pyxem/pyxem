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
from tqdm import tqdm
from transforms3d.euler import euler2mat
import diffpy.structure

from pyxem.libraries.diffraction_library import DiffractionLibrary
from pyxem.libraries.vector_library import DiffractionVectorLibrary


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
                           reciprocal_radius
                           ):
        """Calculates a library of diffraction vectors and pairwise inter-vector
        angles for a library of crystal structures.

        Each structure in the structure library is rotated to each associated
        orientation and the diffraction pattern is calculated each time.

        Parameters
        ----------
        reciprocal_radius : float
            The maximum g-vector magnitude to be included in the simulations.

        Returns
        -------
        vector_library : :class:`DiffractionVectorLibrary`
            Mapping of crystal structure and orientation to diffraction data
            objects.

        """
        rl = structure.lattice.reciprocal_lattice_crystallographic
        recip_pts = rl.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], max_length)
        calc_peaks = np.asarray(sorted(recip_pts, key=lambda i: (i[1], -i[0][0], -i[0][1], -i[0][2])))
        #convert array of unique vectors to polar coordinates
        gpolar = np.array(_cart2polar(vectors.data.T[0], vectors.data.T[1]))
        #iterate through all pairs calculating theoretical interplanar angle
        for comb in itertools.combinations(np.arange(len(vectors)), 2):
            i, j = comb[0], comb[1]
            #get hkl values for all planes in indexed family
            hkls1 = calc_peaks.T[0][np.where(np.isin(calc_peaks.T[1], indexation[i][1][1]))]
            hkls2 = calc_peaks.T[0][np.where(np.isin(calc_peaks.T[1], indexation[j][1][1]))]
            #assign empty array for inter-vector angles
            phis = np.zeros((len(hkls1), len(hkls2)))
            #iterate through all pairs of indices
            for prod in itertools.product(np.arange(len(hkls1)), np.arange(len(hkls2))):
                m, n = prod[0], prod[1]
                hkl1, hkl2 = hkls1[m], hkls2[n]
                phis[m,n] = get_interplanar_angle(structure, hkl1, hkl2)

        vector_library = DiffractionVectorLibrary([gpolar,phis])

        return vector_library
