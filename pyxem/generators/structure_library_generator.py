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

import numpy as np

from pyxem.libraries.structure_library import StructureLibrary
from pyxem.utils.sim_utils import rotation_list_stereographic

"""Generating structure libraries."""

# Inverse pole figure corners for crystal systems
stereographic_corners = {
    'cubic': [(0, 0, 1), (1, 0, 1), (1, 1, 1)],
    'hexagonal': [(0, 0, 0, 1), (1, 0, -1, 0), (1, 1, -2, 0)],
    'orthorombic': [(0, 0, 1), (1, 0, 0), (0, 1, 0)],
    'tetragonal': [(0, 0, 1), (1, 0, 0), (1, 1, 0)],
    'trigonal': [(0, 0, 0, 1), (0, -1, 1, 0), (1, -1, 0, 0)],
    'monoclinic': [(0, 0, 1), (0, 1, 0), (0, -1, 0)]
}


class StructureLibraryGenerator:
    """Generates a structure library for the given phases

    Parameters
    ----------
    phases : list
        Array of three-component phase descriptions, where the phase
        description is [<phase name> : string, <structure> :
        diffpy.structure.Structure, <crystal system> : string], and crystal
        system is one of 'cubic', 'hexagonal', 'orthorombic', 'tetragonal',
        'trigonal', 'monoclinic'.

    Attributes
    ----------
    phase_names : list of string
        List of phase names.
    structures : list of diffpy.structure.Structure
        List of structures.
    systems : list of string
        List of crystal systems.

    Examples
    --------
    >>> gen = StructureLibraryGenerator([
    ...     ('ZB', structure_zb, 'cubic'),
    ...     ('WZ', structure_wz, 'hexagonal')])
    """

    def __init__(self, phases):
        self.phase_names = [phase[0] for phase in phases]
        self.structures = [phase[1] for phase in phases]
        self.systems = [phase[2] for phase in phases]

    def get_orientations_from_list(self, orientations):
        """Create a structure library from a list of rotations.

        Parameters
        ----------
        orientations : list
            A list over identifiers of lists of euler angles (as tuples) in the rzxz
            convention and in degrees.

        Returns
        -------
        structure_library : StructureLibrary
            Structure library for the given phase names, structures and orientations.
        """
        return StructureLibrary(self.phase_names, self.structures, orientations)

    def get_orientations_from_stereographic_triangle(self, inplane_rotations, resolution):
        """
        Create a structure library from the stereographic triangles of the
        given crystal systems.

        Parameters
        ----------
        inplane_rotations : list
            List over identifiers of lists of inplane rotations of the
            diffraction patterns, in degrees.
        resolution : float
            Rotation list resolution in degrees.

        Returns
        -------
        structure_library : StructureLibrary
            Structure library for the given phase names, structures and crystal system.
        """
        rotation_lists = [
            rotation_list_stereographic(structure, *stereographic_corners[system],
                                        np.deg2rad(inplane_rotation), np.deg2rad(resolution))
            for phase_name, structure, system, inplane_rotation in
            zip(self.phase_names, self.structures, self.systems, inplane_rotations)]
        return StructureLibrary(self.phase_names, self.structures, rotation_lists)
