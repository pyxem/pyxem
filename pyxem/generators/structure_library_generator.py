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

from pyxem.libraries.structure_library import StructureLibrary
from pyxem.utils.sim_utils import rotation_list_stereographic

"""Generating structure libraries."""

# Inverse pole figure corners for crystal systems
stereographic_corners = {
        'cubic': [ (0, 0, 1), (1, 0, 1), (1, 1, 1) ],
        'hexagonal': [ (0, 0, 0, 1), (1, 0, -1, 0), (1, 1, -2, 0) ],
        'orthorombic': [ (0, 0, 1), (1, 0, 0), (0, 1, 0) ],
        'tetragonal': [ (0, 0, 1), (1, 0, 0), (1, 1, 0) ],
        'trigonal': [ (0, 0, 0, 1), (0, -1, 1, 0), (1, -1, 0, 0) ],
        'monoclinic': [ (0, 0, 1), (0, 1, 0), (0, -1, 0) ]
}


class StructureLibraryGenerator:
    def __init__(self, phases):
        self.phase_names = [phase[0] for phase in phases]
        self.structures = [phase[1] for phase in phases]
        self.systems = [phase[2] for phase in phases]


    def get_orientations_from_list(self, rotation_lists):
        return StructureLibrary(self.phase_names, self.structures, rotation_lists)


    def get_orientations_from_stereographic_triangle(self, inplane_rotations, resolution):
        rotation_lists = [
                rotation_list_stereographic(structure, *stereographic_corners[system], inplane_rotation, resolution)
                    for phase_name, structure, system, inplane_rotation in
                        zip(self.phase_names, self.structures, self.systems, inplane_rotations)]
        return StructureLibrary(self.phase_names, self.structures, rotation_lists)
