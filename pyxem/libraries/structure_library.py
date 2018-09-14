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

import pyxem as pxm

class StructureLibrary(dict):
    """
    A dictionary containing all the structures and their associated rotations
    in the .struct_lib attribute.

    identifiers: list of strings/ints

    structures: list of diffpy.structure.Structures

    orientations: a list (over identifiers)
    of lists of euler angles (as tuples) in rzxz conventions
    """

    def __init__(self,
                 identifiers,
                 structures,
                 orientations):

        self.identifiers = identifiers
        self.structures  = structures
        self.orientations = orientations
        # create the actual dictionary
        self.struct_lib = dict()
        for ident,struct,ori in zip(identifiers,structures,orientations):
                self.struct_lib[ident] = (struct,ori)
