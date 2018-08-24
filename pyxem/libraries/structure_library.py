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
import pickle

def load_StructureLibrary(filename, safety=False):
    if safety:
        with open(filename, 'rb') as handle:
            return pickle.load(handle)
    else:
        raise RuntimeError('Unpickling is risky, turn safety to True if \
        trust the author of this content')


class StructureLibrary(dict):
    """Maps phase identifiers to crystal structure and corresponding expected or
    allowed orientations.

    """

    def __init__(self,
                 identifiers,
                 structures,
                 orientations,
                 representation='euler',
                 *args, **kwargs):
        """

        """
        self.representation = representation



    def pickle_library(self,filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
